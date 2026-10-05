"""
Pair model (v4): features of the stage-2 kept pairs, grouped-fold XGBoost, calibration and
decision rule at test density, and a single fused pass over the test records.

train
    1. training entities: v3's TRAIN S1s plus background S1s sampled up to --model-s1 (v3 used
       100,000; the background entities' records were already blocked and cut, so this only
       adds training pairs)
    2. training pairs: kept pairs whose S1 is a training entity, from records that are neither
       stress-only nor records of a tune / holdout entity
    3. one XGBoost model per fold, folds grouped by S1 (GPU when available)
    4. every record with a tune or holdout S1 among its kept candidates is scored (fold mean; a
       training pair gets its own fold's model) and keeps its best S1
    5. calibration and the rule on TUNE entities at --tune-density (stress by default: the
       extra look-alike records bring the decoy share up to test's)
predict
    for each test chunk, in one worker: v4 expansion, stage-2 cut, pair features, fold-mean
    score, best S1 per record; then the frozen decision, both submission files, the self-check
    and the test / France diagnostics.
"""
import functools
import json
import os
import shutil
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import blocking as bk
import decision as dc
import expand4 as ex
import features4 as f4
import schema
from pipeline import load_json, log, meta
from pipeline4 import (dense_dir, dirs, fp4, fp_ok, ranker_path, run_pool, stage2_info, v4dir,
                       worker_cfg, write_fp)

DIAG = ["nm_tset", "nm_full_ratio", "ad_tset", "cos_name", "cos_addr", "rank_pos", "s2_rank",
        "lo_tset", "ac_tset"]
GPU_PROCS = 8


def xgb_params(cfg):
    return dict(objective="binary:logistic", eval_metric="logloss", max_depth=cfg.max_depth,
                eta=cfg.eta, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                tree_method="hist", max_bin=256)


def fold_of(sg, n_folds):
    h = (np.asarray(sg, np.uint64) * np.uint64(2654435761)) & np.uint64(0xFFFFFFFF)
    return (h % np.uint64(n_folds)).astype(np.int8)


# ---------------------------------------------------------------------------
# 1. Label arrays and workers
# ---------------------------------------------------------------------------

def arrays(cfg):
    """Global arrays of the training universe, memory-mapped by workers."""
    d = v4dir(cfg, "labels")
    fp = os.path.join(d, "_fp_model.json")
    if fp_ok(cfg, fp, "train") and os.path.exists(os.path.join(d, "trainset.npy")):
        return d
    w = os.path.join(cfg.work_dir, "train")
    gt = np.load(os.path.join(w, "gt.npz"))
    roles = np.load(os.path.join(w, "roles.npy"))
    extra = pd.read_parquet(os.path.join(w, "ids_q.parquet"), columns=["extra"])["extra"].to_numpy()
    excl = np.zeros(len(extra), bool)
    excl[gt["q"][roles[gt["s"]] >= 2]] = True
    trainset = roles == 1
    bg = np.flatnonzero(roles == 0)
    n_more = max(0, min(len(bg), cfg.model_s1 - int(trainset.sum())))
    if n_more:
        trainset[np.random.default_rng(cfg.seed + 31).choice(bg, n_more, replace=False)] = True
    np.save(os.path.join(d, "roles.npy"), roles)
    np.save(os.path.join(d, "extra.npy"), extra.astype(bool))
    np.save(os.path.join(d, "excl.npy"), excl)
    np.save(os.path.join(d, "trainset.npy"), trainset)
    np.save(os.path.join(d, "truth.npy"), bk.sorted_unique(gt["s"].astype(np.int64) * len(extra) + gt["q"]))
    write_fp(fp, cfg, "train", training_entities=int(trainset.sum()))
    return d


M = {}


def _load_boosters(paths, device):
    import xgboost as xgb
    out = []
    for p in paths:
        b = xgb.Booster()
        b.load_model(p)
        b.set_param({"nthread": 1})
        if device == "cuda":
            b.set_param({"device": "cuda"})
        out.append(b)
    return out


def model_init(W, strings_dir, arrays_dir, models, n_folds, device):
    """Extra worker state for scoring (called inside expand4.init_worker, or alone)."""
    W["S"] = f4.open_strings(os.path.join(strings_dir, "s1.arrow"))
    W["Q"] = f4.open_strings(os.path.join(strings_dir, "q.arrow"))
    W["arr"] = ({k: np.load(os.path.join(arrays_dir, f"{k}.npy"), mmap_mode="r")
                 for k in ("roles", "extra", "excl", "truth", "trainset")} if arrays_dir else None)
    W["m1"] = _load_boosters(models, device) if models else []
    W["n_folds"] = n_folds
    W["device"] = device


def init_model_only(strings_dir, arrays_dir, models, n_folds, device):
    schema.pin_native_threads()
    ex.W.clear()
    model_init(ex.W, strings_dir, arrays_dir, models, n_folds, device)


def _predict(bst, X):
    if ex.W.get("device") == "cuda":
        import xgboost as xgb
        return bst.predict(xgb.DMatrix(X, nthread=1))
    return bst.inplace_predict(X)


def _labels(qg, sg):
    A = ex.W["arr"]
    return bk.isin_sorted(sg.astype(np.int64) * len(A["extra"]) + qg, A["truth"])


def _train_mask(qg, sg):
    A = ex.W["arr"]
    return np.asarray(A["trainset"])[sg] & ~np.asarray(A["extra"])[qg] & ~np.asarray(A["excl"])[qg]


def _score(X, qg=None, sg=None):
    models = ex.W["m1"]
    if not len(X):
        return np.empty(0, np.float32)
    p = np.mean([_predict(m, X) for m in models], axis=0).astype(np.float32)
    if qg is not None and ex.W.get("arr") is not None:
        own = _train_mask(qg, sg)
        if own.any():
            fold = fold_of(sg, ex.W["n_folds"])
            for f, m in enumerate(models):
                r = np.flatnonzero(own & (fold == f))
                if len(r):
                    p[r] = _predict(m, X[r])
    return p


def _read_cut(path, rows=None):
    t = pq.read_table(path, columns=f4.PAIR_COLS)
    d = {c: t[c].to_numpy() for c in f4.PAIR_COLS}
    if rows is not None:
        d = {c: v[rows] for c, v in d.items()}
    return d


def _assigned(d, X, p):
    """Best S1 per record with diagnostics of that pair."""
    b = dc.best_per_record(d["qi"], d["sg"], p)
    b = b[np.argsort(d["qi"][b], kind="stable")]
    out = {"qg": d["qg"][b]}
    out["n_cand"] = np.bincount(np.searchsorted(np.unique(d["qi"]), d["qi"])).astype(np.int16)
    out["sg1"] = d["sg"][b]
    out["p1"] = p[b]
    second = bk.group_max(d["qi"], np.where(bk.group_rank(d["qi"], p) == 1, -1.0, p))
    out["second1"] = np.maximum(second[b], 0).astype(np.float32)
    # runners-up for the cross-encoder (cross.py): the next two S1s by probability
    o = np.lexsort((-p, d["qi"]))
    qo = d["qi"][o]
    start = np.flatnonzero(np.r_[True, qo[1:] != qo[:-1]])
    size = np.diff(np.r_[start, len(qo)])
    got = np.zeros(len(start), np.int64)
    for r in (2, 3):
        out[f"sg{r}"] = np.full(len(start), -1, np.int32)
        out[f"p{r}"] = np.zeros(len(start), np.float32)
    for k in range(3):                              # ranks 1..3; skip the chosen best S1
        has = size > k
        idx = o[start[has] + k]
        rows = np.flatnonzero(has)
        other = d["sg"][idx] != out["sg1"][rows]
        rows, idx = rows[other], idx[other]
        for r in (2, 3):
            put = got[rows] == r - 2
            out[f"sg{r}"][rows[put]] = d["sg"][idx[put]]
            out[f"p{r}"][rows[put]] = p[idx[put]]
            got[rows[put]] += 1
            rows, idx = rows[~put], idx[~put]
    for c in DIAG:
        out[c] = X[b, f4.FEATURES.index(c)]
    return out


def task_trainrows(args):
    path, prefix = args
    t = pq.read_table(path, columns=["qg", "sg"])
    rows = np.flatnonzero(_train_mask(t["qg"].to_numpy(), t["sg"].to_numpy()))
    d = _read_cut(path, rows)
    X = f4.pair_features(d, ex.W["S"], ex.W["Q"])
    cols = {"qg": d["qg"], "sg": d["sg"], "y": _labels(d["qg"], d["sg"])}
    for j, c in enumerate(f4.FEATURES):
        cols[c] = X[:, j]
    pq.write_table(pa.table(cols), prefix + ".train.parquet")
    open(prefix + ".train.done", "w").close()
    return {"pairs": len(rows)}


def task_score(args):
    """Records with a tune / holdout S1 among their kept candidates -> best S1 per record."""
    path, prefix = args
    t = pq.read_table(path, columns=["qi", "sg"])
    qi, sg = t["qi"].to_numpy(), t["sg"].to_numpy()
    ev = np.asarray(ex.W["arr"]["roles"])[sg] >= 2
    rows = np.flatnonzero(np.isin(qi, np.unique(qi[ev])))
    d = _read_cut(path, rows)
    X = f4.pair_features(d, ex.W["S"], ex.W["Q"])
    p = _score(X, d["qg"], d["sg"])
    a = _assigned(d, X, p)
    a["y1"] = _labels(a["qg"], a["sg1"])
    a["extra"] = np.asarray(ex.W["arr"]["extra"])[a["qg"]]
    pq.write_table(pa.table(a), prefix + ".asg.parquet")
    open(prefix + ".asg.done", "w").close()
    return {"pairs": len(rows)}


def task_predict(args):
    """Test chunk, fused: expansion -> stage-2 cut -> pair features -> scores."""
    pairs_path, rec_path, prefix = args
    t0 = time.time()
    lo, hi = ex.chunk_range(rec_path)
    e = ex.expand(ex.W, pairs_path, lo, hi)
    s2 = ex.stage2_scores(ex.W, e["X"])
    keep, rank, resc = ex.cut(ex.W, e["q"], e["X"], s2)
    gap = bk.group_max(e["q"], s2) - s2 if len(s2) else s2
    A = ex.W["ctx"]["arr"]
    k = keep
    d = {"qi": e["q"][k], "si": e["s"][k], "qg": np.asarray(A["q_gid"])[e["q"][k]],
         "sg": np.asarray(A["s_gid"])[e["s"][k]], "s2_score": s2[k], "s2_rank": rank[k].astype(np.float32),
         "s2_gap": gap[k].astype(np.float32), "rescued": resc[k]}
    for j, c in enumerate(ex.S2_FEATS):
        d[c] = e["X"][k, j]
    t1 = time.time()
    X = f4.pair_features(d, ex.W["S"], ex.W["Q"])
    t2 = time.time()
    p = _score(X)
    pq.write_table(pa.table({"qg": d["qg"].astype(np.int32), "sg": d["sg"].astype(np.int32), "p": p}),
                   prefix + ".pred.parquet")
    pq.write_table(pa.table(_assigned(d, X, p)) if len(p) else pa.table({"qg": np.empty(0, np.int32)}),
                   prefix + ".asg.parquet")
    n_exp = np.zeros(hi - lo, np.int32)
    if len(e["q"]):
        rq, inv = np.unique(e["q"], return_inverse=True)
        n_exp[rq - lo] = np.bincount(inv)
    pq.write_table(pa.table({"qi": np.arange(lo, hi, dtype=np.int32), "n_expanded": n_exp,
                             "n_kept": np.bincount(e["q"][keep] - lo, minlength=hi - lo).astype(np.int32)}),
                   prefix + ".rec.parquet")
    open(prefix + ".pred.done", "w").close()
    return {"rows": hi - lo, "kept": int(keep.sum()), "expanded": int(len(e["q"])),
            "t_expand": t1 - t0, "t_features": t2 - t1, "t_score": time.time() - t2}


# ---------------------------------------------------------------------------
# 2. Chunk runners
# ---------------------------------------------------------------------------

def _run_cut_chunks(cfg, out_name, kind, task, models=None):
    """Runs task over the v4 cut chunks of the training universe."""
    arr = arrays(cfg)
    total = 0
    device = cfg.xgb_device if models else "cpu"
    for part in meta(cfg, "train")["partitions"]:
        cut = v4dir(cfg, "train", "cut", part["name"])
        out = v4dir(cfg, "train", out_name, part["name"])
        fpp = os.path.join(out, f"_fp.{kind}.json")
        if not fp_ok(cfg, fpp, "train"):
            for f in os.listdir(out):
                if f".{kind}." in f:
                    os.remove(os.path.join(out, f))
            write_fp(fpp, cfg, "train")
        tasks = []
        for f in sorted(os.listdir(cut)):
            if f.endswith(".pairs.parquet"):
                prefix = os.path.join(out, f[:-len(".pairs.parquet")])
                if not os.path.exists(f"{prefix}.{kind}.done"):
                    tasks.append((os.path.join(cut, f), prefix))
        if not tasks:
            continue
        _, _, _, sd = dirs(cfg, "train", part)
        n = cfg.jobs if device == "cpu" else min(cfg.jobs, GPU_PROCS)
        for r in run_pool(n, (sd, arr, models, cfg.n_folds, device), task, tasks,
                          f"{out_name} {part['name']}", log_every=5, init=init_model_only):
            total += r["pairs"]
    return total


def _outputs(cfg, split, out_name, kind):
    for part in meta(cfg, split)["partitions"]:
        d = os.path.join(v4dir(cfg, split, out_name, part["name"]))
        for f in sorted(x for x in os.listdir(d) if x.endswith(f".{kind}.parquet")):
            yield part, os.path.join(d, f)


def read_outputs(cfg, split, out_name, kind, columns=None):
    tabs = []
    for part, path in _outputs(cfg, split, out_name, kind):
        t = pq.read_table(path, columns=columns)
        if not len(t):
            continue
        lab = pa.DictionaryArray.from_arrays(pa.array(np.zeros(len(t), np.int8)), [part["label"]])
        tabs.append(t.append_column("part", lab))
    if not tabs:
        return None
    return pa.concat_tables(tabs, promote_options="permissive").to_pandas()


def read_matrix(cfg, split, out_name, kind, columns):
    files = [p for _, p in _outputs(cfg, split, out_name, kind)]
    n = sum(pq.ParquetFile(p).metadata.num_rows for p in files)
    X = np.empty((n, len(columns)), np.float32)
    i = 0
    for p in files:
        t = pq.read_table(p, columns=columns)
        for j, c in enumerate(columns):
            X[i:i + len(t), j] = t[c].to_numpy()
        i += len(t)
    return X


# ---------------------------------------------------------------------------
# 3. Training
# ---------------------------------------------------------------------------

def logloss(y, p):
    p = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def train_folds(cfg, X, y, fold, group, names, es_frac=0.08):
    import xgboost as xgb
    device = cfg.xgb_device
    dev = {"device": "cuda"} if device == "cuda" else {}
    oof = np.zeros(len(y), np.float32)
    models = []
    group = np.asarray(group, np.int64)
    es_mask = np.random.default_rng(cfg.seed + 7).random(int(group.max()) + 1)[group] < es_frac
    for f in range(int(fold.max()) + 1):
        tr, va, es = (fold != f) & ~es_mask, fold == f, (fold != f) & es_mask
        dtr = xgb.QuantileDMatrix(X[tr], label=y[tr], nthread=cfg.jobs, feature_names=names)
        des = xgb.QuantileDMatrix(X[es], label=y[es], ref=dtr, nthread=cfg.jobs, feature_names=names)
        t0 = time.time()
        bst = xgb.train({**xgb_params(cfg), **dev, "nthread": cfg.jobs, "seed": cfg.seed + f}, dtr,
                        cfg.max_rounds, evals=[(des, "early_stop")], early_stopping_rounds=cfg.early_stop,
                        verbose_eval=False)
        bst = bst[: bst.best_iteration + 1]
        bst.set_param({"nthread": cfg.jobs})
        t1 = time.time()
        oof[va] = (bst.predict(xgb.DMatrix(X[va], nthread=cfg.jobs)) if device == "cuda"
                   else bst.inplace_predict(X[va]))
        log(f"    fold {f}: {bst.num_boosted_rounds()} trees, out-of-fold logloss "
            f"{logloss(y[va], oof[va]):.4f} on {va.sum():,} pairs ({device}: train {t1 - t0:.0f}s)")
        bst.set_param({"device": "cpu"})
        models.append(bst)
        del dtr, des
    return models, oof


def _importance(models, names):
    imp = pd.DataFrame([m.get_score(importance_type="gain") for m in models]).mean()
    return imp.reindex(names).fillna(0).sort_values(ascending=False)


def model_dir(cfg):
    return v4dir(cfg, "model")


def train(cfg):
    import evaluate4
    ok = evaluate4.gates4(cfg)
    if ok is False and cfg.strict_gates:
        raise SystemExit("train refused: the v4 candidate gates fail (see v4/reports/cut_report.txt) "
                         "and --strict-gates is set")
    md = model_dir(cfg)
    mpath = os.path.join(md, "model.json")
    if not fp_ok(cfg, mpath, "train"):
        log("train: features of the training pairs")
        _run_cut_chunks(cfg, "feat", "train", task_trainrows)
        t = read_outputs(cfg, "train", "feat", "train", ["sg", "y"])
        if t is None or not len(t):
            raise SystemExit("no training pairs")
        X = read_matrix(cfg, "train", "feat", "train", f4.FEATURES)
        y = t["y"].to_numpy().astype(np.float32)
        sg = t["sg"].to_numpy()
        fold = fold_of(sg, cfg.n_folds)
        log(f"train: {len(y):,} pairs of {len(np.unique(sg)):,} training entities, {int(y.sum()):,} true, "
            f"{len(f4.FEATURES)} features, {cfg.n_folds} folds grouped by S1")
        models, oof = train_folds(cfg, X, y, fold, sg, f4.FEATURES)
        info = {"features": f4.FEATURES, "pass1": [], "oof_logloss_pass1": logloss(y, oof),
                "training_pairs": int(len(y)), "training_entities": int(len(np.unique(sg)))}
        for i, m in enumerate(models):
            m.save_model(os.path.join(md, f"pass1_{i}.json"))
            info["pass1"].append(f"pass1_{i}.json")
        imp = _importance(models, f4.FEATURES)
        log("  top features by gain:\n" + imp.head(20).round(1).to_string())
        info["importance_pass1"] = imp.round(3).to_dict()
        del X, t
        write_fp(mpath, cfg, "train", **info)
    info = load_json(mpath)
    models = [os.path.join(md, p) for p in info["pass1"]]
    log("train: scoring every record with a tune or holdout S1 among its kept candidates")
    _run_cut_chunks(cfg, "score", "asg", task_score, models=models)
    tune_decision(cfg)


def label_pairs(cfg, qg, sg):
    t = np.load(os.path.join(arrays(cfg), "truth.npy"), mmap_mode="r")
    n_q = meta(cfg, "train")["n_q"]
    return bk.isin_sorted(np.asarray(sg, np.int64) * n_q + np.asarray(qg, np.int64), t)


def train_asg(cfg, crossed=True):
    """Best S1 per evaluation record; after the cross-encoder when it is on."""
    if crossed:
        import cross
        a = cross.train_asg(cfg)
        if a is not None:
            return a
    return read_outputs(cfg, "train", "score", "asg")


def eval_inputs(cfg):
    w = os.path.join(cfg.work_dir, "train")
    gt = np.load(os.path.join(w, "gt.npz"))
    roles = np.load(os.path.join(w, "roles.npy"))
    return gt["s"].astype(np.int64), gt["q"].astype(np.int64), roles, meta(cfg, "train")["n_q"]


def tune_decision(cfg):
    import cross
    a = train_asg(cfg, crossed=cross.active(cfg) and fp_ok(cfg, os.path.join(cross.cross_dir(cfg), "_fp.json"), "cross"))
    gt_s, gt_q, roles, n_q = eval_inputs(cfg)
    std = a if cfg.tune_density == "stress" else a[~a.extra.to_numpy()]
    tune_ids = np.flatnonzero(roles == 2)
    s, q, p, y = std.sg1.to_numpy(), std.qg.to_numpy(), std.p1.to_numpy(), std.y1.to_numpy()
    on_tune = roles[s] == 2
    cal = dc.fit_calibration(p[on_tune], y[on_tune])
    grid = dc.decision_grid(cal)
    res = dc.tune(s, q, p, gt_s, gt_q, {"tune": tune_ids}, n_q, grid)
    best = int(res.tune.idxmax())
    bt = int(res[res["mode"] == "threshold"].tune.idxmax())
    be = int(res[res["mode"] == "expected_f"].tune.idxmax())
    log(f"decision (tune entities, {cfg.tune_density} density): best {dc.rule_name(grid[best])} "
        f"macro F0.5 {res.tune[best]:.5f}; best threshold {dc.rule_name(grid[bt])} {res.tune[bt]:.5f}; "
        f"best expected-F {dc.rule_name(grid[be])} {res.tune[be]:.5f}")
    out = {"pass1": {"cal": cal, "rule": grid[best], "best_threshold": grid[bt],
                     "best_expected_f": grid[be], "tune_table": res.to_dict(orient="records")},
           "final_pass": 1, "tune_density": cfg.tune_density, "fp": fp4(cfg, "train"),
           "cross": bool("crossed" in a.columns)}
    np.savez(os.path.join(model_dir(cfg), "tune_assigned_pass1.npz"),
             p=dc.apply_calibration(p[on_tune], cal), y=y[on_tune])
    bk.save_json(os.path.join(model_dir(cfg), "decision.json"), out)
    return out


def final_rule(cfg, dec):
    rule = dict(dec["pass1"]["rule"])
    if cfg.decision_mode:
        rule = dict(dec["pass1"]["best_expected_f" if cfg.decision_mode == "expected_f" else "best_threshold"])
    if cfg.min_prob is not None:
        rule["min_prob"] = cfg.min_prob
    return rule


# ---------------------------------------------------------------------------
# 4. Predict (test, fused)
# ---------------------------------------------------------------------------

def _set_model(W, strings_dir, models, n_folds, device):
    model_init(W, strings_dir, None, models, n_folds, device)


def predict(cfg):
    md = model_dir(cfg)
    mpath = os.path.join(md, "model.json")
    dpath = os.path.join(md, "decision.json")
    if not fp_ok(cfg, mpath, "train") or not os.path.exists(dpath):
        raise SystemExit("no model for the current settings: run --stage train first")
    info = load_json(mpath)
    dec = load_json(dpath)
    import cross
    if dec.get("cross", False) != cross.active(cfg):
        raise SystemExit("the tuned decision does not match --cross: run --stage "
                         + ("cross" if cross.active(cfg) else "train") + " first")
    models = [os.path.join(md, p) for p in info["pass1"]]
    s2 = stage2_info(cfg)
    wcfg = worker_cfg(cfg, s2["floor"])
    rp = ranker_path(cfg)
    device = cfg.xgb_device if cfg.predict_gpu else "cpu"
    log("predict: v4 expansion, stage-2 cut, features and scores of every test record")
    for part in meta(cfg, "test")["partitions"]:
        ctxd, c4d, cand, sd = dirs(cfg, "test", part)
        out = v4dir(cfg, "test", "pred", part["name"])
        fpp = os.path.join(out, "_fp.json")
        if not fp_ok(cfg, fpp, "train"):
            shutil.rmtree(out, ignore_errors=True)
            os.makedirs(out)
            write_fp(fpp, cfg, "train")
        tasks = []
        for pairs, rec in ex.chunks_of(cand):
            prefix = os.path.join(out, os.path.basename(rec)[:-len(".rec.parquet")])
            if not os.path.exists(prefix + ".pred.done"):
                tasks.append((pairs, rec, prefix))
        if not tasks:
            log(f"  predict test/{part['name']}: cached")
            continue
        n = cfg.procs if device == "cpu" else min(cfg.procs, GPU_PROCS)
        init = (ctxd, c4d, wcfg, rp, s2["path"], dense_dir(cfg, "test", part),
                functools.partial(_set_model, strings_dir=sd, models=models, n_folds=cfg.n_folds,
                                  device=device))
        tot = {"rows": 0, "kept": 0, "expanded": 0, "t_expand": 0.0, "t_features": 0.0, "t_score": 0.0}
        t0 = time.time()
        for st in run_pool(n, init, task_predict, tasks, f"predict test/{part['name']}"):
            for k_ in tot:
                tot[k_] += st[k_]
        el = time.time() - t0
        tt = max(tot["t_expand"] + tot["t_features"] + tot["t_score"], 1e-9)
        log(f"    {tot['rows']:,} records, {tot['expanded'] / max(tot['rows'], 1):.1f} expanded and "
            f"{tot['kept'] / max(tot['rows'], 1):.1f} kept per record, {tot['rows'] / max(el, 1e-9):,.0f} "
            f"records/s; worker time: expansion {tot['t_expand'] / tt:.0%}, features "
            f"{tot['t_features'] / tt:.0%}, scoring {tot['t_score'] / tt:.0%}")
    rule = final_rule(cfg, dec)
    pr = read_outputs(cfg, "test", "pred", "pred", ["qg", "sg", "p"])
    asg = read_outputs(cfg, "test", "pred", "asg")
    m = meta(cfg, "test")
    if pr is None:
        pr = pd.DataFrame({"qg": np.empty(0, np.int32), "sg": np.empty(0, np.int32), "p": np.empty(0, np.float32)})
    if asg is not None and cross.active(cfg):
        asg = cross.apply_test(cfg, asg)
    if asg is not None:
        ps, pq_, pp = dc.decide_assigned(asg.sg1.to_numpy(), asg.qg.to_numpy(), asg.p1.to_numpy(), rule)
    else:
        ps = pq_ = pp = np.empty(0, np.int64)
    log(f"decision {dc.rule_name(rule)}: {len(ps):,} matches for {len(bk.sorted_unique(ps)):,} of "
        f"{m['n_s1']:,} S1 entities")
    d = os.path.join(cfg.work_dir, "test")
    s1_ids = pd.read_parquet(os.path.join(d, "ids_s1.parquet"))["entity_id"].to_numpy(dtype=object)
    q_ids = pd.read_parquet(os.path.join(d, "ids_q.parquet"))["entity_id"].to_numpy(dtype=object)
    os.makedirs(cfg.out_dir, exist_ok=True)
    mfile = os.path.join(cfg.out_dir, schema.MATCH_FILE)
    cfile = os.path.join(cfg.out_dir, schema.CAND_FILE)
    schema.write_lists(mfile, schema.MATCH_COL, s1_ids, ps, pq_, q_ids, pp)
    schema.write_lists(cfile, schema.CAND_COL, s1_ids, pr.sg.to_numpy(), pr.qg.to_numpy(), q_ids,
                       pr.p.to_numpy())
    log(f"wrote {mfile} and {cfile}")
    issues = schema.check_submission(mfile, cfile, os.path.join(cfg.data_dir, "test"),
                                     smoke=m["smoke_test"], log=log)
    import evaluate4
    evaluate4.test_diagnostics(cfg, asg, rule, ps, pq_, pp)
    return issues
