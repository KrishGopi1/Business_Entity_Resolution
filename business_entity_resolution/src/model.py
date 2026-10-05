"""
Pair model (v3): features, grouped-fold XGBoost, scoring, decision tuning, test prediction.

train
    1. refuses while the blocking gates fail (unless --allow-low-recall)
    2. features of the training pairs: kept candidates whose S1 is a TRAIN entity, from records
       that are not stress-only and do not belong to a tune or holdout entity
    3. one XGBoost model per fold, folds grouped by S1 (out-of-fold scores are honest)
    4. scores every record that has a tune or holdout S1 among its candidates (fold mean; a
       training pair gets its own fold's model, i.e. its out-of-fold score) and keeps each
       record's best S1 (exclusive assignment)
    5. calibration and the decision rule, chosen on TUNE entities only; holdout is not read
   --two-pass (ablation) adds a second model with the first model's score and its rank and
   margins inside the record.
predict
    features and scores of every test candidate, the frozen decision, both submission files,
    the self-check, test / France diagnostics and an audit sample.

Features are recomputed when needed rather than stored: storing them for every candidate would
take tens of GB, computing them takes minutes. Workers are spawned processes with one native
thread each; the partition's strings are memory-mapped Arrow files shared through the page cache.
"""
import json
import multiprocessing as mp
import os
import shutil
import time
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import blocking as bk
import decision as dc
import features as ft
import schema
from pipeline import _dir, _fp_ok, _write_fp, fingerprint, load_json, log, meta

XGB_PARAMS = dict(objective="binary:logistic", eval_metric="logloss", max_depth=8, eta=0.1,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=5, tree_method="hist",
                  max_bin=256)
CTX = ["p1", "ctx_rank_rec", "ctx_margin_rec", "ctx_gap_other"]
DIAG = ["nm_tset", "nm_full_ratio", "ad_tset", "cos_name", "cos_addr", "rank_pos"]
PAIR_COLS = ["qi", "si", "qg", "sg"] + list(bk.RANK_FEATS) + ["rank_score", "rank_pos"]


def fold_of(sg, n_folds):
    h = (np.asarray(sg, np.uint64) * np.uint64(2654435761)) & np.uint64(0xFFFFFFFF)
    return (h % np.uint64(n_folds)).astype(np.int8)


def _model_dir(cfg):
    return _dir(cfg, "model")


# ---------------------------------------------------------------------------
# 1. Workers
# ---------------------------------------------------------------------------

_W = {}


GPU_PROCS = 8          # scoring workers when the pair model runs on the GPU (one CUDA context
                       # each); with prediction on the GPU the CPU string features set the pace


def _load_boosters(paths, device="cpu"):
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


def _booster_predict(bst, X):
    """Booster scores for a float32 matrix, on the booster's device. A CUDA booster gets a
    DMatrix (XGBoost copies it to the GPU once) instead of inplace_predict's host-array
    fallback, which warns on every call."""
    if _W.get("device", "cpu") == "cuda":
        import xgboost as xgb
        return bst.predict(xgb.DMatrix(X, nthread=_W.get("nthread", 1)))
    return bst.inplace_predict(X)


def _init(strings_dir, arrays_dir, models, n_folds, keep=None, device="cpu"):
    schema.pin_native_threads()
    _W["keep"] = keep
    _W["device"] = device
    _W["S"] = ft.open_strings(os.path.join(strings_dir, "s1.arrow"))
    _W["Q"] = ft.open_strings(os.path.join(strings_dir, "q.arrow"))
    _W["arr"] = ({k: np.load(os.path.join(arrays_dir, f"{k}.npy"), mmap_mode="r")
                  for k in ("roles", "extra", "excl", "truth")} if arrays_dir else None)
    _W["m1"] = _load_boosters(models.get("m1", []), device)
    _W["m2"] = _load_boosters(models.get("m2", []), device)
    _W["n_folds"] = n_folds


def _kept(path):
    """Rows of a candidate file that pass the post-hoc keep policy (all rows when off)."""
    k = _W.get("keep")
    n = pq.ParquetFile(path).metadata.num_rows
    if not k:
        return np.ones(n, bool)
    cols = ["rank_pos", "rank_score"]
    rescued = "addr_rescue" in pq.ParquetFile(path).schema_arrow.names
    t = pq.read_table(path, columns=cols + (["addr_rescue"] if rescued else []))
    rp, rs = t["rank_pos"].to_numpy(), t["rank_score"].to_numpy()
    keep_min, km, fl = k
    keep = (rp <= keep_min) | ((rs >= fl) & (rp <= km))
    if rescued:                     # v3-fast: address-rescued pairs survive the post cut too
        keep |= t["addr_rescue"].to_numpy(zero_copy_only=False).astype(bool)
    return keep


def _read_pairs(path, rows=None):
    t = pq.read_table(path, columns=PAIR_COLS)
    d = {c: t[c].to_numpy() for c in PAIR_COLS}
    if rows is not None:
        d = {c: v[rows] for c, v in d.items()}
    return d


def _labels(qg, sg):
    A = _W["arr"]
    n_q = len(A["extra"])
    return bk.isin_sorted(sg.astype(np.int64) * n_q + qg, A["truth"])


def _train_mask(qg, sg):
    """Pairs that train the model: TRAIN-entity S1, record neither stress-only nor linked to a
    tune/holdout entity."""
    A = _W["arr"]
    return (np.asarray(A["roles"])[sg] == 1) & ~np.asarray(A["extra"])[qg] & ~np.asarray(A["excl"])[qg]


def _score(models, X, fold=None, own=None):
    """Mean of the fold models; rows in `own` get their own fold's model (out-of-fold)."""
    if len(X) == 0:
        return np.empty(0, np.float32)
    p = np.mean([_booster_predict(m, X) for m in models], axis=0).astype(np.float32)
    if own is not None and own.any():
        for f, m in enumerate(models):
            r = np.flatnonzero(own & (fold == f))
            if len(r):
                p[r] = _booster_predict(m, X[r])
    return p


def context(q, p1):
    """Pass-2 context from pass-1 scores inside each record (no density-dependent counts)."""
    rank = bk.group_rank(q, p1)
    best = bk.group_max(q, p1)
    rest = np.where(rank == 1, -1.0, p1).astype(np.float32)
    second = bk.group_max(q, rest)
    other = np.where(rank == 1, np.maximum(second, 0), best)
    return np.column_stack([p1, rank.astype(np.float32), best - p1, p1 - other]).astype(np.float32)


def _task_trainrows(args):
    """Features of the training pairs of one chunk -> <prefix>.train.parquet."""
    pairs_path, prefix = args
    t = pq.read_table(pairs_path, columns=["qg", "sg"])
    qg, sg = t["qg"].to_numpy(), t["sg"].to_numpy()
    rows = np.flatnonzero(_train_mask(qg, sg) & _kept(pairs_path))
    d = _read_pairs(pairs_path, rows)
    X = ft.pair_features(d, _W["S"], _W["Q"])
    cols = {"qg": d["qg"], "sg": d["sg"], "y": _labels(d["qg"], d["sg"])}
    for j, c in enumerate(ft.FEATURES):
        cols[c] = X[:, j]
    pq.write_table(pa.table(cols), prefix + ".train.parquet")
    open(prefix + ".train.done", "w").close()
    return len(rows)


def _pass(d, need_p2):
    """Features, pass-1 (and pass-2) scores of the pairs in d."""
    X = ft.pair_features(d, _W["S"], _W["Q"])
    qg, sg = d["qg"], d["sg"]
    own = _train_mask(qg, sg) if _W["arr"] is not None else None
    fold = fold_of(sg, _W["n_folds"]) if own is not None else None
    p1 = _score(_W["m1"], X, fold, own)
    out = {"X": X, "p1": p1, "own": own}
    if need_p2 or _W["m2"]:
        C = context(d["qi"], p1)
        out["C"] = C
        if _W["m2"]:
            out["p2"] = _score(_W["m2"], np.hstack([X, C[:, :]]), fold, own)
    return out


def _task_ctx(args):
    """Pass-2 context of the training pairs of one chunk (same order as .train.parquet)."""
    pairs_path, prefix = args
    t = pq.read_table(pairs_path, columns=["qi", "qg", "sg"])
    qi, qg, sg = t["qi"].to_numpy(), t["qg"].to_numpy(), t["sg"].to_numpy()
    kp = _kept(pairs_path)
    tm = _train_mask(qg, sg) & kp
    rows = np.flatnonzero(kp & np.isin(qi, np.unique(qi[tm])))
    r = _pass(_read_pairs(pairs_path, rows), need_p2=True)
    C = r["C"][tm[rows]]
    pq.write_table(pa.table({c: C[:, j] for j, c in enumerate(CTX)}), prefix + ".ctx.parquet")
    open(prefix + ".ctx.done", "w").close()
    return len(C)


def _assigned(d, r, passes):
    """Best S1 per record for each pass, with diagnostics of the final pass's best pair."""
    out = {}
    X = r["X"]
    fidx = {c: ft.FEATURES.index(c) for c in DIAG}
    for k in passes:
        p = r[f"p{k}"]
        b = dc.best_per_record(d["qi"], d["sg"], p)
        b = b[np.argsort(d["qi"][b], kind="stable")]
        if not out:
            out["qg"] = d["qg"][b]
            n_c = np.bincount(np.searchsorted(np.unique(d["qi"]), d["qi"]))
            out["n_cand"] = n_c.astype(np.int16)
        out[f"sg{k}"] = d["sg"][b]
        out[f"p{k}"] = p[b]
        second = bk.group_max(d["qi"], np.where(bk.group_rank(d["qi"], p) == 1, -1.0, p))
        out[f"second{k}"] = np.maximum(second[b], 0).astype(np.float32)
        if k == passes[-1]:
            for c, j in fidx.items():
                out[c] = X[b, j]
    return out


def _task_score(args):
    """Train split: scores records with a tune/holdout S1 among their candidates and writes each
    record's best S1 (per pass) -> <prefix>.asg.parquet."""
    pairs_path, prefix, passes = args
    t = pq.read_table(pairs_path, columns=["qi", "qg", "sg"])
    qi, sg = t["qi"].to_numpy(), t["sg"].to_numpy()
    kp = _kept(pairs_path)
    rows = np.flatnonzero(kp & np.isin(qi, np.unique(qi[kp & (np.asarray(_W["arr"]["roles"])[sg] >= 2)])))
    d = _read_pairs(pairs_path, rows)
    r = _pass(d, need_p2=False)
    a = _assigned(d, r, passes)
    A = _W["arr"]
    for k in passes:
        a[f"y{k}"] = _labels(a["qg"], a[f"sg{k}"])
    a["extra"] = np.asarray(A["extra"])[a["qg"]]
    pq.write_table(pa.table(a), prefix + ".asg.parquet")
    open(prefix + ".asg.done", "w").close()
    return len(a["qg"])


def _task_predict(args):
    """Test split: scores every candidate -> <prefix>.pred.parquet (qg, sg, p) and each
    record's best S1 with diagnostics -> <prefix>.asg.parquet."""
    pairs_path, prefix, passes = args
    d = _read_pairs(pairs_path, np.flatnonzero(_kept(pairs_path)))
    r = _pass(d, need_p2=False)
    k = passes[-1]
    pq.write_table(pa.table({"qg": d["qg"], "sg": d["sg"], "p": r[f"p{k}"]}), prefix + ".pred.parquet")
    pq.write_table(pa.table(_assigned(d, r, passes)), prefix + ".asg.parquet")
    open(prefix + ".pred.done", "w").close()
    return len(d["qg"])


# ---------------------------------------------------------------------------
# 2. Chunk runner
# ---------------------------------------------------------------------------

def _strings(cfg, split, part):
    """Arrow string files of a partition (built once, reused by every pass)."""
    d = os.path.join(cfg.work_dir, split, "strings", part["name"])
    fp = os.path.join(d, "_fp.json")
    if not _fp_ok(cfg, fp, "prepare") or load_json(fp).get("cols") != list(ft.S1_STR):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
        nd = os.path.join(cfg.work_dir, split, "norm")
        for side, f in (("s1", f"s1_{part['name']}.parquet"), ("q", f"q_{part['name']}.parquet")):
            ft.write_strings(pq.read_table(os.path.join(nd, f), columns=list(ft.SRC)),
                             os.path.join(d, f"{side}.arrow"))
        _write_fp(fp, cfg, "prepare", cols=list(ft.S1_STR))
    return d


def _arrays(cfg):
    """Global label arrays of the training universe for the workers (memory-mapped)."""
    d = _dir(cfg, "train", "labels")
    fp = os.path.join(d, "_fp.json")
    if _fp_ok(cfg, fp, "prepare"):
        return d
    w = os.path.join(cfg.work_dir, "train")
    gt = np.load(os.path.join(w, "gt.npz"))
    roles = np.load(os.path.join(w, "roles.npy"))
    extra = pd.read_parquet(os.path.join(w, "ids_q.parquet"), columns=["extra"])["extra"].to_numpy()
    excl = np.zeros(len(extra), bool)
    excl[gt["q"][roles[gt["s"]] >= 2]] = True
    np.save(os.path.join(d, "roles.npy"), roles)
    np.save(os.path.join(d, "extra.npy"), extra.astype(bool))
    np.save(os.path.join(d, "excl.npy"), excl)
    np.save(os.path.join(d, "truth.npy"),
            bk.sorted_unique(gt["s"].astype(np.int64) * len(extra) + gt["q"]))
    _write_fp(fp, cfg, "prepare")
    return d


def _keep_args(cfg):
    from pipeline import post_active
    if not post_active(cfg):
        return None
    return (cfg.keep_min, cfg.post_keep_max if cfg.post_keep_max > 0 else cfg.keep_max,
            max(cfg.post_floor, 0.0))


def run_chunks(cfg, split, out_name, stage, task, kind, models=None, extra_args=(), label=""):
    """Runs `task` over every candidate chunk of every partition of `split`; outputs go to
    work/<split>/<out_name>/<part>/cXXXXX.<kind>.*; finished chunks are skipped while the
    fingerprint of `stage` is unchanged."""
    arrays = _arrays(cfg) if split == "train" else None
    total = 0
    for part in meta(cfg, split)["partitions"]:
        cand = os.path.join(cfg.work_dir, split, "cand", part["name"])
        if not os.path.isdir(cand):
            continue
        out = os.path.join(cfg.work_dir, split, out_name, part["name"])
        fpp = os.path.join(out, f"_fp.{kind}.json")
        if not _fp_ok(cfg, fpp, stage):
            if os.path.isdir(out):
                for f in os.listdir(out):
                    if f".{kind}." in f:
                        os.remove(os.path.join(out, f))
            os.makedirs(out, exist_ok=True)
            _write_fp(fpp, cfg, stage)
        files = bk.pair_files(cand)
        tasks = []
        for f in files:
            prefix = os.path.join(out, os.path.basename(f).replace(".pairs.parquet", ""))
            if not os.path.exists(f"{prefix}.{kind}.done"):
                tasks.append((f, prefix) + tuple(extra_args))
        if not tasks:
            continue
        sdir = _strings(cfg, split, part)
        t0 = time.time()
        device = cfg.xgb_device if models else "cpu"
        n = min(cfg.jobs if device == "cpu" else min(cfg.jobs, GPU_PROCS), len(tasks))
        init = (sdir, arrays, models or {}, cfg.n_folds, _keep_args(cfg), device)
        if n <= 1:
            _init(*init)
            it = map(task, tasks)
            pool = None
        else:
            pool = mp.get_context("spawn").Pool(n, initializer=_init, initargs=init)
            it = pool.imap_unordered(task, tasks)
        try:
            for i, k in enumerate(it):
                total += k
                if (i + 1) % max(1, len(tasks) // 5) == 0 or i + 1 == len(tasks):
                    log(f"  {label} {split}/{part['name']}: {i + 1}/{len(tasks)} chunks "
                        f"({(time.time() - t0) / 60:.1f} min)")
        finally:
            if pool is not None:
                pool.close()
                pool.join()
    return total


def _output_files(cfg, split, out_name, kind):
    for part in meta(cfg, split)["partitions"]:
        d = os.path.join(cfg.work_dir, split, out_name, part["name"])
        if os.path.isdir(d):
            for f in sorted(f for f in os.listdir(d) if f.endswith(f".{kind}.parquet")):
                yield part, os.path.join(d, f)


def read_outputs(cfg, split, out_name, kind, columns=None):
    """All chunk outputs of one kind as a DataFrame, with the partition label ('part',
    categorical)."""
    tabs = []
    for part, path in _output_files(cfg, split, out_name, kind):
        t = pq.read_table(path, columns=columns)
        lab = pa.DictionaryArray.from_arrays(pa.array(np.zeros(len(t), np.int8)), [part["label"]])
        tabs.append(t.append_column("part", lab))
    if not tabs:
        return None
    return pa.concat_tables(tabs, promote_options="permissive").to_pandas()


def read_matrix(cfg, split, out_name, kind, columns):
    """Float32 matrix of the given columns over all chunk outputs, filled in place (no
    intermediate DataFrame)."""
    files = [p for _, p in _output_files(cfg, split, out_name, kind)]
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
# 3. XGBoost with grouped folds
# ---------------------------------------------------------------------------

def logloss(y, p):
    p = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def train_folds(X, y, fold, group, n_jobs, max_rounds, early_stop, seed, names, es_frac=0.1,
                device="cpu"):
    """One model per fold on the other folds; early stopping on an inner slice of the training
    folds (es_frac of their S1 groups), never on the fold being predicted. device 'cuda'
    trains on the GPU (same parameters; GPU histograms give slightly different trees)."""
    import xgboost as xgb
    dev = {"device": "cuda"} if device == "cuda" else {}
    oof = np.zeros(len(y), np.float32)
    models = []
    group = np.asarray(group, np.int64)
    es_mask = np.random.default_rng(seed + 7).random(int(group.max()) + 1)[group] < es_frac
    for f in range(int(fold.max()) + 1):
        tr, va, es = (fold != f) & ~es_mask, fold == f, (fold != f) & es_mask
        dtr = xgb.QuantileDMatrix(X[tr], label=y[tr], nthread=n_jobs, feature_names=names)
        des = xgb.QuantileDMatrix(X[es], label=y[es], ref=dtr, nthread=n_jobs, feature_names=names)
        t0 = time.time()
        bst = xgb.train({**XGB_PARAMS, **dev, "nthread": n_jobs, "seed": seed + f}, dtr, max_rounds,
                        evals=[(des, "early_stop")], early_stopping_rounds=early_stop,
                        verbose_eval=False)
        bst = bst[: bst.best_iteration + 1]
        bst.set_param({"nthread": n_jobs})
        t1 = time.time()
        oof[va] = (bst.predict(xgb.DMatrix(X[va], nthread=n_jobs)) if device == "cuda"
                   else bst.inplace_predict(X[va]))
        log(f"    fold {f}: {bst.num_boosted_rounds()} trees, out-of-fold logloss "
            f"{logloss(y[va], oof[va]):.4f} on {va.sum():,} pairs ({device}: train {t1 - t0:.0f}s, "
            f"predict {time.time() - t1:.0f}s)")
        models.append(bst)
        del dtr, des
    return models, oof


def _importance(models, names):
    imp = pd.DataFrame([m.get_score(importance_type="gain") for m in models]).mean()
    return imp.reindex(names).fillna(0).sort_values(ascending=False)


def _load_train_rows(cfg):
    t = read_outputs(cfg, "train", "feat", "train", ["sg", "y"])
    if t is None or not len(t):
        raise SystemExit("no training pairs: are there TRAIN entities in the universe?")
    return t, read_matrix(cfg, "train", "feat", "train", ft.FEATURES)


def _load_ctx(cfg):
    return read_matrix(cfg, "train", "feat", "ctx", CTX)


# ---------------------------------------------------------------------------
# 4. Stages
# ---------------------------------------------------------------------------

def train(cfg):
    import evaluate
    ok = evaluate.gates_passed(cfg)
    if ok is not True and not cfg.allow_low_recall:
        why = ("no blocking report for the current blocking settings: run --stage evaluate-blocking"
               if ok is None else "the blocking gates FAIL (see work/reports/blocking_report.txt)")
        raise SystemExit(f"train refused: {why}. Fix blocking first, or pass --allow-low-recall "
                         "to train anyway.")
    if ok is not True:
        log("WARNING: training although the blocking gates do not pass (--allow-low-recall)")
    md = _model_dir(cfg)
    mpath = os.path.join(md, "model.json")
    passes = [1, 2] if cfg.two_pass else [1]
    if not _fp_ok(cfg, mpath, "train"):
        log("train: features of the training pairs")
        run_chunks(cfg, "train", "feat", "features", _task_trainrows, "train", label="features")
        t, X = _load_train_rows(cfg)
        y = t["y"].to_numpy().astype(np.float32)
        sg = t["sg"].to_numpy()
        fold = fold_of(sg, cfg.n_folds)
        log(f"train: {len(y):,} pairs of {len(np.unique(sg)):,} TRAIN entities, {int(y.sum()):,} true, "
            f"{len(ft.FEATURES)} features, {cfg.n_folds} folds grouped by S1")
        m1, oof1 = train_folds(X, y, fold, sg, cfg.jobs, cfg.max_rounds, cfg.early_stop, cfg.seed,
                               ft.FEATURES, device=cfg.xgb_device)
        info = {"features": ft.FEATURES, "pass1": [], "pass2": [], "ctx": CTX,
                "oof_logloss_pass1": logloss(y, oof1)}
        for i, m in enumerate(m1):
            m.save_model(os.path.join(md, f"pass1_{i}.json"))
            info["pass1"].append(f"pass1_{i}.json")
        imp = _importance(m1, ft.FEATURES)
        log("  top features by gain:\n" + imp.head(15).round(1).to_string())
        info["importance_pass1"] = imp.round(3).to_dict()
        if cfg.two_pass:
            log("train: pass-2 context (ablation)")
            run_chunks(cfg, "train", "feat", "train", _task_ctx, "ctx",
                       models={"m1": [os.path.join(md, p) for p in info["pass1"]]}, label="context")
            C = _load_ctx(cfg)
            X2 = np.hstack([X, C])
            names2 = ft.FEATURES + CTX
            m2, oof2 = train_folds(X2, y, fold, sg, cfg.jobs, cfg.max_rounds, cfg.early_stop,
                                   cfg.seed + 100, names2, device=cfg.xgb_device)
            for i, m in enumerate(m2):
                m.save_model(os.path.join(md, f"pass2_{i}.json"))
                info["pass2"].append(f"pass2_{i}.json")
            info["oof_logloss_pass2"] = logloss(y, oof2)
            info["importance_pass2"] = _importance(m2, names2).round(3).to_dict()
            del X2
        del X, t
        _write_fp(mpath, cfg, "train", **info)
    info = load_json(mpath)
    models = {"m1": [os.path.join(md, p) for p in info["pass1"]],
              "m2": [os.path.join(md, p) for p in info["pass2"]]}
    log("train: scoring every record with a tune or holdout S1 among its candidates")
    run_chunks(cfg, "train", "score", "train", _task_score, "asg", models=models,
               extra_args=(passes,), label="score")
    tune_decision(cfg, passes)


def _eval_inputs(cfg):
    w = os.path.join(cfg.work_dir, "train")
    gt = np.load(os.path.join(w, "gt.npz"))
    roles = np.load(os.path.join(w, "roles.npy"))
    n_q = meta(cfg, "train")["n_q"]
    return gt["s"].astype(np.int64), gt["q"].astype(np.int64), roles, n_q


def tune_decision(cfg, passes):
    """Calibration and rule per pass, on TUNE entities at cfg.tune_density: 'stress' (with the
    stress-only records, closer to test density; default) or 'standard'. Holdout is not looked
    at."""
    a = read_outputs(cfg, "train", "score", "asg")
    gt_s, gt_q, roles, n_q = _eval_inputs(cfg)
    std = a if cfg.tune_density == "stress" else a[~a.extra.to_numpy()]
    tune_ids = np.flatnonzero(roles == 2)
    out = {}
    for k in passes:
        s, q, p, y = (std[f"sg{k}"].to_numpy(), std.qg.to_numpy(), std[f"p{k}"].to_numpy(),
                      std[f"y{k}"].to_numpy())
        on_tune = roles[s] == 2
        cal = dc.fit_calibration(p[on_tune], y[on_tune])
        grid = dc.decision_grid(cal)
        res = dc.tune(s, q, p, gt_s, gt_q, {"tune": tune_ids}, n_q, grid)
        best = int(res.tune.idxmax())
        bt = int(res[res["mode"] == "threshold"].tune.idxmax())
        be = int(res[res["mode"] == "expected_f"].tune.idxmax())
        log(f"decision pass {k} (tune entities, {cfg.tune_density} density): best {dc.rule_name(grid[best])} "
            f"macro F0.5 {res.tune[best]:.5f}; best threshold {dc.rule_name(grid[bt])} {res.tune[bt]:.5f}; "
            f"best expected-F {dc.rule_name(grid[be])} {res.tune[be]:.5f}")
        out[f"pass{k}"] = {"cal": cal, "rule": grid[best], "best_threshold": grid[bt], "best_expected_f": grid[be],
                           "tune_table": res.to_dict(orient="records")}
        np.savez(os.path.join(_model_dir(cfg), f"tune_assigned_pass{k}.npz"),
                 p=dc.apply_calibration(p[on_tune], cal), y=y[on_tune])
    out["final_pass"] = passes[-1]
    bk.save_json(os.path.join(_model_dir(cfg), "decision.json"), out)
    return out


def _final_rule(cfg, dec):
    k = dec["final_pass"]
    rule = dict(dec[f"pass{k}"]["rule"])
    if cfg.decision_mode:
        src = dec[f"pass{k}"]["best_expected_f" if cfg.decision_mode == "expected_f" else "best_threshold"]
        rule = dict(src)
    if cfg.min_prob is not None:
        rule["min_prob"] = cfg.min_prob
    return k, rule


def predict(cfg):
    md = _model_dir(cfg)
    mpath = os.path.join(md, "model.json")
    if not _fp_ok(cfg, mpath, "train") or not os.path.exists(os.path.join(md, "decision.json")):
        raise SystemExit("no model for the current settings: run --stage train first")
    info = load_json(mpath)
    dec = load_json(os.path.join(md, "decision.json"))
    passes = [1, 2] if info["pass2"] else [1]
    models = {"m1": [os.path.join(md, p) for p in info["pass1"]],
              "m2": [os.path.join(md, p) for p in info["pass2"]]}
    log("predict: features and scores of every test candidate")
    run_chunks(cfg, "test", "pred", "train", _task_predict, "pred", models=models,
               extra_args=(passes,), label="predict")
    k, rule = _final_rule(cfg, dec)
    pr = read_outputs(cfg, "test", "pred", "pred", ["qg", "sg", "p"])
    asg = read_outputs(cfg, "test", "pred", "asg")
    m = meta(cfg, "test")
    if pr is None:
        pr = pd.DataFrame({"qg": np.empty(0, np.int32), "sg": np.empty(0, np.int32), "p": np.empty(0, np.float32)})
    ps, pq_, pp = dc.decide_assigned(asg[f"sg{k}"].to_numpy(), asg.qg.to_numpy(), asg[f"p{k}"].to_numpy(), rule) \
        if asg is not None else (np.empty(0, np.int64),) * 3
    log(f"decision {dc.rule_name(rule)} (pass {k}): {len(ps):,} matches for "
        f"{len(bk.sorted_unique(ps)):,} of {m['n_s1']:,} S1 entities")
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
    import evaluate
    evaluate.test_diagnostics(cfg, asg, k, rule, ps, pq_, pp)
    return issues


# ---------------------------------------------------------------------------
# 5. Submission zip
# ---------------------------------------------------------------------------

def make_zip(cfg, team):
    """<team>_submission.zip in the package root (the folder holding code/ and
    Documentation_template.md): output/ with both files, code/business_entity_resolution/
    {src/, README.md, requirements.txt}, Documentation_template.md. Caches, tests' data and
    __pycache__ are left out. Not written if the self-check fails."""
    code = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = os.path.dirname(os.path.dirname(code))
    files = {f"output/{n}": os.path.join(cfg.out_dir, n) for n in (schema.MATCH_FILE, schema.CAND_FILE)}
    missing = [p for p in files.values() if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"run the pipeline first; not found: {missing}")
    if schema.check_submission(files[f"output/{schema.MATCH_FILE}"], files[f"output/{schema.CAND_FILE}"],
                               os.path.join(cfg.data_dir, "test"), log=log):
        raise SystemExit("the output files do not pass the check; zip not written")
    for name in ("README.md", "requirements.txt"):
        if os.path.exists(os.path.join(code, name)):
            files[f"code/business_entity_resolution/{name}"] = os.path.join(code, name)
    for sub in ("src", "tests"):
        for dirpath, dirnames, filenames in os.walk(os.path.join(code, sub)):
            dirnames[:] = sorted(x for x in dirnames if x != "__pycache__" and not x.startswith("."))
            for name in sorted(filenames):
                if name.endswith(".py"):
                    full = os.path.join(dirpath, name)
                    files[f"code/business_entity_resolution/{os.path.relpath(full, code)}"] = full
    # trained weights (v4): v3 ranker, stage-2 ranker, pair model and rule, bi- / cross-encoder
    v4 = cfg.v4_dir or os.path.join(cfg.work_dir, "v4")
    for arc, d in (("models/v3_ranker", os.path.join(cfg.work_dir, "train", "ranker")),
                   ("models/stage2", os.path.join(v4, "stage2")), ("models/pair", os.path.join(v4, "model")),
                   ("models/dense", os.path.join(v4, "dense", "model")),
                   ("models/cross", os.path.join(v4, "cross", "model"))):
        if not os.path.isdir(d):
            continue
        for dirpath, _, filenames in os.walk(d):
            for name in sorted(filenames):
                if name.endswith((".json", ".ubj", ".safetensors", ".txt", ".model", ".bin")) \
                        and not name.endswith(".npz"):
                    full = os.path.join(dirpath, name)
                    files[f"{arc}/{os.path.relpath(full, d)}"] = full
    doc = os.path.join(root, "Documentation_template.md")
    if os.path.exists(doc):
        files["Documentation_template.md"] = doc
        if "[run]" in open(doc, encoding="utf-8").read():
            log("warning: Documentation_template.md still has [run] placeholders; fill them in "
                "from work/reports/final_report.txt and zip again")
    else:
        log(f"warning: no Documentation_template.md in {root}")
    path = os.path.join(root, f"{team}_submission.zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for arc, full in files.items():
            z.write(full, arc)
    log(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB, {len(files)} files)")
    return path
