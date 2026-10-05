"""
v4 stages on top of v3's cached work/ folder (read only): v4 contexts, stage-2 re-ranker and
cut, then the pair model (model4.py) and reports (evaluate4.py). Everything v4 writes goes to
<work>/v4 (or --v4-dir), with fingerprints of its own chained to v3's block fingerprint, so a
v4 setting or code change never invalidates v3's 6-hour blocking output.
"""
import functools
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import blocking as bk
import context4
import expand4 as ex
import schema
from pipeline import _SRC, fingerprint, load_json, log, meta

CACHE_HELP = (
    "v3's cached work/ does not match these settings, so v3's prepare / block stages would run "
    "again (6+ hours). Check that --data is the SAME path you used for v3 (the fingerprints "
    "include its absolute path), that --work points at v3's work folder, and that you did not "
    "change any v3 setting (k, keep_max, roles ...). If you are sure the cached files are the "
    "right ones, add --trust-v3-cache; to rebuild them on purpose, add --allow-reblock.")

STAGE4_KEYS = {
    "context4": (["df_cap_frac", "min_df_cap", "fallback_share", "n_fallback", "n_always"],
                 ["views4.py", "context4.py", "features.py"]),
    "dense": (["dense", "dense_model", "dense_k", "dense_epochs", "dense_pairs"], ["dense.py"]),
    "stage2": (["s1_keep_min", "s1_keep_max", "s1_floor", "bigkey", "big_cap", "big_top", "k4",
                "src_desp", "src_loose", "src_nlchan", "src_dschan", "s2_sample", "s2_rounds",
                "s2_depth"], ["expand4.py", "stage2.py"]),
    "cut": (["s2_keep_min", "s2_keep_max", "s2_floor", "s2_target", "addr_rescue", "rescue_min_cos",
             "s2_keep_noaddr"], []),
    "train": (["model_s1", "max_rounds", "early_stop", "n_folds", "tune_density", "eta", "max_depth"],
              ["features4.py", "model4.py", "decision.py"]),
    "cross": (["cross", "cross_model", "cross_pairs", "cross_epochs", "cross_lo", "cross_hi",
               "cross_margin"], ["cross.py"]),
}


def v4dir(cfg, *parts):
    p = os.path.join(cfg.v4_dir or os.path.join(cfg.work_dir, "v4"), *parts)
    os.makedirs(p, exist_ok=True)
    return p


def fp4(cfg, stage):
    h = hashlib.sha1(fingerprint(cfg, "block").encode())
    order = list(STAGE4_KEYS)
    for st in order[: order.index(stage) + 1]:
        keys, files = STAGE4_KEYS[st]
        vals = {k: getattr(cfg, k) for k in keys}
        if (st == "dense" and cfg.dense == "off") or (st == "cross" and cfg.cross == "off"):
            vals, files = {st: "off"}, []           # an optional stage that is off changes nothing
        h.update(json.dumps(vals, sort_keys=True, default=list).encode())
        for f in files:
            with open(os.path.join(_SRC, f), "rb") as fh:
                h.update(hashlib.sha1(fh.read()).digest())
        if st in ("stage2", "train"):
            h.update(cfg.xgb_device.encode())
    return h.hexdigest()[:16]


def fp_ok(cfg, path, stage):
    if cfg.force or not os.path.exists(path):
        return False
    return load_json(path).get("fp") == fp4(cfg, stage)


def write_fp(path, cfg, stage, **extra):
    bk.save_json(path, {"fp": fp4(cfg, stage), **extra})


# ---------------------------------------------------------------------------
# v3 cache check
# ---------------------------------------------------------------------------

def v3_cache_state(cfg, verbose=False):
    """'ok' when v3's prepare, contexts, candidates and ranker are cached for these settings,
    'stale' when a cached file exists with another fingerprint, 'missing' when files are only
    missing (v3's stages then compute just those)."""
    problems, found = [], False
    for split in ("train", "test"):
        mp_ = os.path.join(cfg.work_dir, split, "meta.json")
        if not os.path.exists(mp_):
            problems.append(f"{split}/meta.json missing")
            continue
        if load_json(mp_).get("fp") != fingerprint(cfg, "prepare"):
            found = True
            problems.append(f"{split}/meta.json fingerprint differs (prepare settings or --data path)")
        for part in load_json(mp_)["partitions"]:
            for sub, stage in (("ctx", "context"), ("cand", "block")):
                f = os.path.join(cfg.work_dir, split, sub, part["name"], "_fp.json")
                if not os.path.exists(f):
                    problems.append(f"{split}/{sub}/{part['name']} missing")
                elif load_json(f).get("fp") != fingerprint(cfg, stage):
                    found = True
                    problems.append(f"{split}/{sub}/{part['name']} fingerprint differs")
    rj = os.path.join(cfg.work_dir, "train", "ranker", "ranker.json")
    if not os.path.exists(rj):
        problems.append("train/ranker/ranker.json missing")
    elif load_json(rj).get("fp") != fingerprint(cfg, "block"):
        found = True
        problems.append("candidate ranker fingerprint differs")
    if verbose and problems:
        log("v3 cache: " + "; ".join(problems[:8]) + (" ..." if len(problems) > 8 else ""))
    if not problems:
        return "ok"
    return "stale" if found else "missing"


def require_v3(cfg):
    st = v3_cache_state(cfg, verbose=True)
    if st == "ok":
        return
    if cfg.trust_v3_cache:
        log("v3 cache: fingerprints differ, using the cached files anyway (--trust-v3-cache)")
        return
    raise SystemExit(CACHE_HELP if st == "stale" else
                     "v3's prepare / block output is missing: run --stage prepare and --stage block "
                     "first (or point --work at the v3 work folder)")


# ---------------------------------------------------------------------------
# Worker settings and pools
# ---------------------------------------------------------------------------

def worker_cfg(cfg, s2_floor=None):
    d = {k: getattr(cfg, k) for k in ("block_cap", "big_cap", "big_top", "bigkey", "k4", "src_desp",
                                      "src_loose", "src_nlchan", "src_dschan", "s1_keep_min",
                                      "s1_keep_max", "s1_floor", "s2_keep_min", "s2_keep_max",
                                      "addr_rescue", "rescue_min_cos", "s2_keep_noaddr")}
    d["s2_floor"] = float(s2_floor if s2_floor is not None else -np.inf)
    return d


def dirs(cfg, split, part):
    return (os.path.join(cfg.work_dir, split, "ctx", part["name"]),
            v4dir(cfg, split, "ctx4", part["name"]),
            os.path.join(cfg.work_dir, split, "cand", part["name"]),
            v4dir(cfg, split, "strings", part["name"]))


def ranker_path(cfg):
    info = load_json(os.path.join(cfg.work_dir, "train", "ranker", "ranker.json"))
    return os.path.join(cfg.work_dir, "train", "ranker", info["model"])


def dense_dir(cfg, split, part):
    if cfg.dense == "off":
        return None
    return os.path.join(cfg.v4_dir or os.path.join(cfg.work_dir, "v4"), split, "dense", part["name"])


def run_pool(n_procs, init_args, func, tasks, label, log_every=10, init=None):
    """Runs func over tasks in spawned workers initialized with expand4.init_worker(*init_args)
    (or init(*init_args)). Yields results in completion order."""
    init = init or ex.init_worker
    t0 = time.time()
    if n_procs <= 1 or len(tasks) <= 1:
        init(*init_args)
        it, pool = map(func, tasks), None
    else:
        pool = mp.get_context("spawn").Pool(min(n_procs, len(tasks)), initializer=init, initargs=init_args)
        it = pool.imap_unordered(func, tasks)
    try:
        for i, r in enumerate(it):
            yield r
            if (i + 1) % max(1, len(tasks) // log_every) == 0 or i + 1 == len(tasks):
                el = time.time() - t0
                log(f"    {label}: {i + 1}/{len(tasks)} chunks ({el / 60:.1f} min, "
                    f"~{el * (len(tasks) - i - 1) / (i + 1) / 60:.0f} min left)")
    finally:
        if pool is not None:
            pool.close()
            pool.join()


# ---------------------------------------------------------------------------
# Stage context4
# ---------------------------------------------------------------------------

def build_context4(cfg):
    ccfg = {k: getattr(cfg, k) for k in ("df_cap_frac", "min_df_cap", "fallback_share", "n_fallback",
                                         "n_always")}
    for split in ("train", "test"):
        for part in meta(cfg, split)["partitions"]:
            _, c4d, _, sd = dirs(cfg, split, part)
            fpp = os.path.join(c4d, "_fp.json")
            if fp_ok(cfg, fpp, "context4") and os.path.exists(os.path.join(sd, "q.arrow")):
                continue
            shutil.rmtree(c4d, ignore_errors=True)
            shutil.rmtree(sd, ignore_errors=True)
            log(f"context4 {split}/{part['name']}: {part['n_s1']:,} S1 x {part['n_q']:,} records")
            context4.build(os.path.join(cfg.work_dir, split, "norm"), part["name"], c4d, sd, ccfg,
                           cfg.jobs, log)
            write_fp(fpp, cfg, "context4")


# ---------------------------------------------------------------------------
# Stage stage2
# ---------------------------------------------------------------------------

def excluded_records(cfg):
    """Training-universe records that never train anything: stress-only records and records of
    tune / holdout entities (v3's rule)."""
    p = os.path.join(v4dir(cfg, "labels"), "excluded.npy")
    if os.path.exists(p) and not cfg.force:
        return p
    w = os.path.join(cfg.work_dir, "train")
    gt = np.load(os.path.join(w, "gt.npz"))
    roles = np.load(os.path.join(w, "roles.npy"))
    ex_ = pd.read_parquet(os.path.join(w, "ids_q.parquet"), columns=["extra"])["extra"].to_numpy().copy()
    ex_[gt["q"][roles[gt["s"]] >= 2]] = True
    np.save(p, ex_)
    return p


def _set_sample(W, excluded_path, q_offset):
    W["excluded"] = np.load(excluded_path, mmap_mode="r")
    W["q_offset"] = q_offset


def fit_stage2(cfg):
    d = v4dir(cfg, "stage2")
    info_path = os.path.join(d, "stage2.json")
    if fp_ok(cfg, info_path, "stage2"):
        return load_json(info_path)
    import stage2 as s2m
    excl = excluded_records(cfg)
    parts = [p for p in meta(cfg, "train")["partitions"] if p["label"] != "*"]
    n_all = sum(p["n_q"] for p in parts)
    rng = np.random.default_rng(cfg.seed + 21)
    rp = ranker_path(cfg)
    wcfg = worker_cfg(cfg)
    got = {"q": [], "X": [], "y": [], "src4": []}
    qoff = 0
    log(f"stage 2: expanding up to {cfg.s2_sample:,} sampled training records")
    for part in parts:
        ctxd, c4d, cand, _ = dirs(cfg, "train", part)
        chunks = ex.chunks_of(cand)
        n_take = int(round(cfg.s2_sample * part["n_q"] / max(n_all, 1)))
        per = max(1, part["n_q"] // max(len(chunks), 1))
        pick = rng.permutation(len(chunks))[: max(1, -(-n_take // per))]
        tasks = [chunks[i] for i in sorted(pick)]
        init = (ctxd, c4d, wcfg, rp, None, dense_dir(cfg, "train", part),
                functools.partial(_set_sample, excluded_path=excl, q_offset=qoff))
        for r in run_pool(cfg.procs, init, ex.sample_task, tasks, f"stage-2 sample {part['name']}"):
            for k in got:
                got[k].append(r[k])
        qoff += int(part["n_q"])
    q = np.concatenate(got["q"])
    order = np.argsort(q, kind="stable")
    q = q[order]
    X = np.concatenate(got["X"])[order]
    y = np.concatenate(got["y"]).astype(bool)[order]
    src4 = np.concatenate(got["src4"])[order]
    del got
    n_rec = len(np.unique(q))
    new = (src4 & ex.BIT4["v3"]) == 0
    log(f"  sampled {n_rec:,} records: {len(q) / max(n_rec, 1):.1f} candidates per record "
        f"({new.mean():.1%} new), {int(y.sum()):,} true pairs, {int(y[new].sum()):,} of them only "
        f"from new sources")
    fit_half = (q * 2654435761 % 1000) < 500
    path = os.path.join(d, "stage2_model.json")
    bst, imp = s2m.fit(X[fit_half], y[fit_half], q[fit_half], cfg.jobs, cfg.s2_rounds, cfg.s2_depth,
                       cfg.seed, path, cfg.xgb_device, log)
    ho = ~fit_half
    s2 = bst.inplace_predict(X[ho]).astype(np.float32)
    cur = s2m.curve(q[ho], s2, y[ho], X[ho], cfg.s2_keep_min, cfg.addr_rescue, cfg.rescue_min_cos)
    if cfg.s2_floor >= 0:
        floor = cfg.s2_floor
        row = cur[(cur.keep_max == cfg.s2_keep_max)].iloc[(cur[cur.keep_max == cfg.s2_keep_max].floor - floor).abs().argmin()]
    else:
        floor, row = s2m.choose(cur, cfg.s2_keep_max, cfg.s2_target)
    rp_col = ex.S2_FEATS.index("rank_pos")
    v3c = s2m.v3_curve(q[ho], X[ho][:, rp_col], y[ho], cfg.keep_min)
    piv = cur.pivot(index="floor", columns="keep_max", values="true_kept")
    ppr = cur.pivot(index="floor", columns="keep_max", values="pairs_per_record")
    log("  stage-2 cut on held-out training records: share of the expanded set's true pairs kept\n"
        + piv.round(5).to_string() + "\n  pairs per record\n" + ppr.round(1).to_string()
        + "\n  the same records cut by v3's stage-1 rank instead\n" + v3c.round(5).to_string(index=False))
    log(f"  cut: keep_min {cfg.s2_keep_min}, keep_max {cfg.s2_keep_max}, floor {floor:.4g}, address "
        f"rescue {cfg.addr_rescue}: keeps {row.true_kept:.5f} of the expanded true pairs at "
        f"{row.pairs_per_record:.1f} per record")
    info = {"model": os.path.basename(path), "floor": float(floor if np.isfinite(floor) else -1e30),
            "held_out_true_kept": float(row.true_kept), "held_out_pairs_per_record": float(row.pairs_per_record),
            "sample_records": int(n_rec), "sample_pairs_per_record": float(len(q) / max(n_rec, 1)),
            "sample_new_share": float(new.mean()), "sample_true_new_only": int(y[new].sum()),
            "importance_gain": imp, "curve": cur.replace([np.inf, -np.inf], -1e30).to_dict(orient="records"),
            "v3_curve": v3c.to_dict(orient="records")}
    write_fp(info_path, cfg, "stage2", **info)
    return load_json(info_path)


def stage2_info(cfg):
    p = os.path.join(v4dir(cfg, "stage2"), "stage2.json")
    if not fp_ok(cfg, p, "stage2"):
        raise SystemExit("no stage-2 model for these settings: run --stage stage2 first")
    info = load_json(p)
    info["path"] = os.path.join(v4dir(cfg, "stage2"), info["model"])
    return info


# ---------------------------------------------------------------------------
# Stage cut (training universe; test is cut inside predict)
# ---------------------------------------------------------------------------

def run_cut(cfg, split="train"):
    info = stage2_info(cfg)
    wcfg = worker_cfg(cfg, info["floor"])
    rp = ranker_path(cfg)
    for part in meta(cfg, split)["partitions"]:
        ctxd, c4d, cand, _ = dirs(cfg, split, part)
        out = v4dir(cfg, split, "cut", part["name"])
        fpp = os.path.join(out, "_fp.json")
        if fp_ok(cfg, fpp, "cut"):
            log(f"cut {split}/{part['name']}: cached")
            continue
        partial = os.path.join(out, "_fp.partial.json")
        if not (os.path.exists(partial) and load_json(partial).get("fp") == fp4(cfg, "cut") and not cfg.force):
            shutil.rmtree(out, ignore_errors=True)
            os.makedirs(out)
            write_fp(partial, cfg, "cut")
        tasks = []
        for pairs, rec in ex.chunks_of(cand):
            prefix = os.path.join(out, os.path.basename(rec)[:-len(".rec.parquet")])
            if not os.path.exists(prefix + ".done"):
                tasks.append((pairs, rec, prefix))
        log(f"cut {split}/{part['name']}: {len(tasks)} chunks to do")
        tot = {"rows": 0, "expanded": 0, "new": 0, "kept": 0, "seconds_expand": 0.0, "seconds": 0.0}
        init = (ctxd, c4d, wcfg, rp, info["path"], dense_dir(cfg, split, part))
        t0 = time.time()
        for st in run_pool(cfg.procs, init, ex.cut_task, tasks, f"cut {split}/{part['name']}"):
            for k in tot:
                tot[k] += st[k]
        if tot["rows"]:
            el = time.time() - t0
            log(f"    {tot['rows']:,} records: expanded {tot['expanded'] / tot['rows']:.1f} per record "
                f"({tot['new'] / tot['rows']:.2f} new), kept {tot['kept'] / tot['rows']:.1f}; "
                f"{tot['rows'] / max(el, 1e-9):,.0f} records/s; expansion {tot['seconds_expand'] / max(tot['seconds'], 1e-9):.0%} of worker time")
        os.replace(partial, fpp)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def run_stage(cfg, stage):
    require_v3(cfg)
    if stage == "context4":
        build_context4(cfg)
    elif stage == "dense":
        if cfg.dense == "off":
            log("dense: off (enable with --dense pretrained or --dense finetune)")
        else:
            import dense
            dense.build(cfg)
    elif stage == "stage2":
        fit_stage2(cfg)
    elif stage == "cut":
        run_cut(cfg, "train")
    elif stage == "report4":
        import evaluate4
        evaluate4.cut_report(cfg)
    elif stage == "train":
        import model4
        model4.train(cfg)
    elif stage == "cross":
        if cfg.cross == "off":
            log("cross: off (enable with --cross finetune)")
        else:
            import cross
            cross.build(cfg)
    elif stage == "evaluate":
        import evaluate4
        evaluate4.final_report(cfg)
    elif stage == "predict":
        import model4
        model4.predict(cfg)
    else:
        raise ValueError(stage)
