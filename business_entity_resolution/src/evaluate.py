"""
Evaluation (v3): blocking recall report and gates, oracle candidate ceiling, the final
holdout report, density-stress validation and label-free France / test diagnostics.

Populations. Recall and ceiling gates are computed on the true pairs of TUNE + HOLDOUT
entities ("unseen"): the candidate ranker never saw their labels. Numbers for training
entities are shown next to them only as a check for overfitting.
"""
import json
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import blocking as bk
import indexing as ix
from pipeline import load_json, log, meta, post_active, post_keep

KEEP_MAXES = (20, 30, 50, 80, 100, 120, 150, 200)


def _report_dir(cfg):
    d = os.path.join(cfg.work_dir, "reports")
    os.makedirs(d, exist_ok=True)
    return d


def f05_ceiling(n_true, n_reached):
    """Macro-F0.5 terms of a perfect model restricted to the candidates: precision 1, recall
    n_reached / n_true; an entity with no true record scores 1 (predicted empty)."""
    r = n_reached / np.maximum(n_true, 1)
    f = np.where(r > 0, 1.25 * r / (0.25 + r), 0.0)
    return np.where(n_true == 0, 1.0, f)


def _read_all(cfg, split, kind, columns=None):
    tabs = []
    for part in meta(cfg, split)["partitions"]:
        d = os.path.join(cfg.work_dir, split, "cand", part["name"])
        if os.path.isdir(d):
            t = bk.read_parts(d, kind, columns)
            if t is not None:
                df = t.to_pandas()
                df["part"] = part["label"]
                tabs.append(df)
    return pd.concat(tabs, ignore_index=True) if tabs else pd.DataFrame(columns=(columns or []) + ["part"])


def true_pair_table(cfg):
    """One row per true pair of the training universe: role, country, source, segment flags,
    whether the union found it, whether it was kept, which sources found it, ranker rank."""
    d = os.path.join(cfg.work_dir, "train")
    gt = np.load(os.path.join(d, "gt.npz"))
    roles = np.load(os.path.join(d, "roles.npy"))
    ids_s1 = pd.read_parquet(os.path.join(d, "ids_s1.parquet"))
    ids_q = pd.read_parquet(os.path.join(d, "ids_q.parquet"))
    t = pd.DataFrame({"sg": gt["s"].astype(np.int64), "qg": gt["q"].astype(np.int64)})
    t["role"] = roles[t.sg]
    t["country"] = ids_s1["country"].to_numpy()[t.sg]
    t["source"] = np.where(ids_q["src"].to_numpy()[t.qg] == 3, "S3", "S2")
    u = _read_all(cfg, "train", "truth", ["qg", "sg", "src", "rank_score", "rank_pos", "kept"])
    u = u.drop_duplicates(["qg", "sg"])
    t = t.merge(u[["qg", "sg", "src", "rank_score", "rank_pos", "kept"]], on=["qg", "sg"], how="left")
    t["in_union"] = t["src"].notna()
    t["kept"] = t["kept"].fillna(False).astype(bool) & post_keep(
        cfg, t["rank_pos"].fillna(1e9).to_numpy(), t["rank_score"].to_numpy())
    t["src"] = t["src"].fillna(0).astype(np.int64)
    # segment flags from the normalized records
    nd = os.path.join(d, "norm")
    s_tab = pq.read_table(os.path.join(nd, "s1_all.parquet"), columns=["name_tokens_sorted", "name_base"])
    q_tab = pq.read_table(os.path.join(nd, "q_all.parquet"),
                          columns=["name_tokens_sorted", "name_nonlatin", "address_missing"])
    Ms = ix.binary_tokens(s_tab["name_tokens_sorted"].to_pylist(), cfg.jobs)
    Mq = ix.binary_tokens(q_tab["name_tokens_sorted"].to_pylist(), cfg.jobs)
    inter = bk.rowwise_dot(Mq, Ms, t.qg.to_numpy(), t.sg.to_numpy())
    t["seg_zero_name_overlap"] = inter == 0
    t["seg_nonlatin_name"] = q_tab["name_nonlatin"].to_numpy(zero_copy_only=False)[t.qg]
    t["seg_missing_address"] = q_tab["address_missing"].to_numpy(zero_copy_only=False)[t.qg]
    base = pd.Series(s_tab["name_base"].to_pylist()) + "|" + ids_s1["country"].astype(str)
    cnt = base.map(base.value_counts()).to_numpy()
    t["seg_common_name"] = cnt[t.sg] > 20
    n_true = np.bincount(gt["s"], minlength=len(roles))
    t["seg_hub_entity"] = n_true[t.sg] > 30
    return t, roles, n_true


def _post_counts(cfg, rec):
    """Candidates per record under the post-hoc keep policy, from the per-record counts of
    union pairs at or above each floor of the grid (exact when post_floor is on the grid)."""
    if not post_active(cfg) or rec is None or not len(rec):
        return rec
    fl = max(cfg.post_floor, 0.0)
    hit = np.flatnonzero(np.isclose(bk.FLOOR_GRID, fl))
    if not len(hit):
        log(f"post_floor {fl:g} is not on the report grid {list(bk.FLOOR_GRID)}: candidate counts "
            "shown are the blocked ones")
        return rec
    km = cfg.post_keep_max if cfg.post_keep_max > 0 else cfg.keep_max
    nu = rec.n_union.to_numpy()
    per = np.maximum(np.minimum(nu, cfg.keep_min), np.minimum(rec[f"n_ge_{hit[0]}"].to_numpy(), km))
    rec = rec.copy()
    rec["n_kept"] = np.minimum(per, rec.n_kept.to_numpy())
    return rec


def _recall(df):
    return {"true_pairs": int(len(df)),
            "union_recall": float(df.in_union.mean()) if len(df) else np.nan,
            "retained_recall": float(df.kept.mean()) if len(df) else np.nan}


def blocking_report(cfg):
    """Writes work/reports/blocking_report.{txt,json} and blocking_gates.json."""
    t, roles, n_true = true_pair_table(cfg)
    lines, rep = [], {}
    pops = {"unseen (tune+holdout)": t.role >= 2, "training entities": t.role == 1,
            "all universe": np.ones(len(t), bool)}
    tab = pd.DataFrame({k: _recall(t[m]) for k, m in pops.items()}).T
    lines.append("TRUE-PAIR RECALL (union = found by any channel or block; retained = kept "
                 "after the ranker)\n" + tab.round(5).to_string())
    rep["recall"] = tab.to_dict(orient="index")
    un = t[t.role >= 2]

    seg_rows = []
    for name, col in (("country", "country"), ("source", "source")):
        for val, g in un.groupby(col):
            seg_rows.append({"segment": f"{name}={val}", **_recall(g)})
    for c in [c for c in t.columns if c.startswith("seg_")]:
        seg_rows.append({"segment": c[4:], **_recall(un[un[c]])})
    seg = pd.DataFrame(seg_rows)
    lines.append("RECALL BY SEGMENT (unseen true pairs)\n" + seg.round(5).to_string(index=False))
    rep["segments"] = seg.to_dict(orient="records")

    iu = un[un.in_union]
    src_rows = []
    for i, name in enumerate(ix.SOURCES):
        bit = 1 << i
        found = (iu.src & bit) > 0
        only = iu.src == bit
        src_rows.append({"source": name, "found": found.mean(), "only_this": only.mean(),
                         "union_recall_without_it": (un.in_union.sum() - only.sum()) / max(len(un), 1)})
    src = pd.DataFrame(src_rows)
    lines.append("RETRIEVAL SOURCES (share of the unseen true pairs in the union found by each "
                 "source, found ONLY by it, and union recall if it were removed)\n"
                 + src.round(5).to_string(index=False))
    rep["sources"] = src.to_dict(orient="records")

    # keep-policy table: retained recall and pairs per record at other keep_max / floors
    rec = _post_counts(cfg, _read_all(cfg, "train", "rec"))
    ids_q = pd.read_parquet(os.path.join(cfg.work_dir, "train", "ids_q.parquet"), columns=["extra"])
    rinfo = load_json(os.path.join(cfg.work_dir, "train", "ranker", "ranker.json"))
    rows = []
    rank = un.rank_pos.fillna(1e9).to_numpy()
    score = un.rank_score.fillna(-1).to_numpy()
    nu = rec.n_union.to_numpy()
    for km in KEEP_MAXES:
        for ti, fl in enumerate(bk.FLOOR_GRID):
            kept = un.in_union.to_numpy() & ((rank <= cfg.keep_min) | ((score >= fl) & (rank <= km)))
            per = np.maximum(np.minimum(nu, cfg.keep_min), np.minimum(rec[f"n_ge_{ti}"].to_numpy(), km))
            rows.append({"keep_max": km, "floor": fl, "retained_recall": kept.mean(),
                         "pairs_per_record": per.mean()})
    pol = pd.DataFrame(rows)
    lines.append("KEEP POLICY (unseen true pairs retained, and mean candidates per record, for "
                 f"other keep_max / floor values; keep_min {cfg.keep_min}; current keep_max "
                 f"{cfg.keep_max}, floor {rinfo['floor']:g})\nretained recall:\n"
                 + pol.pivot(index="floor", columns="keep_max", values="retained_recall").round(5).to_string()
                 + "\npairs per record:\n"
                 + pol.pivot(index="floor", columns="keep_max", values="pairs_per_record").round(1).to_string())
    rep["keep_policy"] = pol.to_dict(orient="records")

    # oracle candidate ceiling (macro F0.5 of a perfect model on the kept candidates)
    ent = np.flatnonzero(roles >= 2)
    reach_kept = np.bincount(t.sg[t.kept], minlength=len(roles))
    reach_union = np.bincount(t.sg[t.in_union], minlength=len(roles))
    ceil_kept = f05_ceiling(n_true[ent], reach_kept[ent])
    ceil_union = f05_ceiling(n_true[ent], reach_union[ent])
    ids_s1 = pd.read_parquet(os.path.join(cfg.work_dir, "train", "ids_s1.parquet"))
    cdf = pd.DataFrame({"country": ids_s1.country.to_numpy()[ent], "kept": ceil_kept, "union": ceil_union})
    ctab = cdf.groupby("country")[["kept", "union"]].mean()
    ctab.loc["ALL"] = [ceil_kept.mean(), ceil_union.mean()]
    lines.append("ORACLE CANDIDATE CEILING (macro F0.5 of a perfect model on tune+holdout "
                 "entities, singletons included)\n" + ctab.round(5).to_string())
    rep["ceiling"] = {"kept": float(ceil_kept.mean()), "union": float(ceil_union.mean()),
                      "by_country": ctab.to_dict(orient="index")}

    # candidates per record, train (standard and stress records) and test
    ext = ids_q["extra"].to_numpy()
    q_gid = _record_gids(cfg, "train")
    rec["extra"] = ext[q_gid] if len(q_gid) == len(rec) else False
    dist = [_dist(f"train {p}", g) for p, g in rec[~rec.extra].groupby("part")]
    if rec.extra.any():
        dist.append(_dist("train stress-only records", rec[rec.extra]))
    test_rec = _post_counts(cfg, _read_all(cfg, "test", "rec"))
    dist += [_dist(f"test {p}", g) for p, g in test_rec.groupby("part")] if len(test_rec) else []
    dtab = pd.DataFrame(dist)
    lines.append("CANDIDATES PER RECORD (kept, and union before the ranker)\n"
                 + dtab.round(1).to_string(index=False))
    rep["candidates_per_record"] = dtab.to_dict(orient="records")
    m = meta(cfg, "train")
    if m.get("universe"):
        lines.append("DENSITY (S2/S3 records per S1: training universe, with stress records, test)\n"
                     + pd.DataFrame(m["universe"]).T.to_string())

    # gates
    r_un = rep["recall"]["unseen (tune+holdout)"]
    checks = [("union recall", r_un["union_recall"], cfg.gate_union),
              ("retained recall", r_un["retained_recall"], cfg.gate_retained),
              ("oracle candidate ceiling", rep["ceiling"]["kept"], cfg.gate_ceiling)]
    for r in seg_rows:
        if r["true_pairs"] >= cfg.gate_min_segment_pairs:
            checks.append((f"retained recall, {r['segment']}", r["retained_recall"], cfg.gate_segment))
    gate_rows = [{"check": c, "value": v, "required": req, "pass": bool(v >= req)} for c, v, req in checks]
    passed = all(g["pass"] for g in gate_rows)
    if post_active(cfg):
        lines.insert(0, f"KEEP POLICY IN USE: post-hoc keep_max {cfg.post_keep_max or cfg.keep_max}, "
                        f"floor {max(cfg.post_floor, 0):g}, keep_min {cfg.keep_min} (applied to the "
                        "blocked candidates; retained recall, ceiling and counts below use it)")
    lines.insert(0, ("BLOCKING GATES: PASS" if passed else "BLOCKING GATES: FAIL") + "\n"
                 + pd.DataFrame(gate_rows).round(5).to_string(index=False))
    rd = _report_dir(cfg)
    text = "\n\n".join(lines) + "\n"
    with open(os.path.join(rd, "blocking_report.txt"), "w") as f:
        f.write(text)
    bk.save_json(os.path.join(rd, "blocking_report.json"), rep)
    from pipeline import fingerprint
    bk.save_json(os.path.join(rd, "blocking_gates.json"),
                 {"passed": passed, "checks": gate_rows, "block_fp": fingerprint(cfg, "keep")})
    log("blocking report:\n" + text)
    return passed


def _record_gids(cfg, split):
    """Global record id of every row of the rec tables, in the order _read_all returns them."""
    out = []
    for part in meta(cfg, split)["partitions"]:
        d = os.path.join(cfg.work_dir, split, "cand", part["name"])
        if os.path.isdir(d):
            g = pq.read_table(os.path.join(cfg.work_dir, split, "norm", f"q_{part['name']}.parquet"),
                              columns=["gid"])["gid"].to_numpy()
            t = bk.read_parts(d, "rec", ["qi"])
            if t is not None:
                out.append(g[t["qi"].to_numpy()])
    return np.concatenate(out) if out else np.empty(0, np.int64)


def _dist(name, g):
    k, u = g.n_kept.to_numpy(), g.n_union.to_numpy()
    return {"records": name, "n": len(g), "kept_mean": k.mean(), "kept_p50": np.median(k),
            "kept_p90": np.percentile(k, 90), "kept_max": k.max(), "union_mean": u.mean(),
            "union_p99": np.percentile(u, 99), "no_candidate": float((k == 0).mean())}


def gates_passed(cfg):
    from pipeline import fingerprint
    p = os.path.join(cfg.work_dir, "reports", "blocking_gates.json")
    if not os.path.exists(p):
        return None
    g = load_json(p)
    if g.get("block_fp") != fingerprint(cfg, "keep"):
        return None
    return bool(g["passed"])


# ---------------------------------------------------------------------------
# Final report: holdout (after the rule is frozen on tune), density stress, error breakdown
# ---------------------------------------------------------------------------

def _nbucket(n):
    return pd.cut(n, [-1, 0, 1, 3, 10, 30, np.inf], labels=["0", "1", "2-3", "4-10", "11-30", "31+"])


def final_report(cfg):
    """Writes work/reports/final_report.{txt,json}. Every decision (model, calibration, rule)
    was fixed on train / tune entities before this reads the holdout entities."""
    import decision as dc
    import model
    md = os.path.join(cfg.work_dir, "model")
    if not os.path.exists(os.path.join(md, "decision.json")):
        raise SystemExit("no tuned decision: run --stage train first")
    dec = load_json(os.path.join(md, "decision.json"))
    info = load_json(os.path.join(md, "model.json"))
    a = model.read_outputs(cfg, "train", "score", "asg")
    gt_s, gt_q, roles, n_q = model._eval_inputs(cfg)
    passes = [1, 2] if info["pass2"] else [1]
    tune_ids, hold_ids = np.flatnonzero(roles == 2), np.flatnonzero(roles == 3)
    country = pd.read_parquet(os.path.join(cfg.work_dir, "train", "ids_s1.parquet"))["country"].to_numpy()
    std = a[~a.extra.to_numpy()]
    lines, rep = [], {}

    rows = []
    for k in passes:
        seen = {}
        for which, label in (("rule", "chosen on tune"), ("best_threshold", "best threshold on tune"),
                             ("best_expected_f", "best expected-F on tune")):
            rule = dec[f"pass{k}"][which]
            key = json.dumps(rule, sort_keys=True)
            if key in seen:
                seen[key]["rule_role"] += f" = {label}"
                continue
            row = {"pass": k, "rule_role": label, "rule": dc.rule_name(rule)}
            for dens, df in (("standard", std), ("stress", a)):
                df = df[roles[df[f"sg{k}"].to_numpy()] >= 2]      # rules act per S1
                ps, pq_, _ = dc.decide_assigned(df[f"sg{k}"].to_numpy(), df.qg.to_numpy(),
                                                df[f"p{k}"].to_numpy(), rule)
                for es_name, es in (("tune", tune_ids), ("holdout", hold_ids)):
                    row[f"{es_name}_{dens}"] = dc.macro_f05(ps, pq_, gt_s, gt_q, es, n_q)
            seen[key] = row
            rows.append(row)
    res = pd.DataFrame(rows)
    fk = dec["final_pass"]
    lines.append("MACRO F0.5 (standard = training-universe density; stress = with the extra "
                 "look-alike records up to test density; tune scores are in-sample for the rule, "
                 f"which was tuned at {getattr(cfg, 'tune_density', 'standard')} density)\n"
                 + res.round(5).to_string(index=False)
                 + f"\nfinal model: pass {fk}; out-of-fold logloss pass 1 {info['oof_logloss_pass1']:.4f}"
                 + (f", pass 2 {info['oof_logloss_pass2']:.4f}" if info.get("pass2") else ""))
    rep["scores"] = res.to_dict(orient="records")

    tr = _read_all(cfg, "train", "truth", ["qg", "sg", "kept", "rank_pos", "rank_score"]
                   ).drop_duplicates(["qg", "sg"])
    ok = tr.kept.astype(bool).to_numpy() & post_keep(cfg, tr.rank_pos.to_numpy(), tr.rank_score.to_numpy())
    reached = np.bincount(tr.sg[ok].astype(np.int64), minlength=len(roles))
    rule = dec[f"pass{fk}"]["rule"]
    for dens, df in (("standard", std), ("stress", a)):
        df = df[roles[df[f"sg{fk}"].to_numpy()] == 3]
        ps, pq_, _ = dc.decide_assigned(df[f"sg{fk}"].to_numpy(), df.qg.to_numpy(),
                                        df[f"p{fk}"].to_numpy(), rule)
        r = dc.entity_table(ps, pq_, gt_s, gt_q, hold_ids, n_q, reached[hold_ids])
        r["country"] = country[r.s1]
        r["n_true_bucket"] = _nbucket(r.n_true)
        lines.append(f"HOLDOUT LOSS BREAKDOWN ({dens} density, pass {fk}, {dc.rule_name(rule)}): macro "
                     f"F0.5 {r.f05.mean():.5f}, candidate ceiling {r.ceiling.mean():.5f}, model gap "
                     f"{r.ceiling.mean() - r.f05.mean():.5f}\n" + dc.loss_breakdown(r).round(5).to_string())
        by = []
        for col in ("country", "n_true_bucket"):
            g = r.groupby(col, observed=True).agg(entities=("f05", "size"), f05=("f05", "mean"),
                                                  ceiling=("ceiling", "mean"))
            g["macro_points_lost"] = r.groupby(col, observed=True)["f05"].apply(lambda x: (1 - x).sum()) / len(r)
            g.index = [f"{col}={v}" for v in g.index]
            by.append(g)
        lines.append(f"HOLDOUT BY SEGMENT ({dens} density)\n" + pd.concat(by).round(5).to_string())
        rep[f"holdout_{dens}"] = {"macro_f05": float(r.f05.mean()), "ceiling": float(r.ceiling.mean())}

    # reliability of the calibrated assigned score on holdout (standard density)
    s, p, y = std[f"sg{fk}"].to_numpy(), std[f"p{fk}"].to_numpy(), std[f"y{fk}"].to_numpy()
    on = roles[s] == 3
    pc_ = dc.apply_calibration(p[on], dec[f"pass{fk}"]["cal"])
    b = pd.cut(pc_, [0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.97, 1.0001], include_lowest=True)
    rel = pd.DataFrame({"bin": b, "p": pc_, "y": y[on]}).groupby("bin", observed=True).agg(
        records=("y", "size"), mean_calibrated_p=("p", "mean"), true_share=("y", "mean"))
    lines.append("CALIBRATION ON HOLDOUT (records assigned to a holdout S1; calibration was fitted "
                 "on tune)\n" + rel.round(4).to_string())
    m = meta(cfg, "train")
    if m.get("universe"):
        lines.append("DENSITY (S2/S3 per S1)\n" + pd.DataFrame(m["universe"]).T[
            ["records_per_S1", "stress_records_per_S1", "test_records_per_S1"]].to_string())
    text = "\n\n".join(lines) + "\n"
    rd = _report_dir(cfg)
    with open(os.path.join(rd, "final_report.txt"), "w") as f:
        f.write(text)
    bk.save_json(os.path.join(rd, "final_report.json"), rep)
    log("final report:\n" + text)
    return rep


# ---------------------------------------------------------------------------
# Test / France diagnostics (no labels): test partitions next to the holdout reference
# ---------------------------------------------------------------------------

def _diag_rows(label, asg, k, n_rec, sel_s, sel_q, s1_pool, extra_note=""):
    p = asg[f"p{k}"].to_numpy()
    n_cand = asg.n_cand.to_numpy()
    pred_rec = bk.isin_sorted(asg.qg.to_numpy(), bk.sorted_unique(sel_q))
    per_s1 = np.bincount(np.searchsorted(s1_pool, sel_s), minlength=len(s1_pool)) if len(s1_pool) else np.zeros(0)
    ne = per_s1 > 0
    return {"population": label + extra_note, "records": int(n_rec), "S1": int(len(s1_pool)),
            "records_per_S1": n_rec / max(len(s1_pool), 1),
            "no_candidate_share": 1 - len(asg) / max(n_rec, 1),
            "cands_mean": n_cand.mean() if len(asg) else np.nan,
            "cands_p90": np.percentile(n_cand, 90) if len(asg) else np.nan,
            "best_p_p25": np.percentile(p, 25) if len(p) else np.nan,
            "best_p_p50": np.percentile(p, 50) if len(p) else np.nan,
            "best_p_ge_0.5": (p >= 0.5).mean() if len(p) else np.nan,
            "margin_p50": np.median(p - asg[f"second{k}"].to_numpy()) if len(p) else np.nan,
            "records_matched": pred_rec.mean() if len(p) else np.nan,
            "S1_nonempty": ne.mean() if len(ne) else np.nan,
            "matches_per_nonempty_S1": per_s1[ne].mean() if ne.any() else np.nan,
            "max_matches_S1": int(per_s1.max()) if len(per_s1) else 0,
            "match_nm_tset_p10": np.nanpercentile(asg.nm_tset[pred_rec], 10) if pred_rec.any() else np.nan,
            "match_nm_tset_p50": np.nanmedian(asg.nm_tset[pred_rec]) if pred_rec.any() else np.nan,
            "match_ad_tset_p50": np.nanmedian(asg.ad_tset[pred_rec]) if pred_rec.any() else np.nan,
            "match_cos_name_p50": np.nanmedian(asg.cos_name[pred_rec]) if pred_rec.any() else np.nan}


def test_diagnostics(cfg, asg, k, rule, ps, pq_, pp):
    """Label-free comparison of each test country with the holdout reference, the label-shift
    estimate and an audit sample for cfg.audit_country. Writes work/reports/test_diagnostics.txt
    and audit_<country>.tsv."""
    import decision as dc
    import model
    md = os.path.join(cfg.work_dir, "model")
    dec = load_json(os.path.join(md, "decision.json"))
    lines, rows = [], []
    ids_s1 = pd.read_parquet(os.path.join(cfg.work_dir, "test", "ids_s1.parquet"))
    ids_q = pd.read_parquet(os.path.join(cfg.work_dir, "test", "ids_q.parquet"), columns=["country"])
    s_country, q_country = ids_s1.country.to_numpy(), ids_q.country.to_numpy()
    # reference: records whose best S1 is a tune/holdout entity; their S1s
    ref = model.read_outputs(cfg, "train", "score", "asg")
    roles = np.load(os.path.join(cfg.work_dir, "train", "roles.npy"))
    tr_country = pd.read_parquet(os.path.join(cfg.work_dir, "train", "ids_s1.parquet"))["country"].to_numpy()
    for dens in ("standard", "stress"):
        r = ref[roles[ref[f"sg{k}"].to_numpy()] >= 2]
        if dens == "standard":
            r = r[~r.extra.to_numpy()]
        rs, rq, _ = dc.decide_assigned(r[f"sg{k}"].to_numpy(), r.qg.to_numpy(), r[f"p{k}"].to_numpy(), rule)
        for c in sorted(set(tr_country[r[f"sg{k}"].to_numpy()])):
            m = tr_country[r[f"sg{k}"].to_numpy()] == c
            pool = np.flatnonzero((roles >= 2) & (tr_country == c))
            keep = tr_country[rs] == c
            rows.append(_diag_rows(f"train ref {c or '<blank>'}", r[m], k, int(m.sum()), rs[keep], rq[keep],
                                   pool, f" ({dens})"))
    for c in sorted(set(q_country) | set(s_country)):
        m = q_country[asg.qg.to_numpy()] == c
        pool = np.flatnonzero(s_country == c)
        keep = s_country[ps] == c
        rows.append(_diag_rows(f"TEST {c or '<blank>'}", asg[m], k, int((q_country == c).sum()),
                               ps[keep], pq_[keep], pool))
    tab = pd.DataFrame(rows).set_index("population")
    lines.append(f"TEST vs TRAINING REFERENCE (no labels; rule {dc.rule_name(rule)}, pass {k}). Reference "
                 "rows use records whose best S1 is a tune/holdout entity. A test country whose score "
                 "or similarity distribution differs from every reference row is extrapolating.\n"
                 + tab.T.round(3).to_string())
    cal = dec[f"pass{k}"].get("cal")
    tz = np.load(os.path.join(md, f"tune_assigned_pass{k}.npz"))
    shift = {}
    for c in [None] + sorted(set(q_country)):
        m = np.ones(len(asg), bool) if c is None else q_country[asg.qg.to_numpy()] == c
        if m.sum() < 50:
            continue
        pi_s, pi_t = dc.estimate_prior_shift(tz["p"], tz["y"], dc.apply_calibration(asg[f"p{k}"].to_numpy()[m], cal))
        shift["all" if c is None else c] = {"tune_true_share": round(pi_s, 4), "test_estimate": round(pi_t, 4)}
    lines.append("LABEL-SHIFT ESTIMATE (share of assigned records that are true: tune vs EM estimate "
                 "on test; a large drop means more decoys than training saw)\n"
                 + pd.DataFrame(shift).T.to_string())
    rd = _report_dir(cfg)
    audit = _audit(cfg, asg, k, ps, pq_, pp, s_country, rd)
    if audit:
        lines.append(audit)
    text = "\n\n".join(lines) + "\n"
    with open(os.path.join(rd, "test_diagnostics.txt"), "w") as f:
        f.write(text)
    log("test diagnostics:\n" + text)


def _audit(cfg, asg, k, ps, pq_, pp, s_country, rd, n=100):
    c = schema_country(cfg.audit_country)
    pool = np.flatnonzero(s_country == c)
    if not len(pool):
        return f"AUDIT: no test S1 with country {c!r}; no audit sample written"
    rng = np.random.default_rng(cfg.seed + 5)
    pick = np.sort(rng.choice(pool, min(n, len(pool)), replace=False))
    nd = os.path.join(cfg.work_dir, "test", "norm")
    s_tab = pq.read_table(os.path.join(nd, "s1_all.parquet"), columns=["name_raw_clean", "address_raw_clean"])
    q_tab = pq.read_table(os.path.join(nd, "q_all.parquet"), columns=["name_raw_clean", "address_raw_clean"])
    s_ids = pd.read_parquet(os.path.join(cfg.work_dir, "test", "ids_s1.parquet"))["entity_id"].to_numpy(object)
    q_ids = pd.read_parquet(os.path.join(cfg.work_dir, "test", "ids_q.parquet"))["entity_id"].to_numpy(object)
    pred_key = bk.sorted_unique(np.asarray(ps, np.int64) * len(q_ids) + pq_)
    sg, qg = asg[f"sg{k}"].to_numpy(), asg.qg.to_numpy()
    on = bk.isin_sorted(sg, pick)
    rows = []
    a = asg[on].assign(sg=sg[on])
    a["predicted"] = bk.isin_sorted(a.sg.to_numpy().astype(np.int64) * len(q_ids) + a.qg.to_numpy(), pred_key)
    a = a.sort_values(["sg", "predicted", f"p{k}"], ascending=[True, False, False])
    a = a.groupby("sg").apply(lambda g: pd.concat([g[g.predicted], g[~g.predicted].head(3)]),
                              include_groups=False).reset_index(level=0)
    got = set(a.sg.tolist())
    need_s = np.union1d(pick, a.sg.to_numpy()).astype(np.int64)
    st = s_tab.take(pa.array(need_s))
    sn = dict(zip(need_s.tolist(), st["name_raw_clean"].to_pylist()))
    sa = dict(zip(need_s.tolist(), st["address_raw_clean"].to_pylist()))
    need_q = np.unique(a.qg.to_numpy()).astype(np.int64)
    qt = q_tab.take(pa.array(need_q))
    qn = dict(zip(need_q.tolist(), qt["name_raw_clean"].to_pylist()))
    qa = dict(zip(need_q.tolist(), qt["address_raw_clean"].to_pylist()))
    for r in a.itertuples():
        rows.append({"s1_entity_id": s_ids[r.sg], "s1_name": sn[r.sg], "s1_address": sa[r.sg],
                     "predicted": int(r.predicted), "record_id": q_ids[r.qg],
                     "record_name": qn[int(r.qg)], "record_address": qa[int(r.qg)],
                     "p": round(float(getattr(r, f"p{k}")), 4), "nm_tset": round(float(r.nm_tset), 3),
                     "ad_tset": round(float(r.ad_tset), 3), "correct(fill in)": ""})
    for sgi in pick:
        if sgi not in got:
            rows.append({"s1_entity_id": s_ids[sgi], "s1_name": sn[sgi], "s1_address": sa[sgi],
                         "predicted": 0, "record_id": "", "record_name": "(no record assigned)",
                         "record_address": "", "p": np.nan, "nm_tset": np.nan, "ad_tset": np.nan,
                         "correct(fill in)": ""})
    df = pd.DataFrame(rows)
    path = os.path.join(rd, f"audit_{c}.tsv")
    df.to_csv(path, sep="\t", index=False)
    n_pred = df.groupby("s1_entity_id").predicted.sum()
    return (f"AUDIT SAMPLE: {len(pick)} random {c} S1 entities written to {path} with their predicted "
            f"records and up to 3 assigned-but-rejected records each ({(n_pred > 0).mean():.0%} of them "
            f"non-empty). Mark the last column to measure precision by hand.")


def schema_country(c):
    import schema
    return schema.normalize_country([c])[0]
