"""
v4 reports (written to <v4>/reports):

cut_report.txt      recall of the true pairs of TUNE + HOLDOUT entities at every layer (v3 union,
                    v3 stored candidates, v4 expanded set, stage-2 kept set), v3's 24.6-per-record
                    trim for comparison, recall by segment, the new sources, candidates per record,
                    the oracle ceiling of the scored set, and the v4 candidate gate
final_report.txt    holdout macro F0.5 at standard and test (stress) density, the loss breakdown
                    (with the v4 score gates of the architecture, section 7.1), holdout by segment,
                    calibration
test_diagnostics.txt, audit_<country>.tsv
                    label-free test / France diagnostics (as v3)
"""
import json
import os

import numpy as np
import pandas as pd

import blocking as bk
import decision as dc
import evaluate as ev3
import expand4 as ex
from pipeline import load_json, log, meta
from pipeline4 import fp4, v4dir


def report_dir(cfg):
    return v4dir(cfg, "reports")


def _read_cut(cfg, kind, columns=None):
    import pyarrow.parquet as pq
    tabs = []
    for part in meta(cfg, "train")["partitions"]:
        d = os.path.join(v4dir(cfg, "train", "cut", part["name"]))
        files = sorted(f for f in os.listdir(d) if f.endswith(f".{kind}.parquet"))
        for f in files:
            t = pq.read_table(os.path.join(d, f), columns=columns).to_pandas()
            t["part"] = part["label"]
            tabs.append(t)
    return pd.concat(tabs, ignore_index=True) if tabs else pd.DataFrame(columns=(columns or []) + ["part"])


def true_pairs4(cfg):
    """v3's true-pair table (segments, v3 union / stored flags) joined with the v4 cut truth."""
    t, roles, n_true = ev3.true_pair_table(cfg)
    t = t.rename(columns={"kept": "stored_v3"})
    u = _read_cut(cfg, "truth", ["qg", "sg", "src4", "rank_pos", "s2_score", "s2_rank", "kept", "rescued"])
    u = u.drop_duplicates(["qg", "sg"]).rename(columns={"kept": "kept4", "rank_pos": "rank_pos4"})
    t = t.merge(u.drop(columns=["part"]), on=["qg", "sg"], how="left")
    t["in_expanded"] = t["src4"].notna()
    t["kept4"] = t["kept4"].fillna(False).astype(bool)
    t["src4"] = t["src4"].fillna(0).astype(np.int64)
    # v3's trimmed training set: kept_min 5 or (rank <= 100 and score >= 1e-7)
    t["v3_trim_1e7"] = t.stored_v3 & ((t.rank_pos <= 5) | (t.rank_score >= 1e-7))
    return t, roles, n_true


def _rec(df, col):
    return float(df[col].mean()) if len(df) else np.nan


LAYERS = [("in_union", "v3 union (all channels and blocks)"),
          ("stored_v3", "v3 stored candidates (<= 100 per record; v4's stage-1 set)"),
          ("v3_trim_1e7", "v3 trimmed set you trained on (floor 1e-7)"),
          ("in_expanded", "v4 expanded set (stored + new sources)"),
          ("kept4", "v4 kept after the stage-2 cut (what the pair model scores)")]


def cut_report(cfg):
    t, roles, n_true = true_pairs4(cfg)
    lines, rep = [], {}
    un = t[t.role >= 2]
    rows = []
    for col, label in LAYERS:
        rows.append({"layer": label, "unseen (tune+holdout)": _rec(un, col),
                     "training entities": _rec(t[t.role == 1], col)})
    lay = pd.DataFrame(rows)
    lines.append("TRUE-PAIR RECALL BY LAYER (true pairs of tune + holdout entities, never seen by "
                 "the v3 ranker or stage 2)\n" + lay.round(5).to_string(index=False))
    rep["layers"] = lay.to_dict(orient="records")

    seg_rows = []
    for name, col in (("country", "country"), ("source", "source")):
        for val, g in un.groupby(col):
            seg_rows.append({"segment": f"{name}={val}", "pairs": len(g),
                             **{c: _rec(g, c) for c, _ in LAYERS}})
    for c in [c for c in t.columns if c.startswith("seg_")]:
        g = un[un[c]]
        seg_rows.append({"segment": c[4:], "pairs": len(g), **{k: _rec(g, k) for k, _ in LAYERS}})
    seg = pd.DataFrame(seg_rows)
    lines.append("RECALL BY SEGMENT (unseen true pairs; columns as the layers above)\n"
                 + seg.round(5).to_string(index=False))
    rep["segments"] = seg.to_dict(orient="records")

    src_rows = []
    ie = un[un.in_expanded]
    for name, bit in ex.BIT4.items():
        found = (ie.src4 & bit) > 0
        only = ie.src4 == bit
        src_rows.append({"source": name, "found": found.mean() if len(ie) else np.nan,
                         "found_only_by_it": int(only.sum()),
                         "kept_found_only_by_it": int((only & ie.kept4).sum())})
    src = pd.DataFrame(src_rows)
    lines.append("V4 SOURCES (unseen true pairs in the expanded set: share found by each source, "
                 "and pairs found by it alone, before and after the cut)\n" + src.round(5).to_string(index=False))
    rep["sources"] = src.to_dict(orient="records")

    ent = np.flatnonzero(roles >= 2)
    ids_s1 = pd.read_parquet(os.path.join(cfg.work_dir, "train", "ids_s1.parquet"))
    ceil = {}
    for col, label in LAYERS:
        reach = np.bincount(t.sg[t[col]], minlength=len(roles))
        ceil[col] = ev3.f05_ceiling(n_true[ent], reach[ent])
    cdf = pd.DataFrame({"country": ids_s1.country.to_numpy()[ent], **ceil})
    ctab = cdf.groupby("country")[list(ceil)].mean()
    ctab.loc["ALL"] = [ceil[c].mean() for c in ceil]
    lines.append("ORACLE CANDIDATE CEILING (macro F0.5 of a perfect model on tune+holdout entities, "
                 "singletons included; columns as the layers above)\n" + ctab.round(5).to_string())
    rep["ceiling"] = {c: float(ceil[c].mean()) for c in ceil}

    rec = _read_cut(cfg, "rec")
    q_gid = []
    import pyarrow.parquet as pq
    for part in meta(cfg, "train")["partitions"]:
        g = pq.read_table(os.path.join(cfg.work_dir, "train", "norm", f"q_{part['name']}.parquet"),
                          columns=["gid"])["gid"].to_numpy()
        d = v4dir(cfg, "train", "cut", part["name"])
        for f in sorted(x for x in os.listdir(d) if x.endswith(".rec.parquet")):
            q_gid.append(g[pq.read_table(os.path.join(d, f), columns=["qi"])["qi"].to_numpy()])
    q_gid = np.concatenate(q_gid) if q_gid else np.empty(0, np.int64)
    extra = pd.read_parquet(os.path.join(cfg.work_dir, "train", "ids_q.parquet"), columns=["extra"])["extra"].to_numpy()
    if len(rec):
        rec["extra"] = extra[q_gid] if len(q_gid) == len(rec) else False
        dist = []
        for (p, e), g in rec.groupby(["part", "extra"]):
            k, u, nw = g.n_kept.to_numpy(), g.n_expanded.to_numpy(), g.n_new.to_numpy()
            dist.append({"records": f"train {p}" + (" stress-only" if e else ""), "n": len(g),
                         "expanded_mean": u.mean(), "new_mean": nw.mean(), "kept_mean": k.mean(),
                         "kept_p50": np.median(k), "kept_p90": np.percentile(k, 90), "kept_max": k.max(),
                         "no_candidate": float((k == 0).mean())})
        dtab = pd.DataFrame(dist)
        lines.append("CANDIDATES PER RECORD (training universe; test is cut inside predict)\n"
                     + dtab.round(2).to_string(index=False))
        rep["candidates_per_record"] = dtab.to_dict(orient="records")
    s2 = load_json(os.path.join(v4dir(cfg, "stage2"), "stage2.json"))
    lines.append(f"STAGE-2 CUT: keep_min {cfg.s2_keep_min}, keep_max {cfg.s2_keep_max}, no-address "
                 f"records {cfg.s2_keep_noaddr}, floor "
                 f"{s2['floor']:.4g}, address rescue {cfg.addr_rescue} (held-out training records: "
                 f"{s2['held_out_true_kept']:.5f} of expanded true pairs kept at "
                 f"{s2['held_out_pairs_per_record']:.1f} per record)")
    passed = rep["ceiling"]["kept4"] >= cfg.gate4_ceiling
    lines.insert(0, f"V4 CANDIDATE GATE: {'PASS' if passed else 'FAIL'} (oracle ceiling of the scored "
                    f"set {rep['ceiling']['kept4']:.5f}, required {cfg.gate4_ceiling}; the score gates "
                    "are in final_report.txt)")
    text = "\n\n".join(lines) + "\n"
    rd = report_dir(cfg)
    with open(os.path.join(rd, "cut_report.txt"), "w") as f:
        f.write(text)
    bk.save_json(os.path.join(rd, "cut_report.json"), rep)
    bk.save_json(os.path.join(rd, "gates4.json"), {"passed": bool(passed), "fp": fp4(cfg, "cut"),
                                                   "ceiling_kept": rep["ceiling"]["kept4"]})
    log("cut report:\n" + text)
    return passed


def gates4(cfg):
    p = os.path.join(report_dir(cfg), "gates4.json")
    if not os.path.exists(p):
        return None
    g = load_json(p)
    return bool(g["passed"]) if g.get("fp") == fp4(cfg, "cut") else None


def _reached(cfg, n_s):
    tr = _read_cut(cfg, "truth", ["qg", "sg", "kept"]).drop_duplicates(["qg", "sg"])
    ok = tr.kept.astype(bool).to_numpy()
    return np.bincount(tr.sg[ok].astype(np.int64), minlength=n_s)


def final_report(cfg):
    import model4
    md = model4.model_dir(cfg)
    if not os.path.exists(os.path.join(md, "decision.json")):
        raise SystemExit("no tuned decision: run --stage train first")
    dec = load_json(os.path.join(md, "decision.json"))
    info = load_json(os.path.join(md, "model.json"))
    import cross
    if dec.get("cross", False) != cross.active(cfg):
        raise SystemExit("the tuned decision does not match --cross: run --stage "
                         + ("cross" if cross.active(cfg) else "train") + " first")
    a = model4.train_asg(cfg)
    gt_s, gt_q, roles, n_q = model4.eval_inputs(cfg)
    tune_ids, hold_ids = np.flatnonzero(roles == 2), np.flatnonzero(roles == 3)
    country = pd.read_parquet(os.path.join(cfg.work_dir, "train", "ids_s1.parquet"))["country"].to_numpy()
    std = a[~a.extra.to_numpy()]
    lines, rep = [], {}
    rows, seen = [], {}
    for which, label in (("rule", "chosen on tune"), ("best_threshold", "best threshold on tune"),
                         ("best_expected_f", "best expected-F on tune")):
        rule = dec["pass1"][which]
        key = json.dumps(rule, sort_keys=True)
        if key in seen:
            seen[key]["rule_role"] += f" = {label}"
            continue
        row = {"rule_role": label, "rule": dc.rule_name(rule)}
        for dens, df in (("standard", std), ("stress", a)):
            df = df[roles[df.sg1.to_numpy()] >= 2]
            ps, pq_, _ = dc.decide_assigned(df.sg1.to_numpy(), df.qg.to_numpy(), df.p1.to_numpy(), rule)
            for es_name, es in (("tune", tune_ids), ("holdout", hold_ids)):
                row[f"{es_name}_{dens}"] = dc.macro_f05(ps, pq_, gt_s, gt_q, es, n_q)
        seen[key] = row
        rows.append(row)
    res = pd.DataFrame(rows)
    lines.append("MACRO F0.5 (standard = training-universe density; stress = with the extra look-alike "
                 "records up to test density; tune is in-sample for the rule, tuned at "
                 f"{dec.get('tune_density', cfg.tune_density)} density)\n" + res.round(5).to_string(index=False)
                 + f"\nout-of-fold logloss {info['oof_logloss_pass1']:.4f}; {info['training_pairs']:,} "
                 f"training pairs of {info['training_entities']:,} entities")
    rep["scores"] = res.to_dict(orient="records")
    reached = _reached(cfg, len(roles))
    rule = dec["pass1"]["rule"]
    for dens, df in (("standard", std), ("stress", a)):
        df = df[roles[df.sg1.to_numpy()] == 3]
        ps, pq_, _ = dc.decide_assigned(df.sg1.to_numpy(), df.qg.to_numpy(), df.p1.to_numpy(), rule)
        r = dc.entity_table(ps, pq_, gt_s, gt_q, hold_ids, n_q, reached[hold_ids])
        r["country"] = country[r.s1]
        r["n_true_bucket"] = ev3._nbucket(r.n_true)
        lines.append(f"HOLDOUT LOSS BREAKDOWN ({dens} density, {dc.rule_name(rule)}): macro F0.5 "
                     f"{r.f05.mean():.5f}, candidate ceiling {r.ceiling.mean():.5f}, model gap "
                     f"{r.ceiling.mean() - r.f05.mean():.5f}\n" + dc.loss_breakdown(r).round(5).to_string())
        by = []
        for col in ("country", "n_true_bucket"):
            g = r.groupby(col, observed=True).agg(entities=("f05", "size"), f05=("f05", "mean"),
                                                  ceiling=("ceiling", "mean"))
            g["macro_points_lost"] = r.groupby(col, observed=True)["f05"].apply(lambda x: (1 - x).sum()) / len(r)
            g.index = [f"{col}={v}" for v in g.index]
            by.append(g)
        seg = pd.concat(by)
        lines.append(f"HOLDOUT BY SEGMENT ({dens} density)\n" + seg.round(5).to_string())
        rep[f"holdout_{dens}"] = {"macro_f05": float(r.f05.mean()), "ceiling": float(r.ceiling.mean()),
                                  "by_segment": seg.round(6).to_dict(orient="index")}
    hs = rep["holdout_stress"]
    india = hs["by_segment"].get("country=india", {}).get("f05", np.nan)
    gates = [("oracle ceiling of the scored set", hs["ceiling"], cfg.gate4_ceiling, ">="),
             ("holdout F0.5 at test density", hs["macro_f05"], cfg.gate4_holdout_stress, ">="),
             ("model gap (ceiling - holdout)", hs["ceiling"] - hs["macro_f05"], cfg.gate4_gap, "<="),
             ("India holdout F0.5 at test density", india, 0.990, ">=")]
    gtab = pd.DataFrame([{"gate": g, "value": v, "required": f"{op} {req}",
                          "pass": bool(v >= req if op == ">=" else v <= req)} for g, v, req, op in gates])
    lines.insert(0, "V4 SCORE GATES (architecture 7.1)\n" + gtab.round(5).to_string(index=False))
    rep["gates"] = gtab.to_dict(orient="records")
    s, p, y = std.sg1.to_numpy(), std.p1.to_numpy(), std.y1.to_numpy()
    on = roles[s] == 3
    pc_ = dc.apply_calibration(p[on], dec["pass1"]["cal"])
    b = pd.cut(pc_, [0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.97, 1.0001], include_lowest=True)
    rel = pd.DataFrame({"bin": b, "p": pc_, "y": y[on]}).groupby("bin", observed=True).agg(
        records=("y", "size"), mean_calibrated_p=("p", "mean"), true_share=("y", "mean"))
    lines.append("CALIBRATION ON HOLDOUT (standard density; fitted on tune at "
                 f"{dec.get('tune_density', cfg.tune_density)} density)\n" + rel.round(4).to_string())
    imp = pd.Series(info["importance_pass1"]).head(25)
    lines.append("TOP FEATURES BY GAIN\n" + imp.round(1).to_string())
    text = "\n\n".join(lines) + "\n"
    rd = report_dir(cfg)
    with open(os.path.join(rd, "final_report.txt"), "w") as f:
        f.write(text)
    bk.save_json(os.path.join(rd, "final_report.json"), rep)
    log("final report:\n" + text)
    model_loss(cfg, a, dec["pass1"]["rule"] if not cfg.decision_mode else model4.final_rule(cfg, dec),
               roles, gt_s, gt_q, n_q)
    return rep


LOSS_KINDS = ["not retrieved (not in the expanded set)", "cut out by stage 2",
              "record went to another S1", "below the rule (assigned, not selected)"]


def model_loss(cfg, a, rule, roles, gt_s, gt_q, n_q, n_sample=60):
    """Holdout (test density): every true pair by outcome, every selected false pair by kind,
    the macro F0.5 with each kind of error removed, and samples with the texts
    (model_loss.txt, model_loss_sample.tsv)."""
    import cross
    hold_ids = np.flatnonzero(roles == 3)
    t, _, _ = true_pairs4(cfg)
    t = t[t.role == 3].reset_index(drop=True)
    h = a[roles[a.sg1.to_numpy()] == 3]
    ps, pq_, pp = dc.decide_assigned(h.sg1.to_numpy(), h.qg.to_numpy(), h.p1.to_numpy(), rule)
    base = dc.macro_f05(ps, pq_, gt_s, gt_q, hold_ids, n_q)
    n = n_q + 1
    sel = pd.Index(ps.astype(np.int64) * n + pq_)
    key = t.sg.to_numpy() * n + t.qg.to_numpy()
    found = pd.Index(key).isin(sel)
    asg = a.drop_duplicates("qg").set_index("qg")
    sg1 = asg.sg1.reindex(t.qg.to_numpy()).to_numpy()
    p1 = dc.apply_calibration(asg.p1.reindex(t.qg.to_numpy()).fillna(0).to_numpy(), rule.get("cal"))
    kind = np.full(len(t), "found", object)
    kind[~found & ~t.in_expanded.to_numpy()] = LOSS_KINDS[0]
    kind[~found & t.in_expanded.to_numpy() & ~t.kept4.to_numpy()] = LOSS_KINDS[1]
    rest = ~found & t.kept4.to_numpy()
    kind[rest & (sg1 != t.sg.to_numpy())] = LOSS_KINDS[2]
    kind[rest & (sg1 == t.sg.to_numpy())] = LOSS_KINDS[3]
    t["kind"], t["p_best"], t["other_sg"] = kind, p1, np.where(sg1 != t.sg.to_numpy(), sg1, -1)
    # false selected pairs
    truth = pd.Index(gt_s.astype(np.int64) * n + gt_q)
    false = ~pd.Index(ps.astype(np.int64) * n + pq_).isin(truth)
    has_true = np.zeros(n_q, bool)
    has_true[gt_q] = True
    fk = np.where(has_true[pq_[false]], "false match: record belongs to another S1",
                  "false match: record matches no S1 (decoy)")
    rows = []
    for k in LOSS_KINDS:
        m = t.kind.to_numpy() == k
        ps2, pq2 = np.r_[ps, t.sg.to_numpy()[m]], np.r_[pq_, t.qg.to_numpy()[m]]
        rows.append({"kind": k, "pairs": int(m.sum()), "share_of_true_pairs": m.mean(),
                     "macro_f05_if_fixed": dc.macro_f05(ps2, pq2, gt_s, gt_q, hold_ids, n_q)})
    for k in np.unique(fk):
        m = np.zeros(len(ps), bool)
        m[np.flatnonzero(false)[fk == k]] = True
        rows.append({"kind": k, "pairs": int(m.sum()), "share_of_true_pairs": np.nan,
                     "macro_f05_if_fixed": dc.macro_f05(ps[~m], pq_[~m], gt_s, gt_q, hold_ids, n_q)})
    tab = pd.DataFrame(rows)
    tab["gain"] = tab.macro_f05_if_fixed - base
    seg_cols = [c for c in t.columns if c.startswith("seg_")] + ["country"]
    segs = []
    for c in seg_cols:
        g = t.groupby(t[c].astype(str)).kind.value_counts(normalize=True).unstack(fill_value=0)
        g.index = [f"{c.replace('seg_', '')}={v}" for v in g.index]
        segs.append(g)
    seg = pd.concat(segs)
    seg = seg[[k for k in ["found"] + LOSS_KINDS if k in seg.columns]]
    text = (f"HOLDOUT LOSS BY KIND (test density, {dc.rule_name(rule)}, cross-encoder "
            f"{'on' if cross.active(cfg) else 'off'}): macro F0.5 {base:.5f}\n"
            + tab.round(5).to_string(index=False)
            + "\n\nOUTCOME OF HOLDOUT TRUE PAIRS BY SEGMENT (row shares)\n" + seg.round(4).to_string()
            + "\n\nCALIBRATED BEST PROBABILITY OF 'below the rule' PAIRS\n"
            + pd.cut(t.p_best[t.kind == LOSS_KINDS[3]], [0, .1, .3, .5, .7, .8, .9, 1.0001],
                     include_lowest=True).value_counts().sort_index().to_string() + "\n")
    # samples with the texts
    tx = cross.Texts(cfg, "train")
    rng = np.random.default_rng(cfg.seed + 71)
    smp = []
    for k in LOSS_KINDS:
        idx = np.flatnonzero(t.kind.to_numpy() == k)
        for i in rng.choice(idx, min(n_sample, len(idx)), replace=False):
            r = t.iloc[i]
            smp.append({"kind": k, "country": r.country, "p_best": round(float(r.p_best), 4),
                        "s1": tx.s[r.sg], "record": tx.q[r.qg],
                        "other_s1": tx.s[r.other_sg] if r.other_sg >= 0 else ""})
    fi = np.flatnonzero(false)
    for k in np.unique(fk):
        idx = fi[fk == k]
        for i in rng.choice(idx, min(n_sample, len(idx)), replace=False):
            smp.append({"kind": k, "country": "", "p_best": round(float(pp[i]), 4),
                        "s1": tx.s[ps[i]], "record": tx.q[pq_[i]], "other_s1": ""})
    rd = report_dir(cfg)
    with open(os.path.join(rd, "model_loss.txt"), "w") as f:
        f.write(text)
    pd.DataFrame(smp).to_csv(os.path.join(rd, "model_loss_sample.tsv"), sep="\t", index=False)
    log("model loss:\n" + text)



def test_diagnostics(cfg, asg, rule, ps, pq_, pp):
    """Label-free test diagnostics as v3 (test countries next to the holdout reference, the
    label-shift estimate, the France audit sample)."""
    import model4
    if asg is None or not len(asg):
        return
    k = 1
    md = model4.model_dir(cfg)
    dec = load_json(os.path.join(md, "decision.json"))
    lines, rows = [], []
    ids_s1 = pd.read_parquet(os.path.join(cfg.work_dir, "test", "ids_s1.parquet"))
    ids_q = pd.read_parquet(os.path.join(cfg.work_dir, "test", "ids_q.parquet"), columns=["country"])
    s_country, q_country = ids_s1.country.to_numpy(), ids_q.country.to_numpy()
    ref = model4.read_outputs(cfg, "train", "score", "asg")
    roles = np.load(os.path.join(cfg.work_dir, "train", "roles.npy"))
    tr_country = pd.read_parquet(os.path.join(cfg.work_dir, "train", "ids_s1.parquet"))["country"].to_numpy()
    for dens in ("standard", "stress"):
        r = ref[roles[ref.sg1.to_numpy()] >= 2]
        if dens == "standard":
            r = r[~r.extra.to_numpy()]
        rs, rq, _ = dc.decide_assigned(r.sg1.to_numpy(), r.qg.to_numpy(), r.p1.to_numpy(), rule)
        for c in sorted(set(tr_country[r.sg1.to_numpy()])):
            m = tr_country[r.sg1.to_numpy()] == c
            pool = np.flatnonzero((roles >= 2) & (tr_country == c))
            keep = tr_country[rs] == c
            rows.append(ev3._diag_rows(f"train ref {c or '<blank>'}", r[m], k, int(m.sum()), rs[keep],
                                       rq[keep], pool, f" ({dens})"))
    for c in sorted(set(q_country) | set(s_country)):
        m = q_country[asg.qg.to_numpy()] == c
        pool = np.flatnonzero(s_country == c)
        keep = s_country[ps] == c
        rows.append(ev3._diag_rows(f"TEST {c or '<blank>'}", asg[m], k, int((q_country == c).sum()),
                                   ps[keep], pq_[keep], pool))
    tab = pd.DataFrame(rows).set_index("population")
    lines.append(f"TEST vs TRAINING REFERENCE (no labels; rule {dc.rule_name(rule)}). A test country whose "
                 "score or similarity distribution differs from every reference row is extrapolating.\n"
                 + tab.T.round(3).to_string())
    cal = dec["pass1"].get("cal")
    tz = np.load(os.path.join(md, "tune_assigned_pass1.npz"))
    shift = {}
    for c in [None] + sorted(set(q_country)):
        m = np.ones(len(asg), bool) if c is None else q_country[asg.qg.to_numpy()] == c
        if m.sum() < 50:
            continue
        pi_s, pi_t = dc.estimate_prior_shift(tz["p"], tz["y"], dc.apply_calibration(asg.p1.to_numpy()[m], cal))
        shift["all" if c is None else c] = {"tune_true_share": round(pi_s, 4), "test_estimate": round(pi_t, 4)}
    lines.append("LABEL-SHIFT ESTIMATE (share of assigned records that are true: tune vs EM estimate on "
                 "test)\n" + pd.DataFrame(shift).T.to_string())
    rd = report_dir(cfg)
    wk = cfg.work_dir
    try:
        audit = ev3._audit(cfg, asg, k, ps, pq_, pp, s_country, rd)
    finally:
        cfg.work_dir = wk
    if audit:
        lines.append(audit)
    text = "\n\n".join(lines) + "\n"
    with open(os.path.join(rd, "test_diagnostics.txt"), "w") as f:
        f.write(text)
    log("test diagnostics:\n" + text)
