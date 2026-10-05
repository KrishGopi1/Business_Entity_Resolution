"""
Decision layer (v3): exclusive assignment, calibration on tune, per-S1 F0.5 rule, metric.

1. Exclusive assignment: each S2/S3 record keeps only its best S1 (the 1:N constraint).
2. Calibration: an isotonic map from the assigned score to P(true), fitted on tune entities.
3. Rule, chosen on the TUNE entities only:
     threshold   every assigned record with p >= t (one global threshold); empty allowed
     expected_f  per S1, the prefix (best k records by p, k = 0 allowed) with the highest exact
                 expected F0.5 under the calibrated probabilities; u is the probability that
                 the S1 has a true record it cannot see (lost in blocking or to another S1)
   The holdout entities are scored only after the rule is frozen, and both the best threshold
   rule and the best expected-F rule are reported there.
"""
import numpy as np
import pandas as pd

from blocking import isin_sorted, sorted_unique


def desc_order(group, score):
    """Permutation sorting by group (ascending) then score (descending)."""
    return np.lexsort((-np.asarray(score, np.float64), np.asarray(group)))


def best_per_record(qg, sg, p):
    """Each record keeps only its best S1. Returns index positions of the kept pairs."""
    if len(p) == 0:
        return np.empty(0, np.int64)
    order = desc_order(qg, p)
    g = np.asarray(qg)[order]
    return order[np.r_[True, g[1:] != g[:-1]]]


def fit_calibration(p, y):
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(y_min=1e-4, y_max=1 - 1e-4, out_of_bounds="clip")
    iso.fit(np.asarray(p, np.float64), np.asarray(y, np.float64))
    return {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}


def apply_calibration(p, cal):
    if not cal:
        return np.asarray(p, np.float32)
    return np.interp(p, np.asarray(cal["x"]), np.asarray(cal["y"])).astype(np.float32)


def _pb_rows(P):
    m, n = P.shape
    cur = np.ones((m, 1))
    out = [cur]
    for j in range(n):
        p = P[:, j:j + 1]
        nxt = np.zeros((m, j + 2))
        nxt[:, :-1] += cur * (1 - p)
        nxt[:, 1:] += cur * p
        out.append(nxt)
        cur = nxt
    return out


def expected_f05_best_k(P, u=0.0):
    """P: (m, n) probabilities sorted descending per row. Best prefix length k per row and its
    expected F0.5 (exact under independence). F0.5 of the top k = 1.25 TP / (0.25 N_true + k);
    k = 0 scores 1 only if N_true = 0."""
    m, n = P.shape
    pre = _pb_rows(P)
    suf = [None] * (n + 1)
    cur = np.column_stack([np.full(m, 1 - u), np.full(m, u)])
    suf[n] = cur
    for j in range(n - 1, -1, -1):
        p = P[:, j:j + 1]
        nxt = np.zeros((m, cur.shape[1] + 1))
        nxt[:, :-1] += cur * (1 - p)
        nxt[:, 1:] += cur * p
        suf[j] = nxt
        cur = nxt
    E = np.zeros((m, n + 1))
    E[:, 0] = suf[0][:, 0]
    for k in range(1, n + 1):
        a = np.arange(k + 1)[:, None]
        b = np.arange(suf[k].shape[1])[None, :]
        W = 1.25 * a / (0.25 * (a + b) + k)
        E[:, k] = np.einsum("ma,mb,ab->m", pre[k], suf[k], W)
    kbest = np.argmax(E, axis=1)
    return kbest, E[np.arange(m), kbest]


def _approx_best_k(p, u=0.0):
    tp = np.cumsum(p)
    n_true = p.sum() + u
    f = 1.25 * tp / (0.25 * n_true + np.arange(1, len(p) + 1))
    e0 = np.exp(np.log1p(-np.minimum(p, 1 - 1e-9)).sum()) * (1 - u)
    k = int(np.argmax(f)) + 1
    return (k, f[k - 1]) if f[k - 1] > e0 else (0, e0)


def decide_assigned(s, q, p, rule, n_exact=40):
    """Applies a rule to already-assigned pairs (one per record). Returns (s, q, p) selected."""
    pp = apply_calibration(p, rule.get("cal"))
    keep = pp >= rule.get("min_prob", 0.0)
    s, q, pp = s[keep], q[keep], pp[keep].astype(np.float64)
    if rule["mode"] == "threshold" or len(pp) == 0:
        return s, q, pp.astype(np.float32)
    o = desc_order(s, pp)
    s, q, pp = s[o], q[o], pp[o]
    start = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    size = np.diff(np.r_[start, len(s)])
    kbest = np.zeros(len(start), np.int64)
    u = float(rule.get("u", 0.0))
    for n in np.unique(size):
        gi = np.flatnonzero(size == n)
        if n <= n_exact:
            for b in range(0, len(gi), 200_000 // max(n, 1) + 1):
                g = gi[b:b + 200_000 // max(n, 1) + 1]
                P = pp[start[g][:, None] + np.arange(n)[None, :]]
                kbest[g] = expected_f05_best_k(P, u)[0]
        else:
            for g in gi:
                kbest[g] = _approx_best_k(pp[start[g]:start[g] + n], u)[0]
    pos = np.arange(len(s)) - np.repeat(start, size)
    sel = pos < np.repeat(kbest, size)
    return s[sel], q[sel], pp[sel].astype(np.float32)


def decide(qg, sg, p, rule):
    """Exclusive assignment, then the rule. Returns selected (s1 id, record id, probability)."""
    b = best_per_record(qg, sg, p)
    return decide_assigned(np.asarray(sg)[b], np.asarray(qg)[b], np.asarray(p, np.float32)[b], rule)


def rule_name(r):
    if r["mode"] == "threshold":
        return f"threshold p >= {r['min_prob']:g}" + (" (calibrated)" if r.get("cal") else "")
    return f"expected_f u={r['u']:g}" + (" calibrated" if r.get("cal") else "")


def decision_grid(cal):
    grid = [{"mode": "threshold", "min_prob": round(float(t), 3), "u": 0.0, "cal": None}
            for t in np.arange(0.20, 0.90, 0.025)]
    for c in ([cal] if cal else []) + [None]:
        for u in (0.0, 0.02, 0.05, 0.1, 0.2):
            grid.append({"mode": "expected_f", "min_prob": 0.01, "u": u, "cal": c})
    return grid


# ---------------------------------------------------------------------------
# Metric (as the challenge defines it) and loss breakdown
# ---------------------------------------------------------------------------

def _f05(prec, rec):
    return np.where((prec > 0) & (rec > 0),
                    1.25 * prec * rec / np.maximum(0.25 * prec + rec, 1e-12), 0.0)


def per_entity_f05(pred_s, pred_q, gt_s, gt_q, eval_s, n_q):
    """Per-entity F0.5 for the sorted unique S1 ids eval_s. An entity with no true record scores
    1 when predicted empty, 0 otherwise. Returns (f, n_true, n_pred, tp)."""
    E = len(eval_s)

    def locate(x):
        i = np.searchsorted(eval_s, x)
        ok = i < E
        ok[ok] = eval_s[i[ok]] == x[ok]
        return i, ok

    pred_s, pred_q = np.asarray(pred_s, np.int64), np.asarray(pred_q, np.int64)
    gt_s, gt_q = np.asarray(gt_s, np.int64), np.asarray(gt_q, np.int64)
    ip, okp = locate(pred_s)
    it, okt = locate(gt_s)
    pk = pred_s[okp] * n_q + pred_q[okp]
    tk = sorted_unique(gt_s[okt] * n_q + gt_q[okt])
    hit = isin_sorted(pk, tk)
    n_pred = np.bincount(ip[okp], minlength=E)
    n_true = np.bincount(it[okt], minlength=E)
    tp = np.bincount(ip[okp][hit], minlength=E)
    f = _f05(tp / np.maximum(n_pred, 1), tp / np.maximum(n_true, 1))
    f = np.where((n_true == 0) & (n_pred == 0), 1.0, f)
    return f, n_true, n_pred, tp


def macro_f05(pred_s, pred_q, gt_s, gt_q, eval_s, n_q):
    eval_s = sorted_unique(np.asarray(eval_s, np.int64))
    return float(per_entity_f05(pred_s, pred_q, gt_s, gt_q, eval_s, n_q)[0].mean())


def tune(s, q, p, gt_s, gt_q, eval_sets, n_q, grid):
    """Macro F0.5 of each rule on each evaluation set (name -> S1 ids), from assigned pairs.
    After assignment every rule acts on each S1 separately, so only the pairs assigned to the
    evaluation set's S1s are decided."""
    sub = {}
    for name, es in eval_sets.items():
        es = sorted_unique(np.asarray(es, np.int64))
        m = isin_sorted(np.asarray(s, np.int64), es)
        sub[name] = (es, s[m], q[m], p[m])
    rows = []
    for i, rule in enumerate(grid):
        row = {"rule": i, "name": rule_name(rule), "mode": rule["mode"]}
        for name, (es, ss, qq, pp) in sub.items():
            ps, pq_, _ = decide_assigned(ss, qq, pp, rule)
            row[name] = macro_f05(ps, pq_, gt_s, gt_q, es, n_q)
        rows.append(row)
    return pd.DataFrame(rows)


def estimate_prior_shift(p_src, y_src, p_tgt, iters=200):
    """EM label-shift estimate (Saerens et al. 2002) of the share of assigned records that are
    true: returns (source share, estimated target share). A diagnostic only."""
    pi_s = float(np.clip(np.mean(y_src), 1e-4, 1 - 1e-4))
    pi_t = pi_s
    p = np.clip(np.asarray(p_tgt, np.float64), 1e-6, 1 - 1e-6)
    for _ in range(iters):
        a = pi_t / pi_s * p
        b = (1 - pi_t) / (1 - pi_s) * (1 - p)
        new = float(np.mean(a / (a + b)))
        if abs(new - pi_t) < 1e-7:
            break
        pi_t = new
    return pi_s, pi_t


def entity_table(pred_s, pred_q, gt_s, gt_q, eval_s, n_q, reached=None):
    """Per-entity outcome for the loss breakdown. reached: true pairs present among the
    candidates per entity (aligned with eval_s) for the blocking ceiling."""
    eval_s = sorted_unique(np.asarray(eval_s, np.int64))
    f, n_true, n_pred, tp = per_entity_f05(pred_s, pred_q, gt_s, gt_q, eval_s, n_q)
    kind = np.select(
        [(n_true == 0) & (n_pred == 0), n_true == 0, f >= 1.0, n_pred == 0, tp < n_pred],
        ["no match, predicted empty", "no match, false match predicted", "perfect",
         "predicted empty (missed all)", "has a wrong match"], "missed some")
    r = pd.DataFrame({"s1": eval_s, "kind": kind, "f05": f, "n_true": n_true, "n_pred": n_pred, "tp": tp})
    if reached is not None:
        rr = reached / np.maximum(n_true, 1)
        r["ceiling"] = np.where(n_true == 0, 1.0, _f05(np.ones(len(r)), rr))
    return r


def loss_breakdown(r):
    loss = max(float((1 - r.f05).sum()), 1e-9)
    k = r.groupby("kind").agg(entities=("f05", "size"), mean_f05=("f05", "mean"))
    k["share_of_loss"] = r.groupby("kind")["f05"].apply(lambda x: (1 - x).sum()) / loss
    k["macro_points_lost"] = r.groupby("kind")["f05"].apply(lambda x: (1 - x).sum()) / len(r)
    return k.sort_values("share_of_loss", ascending=False)
