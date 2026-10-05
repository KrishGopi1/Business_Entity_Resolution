"""
Stage-2 re-ranker (v4 architecture 5.3): decides the cut instead of v3's ranker, which stays
as a cheap stage-1 feature.

  * objective: XGBoost rank:pairwise grouped by record (the job is ordering the candidates of
    one record; a pointwise loss spends its capacity on easy negatives)
  * features: v3's 44 ranker features (record-relative ones recomputed over the expanded set),
    the stage-1 score and rank, and the cheap v4 features (expand4.CHEAP4)
  * training: records of training entities only (v3's exclusion rule: no record of a tune or
    holdout entity, no stress-only record), whole v3 chunks sampled at random, the whole
    expanded candidate list of each record; half of the sampled records fit the model, the
    other half choose the floor
  * cut: keep the s2_keep_min best, plus score >= floor and rank <= s2_keep_max, plus the
    address rescue; the floor is the highest that keeps s2_target of the held-out records'
    expanded true pairs
"""
import numpy as np
import pandas as pd

import blocking as bk
import expand4 as ex

PARAMS = dict(objective="rank:pairwise", lambdarank_pair_method="mean",
              lambdarank_num_pair_per_sample=8, eval_metric="ndcg@10", eta=0.1, max_depth=6,
              subsample=0.8, colsample_bytree=0.8, min_child_weight=1, tree_method="hist")
KEEP_MAXES = (10, 15, 20, 25, 30, 40, 60)
QUANTS = (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)


def fit(X, y, q, n_jobs, rounds, depth, seed, path, device="cpu", log=print):
    """Fits on groups (records) with at least one true pair; early stopping on a slice of them.
    X rows must be grouped by q (they are sorted by record)."""
    import xgboost as xgb
    grp_has = bk.group_max(q, y.astype(np.float32)) > 0
    X, y, q = X[grp_has], y[grp_has], q[grp_has]
    es = (q * 2654435761 % 1000) < 100
    params = {**PARAMS, "max_depth": depth, "nthread": n_jobs, "seed": seed}
    if device == "cuda":
        params["device"] = "cuda"
    dtr = xgb.DMatrix(X[~es], label=y[~es].astype(np.float32), qid=q[~es], nthread=n_jobs,
                      feature_names=list(ex.S2_FEATS))
    des = xgb.DMatrix(X[es], label=y[es].astype(np.float32), qid=q[es], nthread=n_jobs,
                      feature_names=list(ex.S2_FEATS))
    bst = xgb.train(params, dtr, rounds, evals=[(des, "es")], early_stopping_rounds=30,
                    verbose_eval=False)
    bst = bst[: bst.best_iteration + 1]
    bst.set_param({"device": "cpu"})
    bst.save_model(path)
    log(f"  stage 2: {bst.num_boosted_rounds()} trees on {len(np.unique(q[~es])):,} records "
        f"({(~es).sum():,} pairs)")
    imp = bst.get_score(importance_type="gain")
    return bst, dict(sorted(imp.items(), key=lambda kv: -kv[1]))


def curve(q, s2, y, X, keep_min, rescue, rescue_cos):
    """Share of the held-out records' expanded true pairs kept, and kept pairs per record, for
    keep_max in KEEP_MAXES and floors at quantiles of the stage-2 score."""
    rank = bk.group_rank(q, s2)
    resc = np.zeros(len(q), bool)
    if rescue > 0:
        resc = bk.address_rescue(X[:, :len(bk.RANK_FEATS)], rescue, rescue_cos)
    n_q = max(len(np.unique(q)), 1)
    floors = sorted(set(np.quantile(s2, QUANTS).tolist())) if len(s2) else [0.0]
    rows = []
    for km in KEEP_MAXES:
        for t in [-np.inf] + floors:
            keep = (rank <= keep_min) | ((s2 >= t) & (rank <= km)) | resc
            rows.append({"keep_max": km, "floor": float(t), "true_kept": float(keep[y].mean()) if y.any() else 1.0,
                         "pairs_per_record": float(keep.sum() / n_q)})
    return pd.DataFrame(rows)


def choose(cur, keep_max, target):
    c = cur[cur.keep_max == keep_max].sort_values("floor")
    goal = min(target, float(c.true_kept.max()))
    ok = c[c.true_kept >= goal - 1e-12]
    row = ok.iloc[-1]
    return float(row.floor), row


def v3_curve(q, rank_pos, y, keep_min, maxes=(20, 25, 50, 100)):
    """The same held-out records cut by v3's stage-1 rank, for comparison."""
    n_q = max(len(np.unique(q)), 1)
    rows = []
    for km in maxes:
        keep = rank_pos <= max(km, keep_min)
        rows.append({"keep_max": km, "true_kept": float(keep[y].mean()) if y.any() else 1.0,
                     "pairs_per_record": float(keep.sum() / n_q)})
    return pd.DataFrame(rows)
