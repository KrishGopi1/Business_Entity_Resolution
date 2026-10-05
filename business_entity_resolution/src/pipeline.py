"""
Stages of the v4 pipeline. The first three are v3's, unchanged, and are normally NOT run again:
v4 reads their cached results from your existing work/ folder (fingerprints must match; see
check_v3_cache in pipeline4.py).

    prepare            (v3) read TSVs, training universe, S1 roles, normalization, partitions
    block              (v3) retrieval contexts, candidate ranker, up to 100 candidates per record
    evaluate-blocking  (v3) v3's recall report
    context4           v4 views and matrices per partition (new keys, loose romanization ...)
    dense              optional bi-encoder channel and cosine feature (--dense)
    stage2             new retrieval sources added to v3's candidates; stage-2 re-ranker fitted
    cut                stage-2 cut of every training-universe record (test is cut inside predict)
    report4            recall / ceiling report of the v4 candidate set, v4 gates
    train              pair features, grouped-fold XGBoost, calibration and rule at test density
    cross              optional cross-encoder on contested records, stacker, rule re-fitted (--cross)
    evaluate           holdout report
    predict            test: expansion, stage 2, pair model, decision, submission files
"""
import hashlib
import json
import os
import re
import shutil
import time
import zipfile
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import blocking as bk
import indexing as ix
import normalize as nz
import schema

V3_STAGES = ["prepare", "block", "evaluate-blocking"]
V4_STAGES = ["context4", "dense", "stage2", "cut", "report4", "train", "cross", "evaluate", "predict"]
STAGES = V3_STAGES + V4_STAGES


@dataclass
class Config:
    data_dir: str = "dataset"
    work_dir: str = "work"
    out_dir: str = "output"
    n_jobs: int = 0                  # 0 = CPU quota - 1 (reads the cgroup limit)
    seed: int = 42
    force: bool = False
    # training universe
    train_scale: float = 1.0         # S1 per train country = test S1 count x this (max: all)
    fallback_frac: float = 0.3       # for a train country absent from test
    stress: bool = True              # also block extra decoy records for the density-stress check
    train_s1: int = 100_000          # S1 entities whose pairs train the model
    tune_s1: int = 40_000            # calibration, keep settings and decision rule
    holdout_s1: int = 40_000         # untouched final score
    test_frac: float = 1.0           # < 1 only for smoke tests
    # retrieval
    channels: tuple = ix.CHANNELS
    blocks: tuple = ix.BLOCKS
    k_name: int = 40
    k_addr: int = 40
    k_joint: int = 40
    k_extra: int = 10                # fetched beyond k so ties with the k-th can be kept
    k_noaddr: int = 120              # name top-k for records with no searchable address
    block_cap: int = 100             # a rescue-block key is used only if <= this many S1 share it
    df_cap_frac: float = 0.02        # soft df cap: share of the partition's S1 records
    min_df_cap: int = 2000
    fallback_share: float = 0.5
    n_fallback: int = 4
    n_always: int = 2                # rarest common features every record keeps for retrieval
    # candidate ranker and keep policy
    ranker_sample: int = 200_000     # training records whose union fits the ranker
    ranker_rounds: int = 80
    keep_min: int = 5                # a record's best candidates by ranker score, always kept
    keep_max: int = 100              # never more than this many per record
    floor: float = -1.0              # ranker score floor; -1 = choose for target_kept_share
    target_kept_share: float = 0.999 # share of the union's true pairs the floor must keep
    post_keep_max: int = 0           # >0: model stages use only candidates ranked <= this ...
    post_floor: float = -1.0         # ... and with ranker score >= this (a subset of the blocked
                                     # candidates, so no re-blocking); keep_min best always kept
    chunk_rows: int = 10_000
    block_procs: int = 0             # 0 = one per usable CPU (the main process only waits)
    # address rescue (v3-fast): keep a pair on address evidence alone, whatever its ranker
    # score: among the record's addr_rescue best address matches, a shared non-postcode
    # number, address cosine >= rescue_min_cos, postcodes not different. 0 = off (= v3).
    addr_rescue: int = 2
    rescue_min_cos: float = 0.4
    # XGBoost device for the pair model's training and scoring (the small candidate ranker
    # stays on the CPU, so it is identical to v3): auto = cuda when XGBoost can use a GPU here
    device: str = "auto"
    # recall gates (evaluated on tune + holdout entities, never seen by the ranker)
    gate_union: float = 0.998
    gate_retained: float = 0.996
    gate_ceiling: float = 0.995
    gate_segment: float = 0.990
    gate_min_segment_pairs: int = 500
    allow_low_recall: bool = False
    # model
    max_rounds: int = 1500
    early_stop: int = 50
    n_folds: int = 3
    two_pass: bool = False           # ablation only
    tune_density: str = "stress"     # density the decision rule is tuned at: stress | standard
    decision_mode: str = None        # override the tuned rule at predict time
    min_prob: float = None
    audit_country: str = "france"    # test country for the diagnostics audit sample
    tune_density: str = "stress"     # density the decision rule is tuned at: stress | standard
    # ---- v4 ----------------------------------------------------------------------------
    v4_dir: str = ""                 # v4 outputs (default <work>/v4); v3's work/ is only read
    trust_v3_cache: bool = False     # use v3's cached files even if their fingerprints differ
    allow_reblock: bool = False      # let the v3 block stage run again (6+ hours) if stale
    # stage-1 set: v3's stored candidates, optionally trimmed (100 / 0 = all of them)
    s1_keep_min: int = 5
    s1_keep_max: int = 100
    s1_floor: float = 0.0
    # new retrieval sources
    bigkey: bool = True              # ranked big-key blocks
    big_cap: int = 5000              # keys shared by more S1s than this stay skipped
    big_top: int = 20                # S1s kept per big key, by address cosine
    k4: int = 40                     # top-k of the v4 name channels
    src_desp: bool = True
    src_loose: bool = True
    src_nlchan: bool = True
    src_dschan: bool = True
    # bi-encoder (optional): off | pretrained | finetune
    dense: str = "off"
    dense_model: str = "intfloat/multilingual-e5-small"
    dense_k: int = 20
    dense_epochs: int = 1
    dense_pairs: int = 1_000_000
    dense_batch: int = 256
    cross: str = "off"               # off | finetune | pretrained | kaggle (cross-encoder on contested records)
    cross_scores: str = ""           # --cross kaggle: cross_scores.parquet from kaggle_cross.py
    cross_model: str = "intfloat/multilingual-e5-small"
    cross_pairs: int = 600_000
    cross_epochs: int = 1
    cross_batch: int = 64
    cross_lo: float = 0.05           # contested: best p in (cross_lo, cross_hi) ...
    cross_hi: float = 0.95
    cross_margin: float = 0.2        # ... or best minus second < cross_margin
    # stage-2 re-ranker and cut
    s2_sample: int = 150_000         # training records whose expanded sets fit stage 2
    s2_rounds: int = 400
    s2_depth: int = 6
    s2_keep_min: int = 5
    s2_keep_max: int = 25
    s2_keep_noaddr: int = 25         # records without an address keep their best N regardless of the floor
    s2_floor: float = -1.0           # -1 = choose for s2_target on held-out training records
    s2_target: float = 0.998         # share of the expanded set's true pairs the cut keeps
    # pair model
    model_s1: int = 300_000          # training entities: TRAIN role + background S1 up to this
    eta: float = 0.08
    max_depth: int = 8
    # v4 gates (reported; train refuses only with --strict-gates)
    gate4_ceiling: float = 0.9975
    gate4_holdout_stress: float = 0.993
    gate4_gap: float = 0.005
    strict_gates: bool = False
    predict_gpu: bool = False        # pair model on the GPU inside predict (limits predict to 8
                                     # processes; the CPU default uses every core for features)

    @classmethod
    def quick(cls, **kw):
        base = dict(work_dir="work_quick", out_dir="output_quick", train_scale=0.03,
                    train_s1=8_000, tune_s1=3_000, holdout_s1=3_000, test_frac=0.03,
                    ranker_sample=30_000, max_rounds=300, min_df_cap=200)
        base.update(kw)
        return cls(**base)

    @property
    def jobs(self):
        return schema.default_jobs(self.n_jobs)

    @property
    def procs(self):
        return self.block_procs or schema.usable_cpus()

    @property
    def xgb_device(self):
        return resolve_device(self.device)

    def retrieval(self):
        return {"channels": list(self.channels), "blocks": list(self.blocks),
                "k": {"name": self.k_name, "addr": self.k_addr, "joint": self.k_joint},
                "k_noaddr": self.k_noaddr,
                "k_extra": self.k_extra, "block_cap": self.block_cap,
                "keep_min": self.keep_min, "keep_max": self.keep_max, "floor": 0.0,
                # v4: the block stage is v3's exactly; the address rescue runs in the stage-2 cut
                "addr_rescue": 0, "rescue_min_cos": self.rescue_min_cos}


_T0 = time.time()
_DEVICE = {}


def resolve_device(device):
    """'cuda' if XGBoost here was built with CUDA and can train on a visible GPU, else 'cpu'.
    'auto' probes once per process; 'cuda' that does not work stops with the reason."""
    if device == "cpu":
        return "cpu"
    if device not in _DEVICE:
        import warnings
        import xgboost as xgb
        ok, why = False, "XGBoost was built without CUDA"
        if xgb.build_info().get("USE_CUDA"):
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    X = np.random.default_rng(0).random((256, 4), dtype=np.float32)
                    d = xgb.QuantileDMatrix(X, label=(X[:, 0] > 0.5).astype(np.float32))
                    xgb.train({"device": "cuda", "tree_method": "hist", "verbosity": 1}, d, 2)
                bad = [str(x.message) for x in w if "cuda" in str(x.message).lower()
                       or "gpu" in str(x.message).lower()]
                ok, why = not bad, "; ".join(bad)
            except Exception as e:                               # noqa: BLE001
                why = str(e).splitlines()[0]
        _DEVICE[device] = "cuda" if ok else "cpu"
        if not ok and device == "cuda":
            raise SystemExit(f"--device cuda: XGBoost cannot use a GPU here ({why})")
        if device == "auto":
            log(f"XGBoost device: {_DEVICE[device]}" + ("" if ok else f" ({why})"))
    return _DEVICE[device]


def log(*msg):
    print(f"[{time.strftime('%H:%M:%S')} +{(time.time() - _T0) / 60:6.1f}m]", *msg, flush=True)


def _dir(cfg, *parts):
    p = os.path.join(cfg.work_dir, *parts)
    os.makedirs(p, exist_ok=True)
    return p


def load_json(path):
    with open(path) as f:
        return json.load(f)


def meta(cfg, split):
    p = os.path.join(cfg.work_dir, split, "meta.json")
    if not os.path.exists(p):
        raise SystemExit(f"{p} is missing: run --stage prepare first")
    return load_json(p)


# ---------------------------------------------------------------------------
# Fingerprints: a cached result is reused only if its settings and code are unchanged
# ---------------------------------------------------------------------------

_SRC = os.path.dirname(os.path.abspath(__file__))
STAGE_KEYS = {
    "prepare": (["seed", "train_scale", "fallback_frac", "stress", "train_s1", "tune_s1",
                 "holdout_s1", "test_frac"], ["normalize.py", "schema.py"]),
    "context": (["channels", "blocks", "df_cap_frac", "min_df_cap", "fallback_share",
                 "n_fallback", "n_always"], ["indexing.py"]),
    "block": (["k_name", "k_addr", "k_joint", "k_extra", "k_noaddr", "block_cap", "ranker_sample",
               "ranker_rounds", "keep_min", "keep_max", "floor", "target_kept_share"],
              ["blocking.py"]),
    "keep": (["post_keep_max", "post_floor"], []),
    "features": ([], ["features.py"]),
    "train": (["max_rounds", "early_stop", "n_folds", "two_pass", "tune_density"], ["model.py"]),
}


# v3-fast keeps v3's caches valid wherever its outputs are unchanged:
#  * files edited only for speed, and checked to write byte-identical outputs to v3
#    (tests/test_v3_equivalence.py), keep contributing their v3 hash. Any later edit that
#    changes what one of these files outputs must remove it from this table.
V3_FILE_SHA1 = {"blocking.py": "499b3b956689ec116494881a3ca7f9022919aad5",
                "model.py": "4d7642581506d6131817a77a56d582c5dd7b5def"}
#  * settings added by v3-fast enter a fingerprint only when they change outputs: the
#    address rescue only when it is on, the XGBoost device only for the trained models
#    (GPU and CPU training give slightly different trees).
# v4: the address rescue is applied in the stage-2 cut, never inside v3's block stage, so the
# block fingerprint is v3's own (his cached candidates stay valid).
NEW_KEYS = {"train": lambda cfg: ({"device": "cuda"} if cfg.xgb_device == "cuda" else {})}


def post_active(cfg):
    return cfg.post_keep_max > 0 or cfg.post_floor >= 0


def post_keep(cfg, rank_pos, rank_score, rescued=None):
    """Post-hoc keep policy on already blocked candidates (all True when it is off): a
    candidate stays if it is among the record's keep_min best, or scores >= post_floor and
    ranks <= post_keep_max. v3-fast: address-rescued pairs (`rescued`) always stay, like the
    keep_min best, since the rescue exists for pairs the ranker scores low."""
    rank_pos = np.asarray(rank_pos, np.float64)
    if not post_active(cfg):
        return np.ones(len(rank_pos), bool)
    km = cfg.post_keep_max if cfg.post_keep_max > 0 else cfg.keep_max
    fl = max(cfg.post_floor, 0.0)
    score = np.nan_to_num(np.asarray(rank_score, np.float64), nan=-1.0)
    keep = (rank_pos <= cfg.keep_min) | ((score >= fl) & (rank_pos <= km))
    if rescued is not None:
        keep |= np.asarray(rescued, bool)
    return keep


def _file_sha1(f):
    if f in V3_FILE_SHA1:
        return bytes.fromhex(V3_FILE_SHA1[f])
    with open(os.path.join(_SRC, f), "rb") as fh:
        return hashlib.sha1(fh.read()).digest()


def fingerprint(cfg, stage, new_keys=True):
    """new_keys=False gives the fingerprint without v3-fast's added settings: the candidate
    ranker uses it, because the address rescue changes which pairs are kept, not the ranker."""
    order = list(STAGE_KEYS)
    h = hashlib.sha1()
    for st in order[: order.index(stage) + 1]:
        keys, files = STAGE_KEYS[st]
        h.update(json.dumps({k: getattr(cfg, k) for k in keys}, sort_keys=True, default=list).encode())
        extra = NEW_KEYS[st](cfg) if new_keys and st in NEW_KEYS else {}
        if extra:
            h.update(json.dumps(extra, sort_keys=True).encode())
        for f in files:
            h.update(_file_sha1(f))
    h.update(os.path.abspath(cfg.data_dir).encode())
    return h.hexdigest()[:16]


def _fp_ok(cfg, path, stage, new_keys=True):
    if cfg.force or not os.path.exists(path):
        return False
    return load_json(path).get("fp") == fingerprint(cfg, stage, new_keys)


def _write_fp(path, cfg, stage, new_keys=True, **extra):
    bk.save_json(path, {"fp": fingerprint(cfg, stage, new_keys), **extra})


# ---------------------------------------------------------------------------
# Stage 1: prepare
# ---------------------------------------------------------------------------

def _part_name(label, used):
    base = re.sub(r"[^a-z0-9]+", "_", label).strip("_") or "blank"
    name, i = base, 1
    while name in used:
        i += 1
        name = f"{base}_{i}"
    used.add(name)
    return name


def sample_universe(cfg, s1_country, q_country, gs, gq, test_counts, rng):
    """Density-matched training universe, per country:
      * S1 entities sampled at frac = min(1, test S1 / train S1) x train_scale, with ALL their
        true records (every sampled S1 keeps its full match set);
      * unmatched (decoy) records sampled at the same frac (standard density);
      * stress records: more real look-alike records whose entity is not in the index (the
        remaining decoys, then records of train S1s left out of the universe), added until
        the country reaches test's S2/S3 per S1, as far as the data allows. They are blocked
        like any record but used only by the density-stress evaluation."""
    labels, codes = np.unique(s1_country, return_inverse=True)
    counts = np.bincount(codes)
    matched = np.zeros(len(q_country), bool)
    matched[gq] = True
    frac = {}
    for lab, n in zip(labels, counts):
        t = test_counts.get(lab, {}).get("s1", 0)
        f = min(1.0, t / n) if t > 0 else cfg.fallback_frac
        frac[lab] = min(1.0, f * cfg.train_scale)
    s1_keep = rng.random(len(codes)) < np.array([frac[x] for x in labels])[codes]
    q_keep = np.zeros(len(q_country), bool)
    q_keep[gq[s1_keep[gs]]] = True
    qf = pd.Series(q_country).map(frac).fillna(cfg.fallback_frac * cfg.train_scale).to_numpy(float)
    decoy_draw = rng.random(len(q_country))
    q_keep |= ~matched & (decoy_draw < qf)
    extra = np.zeros(len(q_country), bool)
    s1_out = np.ones(len(q_country), bool)                 # records of S1s outside the universe
    s1_out[gq[s1_keep[gs]]] = False
    orphan = matched & s1_out
    info = {}
    for lab in labels:
        qm = q_country == lab
        n_s = int(s1_keep[s1_country == lab].sum())
        have = int((q_keep & qm).sum())
        t = test_counts.get(lab, {})
        target = t.get("q", 0) / t["s1"] * n_s if t.get("s1") else have
        need = max(0, int(round(target - have)))
        pool = np.flatnonzero(qm & ~q_keep & (~matched | orphan))
        pool = pool[np.argsort(~matched[pool], kind="stable")[::-1]]  # decoys first, then orphans
        take = pool[:need] if cfg.stress else pool[:0]
        extra[take] = True
        info[lab] = {"s1_frac": round(float(frac[lab]), 4), "S1": n_s, "records": have,
                     "records_per_S1": round(have / max(n_s, 1), 3),
                     "test_records_per_S1": round(t.get("q", 0) / t["s1"], 3) if t.get("s1") else None,
                     "stress_records": int(len(take)),
                     "stress_records_per_S1": round((have + len(take)) / max(n_s, 1), 3)}
    return s1_keep, q_keep | extra, extra, info


def prepare(cfg, split):
    d = _dir(cfg, split)
    mpath = os.path.join(d, "meta.json")
    if _fp_ok(cfg, mpath, "prepare"):
        log(f"prepare {split}: cached")
        return
    if os.path.exists(mpath) and cfg.trust_v3_cache and not cfg.force:
        log(f"prepare {split}: using the cached result (--trust-v3-cache)")
        return
    if os.path.exists(mpath) and not cfg.force and not cfg.allow_reblock:
        import pipeline4
        raise SystemExit(pipeline4.CACHE_HELP)
    for sub in ("norm", "ctx", "cand", "feat", "pred", "score", "strings", "labels"):
        shutil.rmtree(os.path.join(d, sub), ignore_errors=True)
    rng = np.random.default_rng(cfg.seed + (0 if split == "train" else 1))
    log(f"prepare {split}: reading {os.path.join(cfg.data_dir, split)}")
    s1 = schema.read_tsv(schema.source_path(cfg.data_dir, split, 1))
    q = pd.concat([schema.read_tsv(schema.source_path(cfg.data_dir, split, k)).assign(src=k)
                   for k in (2, 3)], ignore_index=True)
    s1_country = schema.normalize_country(s1["country"])
    q_country = schema.normalize_country(q["country"])
    log(f"  S1 {len(s1):,}  S2+S3 {len(q):,}  countries {sorted(set(s1_country))}")
    extra = np.zeros(len(q), bool)
    info = {}
    if split == "train":
        gt_s_id, gt_q_id = schema.explode_truth(schema.read_tsv(schema.source_path(cfg.data_dir, split, "gt")))
        gs = pd.Index(s1["entity_id"]).get_indexer(gt_s_id)
        gq = pd.Index(q["entity_id"]).get_indexer(gt_q_id)
        ok = (gs >= 0) & (gq >= 0)
        if not ok.all():
            log(f"  warning: {(~ok).sum():,} ground-truth ids not found in the source files")
        gs, gq = gs[ok], gq[ok]
        log(f"  true pairs {len(gs):,}; with different S1 / record country labels "
            f"{(s1_country[gs] != q_country[gq]).mean():.4%}")
        tdir = os.path.join(cfg.data_dir, "test")
        test_counts = {}
        if os.path.exists(os.path.join(tdir, "test_source1.tsv")):
            ts = pd.Series(schema.normalize_country(schema.read_tsv(
                os.path.join(tdir, "test_source1.tsv"), ["country"])["country"]))
            tq = pd.Series(np.concatenate([schema.normalize_country(schema.read_tsv(
                os.path.join(tdir, f"test_source{k}.tsv"), ["country"])["country"]) for k in (2, 3)]))
            vs, vq = ts.value_counts().to_dict(), tq.value_counts().to_dict()
            test_counts = {k: {"s1": vs.get(k, 0), "q": vq.get(k, 0)} for k in set(vs) | set(vq)}
        s1_keep, q_keep, extra, info = sample_universe(cfg, s1_country, q_country, gs, gq, test_counts, rng)
        for lab, v in info.items():
            log(f"  universe {lab or '<blank>'}: {v}")
    else:
        s1_keep = rng.random(len(s1)) < cfg.test_frac if cfg.test_frac < 1 else np.ones(len(s1), bool)
        q_keep = rng.random(len(q)) < cfg.test_frac if cfg.test_frac < 1 else np.ones(len(q), bool)
        if cfg.test_frac < 1:
            log(f"  SMOKE TEST: {cfg.test_frac:.0%} of test records (not a valid submission)")
    s1_idx, q_idx = np.flatnonzero(s1_keep), np.flatnonzero(q_keep)
    extra = extra[q_idx]
    s1 = s1.iloc[s1_idx].reset_index(drop=True)
    q = q.iloc[q_idx].reset_index(drop=True)
    s1_country, q_country = s1_country[s1_idx], q_country[q_idx]
    pd.DataFrame({"entity_id": s1["entity_id"], "country": s1_country}).to_parquet(
        os.path.join(d, "ids_s1.parquet"), index=False)
    pd.DataFrame({"entity_id": q["entity_id"], "country": q_country, "src": q["src"].astype("int8"),
                  "extra": extra}).to_parquet(os.path.join(d, "ids_q.parquet"), index=False)
    if split == "train":
        s_new = np.full(len(s1_keep), -1, np.int64)
        s_new[s1_idx] = np.arange(len(s1_idx))
        q_new = np.full(len(q_keep), -1, np.int64)
        q_new[q_idx] = np.arange(len(q_idx))
        keep = s1_keep[gs]
        np.savez(os.path.join(d, "gt.npz"), s=s_new[gs[keep]].astype(np.int32),
                 q=q_new[gq[keep]].astype(np.int32))
        n_u = len(s1_idx)
        perm = rng.permutation(n_u)
        n_tune = min(cfg.tune_s1, n_u // 6)
        n_hold = min(cfg.holdout_s1, n_u // 6)
        n_train = min(cfg.train_s1, n_u - n_tune - n_hold)
        roles = np.zeros(n_u, np.int8)                  # 0 background 1 train 2 tune 3 holdout
        roles[perm[:n_tune]] = 2
        roles[perm[n_tune:n_tune + n_hold]] = 3
        roles[perm[n_tune + n_hold:n_tune + n_hold + n_train]] = 1
        np.save(os.path.join(d, "roles.npy"), roles)
        log(f"  universe: S1 {n_u:,} (train {n_train:,}, tune {n_tune:,}, holdout {n_hold:,}, "
            f"background {n_u - n_train - n_tune - n_hold:,}); records {len(q_idx):,} "
            f"(stress-only {int(extra.sum()):,}); true pairs {int(keep.sum()):,}")

    log(f"  normalizing {len(s1):,} S1 and {len(q):,} S2/S3 records with {cfg.jobs} processes")
    s1_tab = nz.normalize_records(s1["business_name"], s1["business_address"], cfg.jobs)
    s1_tab = s1_tab.append_column("gid", pa.array(np.arange(len(s1), dtype=np.int32)))
    q_tab = nz.normalize_records(q["business_name"], q["business_address"], cfg.jobs)
    q_tab = q_tab.append_column("gid", pa.array(np.arange(len(q), dtype=np.int32)))
    q_tab = q_tab.append_column("src", pa.array(q["src"].to_numpy(np.int8)))
    del s1, q
    nd = _dir(cfg, split, "norm")
    pq.write_table(s1_tab, os.path.join(nd, "s1_all.parquet"))
    pq.write_table(q_tab, os.path.join(nd, "q_all.parquet"))
    blank = s1_country == ""
    labels = sorted(set(s1_country) - {""})
    used, parts = set(), []
    for lab in labels:
        name = _part_name(lab, used)
        sm, qm = (s1_country == lab) | blank, q_country == lab
        pq.write_table(s1_tab.filter(pa.array(sm)), os.path.join(nd, f"s1_{name}.parquet"))
        pq.write_table(q_tab.filter(pa.array(qm)), os.path.join(nd, f"q_{name}.parquet"))
        parts.append({"name": name, "label": lab, "n_s1": int(sm.sum()), "n_q": int(qm.sum())})
    orphan = ~pd.Series(q_country).isin(set(labels)).to_numpy()
    if orphan.any():                                   # blank / unknown country: all S1
        name = _part_name("unknown_country", used)
        pq.write_table(s1_tab, os.path.join(nd, f"s1_{name}.parquet"))
        pq.write_table(q_tab.filter(pa.array(orphan)), os.path.join(nd, f"q_{name}.parquet"))
        parts.append({"name": name, "label": "*", "n_s1": int(len(s1_country)), "n_q": int(orphan.sum())})
    for p in parts:
        log(f"  partition {p['label']}: S1 {p['n_s1']:,}  S2/S3 {p['n_q']:,}")
    m = {"split": split, "n_s1": int(len(s1_country)), "n_q": int(len(q_country)),
         "partitions": parts, "universe": info,
         "smoke_test": bool(split == "test" and cfg.test_frac < 1)}
    _write_fp(mpath, cfg, "prepare", **m)


def load_tables(cfg, split, part, s1_cols=None, q_cols=None):
    nd = os.path.join(cfg.work_dir, split, "norm")
    return (pq.read_table(os.path.join(nd, f"s1_{part['name']}.parquet"), columns=s1_cols),
            pq.read_table(os.path.join(nd, f"q_{part['name']}.parquet"), columns=q_cols))


# ---------------------------------------------------------------------------
# Stage 2: block
# ---------------------------------------------------------------------------

def _local_truth(s_gid, q_gid, gt):
    s_loc = np.full(int(max(s_gid.max(initial=0), gt["s"].max(initial=0))) + 1, -1, np.int64)
    s_loc[s_gid] = np.arange(len(s_gid))
    q_loc = np.full(int(max(q_gid.max(initial=0), gt["q"].max(initial=0))) + 1, -1, np.int64)
    q_loc[q_gid] = np.arange(len(q_gid))
    ts, tq = s_loc[gt["s"]], q_loc[gt["q"]]
    ok = (ts >= 0) & (tq >= 0)
    return bk.sorted_unique(tq[ok] * len(s_gid) + ts[ok])


def build_contexts(cfg, split):
    parts = meta(cfg, split)["partitions"]
    gt = np.load(os.path.join(cfg.work_dir, "train", "gt.npz")) if split == "train" else None
    for part in parts:
        cdir = os.path.join(cfg.work_dir, split, "ctx", part["name"])
        fpp = os.path.join(cdir, "_fp.json")
        if _fp_ok(cfg, fpp, "context"):
            continue
        shutil.rmtree(cdir, ignore_errors=True)
        log(f"context {split}/{part['name']}: {part['n_s1']:,} S1 x {part['n_q']:,} S2/S3")
        s_tab, q_tab = load_tables(cfg, split, part)
        truth = _local_truth(s_tab["gid"].to_numpy(), q_tab["gid"].to_numpy(), gt) if gt is not None else None
        ix.build_context(s_tab, q_tab, cdir, cfg.df_cap_frac, cfg.min_df_cap, cfg.fallback_share,
                         cfg.n_fallback, cfg.n_always, cfg.jobs, truth, log)
        _write_fp(fpp, cfg, "context")


def fit_candidate_ranker(cfg):
    """Fits the ranker on the union of sampled training records (never records of tune or
    holdout entities, never stress-only records); half of the sampled records fit it, the
    other half choose the score floor."""
    rdir = _dir(cfg, "train", "ranker")
    info_path = os.path.join(rdir, "ranker.json")
    if _fp_ok(cfg, info_path, "block", new_keys=False):
        return load_json(info_path)
    m = meta(cfg, "train")
    gt = np.load(os.path.join(cfg.work_dir, "train", "gt.npz"))
    roles = np.load(os.path.join(cfg.work_dir, "train", "roles.npy"))
    ids_q = pd.read_parquet(os.path.join(cfg.work_dir, "train", "ids_q.parquet"), columns=["extra"])
    excluded = ids_q["extra"].to_numpy().copy()
    excluded[gt["q"][roles[gt["s"]] >= 2]] = True
    parts = [p for p in m["partitions"] if p["label"] != "*"]
    n_all = sum(p["n_q"] for p in parts)
    rng = np.random.default_rng(cfg.seed + 11)
    Xs, ys, gs, srcs, qoff = [], [], [], [], 0
    rcfg = cfg.retrieval()
    log(f"ranker: sampling up to {cfg.ranker_sample:,} training records")
    for part in parts:
        q_gid = pq.read_table(os.path.join(cfg.work_dir, "train", "norm", f"q_{part['name']}.parquet"),
                              columns=["gid"])["gid"].to_numpy()
        ex = excluded[q_gid]
        n_take = min(int((~ex).sum()), int(round(cfg.ranker_sample * part["n_q"] / max(n_all, 1))))
        # whole blocks of consecutive records: the union is computed per run of consecutive
        # rows, and each call has a fixed cost (about 0.1 s on a large partition), so single
        # scattered rows would be ~1000x slower per record than blocks
        block_len = 1000
        rows, got = [], 0
        for b in rng.permutation(-(-len(q_gid) // block_len)):
            if got >= n_take:
                break
            r = np.arange(b * block_len, min(len(q_gid), (b + 1) * block_len))
            rows.append(r)
            got += int((~ex[r]).sum())
        rows = np.sort(np.concatenate(rows)) if rows else np.empty(0, np.int64)
        u = bk.sample_union(os.path.join(cfg.work_dir, "train", "ctx", part["name"]), rows, rcfg, cfg.procs)
        if u is None:
            continue
        keep = ~ex[u["q"]]                                  # drop tune / holdout / stress records
        u = {k: v[keep] for k, v in u.items()}
        log(f"  {part['name']}: {n_take:,} records, union {len(u['q']) / max(n_take, 1):.1f} per record")
        Xs.append(u["X"])
        ys.append(u["y"].astype(bool))
        gs.append(u["q"] + qoff)
        srcs.append(u["src"][u["y"].astype(bool)])
        qoff += int(part["n_q"])
    X, y, g = np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)
    fit_half = (g * 2654435761 % 1000) < 500
    log(f"  sampled union: {len(y):,} pairs, {y.sum():,} true; fitting on {fit_half.sum():,} pairs")
    path = os.path.join(rdir, "ranker_model.json")
    bst, imp = bk.fit_ranker(X[fit_half], y[fit_half], cfg.jobs, cfg.ranker_rounds, cfg.seed, path)
    score = bst.inplace_predict(X[~fit_half])
    curve = bk.keep_curve(g[~fit_half], score, y[~fit_half], cfg.keep_min,
                          sorted({30, 50, 80, cfg.keep_max, 150}), bk.FLOOR_GRID)
    if cfg.floor >= 0:
        floor, best = cfg.floor, float(curve[curve.keep_max == cfg.keep_max].union_share_kept.max())
    else:
        floor, best = bk.choose_floor(curve, cfg.keep_max, cfg.target_kept_share)
    sel = curve[(curve.keep_max == cfg.keep_max) & (curve.floor == floor)].iloc[0]
    log("  ranker keep curve (held-out sampled records; share of the union's true pairs kept):\n"
        + curve.pivot(index="floor", columns="keep_max", values="union_share_kept").round(5).to_string())
    log(f"  floor {floor:g} with keep_max {cfg.keep_max}: keeps {sel.union_share_kept:.5f} of the "
        f"union's true pairs (best possible {best:.5f}), {sel.pairs_per_record:.1f} pairs per record")
    info = {"model": os.path.basename(path), "floor": floor, "keep_max": cfg.keep_max,
            "keep_min": cfg.keep_min, "held_out_share_kept": float(sel.union_share_kept),
            "held_out_pairs_per_record": float(sel.pairs_per_record),
            "importance_gain": imp, "curve": curve.to_dict(orient="records")}
    _write_fp(info_path, cfg, "block", new_keys=False, **info)
    return load_json(info_path)


def block(cfg, split):
    build_contexts(cfg, split)
    info = fit_candidate_ranker(cfg) if split == "train" else load_json(
        os.path.join(cfg.work_dir, "train", "ranker", "ranker.json"))
    if split == "test" and info.get("fp") != fingerprint(cfg, "block", new_keys=False):
        raise SystemExit("the candidate ranker was fitted with other settings: run --stage block "
                         "on train first (running all stages does this)")
    rcfg = dict(cfg.retrieval(), floor=float(info["floor"]))
    ranker = os.path.join(cfg.work_dir, "train", "ranker", info["model"])
    for part in meta(cfg, split)["partitions"]:
        out = os.path.join(cfg.work_dir, split, "cand", part["name"])
        fpp = os.path.join(out, "_fp.json")
        if _fp_ok(cfg, fpp, "block"):
            log(f"block {split}/{part['name']}: cached")
            continue
        partial = os.path.join(out, "_fp.partial.json")
        if not (os.path.exists(partial) and load_json(partial).get("fp") == fingerprint(cfg, "block")
                and not cfg.force):
            shutil.rmtree(out, ignore_errors=True)
        os.makedirs(out, exist_ok=True)
        _write_fp(partial, cfg, "block")
        log(f"block {split}/{part['name']}: {part['n_q']:,} records against {part['n_s1']:,} S1")
        st = bk.block_partition(os.path.join(cfg.work_dir, split, "ctx", part["name"]), out,
                                part["n_q"], rcfg, ranker, cfg.procs, cfg.chunk_rows, log)
        if st["big_keys"]:
            log(f"    rescue-block keys skipped as larger than {cfg.block_cap}: {st['big_keys']}")
        os.replace(partial, fpp)
    for sub in ("feat", "pred", "score"):
        shutil.rmtree(os.path.join(cfg.work_dir, split, sub), ignore_errors=True)


# ---------------------------------------------------------------------------
# Stage dispatch
# ---------------------------------------------------------------------------

def run(cfg, stages):
    log(f"config: {asdict(cfg)}")
    for stage in stages:
        if stage == "prepare":
            prepare(cfg, "train")
            prepare(cfg, "test")
        elif stage == "block":
            import pipeline4
            state = pipeline4.v3_cache_state(cfg)
            if state == "ok":
                log("block: v3 candidates cached (fingerprints match); nothing to do")
                continue
            if state == "stale" and not cfg.allow_reblock:
                raise SystemExit(pipeline4.CACHE_HELP)
            block(cfg, "train")
            block(cfg, "test")
        elif stage == "evaluate-blocking":
            import evaluate
            evaluate.blocking_report(cfg)
        elif stage in V4_STAGES:
            import pipeline4
            pipeline4.run_stage(cfg, stage)
        else:
            raise ValueError(stage)
    log("done")
