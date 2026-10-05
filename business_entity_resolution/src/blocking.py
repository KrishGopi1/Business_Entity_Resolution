"""
Candidate generation (v3): wide candidate union -> small XGBoost candidate ranker -> bounded
top candidates per S2/S3 record. No S1-side truncation, no fixed top-30 ceiling.

For a chunk of S2/S3 records ("queries") of one partition, a worker:
  1. takes each channel's top-k S1 (plus ties with the k-th score) from its retrieval
     matrices, and every S1 in a rescue block of bounded size (indexing.py);
  2. forms the union, with a source bitmask per pair (which channels / blocks found it);
  3. computes cheap ranker features: exact full-vector cosines of the three channels and
     their ranks and gaps within the record, source flags, name / phonetic / address /
     number token overlap, postcode / prefix / state / house / unit / name / legal-form
     agreement, S1-side name and address frequency, missingness, script and source flags;
  4. scores the union with the ranker (XGBoost's own vectorized predict, one thread per
     worker) and keeps a pair when
         its rank in the record <= keep_min                            (always kept), or
         its score >= floor and its rank in the record <= keep_max;
  5. writes the kept pairs with all their ranker features to parquet, and for training
     partitions also every TRUE pair of the union with its rank and score (so recall at other
     keep settings can be computed later without blocking again) and per-record counts of
     union pairs above a grid of floors (so pairs-per-record at other settings is exact too).

Workers are 'spawn' processes that memory-map the saved context, so nothing is forked after
OpenMP has started and large matrices are shared through the page cache, not copied.
"""
import json
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import indexing as ix

try:
    from sparse_dot_topn import sp_matmul_topn
except ImportError:  # pragma: no cover
    sp_matmul_topn = None

BLOCK_BIT = {b: len(ix.CHANNELS) + i for i, b in enumerate(ix.BLOCKS)}
BLOCK_BIT["dba_rev"] = BLOCK_BIT["dba"]
COS = [f"cos_{c}" for c in ix.CHANNELS]
RANK_FEATS = (
    COS + [f"rk_{c}" for c in ix.CHANNELS] + [f"gap_{c}" for c in ix.CHANNELS]
    + ["n_channels", "n_blocks"] + [f"b_{b}" for b in ix.BLOCKS]
    + ["nm_jacc", "ph_jacc", "ad_jacc", "num_jacc", "num_any",
       "pc_same", "pc3_same", "st_same", "house_same", "unit_same", "nsort_same", "nbase_same",
       "legal_same", "s_name_freq", "q_name_freq", "s_addr_freq", "q_addr_missing",
       "s_addr_missing", "q_nonlatin", "q_src3", "q_union"])
FLOOR_GRID = np.array([0.0, 1e-7, 1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3],
                      np.float64)
_POP = np.array([bin(x).count("1") for x in range(1 << 16)], np.float32)


# ---------------------------------------------------------------------------
# 1. Array helpers
# ---------------------------------------------------------------------------

def sorted_unique(x):
    x = np.sort(np.asarray(x))
    return x[np.r_[True, x[1:] != x[:-1]]] if len(x) else x


def isin_sorted(a, b):
    """np.isin(a, b) for sorted unique b."""
    if len(b) == 0:
        return np.zeros(len(a), bool)
    i = np.searchsorted(b, a)
    i[i >= len(b)] = 0
    return b[i] == a


def _desc_key32(score):
    """uint32 key whose ascending order is score's descending order (float32, no NaN);
    -0.0 and 0.0 map to the same key, as they compare equal."""
    f = (-np.asarray(score, np.float32) + np.float32(0)).view(np.uint32)
    return np.where(f >> np.uint32(31), ~f, f | np.uint32(0x80000000))


def _group_order(group, score):
    """np.lexsort((-score, group)) as one stable sort of a 64-bit key (group offset in the
    high 32 bits, a float32 order key in the low 32), about 6x faster. Falls back to lexsort
    when the key cannot be exact: score not float32 / integer, NaN present, or a group span
    of 2^32 or more."""
    score = np.asarray(score)
    if len(group) and score.dtype in (np.float32, np.float16) and group.min() >= 0 \
            and int(group.max()) - int(group.min()) < 2 ** 32 and not np.isnan(score).any():
        key = ((group - group.min()).astype(np.uint64) << np.uint64(32)) \
            | _desc_key32(score).astype(np.uint64)
        return np.argsort(key, kind="stable")
    return np.lexsort((-np.asarray(score, np.float64), group))


def group_rank(group, score):
    """1-based rank of score (descending) within each group; ties keep input order."""
    group = np.asarray(group, np.int64)
    order = _group_order(group, score)
    g = group[order]
    start = np.flatnonzero(np.r_[True, g[1:] != g[:-1]]) if len(g) else np.empty(0, np.int64)
    pos = np.arange(len(g)) - np.repeat(start, np.diff(np.r_[start, len(g)]))
    rank = np.empty(len(g), np.int32)
    rank[order] = pos + 1
    return rank


def group_max(group, x):
    u, inv = np.unique(group, return_inverse=True)
    m = np.full(len(u), -np.inf, np.float32)
    np.maximum.at(m, inv, x)
    return m[inv]


def rowwise_dot(A, B, ia, ib, batch=400_000):
    """A[ia[j]] . B[ib[j]] for every j (a cosine when rows are L2-normalized)."""
    out = np.empty(len(ia), np.float32)
    for s in range(0, len(ia), batch):
        a, b = A[ia[s:s + batch]], B[ib[s:s + batch]]
        out[s:s + batch] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    return out


def _plain(M):
    """CSR with plain (not memory-mapped) float32 / int32 arrays, as sparse_dot_topn requires."""
    from scipy import sparse
    big = M.nnz >= 2 ** 31
    it = np.int64 if big else np.int32
    return sparse.csr_matrix((np.asarray(M.data, np.float32), np.asarray(M.indices, it),
                              np.asarray(M.indptr, it)), shape=M.shape, copy=False)


def topk(Q, ST, k):
    """Top-k columns of Q @ ST per row (CSR, rows unsorted), single-threaded."""
    if sp_matmul_topn is not None:
        return sp_matmul_topn(_plain(Q), _plain(ST), top_n=k, sort=False).tocsr()
    M = (Q @ ST).tocsr()                                    # scipy fallback (small data only)
    rows = np.repeat(np.arange(M.shape[0]), np.diff(M.indptr))
    keep = np.sort(np.flatnonzero(group_rank(rows, M.data) <= k))
    from scipy import sparse
    ip = np.r_[0, np.cumsum(np.bincount(rows[keep], minlength=M.shape[0]))]
    return sparse.csr_matrix((M.data[keep], M.indices[keep], ip), shape=M.shape)


def trim_ties(R, k, eps=1e-6):
    """(row, col, value) of each row's top k plus every result tied with the k-th value."""
    rows = np.repeat(np.arange(R.shape[0]), np.diff(R.indptr))
    if len(rows) == 0:
        return rows, R.indices, R.data
    rank = group_rank(rows, R.data)
    kth = np.full(R.shape[0], -np.inf, np.float32)
    m = rank == k
    kth[rows[m]] = R.data[m]
    keep = (rank <= k) | (R.data >= kth[rows] - eps)
    return rows[keep], R.indices[keep], R.data[keep]


def _tri(a, b):
    out = (a == b).astype(np.float32)
    out[(a < 0) | (b < 0)] = np.nan
    return out


_NNZ = {}


def row_nnz(M):
    """Stored entries per row of a CSR, cached per matrix object (np.diff of a whole
    partition's indptr is millions of elements, and rank_features needs it for every chunk)."""
    k = id(M)
    hit = _NNZ.get(k)
    if hit is None or hit[0] is not M:
        hit = (M, np.diff(M.indptr))
        _NNZ[k] = hit
    return hit[1]


def _jaccard(A, B, ia, ib):
    inter = rowwise_dot(A, B, ia, ib)
    na = row_nnz(A)[ia].astype(np.float32)
    nb = row_nnz(B)[ib].astype(np.float32)
    out = inter / np.maximum(na + nb - inter, 1)
    out[(na == 0) | (nb == 0)] = np.nan
    return out.astype(np.float32), inter


# ---------------------------------------------------------------------------
# 2. Union and ranker features for a range of query rows
# ---------------------------------------------------------------------------

def union(ctx, lo, hi, cfg):
    """Candidate union of query rows lo..hi-1: dict of arrays q, s (partition rows) and src."""
    n_s = ctx["n_s"]
    keys, bits, big = [], [], {}
    for ci, ch in enumerate(ix.CHANNELS):
        if ch not in cfg["channels"]:
            continue
        m = ctx["mats"][ch]
        k = cfg["k"][ch]
        R = topk(m["Qr"][lo:hi], m["STr"], k + cfg["k_extra"])
        r, c, _ = trim_ties(R, k)
        if ch == "name" and cfg.get("k_noaddr", 0) > k:     # adaptive k: no searchable address
            na = np.flatnonzero(row_nnz(ctx["mats"]["addr"]["Qr"])[lo:hi] == 0)
            if len(na):
                R2 = topk(m["Qr"][lo:hi][na], m["STr"], cfg["k_noaddr"] + cfg["k_extra"])
                r2, c2, _ = trim_ties(R2, cfg["k_noaddr"])
                r, c = np.r_[r, na[r2]], np.r_[c, c2]
        keys.append((r.astype(np.int64) + lo) * n_s + c)
        bits.append(np.full(len(r), 1 << ci, np.int32))
    for b, blk in ctx["blocks"].items():
        name = "dba" if b == "dba_rev" else b
        if name not in cfg["blocks"]:
            continue
        qr, sr, n_big = ix.block_lookup(blk, lo, hi, cfg["block_cap"])
        big[name] = big.get(name, 0) + n_big
        keys.append(qr * n_s + sr)
        bits.append(np.full(len(qr), 1 << BLOCK_BIT[b], np.int32))
    if not keys:
        return {"q": np.empty(0, np.int64), "s": np.empty(0, np.int64),
                "src": np.empty(0, np.int32)}, big
    allk, allb = np.concatenate(keys), np.concatenate(bits)
    order = np.argsort(allk, kind="stable")
    allk, allb = allk[order], allb[order]
    start = np.flatnonzero(np.r_[True, allk[1:] != allk[:-1]])
    src = np.bitwise_or.reduceat(allb, start) if len(allk) else allb
    uk = allk[start]
    return {"q": uk // n_s, "s": uk % n_s, "src": src.astype(np.int32)}, big


def rank_features(ctx, u):
    """Ranker feature matrix (RANK_FEATS order) for union pairs u."""
    q, s, src = u["q"], u["s"], u["src"]
    n = len(q)
    if n == 0:
        return np.empty((0, len(RANK_FEATS)), np.float32)
    A = ctx["arr"]
    F = {}
    for ch in ix.CHANNELS:
        m = ctx["mats"][ch]
        v = rowwise_dot(m["Q"], m["S"], q, s)
        empty = (row_nnz(m["Q"])[q] == 0) | (row_nnz(m["S"])[s] == 0)
        v[empty] = np.nan
        F[f"cos_{ch}"] = v
        z = np.nan_to_num(v, nan=0.0)
        F[f"rk_{ch}"] = group_rank(q, z).astype(np.float32)
        F[f"gap_{ch}"] = group_max(q, z) - z
    nch = len(ix.CHANNELS)
    F["n_channels"] = _POP[src & ((1 << nch) - 1)]
    F["n_blocks"] = _POP[src >> nch]
    for b in ix.BLOCKS:
        F[f"b_{b}"] = ((src >> BLOCK_BIT[b]) & 1).astype(np.float32)
    for tag, key in (("nm", "nm"), ("ph", "ph"), ("ad", "ad")):
        S_, Q_ = ctx["tok"][key]
        F[f"{tag}_jacc"] = _jaccard(Q_, S_, q, s)[0]
    S_, Q_ = ctx["tok"]["num"]
    nj, inter = _jaccard(Q_, S_, q, s)
    F["num_jacc"] = nj
    F["num_any"] = np.where(np.isnan(nj), np.nan, (inter > 0).astype(np.float32))
    for name, f in (("pc_same", "postcode"), ("pc3_same", "postcode_prefix"), ("st_same", "state"),
                    ("house_same", "house_number"), ("unit_same", "unit_number"),
                    ("nsort_same", "name_tokens_sorted"), ("nbase_same", "name_base"),
                    ("legal_same", "legal_form")):
        F[name] = _tri(A[f"s_{f}"][s], A[f"q_{f}"][q])
    F["s_name_freq"] = A["s_name_freq"][s]
    F["q_name_freq"] = A["q_name_freq"][q]
    F["s_addr_freq"] = A["s_addr_freq"][s]
    F["q_addr_missing"] = A["q_addr_missing"][q]
    F["s_addr_missing"] = A["s_addr_missing"][s]
    F["q_nonlatin"] = A["q_nonlatin"][q]
    F["q_src3"] = A["q_src3"][q]
    _, inv, cnt = np.unique(q, return_inverse=True, return_counts=True)
    F["q_union"] = cnt[inv].astype(np.float32)
    # filled column by column into a Fortran-order matrix: the same values as np.column_stack,
    # without its strided row-major writes (XGBoost predicts from either layout identically)
    X = np.empty((len(RANK_FEATS), n), np.float32)
    for j, c in enumerate(RANK_FEATS):
        X[j] = F[c]
    return X.T


# ---------------------------------------------------------------------------
# 3. Workers (spawned; the context is memory-mapped once per worker)
# ---------------------------------------------------------------------------

_W = {}


def _init_worker(ctx_dir, cfg, ranker_path):
    from schema import pin_native_threads
    pin_native_threads()
    _W["ctx"] = ix.load_context(ctx_dir)
    _W["cfg"] = cfg
    _W["ranker"] = None
    if ranker_path:
        import xgboost as xgb
        b = xgb.Booster()
        b.load_model(ranker_path)
        b.set_param({"nthread": 1})
        _W["ranker"] = b


def _truth_mask(ctx, q, s):
    tk = ctx["arr"].get("truth_keys")
    if tk is None:
        return None
    return isin_sorted(q.astype(np.int64) * ctx["n_s"] + s, tk)


def _sample_task(args):
    """Union + features + labels of the given query rows (for fitting the ranker)."""
    rows = args
    ctx, cfg = _W["ctx"], _W["cfg"]
    out = []
    for lo, hi in _runs(rows):
        u, _ = union(ctx, lo, hi, cfg)
        X = rank_features(ctx, u)
        y = _truth_mask(ctx, u["q"], u["s"])
        out.append((u["q"], u["s"], u["src"], X, y))
    cat = lambda i: np.concatenate([o[i] for o in out]) if out else np.empty(0)
    return {"q": cat(0), "s": cat(1), "src": cat(2),
            "X": np.concatenate([o[3] for o in out]) if out else np.empty((0, len(RANK_FEATS)), np.float32),
            "y": cat(4)}


def _runs(rows):
    """Sorted row ids -> contiguous (lo, hi) runs."""
    rows = np.asarray(rows, np.int64)
    if len(rows) == 0:
        return []
    br = np.flatnonzero(np.diff(rows) != 1)
    starts = np.r_[0, br + 1]
    ends = np.r_[br, len(rows) - 1]
    return [(int(rows[a]), int(rows[b]) + 1) for a, b in zip(starts, ends)]


def keep_rule(q, score, keep_min, keep_max, floor):
    rank = group_rank(q, score)
    return (rank <= keep_min) | ((score >= floor) & (rank <= keep_max)), rank


RESCUE_RANKS = (1, 2, 3)            # address-rank limits the report compares


def address_rescue(X, max_rank, min_cos):
    """Pairs kept on address evidence alone, whatever the ranker score (v3-fast).

    A pair qualifies when the S1 is among the record's `max_rank` best address matches
    (rk_addr, the exact address-channel cosine rank inside the record's union), shares a
    number other than the postcode with the record (num_any), has an address cosine of at
    least `min_cos`, and its postcodes do not disagree (equal, or one side has none).
    Every input is a ranker feature that is already computed, so the rule costs nothing,
    and it adds at most `max_rank` pairs per record."""
    col = RANK_FEATS.index
    rk, cos = X[:, col("rk_addr")], X[:, col("cos_addr")]
    num, pc = X[:, col("num_any")], X[:, col("pc_same")]
    with np.errstate(invalid="ignore"):
        return (rk <= max_rank) & (num == 1) & (cos >= min_cos) & (pc != 0)


def _block_task(args):
    """Blocks query rows lo..hi-1 and writes the chunk's parquet files. Returns stats."""
    lo, hi, out_prefix = args
    ctx, cfg = _W["ctx"], _W["cfg"]
    t0 = time.time()
    u, big = union(ctx, lo, hi, cfg)
    t1 = time.time()
    X = rank_features(ctx, u)
    t2 = time.time()
    q, s = u["q"], u["s"]
    if _W["ranker"] is not None and len(q):
        score = _W["ranker"].inplace_predict(X).astype(np.float32)
    else:                                              # no ranker: summed cosines
        score = np.nansum(X[:, :len(ix.CHANNELS)], axis=1).astype(np.float32)
    keep, rank = keep_rule(q, score, cfg["keep_min"], cfg["keep_max"], cfg["floor"])
    n_resc = cfg.get("addr_rescue", 0)
    if n_resc:
        resc = address_rescue(X, n_resc, cfg["rescue_min_cos"])
        rule_keep = keep
        keep = keep | resc
    t3 = time.time()
    A = ctx["arr"]
    cols = {"qi": q[keep].astype(np.int32), "si": s[keep].astype(np.int32),
            "qg": A["q_gid"][q[keep]].astype(np.int32), "sg": A["s_gid"][s[keep]].astype(np.int32),
            "src": u["src"][keep], "rank_score": score[keep], "rank_pos": rank[keep]}
    for j, c in enumerate(RANK_FEATS):
        cols[c] = X[keep, j]
    if n_resc:
        cols["addr_rescue"] = resc[keep] & ~rule_keep[keep]     # kept only by the rescue
    pq.write_table(pa.table(cols), out_prefix + ".pairs.parquet")
    # per-record counts: union size and union pairs at or above each floor of FLOOR_GRID
    rec_q, inv = np.unique(q, return_inverse=True)
    rec = {"qi": np.arange(lo, hi, dtype=np.int32)}
    n_union = np.zeros(hi - lo, np.int32)
    n_union[rec_q - lo] = np.bincount(inv)
    rec["n_union"] = n_union
    for t_i, t in enumerate(FLOOR_GRID):
        c = np.zeros(hi - lo, np.int32)
        c[rec_q - lo] = np.bincount(inv, weights=(score >= t)).astype(np.int32) if len(q) else 0
        rec[f"n_ge_{t_i}"] = c
    rec["n_kept"] = np.bincount(q[keep] - lo, minlength=hi - lo).astype(np.int32)
    if n_resc:                          # pairs each rescue limit would add beyond the rank rule
        for r in RESCUE_RANKS:
            extra = address_rescue(X, r, cfg["rescue_min_cos"]) & ~rule_keep
            rec[f"n_rescue_{r}"] = np.bincount(q[extra] - lo, minlength=hi - lo).astype(np.int32)
    pq.write_table(pa.table(rec), out_prefix + ".rec.parquet")
    y = _truth_mask(ctx, q, s)
    if y is not None:                                   # every true pair of the union
        tcols = {
            "qi": q[y].astype(np.int32), "si": s[y].astype(np.int32),
            "qg": A["q_gid"][q[y]].astype(np.int32), "sg": A["s_gid"][s[y]].astype(np.int32),
            "src": u["src"][y], "rank_score": score[y], "rank_pos": rank[y], "kept": keep[y]}
        if n_resc:                      # address evidence, so the report can compare limits
            tcols["rule_kept"] = rule_keep[y]
            for c in ("rk_addr", "cos_addr", "num_any", "pc_same"):
                tcols[c] = X[y, RANK_FEATS.index(c)]
        pq.write_table(pa.table(tcols), out_prefix + ".truth.parquet")
    open(out_prefix + ".done", "w").close()
    t4 = time.time()
    return {"rows": hi - lo, "union": int(len(q)), "kept": int(keep.sum()),
            "rescued": int((keep & ~rule_keep).sum()) if n_resc else 0,
            "big_keys": big, "seconds": t4 - t0,
            "phase": {"retrieval": t1 - t0, "features": t2 - t1, "ranker": t3 - t2, "write": t4 - t3}}


def _pool(n_procs, ctx_dir, cfg, ranker_path):
    if n_procs <= 1:
        _init_worker(ctx_dir, cfg, ranker_path)
        return None
    return mp.get_context("spawn").Pool(n_procs, initializer=_init_worker,
                                        initargs=(ctx_dir, cfg, ranker_path))


def _imap(pool, func, tasks):
    return pool.imap(func, tasks) if pool is not None else map(func, tasks)


# ---------------------------------------------------------------------------
# 4. Public entry points
# ---------------------------------------------------------------------------

def sample_union(ctx_dir, rows, cfg, n_procs=1, chunk=2_000):
    """Union pairs with ranker features and labels for sampled query rows."""
    rows = np.sort(np.asarray(rows, np.int64))
    tasks = [rows[i:i + chunk] for i in range(0, len(rows), chunk)]
    pool = _pool(n_procs, ctx_dir, cfg, None)
    try:
        parts = list(_imap(pool, _sample_task, tasks))
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]} if parts else None


def block_partition(ctx_dir, out_dir, n_q, cfg, ranker_path, n_procs=1, chunk_rows=10_000,
                    log=print):
    """Blocks every query row of a partition into out_dir (resumable: finished chunks are
    skipped). Returns summed stats."""
    os.makedirs(out_dir, exist_ok=True)
    tasks = []
    for i, lo in enumerate(range(0, n_q, chunk_rows)):
        prefix = os.path.join(out_dir, f"c{i:05d}")
        if not os.path.exists(prefix + ".done"):
            tasks.append((lo, min(n_q, lo + chunk_rows), prefix))
    n_done = -(-n_q // chunk_rows) - len(tasks)
    if n_done:
        log(f"    resuming: {n_done} chunks already finished")
    tot = {"rows": 0, "union": 0, "kept": 0, "rescued": 0, "big_keys": {}, "phase": {}}
    if not tasks:
        return tot
    t0 = time.time()
    pool = _pool(min(n_procs, len(tasks)), ctx_dir, cfg, ranker_path)
    try:
        for i, st in enumerate(_imap(pool, _block_task, tasks)):
            for k in ("rows", "union", "kept", "rescued"):
                tot[k] += st[k]
            for k, v in st["phase"].items():
                tot["phase"][k] = tot["phase"].get(k, 0.0) + v
            for b, v in st["big_keys"].items():
                tot["big_keys"][b] = tot["big_keys"].get(b, 0) + v
            if (i + 1) % max(1, len(tasks) // 10) == 0 or i + 1 == len(tasks):
                el = time.time() - t0
                ph = tot["phase"]
                tph = max(sum(ph.values()), 1e-9)
                log(f"    {tot['rows']:,} records: union {tot['union'] / max(tot['rows'], 1):.1f} "
                    f"per record, kept {tot['kept'] / max(tot['rows'], 1):.1f}"
                    + (f" (address rescue {tot['rescued'] / max(tot['rows'], 1):.2f})" if cfg.get("addr_rescue") else "")
                    + f" ({el / 60:.1f} min, ~{el * (len(tasks) - i - 1) / (i + 1) / 60:.0f} min left; "
                    f"{tot['rows'] / max(el, 1e-9):,.0f} records/s; worker time "
                    + ", ".join(f"{k} {v / tph:.0%}" for k, v in ph.items()) + ")")
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return tot


def read_parts(out_dir, kind, columns=None):
    """Concatenated chunk tables of one kind ('pairs', 'rec', 'truth') of a partition."""
    files = sorted(f for f in os.listdir(out_dir) if f.endswith(f".{kind}.parquet"))
    if not files:
        return None
    return pa.concat_tables([pq.read_table(os.path.join(out_dir, f), columns=columns) for f in files])


def pair_files(out_dir):
    return sorted(os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".pairs.parquet"))


# ---------------------------------------------------------------------------
# 5. Ranker fit and keep policy
# ---------------------------------------------------------------------------

RANKER_PARAMS = dict(objective="binary:logistic", eval_metric="logloss", max_depth=5, eta=0.15,
                     subsample=0.8, colsample_bytree=0.9, min_child_weight=5, tree_method="hist")


def fit_ranker(X, y, n_jobs, rounds, seed, path):
    import xgboost as xgb
    d = xgb.DMatrix(X, label=y.astype(np.float32), nthread=n_jobs, feature_names=list(RANK_FEATS))
    bst = xgb.train({**RANKER_PARAMS, "nthread": n_jobs, "seed": seed}, d, rounds)
    bst.save_model(path)
    imp = bst.get_score(importance_type="gain")
    return bst, dict(sorted(imp.items(), key=lambda kv: -kv[1]))


def keep_curve(q, score, y, keep_min, keep_maxes, floors):
    """Recall of the union's true pairs and mean kept pairs per record, per (keep_max, floor)."""
    rank = group_rank(q, score)
    n_q = max(len(np.unique(q)), 1)
    rows = []
    for km in keep_maxes:
        for t in floors:
            keep = (rank <= keep_min) | ((score >= t) & (rank <= km))
            rows.append({"keep_max": int(km), "floor": float(t),
                         "union_share_kept": float(keep[y].mean()) if y.any() else 1.0,
                         "pairs_per_record": float(keep.sum() / n_q)})
    return pd.DataFrame(rows)


def choose_floor(curve, keep_max, target):
    """Largest floor whose kept share of the union's true pairs reaches min(target, best)."""
    c = curve[curve.keep_max == keep_max].sort_values("floor")
    goal = min(target, float(c.union_share_kept.max()))
    ok = c[c.union_share_kept >= goal - 1e-12]
    return float(ok.floor.max()), float(c.union_share_kept.max())


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=float)
