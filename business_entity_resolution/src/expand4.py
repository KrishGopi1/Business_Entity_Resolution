"""
v4 candidate expansion and stage-2 cut, one v3 candidate chunk at a time. Nothing here
re-runs v3's retrieval: v3's stored candidates (up to 100 per record, with their 44 ranker
features) are read as they are, and only NEW pairs are added.

For the records lo..hi-1 of a chunk, a worker:
  1. reads the v3 candidates of the chunk (stage-1 set);
  2. adds pairs from the new sources (v4 architecture 5.1, 5.2, 6):
       bigkey  ranked big-key blocks: a v3 rescue key (name_sorted, name_phon, name_full,
               st_house_rare) or a v4 exact key shared by more than block_cap S1s (v3 skipped
               these, which switched name blocks off exactly for common names) now contributes
               the big_top S1s under the key that are closest to the record by address
               cosine, for keys of up to big_cap S1s
       desp    exact despaced legal-free name ('rudrasnack'), keys of up to block_cap S1s
       loose   exact loose-romanization key ('sr laksm'), keys of up to block_cap S1s
       nlchan  top-k of a loose-romanization name channel (character 3-grams), for records
               with a non-Latin or handle-like name or no name overlap with any v3 candidate
       dschan  top-k of a despaced character 4-gram channel, for handle / domain records
       dense   top-k of the bi-encoder (dense.py), when it has been built
  3. computes v3's 44 ranker features for the new pairs, recomputes the record-relative ones
     (ranks, gaps, union size) over the expanded set, and re-scores it with v3's ranker
     (stage 1, a feature);
  4. adds cheap v4 features (vectorized, no Python loops): cosines of the three v4 channels with
     their ranks and gaps inside the record, idf-weighted and plain Jaccard of legal-free
     names, loose-token Jaccard, number-set overlap, state (incl. Indic-script state), exact
     loose / despaced key agreement, source flags, dense cosine;
  5. scores the expanded set with the stage-2 re-ranker and keeps
         the record's s2_keep_min best
       or  stage-2 score >= s2_floor and rank <= s2_keep_max
       or  v3-fast's address rescue (S1 among the record's 2 best by address, a shared
           non-postcode number, address cosine >= 0.4, postcodes not different).
"""
import os
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import blocking as bk
import context4 as c4
import indexing as ix

SRC4 = ("v3", "bigkey", "desp", "loose", "nlchan", "dschan", "dense")
BIT4 = {n: 1 << i for i, n in enumerate(SRC4)}
REL3 = [f"rk_{c}" for c in ix.CHANNELS] + [f"gap_{c}" for c in ix.CHANNELS] + ["q_union"]
CHEAP4 = (["in_v3"] + [f"b4_{n}" for n in SRC4[1:]] + ["n_src4"]
          + [f"{p}_{c}" for c in ("nl", "ds", "ac") for p in ("cos", "rk", "gap")]
          + ["nl_wjacc", "nl_jacc", "lp_jacc", "n4_jacc", "n4_any", "n4_qonly", "n4_sonly",
             "st4_same", "lk_same", "ds_same", "q_handle", "cos_dense", "rk_dense", "gap_dense"])
S2_FEATS = list(bk.RANK_FEATS) + ["rank_score", "rank_pos"] + CHEAP4
BIGKEY_V3 = ("name_sorted", "name_phon", "name_full", "st_house_rare")
_POP = np.array([bin(x).count("1") for x in range(1 << len(SRC4))], np.float32)
V3_COLS = ["qi", "si", "src"] + list(bk.RANK_FEATS) + ["rank_score", "rank_pos"]


# ---------------------------------------------------------------------------
# 1. New sources
# ---------------------------------------------------------------------------

def big_lookup(blk, lo, hi, cap, big_cap):
    """(query row, S1 row) pairs of keys whose S1 group has more than cap and at most big_cap
    members (the keys v3's rescue blocks skipped)."""
    a, b = blk["q_ptr"][lo], blk["q_ptr"][hi]
    if a == b or len(blk["keys"]) == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    qk = np.asarray(blk["q_keys"][a:b])
    qrow = np.repeat(np.arange(lo, hi), np.diff(np.asarray(blk["q_ptr"][lo:hi + 1])))
    pos = np.searchsorted(blk["keys"], qk)
    pos[pos >= len(blk["keys"])] = 0
    hit = blk["keys"][pos] == qk
    size = np.where(hit, blk["size"][pos], 0)
    use = (size > cap) & (size <= big_cap)
    qrow, pos, size = qrow[use], pos[use], size[use]
    if not len(qrow):
        return np.empty(0, np.int64), np.empty(0, np.int64)
    rep = np.repeat(np.arange(len(pos)), size)
    off = np.arange(len(rep)) - np.repeat(np.r_[0, np.cumsum(size)][:-1], size)
    s_rows = np.asarray(blk["s_rows"])[blk["start"][pos][rep] + off]
    return qrow[rep].astype(np.int64), s_rows.astype(np.int64)


def _ranked(ctx, qr, sr, top):
    """The top S1s per record among (qr, sr) by v3 address cosine; records without an address
    get none (a common name with no address cannot be ranked)."""
    if not len(qr):
        return qr, sr
    m = ctx["mats"]["addr"]
    ok = (bk.row_nnz(m["Q"])[qr] > 0) & (bk.row_nnz(m["S"])[sr] > 0)
    qr, sr = qr[ok], sr[ok]
    if not len(qr):
        return qr, sr
    key = qr * ctx["n_s"] + sr
    key, first = np.unique(key, return_index=True)
    qr, sr = qr[first], sr[first]
    score = bk.rowwise_dot(m["Q"], m["S"], qr, sr)
    keep = bk.group_rank(qr, score) <= top
    return qr[keep], sr[keep]


def _topk_rows(ch4, rows, k, k_extra=10):
    if not len(rows):
        return np.empty(0, np.int64), np.empty(0, np.int64)
    m = ch4
    R = bk.topk(m["Qr"][rows], m["STr"], k + k_extra)
    r, c, _ = bk.trim_ties(R, k)
    return rows[r].astype(np.int64), c.astype(np.int64)


def new_sources(W, lo, hi, stored_q, stored_nmj):
    """Keys (q * n_s + s) and source bits of the new sources for rows lo..hi-1."""
    ctx, x4, cfg = W["ctx"], W["ctx4"], W["cfg"]
    n_s = ctx["n_s"]
    keys, bits = [], []
    if cfg["bigkey"]:
        for b in BIGKEY_V3:
            if b in ctx["blocks"]:
                qr, sr = big_lookup(ctx["blocks"][b], lo, hi, cfg["block_cap"], cfg["big_cap"])
                qr, sr = _ranked(ctx, qr, sr, cfg["big_top"])
                keys.append(qr * n_s + sr)
                bits.append(np.full(len(qr), BIT4["bigkey"], np.int32))
    for b in ("desp", "loose"):
        if not cfg.get(f"src_{b}", True):
            continue
        blk = x4["blocks"][b]
        qr, sr, _ = ix.block_lookup(blk, lo, hi, cfg["block_cap"])
        keys.append(qr * n_s + sr)
        bits.append(np.full(len(qr), BIT4[b], np.int32))
        if cfg["bigkey"]:
            qr, sr = big_lookup(blk, lo, hi, cfg["block_cap"], cfg["big_cap"])
            qr, sr = _ranked(ctx, qr, sr, cfg["big_top"])
            keys.append(qr * n_s + sr)
            bits.append(np.full(len(qr), BIT4["bigkey"], np.int32))
    rows = np.arange(lo, hi)
    handle = np.asarray(x4["arr"]["q_handle"][lo:hi]) > 0
    if cfg.get("src_nlchan", True):
        nonlat = np.asarray(ctx["arr"]["q_nonlatin"][lo:hi]) > 0
        best = np.full(hi - lo, -1.0, np.float32)
        if len(stored_q):
            np.maximum.at(best, stored_q - lo, np.nan_to_num(stored_nmj, nan=-1.0))
        need = nonlat | handle | (best <= 0)
        qr, sr = _topk_rows(x4["mats"]["nl"], rows[need], cfg["k4"])
        keys.append(qr * n_s + sr)
        bits.append(np.full(len(qr), BIT4["nlchan"], np.int32))
    if cfg.get("src_dschan", True) and handle.any():
        qr, sr = _topk_rows(x4["mats"]["ds"], rows[handle], cfg["k4"])
        keys.append(qr * n_s + sr)
        bits.append(np.full(len(qr), BIT4["dschan"], np.int32))
    dn = W.get("dense_top")
    if dn is not None:
        a, b = dn["q_ptr"][lo], dn["q_ptr"][hi]
        qr = np.repeat(np.arange(lo, hi), np.diff(np.asarray(dn["q_ptr"][lo:hi + 1]))).astype(np.int64)
        sr = np.asarray(dn["s"][a:b], np.int64)
        keys.append(qr * n_s + sr)
        bits.append(np.full(len(qr), BIT4["dense"], np.int32))
    if not keys:
        return np.empty(0, np.int64), np.empty(0, np.int32)
    return np.concatenate(keys), np.concatenate(bits)


# ---------------------------------------------------------------------------
# 2. Expanded set with features
# ---------------------------------------------------------------------------

def _rel(q, v):
    z = np.nan_to_num(v, nan=0.0).astype(np.float32)
    return bk.group_rank(q, z).astype(np.float32), (bk.group_max(q, z) - z).astype(np.float32)


def _cos(M, q, s):
    v = bk.rowwise_dot(M["Q"], M["S"], q, s)
    v[(bk.row_nnz(M["Q"])[q] == 0) | (bk.row_nnz(M["S"])[s] == 0)] = np.nan
    return v


def _dense_cos(W, q, s, batch=200_000):
    E = W.get("dense_emb")
    if E is None:
        return np.full(len(q), np.nan, np.float32)
    out = np.empty(len(q), np.float32)
    for i in range(0, len(q), batch):
        a = np.asarray(E["q"][q[i:i + batch]], np.float32)
        b = np.asarray(E["s"][s[i:i + batch]], np.float32)
        out[i:i + batch] = (a * b).sum(axis=1)
    return out


def cheap_features(W, q, s, src4):
    ctx, x4 = W["ctx"], W["ctx4"]
    F = {"in_v3": ((src4 & BIT4["v3"]) > 0).astype(np.float32)}
    for n in SRC4[1:]:
        F[f"b4_{n}"] = ((src4 & BIT4[n]) > 0).astype(np.float32)
    F["n_src4"] = _POP[src4]
    for ch in ("nl", "ds", "ac"):
        v = _cos(x4["mats"][ch], q, s)
        F[f"cos_{ch}"] = v
        F[f"rk_{ch}"], F[f"gap_{ch}"] = _rel(q, v)
    Ss, Qs = x4["tok"]["nlw"]
    inter = bk.rowwise_dot(Qs, Ss, q, s)
    sq, ss = x4["arr"]["q_nlw_sum"][q], x4["arr"]["s_nlw_sum"][s]
    w = inter / np.maximum(sq + ss - inter, 1e-6)
    w[(sq == 0) | (ss == 0)] = np.nan
    F["nl_wjacc"] = w.astype(np.float32)
    Ss, Qs = x4["tok"]["nl"]
    F["nl_jacc"] = bk._jaccard(Qs, Ss, q, s)[0]
    Ss, Qs = x4["tok"]["lp"]
    F["lp_jacc"] = bk._jaccard(Qs, Ss, q, s)[0]
    Ss, Qs = x4["tok"]["n4"]
    nj, inter = bk._jaccard(Qs, Ss, q, s)
    nq, ns = bk.row_nnz(Qs)[q].astype(np.float32), bk.row_nnz(Ss)[s].astype(np.float32)
    F["n4_jacc"] = nj
    F["n4_any"] = np.where(np.isnan(nj), np.nan, (inter > 0).astype(np.float32))
    F["n4_qonly"] = np.where(nq > 0, nq - inter, np.nan).astype(np.float32)
    F["n4_sonly"] = np.where(ns > 0, ns - inter, np.nan).astype(np.float32)
    A = x4["arr"]
    F["st4_same"] = bk._tri(A["s_state4"][s], A["q_state4"][q])
    F["lk_same"] = bk._tri(A["s_loose_key"][s], A["q_loose_key"][q])
    F["ds_same"] = bk._tri(A["s_name_despaced"][s], A["q_name_despaced"][q])
    F["q_handle"] = np.asarray(A["q_handle"][q], np.float32)
    d = _dense_cos(W, q, s)
    F["cos_dense"] = d
    F["rk_dense"], F["gap_dense"] = _rel(q, d)
    if W.get("dense_emb") is None:
        F["rk_dense"][:] = np.nan
        F["gap_dense"][:] = np.nan
    return F


def read_stored(W, path, lo, hi):
    """v3 candidates of the chunk (optionally trimmed by the stage-1 settings)."""
    cfg = W["cfg"]
    if path is None or not os.path.exists(path):
        return None
    t = pq.read_table(path, columns=V3_COLS)
    d = {c: t[c].to_numpy() for c in V3_COLS}
    if cfg["s1_keep_max"] < 100 or cfg["s1_floor"] > 0:
        keep = (d["rank_pos"] <= cfg["s1_keep_min"]) | (
            (d["rank_pos"] <= cfg["s1_keep_max"]) & (d["rank_score"] >= cfg["s1_floor"]))
        d = {c: v[keep] for c, v in d.items()}
    return d


def expand(W, path, lo, hi):
    """Expanded candidate set of rows lo..hi-1: dict with q, s (partition rows), src3 (v3 bits),
    src4 (v4 bits) and X (float32, S2_FEATS order)."""
    ctx = W["ctx"]
    n_s = ctx["n_s"]
    st = read_stored(W, path, lo, hi)
    if st is None:
        st = {c: np.empty(0, np.float32) for c in V3_COLS}
        st["qi"] = np.empty(0, np.int64)
        st["si"] = np.empty(0, np.int64)
        st["src"] = np.empty(0, np.int32)
    sq, ss = st["qi"].astype(np.int64), st["si"].astype(np.int64)
    skey = sq * n_s + ss
    nmj = st["nm_jacc"] if len(sq) else np.empty(0, np.float32)
    nkey, nbit = new_sources(W, lo, hi, sq, nmj)
    allk = np.concatenate([skey, nkey])
    allb = np.concatenate([np.full(len(skey), BIT4["v3"], np.int32), nbit])
    order = np.argsort(allk, kind="stable")
    allk, allb = allk[order], allb[order]
    start = np.flatnonzero(np.r_[True, allk[1:] != allk[:-1]]) if len(allk) else np.empty(0, np.int64)
    src4 = np.bitwise_or.reduceat(allb, start) if len(allk) else allb
    key = allk[start] if len(allk) else allk
    q, s = key // n_s, key % n_s
    n = len(q)
    X3 = np.empty((n, len(bk.RANK_FEATS)), np.float32)
    src3 = np.zeros(n, np.int32)
    in3 = (src4 & BIT4["v3"]) > 0
    if len(skey):
        pos = np.searchsorted(key, skey)
        for j, c in enumerate(bk.RANK_FEATS):
            X3[pos, j] = st[c]
        src3[pos] = st["src"]
    new = np.flatnonzero(~in3)
    if len(new):
        X3[new] = bk.rank_features(ctx, {"q": q[new], "s": s[new], "src": np.zeros(len(new), np.int32)})
    col = bk.RANK_FEATS.index
    for ch in ix.CHANNELS:                 # record-relative features over the expanded set
        rk, gap = _rel(q, X3[:, col(f"cos_{ch}")])
        X3[:, col(f"rk_{ch}")] = rk
        X3[:, col(f"gap_{ch}")] = gap
    if n:
        _, inv, cnt = np.unique(q, return_inverse=True, return_counts=True)
        X3[:, col("q_union")] = cnt[inv]
    r1 = W["ranker"].inplace_predict(X3).astype(np.float32) if n else np.empty(0, np.float32)
    F = cheap_features(W, q, s, src4) if n else {c: np.empty(0, np.float32) for c in CHEAP4}
    X = np.empty((len(S2_FEATS), n), np.float32)
    X[:len(bk.RANK_FEATS)] = X3.T
    X[len(bk.RANK_FEATS)] = r1
    X[len(bk.RANK_FEATS) + 1] = bk.group_rank(q, r1) if n else r1
    for j, c in enumerate(CHEAP4):
        X[len(bk.RANK_FEATS) + 2 + j] = F[c]
    return {"q": q, "s": s, "src3": src3, "src4": src4, "X": X.T}


def stage2_scores(W, X):
    b = W.get("s2")
    if b is None or not len(X):
        return np.zeros(len(X), np.float32)
    return b.inplace_predict(X).astype(np.float32)


_QNOADDR = bk.RANK_FEATS.index("q_addr_missing")


def cut(W, q, X, s2):
    """Stage-2 keep rule. Returns (keep, rank, rescued_only)."""
    cfg = W["cfg"]
    rank = bk.group_rank(q, s2)
    keep = (rank <= cfg["s2_keep_min"]) | ((s2 >= cfg["s2_floor"]) & (rank <= cfg["s2_keep_max"]))
    if cfg.get("s2_keep_noaddr", 0) > 0 and len(q):
        # a record without an address can only be told apart from its namesakes by the pair
        # model, not by the address cosine the floor leans on: keep its best s2_keep_noaddr
        noaddr = X[:, _QNOADDR] > 0.5
        keep |= noaddr & (rank <= cfg["s2_keep_noaddr"])
    resc = np.zeros(len(q), bool)
    if cfg["addr_rescue"] > 0 and len(q):
        r = bk.address_rescue(X[:, :len(bk.RANK_FEATS)], cfg["addr_rescue"], cfg["rescue_min_cos"])
        resc = r & ~keep
        keep = keep | r
    return keep, rank, resc


def truth_mask(W, q, s):
    tk = W["ctx"]["arr"].get("truth_keys")
    if tk is None:
        return None
    return bk.isin_sorted(q.astype(np.int64) * W["ctx"]["n_s"] + s, tk)


# ---------------------------------------------------------------------------
# 3. Workers
# ---------------------------------------------------------------------------

W = {}


def init_worker(ctx_dir, ctx4_dir, cfg, ranker_path, s2_path=None, dense_dir=None, extra=None):
    from schema import pin_native_threads
    pin_native_threads()
    import xgboost as xgb
    W.clear()
    W["ctx"] = ix.load_context(ctx_dir)
    W["ctx4"] = c4.load(ctx4_dir)
    W["cfg"] = cfg
    b = xgb.Booster()
    b.load_model(ranker_path)
    b.set_param({"nthread": 1})
    W["ranker"] = b
    W["s2"] = None
    if s2_path:
        b2 = xgb.Booster()
        b2.load_model(s2_path)
        b2.set_param({"nthread": 1})
        W["s2"] = b2
    if dense_dir and os.path.exists(os.path.join(dense_dir, "q_ptr.npy")):
        W["dense_top"] = {k: np.load(os.path.join(dense_dir, f"{k}.npy"), mmap_mode="r")
                          for k in ("q_ptr", "s")}
        W["dense_emb"] = {k: np.load(os.path.join(dense_dir, f"emb_{k}.npy"), mmap_mode="r")
                          for k in ("q", "s")}
    if extra:
        extra(W)


def chunk_range(rec_path):
    t = pq.read_table(rec_path, columns=["qi"])["qi"].to_numpy()
    return (int(t.min()), int(t.max()) + 1) if len(t) else (0, 0)


def chunks_of(cand_dir):
    """(pairs path, rec path) per v3 chunk, in order."""
    out = []
    for f in sorted(os.listdir(cand_dir)):
        if f.endswith(".rec.parquet"):
            pre = os.path.join(cand_dir, f[:-len(".rec.parquet")])
            out.append((pre + ".pairs.parquet", pre + ".rec.parquet"))
    return out


def sample_task(args):
    """Expanded set, stage-2 features and labels of one chunk (for fitting stage 2). Records in
    `excluded_gid` (tune / holdout entity records, stress records) are dropped."""
    pairs_path, rec_path = args
    lo, hi = chunk_range(rec_path)
    e = expand(W, pairs_path, lo, hi)
    y = truth_mask(W, e["q"], e["s"])
    qg = np.asarray(W["ctx"]["arr"]["q_gid"])[e["q"]]
    keep = ~np.asarray(W["excluded"])[qg]
    return {"q": e["q"][keep] + W["q_offset"], "X": e["X"][keep], "y": y[keep],
            "src4": e["src4"][keep]}


def cut_task(args):
    """Expands, scores with stage 2, cuts and writes one chunk:
       <prefix>.pairs.parquet  kept pairs with every stage-2 feature and s2_score / s2_rank / s2_gap
       <prefix>.rec.parquet    per record: expanded and kept counts
       <prefix>.truth.parquet  (train) every true pair of the expanded set, kept or not"""
    pairs_path, rec_path, prefix = args
    t0 = time.time()
    lo, hi = chunk_range(rec_path)
    e = expand(W, pairs_path, lo, hi)
    t1 = time.time()
    s2 = stage2_scores(W, e["X"])
    keep, rank, resc = cut(W, e["q"], e["X"], s2)
    gap = bk.group_max(e["q"], s2) - s2 if len(s2) else s2
    A = W["ctx"]["arr"]
    k = keep
    cols = {"qi": e["q"][k].astype(np.int32), "si": e["s"][k].astype(np.int32),
            "qg": np.asarray(A["q_gid"])[e["q"][k]].astype(np.int32),
            "sg": np.asarray(A["s_gid"])[e["s"][k]].astype(np.int32),
            "src3": e["src3"][k], "src4": e["src4"][k].astype(np.int16),
            "s2_score": s2[k], "s2_rank": rank[k].astype(np.float32),
            "s2_gap": gap[k].astype(np.float32), "rescued": resc[k]}
    for j, c in enumerate(S2_FEATS):
        cols[c] = e["X"][k, j]
    pq.write_table(pa.table(cols), prefix + ".pairs.parquet", compression="zstd")
    rq, inv = np.unique(e["q"], return_inverse=True)
    n_exp = np.zeros(hi - lo, np.int32)
    n_exp[rq - lo] = np.bincount(inv)
    n_new = np.zeros(hi - lo, np.int32)
    newm = (e["src4"] & BIT4["v3"]) == 0
    np.add.at(n_new, e["q"][newm] - lo, 1)
    pq.write_table(pa.table({"qi": np.arange(lo, hi, dtype=np.int32), "n_expanded": n_exp,
                             "n_new": n_new,
                             "n_kept": np.bincount(e["q"][keep] - lo, minlength=hi - lo).astype(np.int32)}),
                   prefix + ".rec.parquet")
    y = truth_mask(W, e["q"], e["s"])
    if y is not None:
        pq.write_table(pa.table({
            "qg": np.asarray(A["q_gid"])[e["q"][y]].astype(np.int32),
            "sg": np.asarray(A["s_gid"])[e["s"][y]].astype(np.int32),
            "src3": e["src3"][y], "src4": e["src4"][y].astype(np.int16),
            "rank_pos": e["X"][y, S2_FEATS.index("rank_pos")],
            "s2_score": s2[y], "s2_rank": rank[y].astype(np.float32), "kept": keep[y],
            "rescued": resc[y]}), prefix + ".truth.parquet")
    open(prefix + ".done", "w").close()
    return {"rows": hi - lo, "expanded": int(len(e["q"])), "new": int(newm.sum()),
            "kept": int(keep.sum()), "seconds_expand": t1 - t0, "seconds": time.time() - t0}
