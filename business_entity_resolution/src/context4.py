"""
v4 per-partition context, built next to v3's (which is reused as it is, memory-mapped):

    views        v4 views of every S1 and record (views4.py), saved as parquet
    strings      one Arrow file per side with every string column the pair features read
                 (v3.1's columns plus the v4 views), memory-mapped by the scoring workers
    mats         TF-IDF matrices (sublinear tf x S1-side idf, rows L2-normalized):
                   nl   character 3-grams of name_loose + its loose tokens (transliteration)
                   ds   character 4-grams of name_despaced (domains, handles, spacing)
                   ac   words of address_core (address without state / region / markers)
                 and retrieval matrices for nl and ds with v3's soft df cap
    tok          0/1 token matrices: nl (name_nolegal words, also idf-weighted), lp (loose
                 tokens), n4 (numbers4)
    blocks       exact keys: desp (name_despaced, 6+ characters) and loose (loose_key, 4+),
                 laid out like v3's rescue blocks (indexing.build_blocks format)
    arr          code arrays (state4, loose_key, despaced), handle flag, idf row sums
"""
import json
import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer

import features as ft3
import indexing as ix
import views4

MATS4 = ("nl", "ds", "ac")
RETR4 = ("nl", "ds")
TOK4 = {"nl": "name_nolegal", "lp": "name_loose", "n4": "numbers4"}
BLOCKS4 = ("desp", "loose")
# string columns of the v4 Arrow files (read by features4)
STR4 = tuple(ft3.SRC[:-1]) + ("name_nolegal", "address_core", "name_loose", "name_despaced",
                              "numbers4", "state4", "postcode")


def _hv(analyzer, n):
    return HashingVectorizer(analyzer=analyzer, ngram_range=(n, n), lowercase=False,
                             alternate_sign=False, norm=None, n_features=ix.N_CHAR, dtype=np.float32)


def _counts(texts, analyzer, n, n_jobs, chunk=200_000):
    hv = _hv(analyzer, n)
    if n_jobs <= 1 or len(texts) <= 2 * chunk:
        return hv.transform(texts).tocsr()
    from joblib import Parallel, delayed
    parts = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(hv.transform)(texts[i:i + chunk]) for i in range(0, len(texts), chunk))
    return sparse.vstack(parts, format="csr")


def _channel_counts(ch, S, Q, n_jobs):
    """Raw term counts (S1 side, record side) of a v4 channel."""
    if ch == "nl":
        Hs = _counts(S["name_loose"], "char_wb", 3, n_jobs)
        Hq = _counts(Q["name_loose"], "char_wb", 3, n_jobs)
        Cs, Cq = ix.exact_counts([S["name_loose"]], [Q["name_loose"]], lambda s: s.split(), n_jobs)
        return sparse.hstack([Hs, Cs], format="csr"), sparse.hstack([Hq, Cq], format="csr")
    if ch == "ds":
        return (_counts(S["name_despaced"], "char", 4, n_jobs),
                _counts(Q["name_despaced"], "char", 4, n_jobs))
    if ch == "ac":
        return ix.exact_counts([S["address_core"]], [Q["address_core"]], lambda s: s.split(), n_jobs)
    raise ValueError(ch)


def _codes(a, b):
    codes, uniq = pd.factorize(np.concatenate([np.asarray(a, object), np.asarray(b, object)]))
    codes = codes.astype(np.int32)
    empty = np.flatnonzero(uniq == "")
    if len(empty):
        codes[codes == empty[0]] = -1
    return codes[:len(a)], codes[len(a):]


def _blocks(S, Q):
    """Exact-key inverted indexes in indexing.build_blocks' layout."""
    out = {}
    for b, field, min_len in (("desp", "name_despaced", 6), ("loose", "loose_key", 4)):
        sk = [(i, [f"{b}|{x}"]) for i, x in enumerate(S[field]) if len(x) >= min_len]
        qk = [(i, [f"{b}|{x}"]) for i, x in enumerate(Q[field]) if len(x) >= min_len]
        s_rows, s_h = ix._multi_keys(sk)
        q_rows, q_h = ix._multi_keys(qk)
        order = np.argsort(s_h, kind="stable")
        s_rows, s_h = s_rows[order], s_h[order]
        uk, start, size = np.unique(s_h, return_index=True, return_counts=True)
        q_ptr = np.r_[0, np.cumsum(np.bincount(q_rows, minlength=len(Q[field])))].astype(np.int64)
        out[b] = {"s_rows": s_rows.astype(np.int32), "keys": uk, "start": start.astype(np.int64),
                  "size": size.astype(np.int64), "q_ptr": q_ptr, "q_keys": q_h}
    return out


def write_strings(norm_tab, views_tab, path):
    cols = {c: pc.fill_null(norm_tab[c], "") for c in ft3.SRC[:-1] + ("postcode",)}
    for c in ("name_nolegal", "address_core", "name_loose", "name_despaced", "numbers4", "state4"):
        cols[c] = views_tab[c]
    t = pa.table({c: cols[c] for c in STR4})
    with pa.OSFile(path, "wb") as f, pa.ipc.new_file(f, t.schema) as w:
        w.write_table(t)


def build(norm_dir, part_name, out_dir, str_dir, cfg, n_jobs=1, log=print):
    """Builds the v4 context of one partition from v3's normalized tables."""
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(str_dir, exist_ok=True)
    cols = list(set(views4.NORM_COLS) | set(ft3.SRC) | {"postcode"})
    s_norm = pq.read_table(os.path.join(norm_dir, f"s1_{part_name}.parquet"), columns=cols)
    q_norm = pq.read_table(os.path.join(norm_dir, f"q_{part_name}.parquet"), columns=cols)
    Sv = views4.compute_views_parallel(s_norm, n_jobs)
    Qv = views4.compute_views_parallel(q_norm, n_jobs)
    pq.write_table(Sv, os.path.join(out_dir, "views_s1.parquet"))
    pq.write_table(Qv, os.path.join(out_dir, "views_q.parquet"))
    write_strings(s_norm, Sv, os.path.join(str_dir, "s1.arrow"))
    write_strings(q_norm, Qv, os.path.join(str_dir, "q.arrow"))
    S = {c: Sv[c].to_pylist() for c in Sv.column_names}
    Q = {c: Qv[c].to_pylist() for c in Qv.column_names}
    n_s = len(S["name_loose"])
    cap = max(int(cfg["min_df_cap"]), int(cfg["df_cap_frac"] * n_s))
    stats = {"n_s1": n_s, "n_q": len(Q["name_loose"]), "channels": {}}
    for ch in MATS4:
        Cs, Cq = _channel_counts(ch, S, Q, n_jobs)
        df = np.bincount(Cs.indices, minlength=Cs.shape[1])
        idf = np.zeros(Cs.shape[1], np.float32)
        ok = df > 0
        idf[ok] = (np.log((1.0 + n_s) / (1.0 + df[ok])) + 1.0).astype(np.float32)
        Ws, Wq = ix._weigh_blocks(Cs, idf, n_jobs), ix._weigh_blocks(Cq, idf, n_jobs)
        del Cs, Cq
        ix.save_csr(out_dir, f"m_{ch}_S", Ws)
        ix.save_csr(out_dir, f"m_{ch}_Q", Wq)
        if ch in RETR4:
            if (df[Wq.indices] > cap).any():
                Sr, Qr, st = ix.soft_cap(Ws, Wq, df, cap, cfg["fallback_share"], cfg["n_fallback"],
                                         cfg["n_always"])
            else:                                   # nothing to cap (small data)
                Sr, Qr, st = Ws, Wq, {"df_cap": int(cap), "capped_features": 0}
            ix.save_csr(out_dir, f"m_{ch}_Qr", Qr)
            ix.save_csr(out_dir, f"m_{ch}_STr", Sr.T.tocsr())
            stats["channels"][ch] = st
        del Ws, Wq
    arr = {}
    for k, f in TOK4.items():
        Bs = ix.binary_tokens(S[f], n_jobs)
        Bq = ix.binary_tokens(Q[f], n_jobs)
        ix.save_csr(out_dir, f"t_{k}_S", Bs)
        ix.save_csr(out_dir, f"t_{k}_Q", Bq)
        if k == "nl":                                  # idf-weighted copies for a weighted Jaccard
            df = np.bincount(Bs.indices, minlength=Bs.shape[1])
            idf = np.where(df > 0, np.log((1.0 + n_s) / (1.0 + df)) + 1.0,
                           np.log(1.0 + n_s) + 1.0).astype(np.float32)
            Ws = Bs.multiply(idf[None, :]).tocsr().astype(np.float32)
            Wq = Bq.multiply(idf[None, :]).tocsr().astype(np.float32)
            ix.save_csr(out_dir, "t_nlw_S", Ws)
            ix.save_csr(out_dir, "t_nlw_Q", Wq)
            arr["s_nlw_sum"] = np.asarray(Ws.sum(axis=1)).ravel().astype(np.float32)
            arr["q_nlw_sum"] = np.asarray(Wq.sum(axis=1)).ravel().astype(np.float32)
    for f in ("state4", "loose_key", "name_despaced"):
        arr[f"s_{f}"], arr[f"q_{f}"] = _codes(S[f], Q[f])
    arr["q_handle"] = np.asarray(Q["q_is_handle"], np.float32)
    os.makedirs(os.path.join(out_dir, "arr"), exist_ok=True)
    for k, v in arr.items():
        np.save(os.path.join(out_dir, "arr", k + ".npy"), v)
    for b, d in _blocks(S, Q).items():
        os.makedirs(os.path.join(out_dir, "blk", b), exist_ok=True)
        for k, v in d.items():
            np.save(os.path.join(out_dir, "blk", b, k + ".npy"), v)
        stats.setdefault("blocks", {})[b] = {"s1_keys": int(len(d["keys"])),
                                            "records_with_key": int((np.diff(d["q_ptr"]) > 0).sum())}
    stats["build_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(out_dir, "context4.json"), "w") as f:
        json.dump(stats, f, indent=1)
    log(f"    v4 context {part_name}: {stats['build_seconds']:.0f}s")
    return stats


def load(d, mmap=True):
    c = {"dir": d, "mats": {}, "tok": {}, "arr": {}, "blocks": {}}
    for ch in MATS4:
        c["mats"][ch] = {k: ix.load_csr(d, f"m_{ch}_{k}", mmap) for k in ("S", "Q")}
        if ch in RETR4:
            c["mats"][ch]["Qr"] = ix.load_csr(d, f"m_{ch}_Qr", mmap)
            c["mats"][ch]["STr"] = ix.load_csr(d, f"m_{ch}_STr", mmap)
    for k in list(TOK4) + ["nlw"]:
        c["tok"][k] = (ix.load_csr(d, f"t_{k}_S", mmap), ix.load_csr(d, f"t_{k}_Q", mmap))
    mode = "c" if mmap else None
    for f in os.listdir(os.path.join(d, "arr")):
        c["arr"][f[:-4]] = np.load(os.path.join(d, "arr", f), mmap_mode=mode)
    for b in os.listdir(os.path.join(d, "blk")):
        bd = os.path.join(d, "blk", b)
        c["blocks"][b] = {f[:-4]: np.load(os.path.join(bd, f), mmap_mode=mode) for f in os.listdir(bd)}
    return c
