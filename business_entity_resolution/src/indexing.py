"""
Retrieval indexes for one country partition (v3): three sparse TF-IDF channels, bounded
exact/structured inverted indexes ("rescue blocks"), and the per-record arrays the candidate
ranker needs. build_context() computes everything once per partition and saves it as plain
.npy files, which blocking workers memory-map (read-only, shared through the page cache, no
fork, no pickling of large matrices).

Channels (sublinear TF x IDF from the S1 side, rows L2-normalized, so a dot is a cosine):
    name   character 3-grams of name_full (legal forms, numbers, DBA and order kept) plus the
           phonetic key of every name token, in one vector. Handles spelling, spacing,
           punctuation, word order and transliteration noise.
    addr   address_full words and numbers, their phonetic keys, and joined neighbours
           ('indira nagar' -> '+indiranagar'). Carries the pairs whose names share nothing.
    joint  name tokens and name phonetic keys + address tokens (prefixed per side) in one
           word vector, so moderately common name and address evidence add up without a
           name x address conjunction vocabulary.

Soft document-frequency cap (replaces v2's max_df, which deleted common words): features in
more than `df_cap` S1 records stay in every vector and every exact cosine, but are left out of
the posting lists used for top-k retrieval, which is where they cost time. A record whose rare
features carry less than half of its vector's weight (a stock name such as 'sai enterprises')
also keeps its rarest common features (one at a time until half the weight is reached, at
most `n_fallback`), and every record keeps its `n_always` rarest common features, because a
typo or transliteration produces rare n-grams that the true S1 does not have. No record ends up unsearchable in a channel because its words are
common. Records with no searchable address get a larger name top-k (adaptive k), since the
name is all they have.

Rescue blocks, each used only when the S1 block for a key has at most `block_cap` members:
    name_full        exact name_full
    name_sorted      exact name_tokens_sorted
    name_phon        exact sorted phonetic key (not in the plan's list; transliterated names)
    addr_full        exact address_full (2+ tokens)
    pc_house         postcode + house_number
    pc_rare_name     postcode + one of the record's 2 rarest name tokens (S1-side frequency)
    st_house_rare    state + house_number + one of the record's 2 rarest address words
    dba              record DBA -> S1 name_base or S1 DBA; record name_base -> S1 DBA
"""
import json
import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize as l2_normalize

from normalize import LEGAL_VALUES, phon_key

CHANNELS = ("name", "addr", "joint")
BLOCKS = ("name_full", "name_sorted", "name_phon", "addr_full", "pc_house", "pc_rare_name",
          "st_house_rare", "dba")
SOURCES = CHANNELS + BLOCKS                 # bit i of a pair's `src` = found by SOURCES[i]
N_CHAR = 2 ** 22
N_BIN = 2 ** 20
CODE_FIELDS = ("postcode", "postcode_prefix", "state", "house_number", "unit_number",
               "name_base", "name_tokens_sorted", "legal_form")
TOKEN_SETS = {"nm": "name_tokens_sorted", "ph": "name_phonetic", "ad": "address_tokens",
              "num": "numeric_tokens"}


# ---------------------------------------------------------------------------
# 1. Tokens per channel
# ---------------------------------------------------------------------------

def _phon_tokens(text):
    return [phon_key(t) for t in text.split() if t not in LEGAL_VALUES and not t.isdigit()]


def _name_phon(full, dba):
    return ["~" + p for p in dict.fromkeys(_phon_tokens(full) + _phon_tokens(dba)) if p]


def _addr_tokens(full, phon):
    toks = full.split()
    out = list(toks)
    out += ["~" + p for p in phon.split()]
    alpha = [t for t in toks if t.isalpha()]
    out += ["+" + a + b for a, b in zip(alpha, alpha[1:])]
    return out


def _joint_tokens(sorted_name, name_phon, addr_tokens):
    return (["n:" + t for t in sorted_name.split()] + ["p:" + t for t in name_phon.split()]
            + ["a:" + t for t in addr_tokens.split()])


TOKENIZERS = {  # channel word part -> (fields, function)
    "name": (("name_full", "name_dba"), _name_phon),
    "addr": (("address_full", "address_phonetic"), _addr_tokens),
    "joint": (("name_tokens_sorted", "name_phonetic", "address_tokens"), _joint_tokens),
}


def _token_hashes(cols, func):
    """Rows of text columns -> (tokens per row, uint64 hash per token), stable across processes."""
    toks = [func(*r) for r in zip(*cols)]
    lens = np.fromiter((len(t) for t in toks), np.int64, len(toks))
    flat = [t for row in toks for t in row]
    h = pd.util.hash_array(np.asarray(flat, dtype=object), categorize=False) if flat \
        else np.empty(0, np.uint64)
    return lens, h


def _hash_all(cols, func, n_jobs, chunk=100_000):
    n = len(cols[0])
    if n_jobs <= 1 or n <= 2 * chunk:
        return _token_hashes(cols, func)
    parts = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_token_hashes)([c[i:i + chunk] for c in cols], func) for i in range(0, n, chunk))
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def _csr(lens, cols, n_cols, binary=False):
    indptr = np.r_[0, np.cumsum(lens)].astype(np.int64)
    M = sparse.csr_matrix((np.ones(len(cols), np.float32), cols.astype(np.int32), indptr),
                          shape=(len(lens), max(n_cols, 1)))
    M.sum_duplicates()
    if binary:
        M.data[:] = 1.0
    return M


def exact_counts(s_cols, q_cols, func, n_jobs=1):
    """Term counts (S, Q) over the exact S1 vocabulary; query tokens absent from S1 are dropped."""
    ls, hs = _hash_all(s_cols, func, n_jobs)
    vocab = np.unique(hs)
    Cs = _csr(ls, np.searchsorted(vocab, hs), len(vocab))
    lq, hq = _hash_all(q_cols, func, n_jobs)
    pos = np.searchsorted(vocab, hq)
    pos[pos >= len(vocab)] = 0
    ok = vocab[pos] == hq if len(vocab) else np.zeros(len(hq), bool)
    row = np.repeat(np.arange(len(lq)), lq)
    Cq = _csr(np.bincount(row[ok], minlength=len(lq)), pos[ok], len(vocab))
    return Cs, Cq


def _char_counts(texts, n_jobs, chunk=200_000):
    hv = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3), lowercase=False,
                           alternate_sign=False, norm=None, n_features=N_CHAR, dtype=np.float32)
    if n_jobs <= 1 or len(texts) <= 2 * chunk:
        return hv.transform(texts).tocsr()
    parts = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(hv.transform)(texts[i:i + chunk]) for i in range(0, len(texts), chunk))
    return sparse.vstack(parts, format="csr")


def binary_tokens(texts, n_jobs=1, chunk=200_000):
    """0/1 hashed word matrix (2^20 columns) for token-overlap features."""
    hv = HashingVectorizer(analyzer="word", token_pattern=r"\S+", lowercase=False,
                           alternate_sign=False, norm=None, binary=True, n_features=N_BIN,
                           dtype=np.float32)
    if n_jobs <= 1 or len(texts) <= 2 * chunk:
        return hv.transform(texts).tocsr()
    parts = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(hv.transform)(texts[i:i + chunk]) for i in range(0, len(texts), chunk))
    return sparse.vstack(parts, format="csr")


def channel_counts(ch, s_tab, q_tab, n_jobs=1):
    fields, func = TOKENIZERS[ch]
    s_cols = [s_tab[f].to_pylist() for f in fields]
    q_cols = [q_tab[f].to_pylist() for f in fields]
    Cs, Cq = exact_counts(s_cols, q_cols, func, n_jobs)
    if ch == "name":                               # char 3-grams + phonetic words
        Hs = _char_counts(s_tab["name_full"].to_pylist(), n_jobs)
        Hq = _char_counts(q_tab["name_full"].to_pylist(), n_jobs)
        Cs = sparse.hstack([Hs, Cs], format="csr")
        Cq = sparse.hstack([Hq, Cq], format="csr")
    return Cs, Cq


# ---------------------------------------------------------------------------
# 2. Weighting and the soft df cap
# ---------------------------------------------------------------------------

def _weigh(C, idf):
    C = C.astype(np.float32, copy=True)
    C.data = (1.0 + np.log(C.data)) * idf[C.indices]
    C.eliminate_zeros()
    return l2_normalize(C, norm="l2", copy=False).astype(np.float32)


def _weigh_blocks(C, idf, n_jobs, chunk=500_000):
    if n_jobs <= 1 or C.shape[0] <= 2 * chunk:
        return _weigh(C, idf)
    parts = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_weigh)(C[i:i + chunk], idf) for i in range(0, C.shape[0], chunk))
    return sparse.vstack(parts, format="csr")


def _keep_cols(M, keep):
    M = M.tocsr(copy=True)
    M.data[~keep[M.indices]] = 0
    M.eliminate_zeros()
    return M


def soft_cap(Ws, Wq, df, cap, share_min=0.5, n_fallback=4, n_always=2):
    """Retrieval matrices (S_ret, Q_ret) and stats. Q_ret keeps every feature with df <= cap;
    a row whose kept features carry less than share_min of its squared weight also keeps its
    rarest capped features, added one at a time until share_min is reached (at most
    n_fallback). Every row keeps at least its n_always rarest capped features: a typo or a
    transliteration makes rare n-grams the true S1 does not have, and without some shared
    common evidence such a record could not reach it. S_ret keeps every column any Q_ret
    row uses."""
    rare = (df <= cap) & (df > 0)
    Qr = _keep_cols(Wq, rare)
    share = np.asarray(Qr.multiply(Qr).sum(axis=1)).ravel()
    has = np.diff(Wq.indptr) > 0
    fb = np.flatnonzero(has)
    add_r, add_c, add_v = [], [], []
    if len(fb):
        sub = Wq[fb].tocoo()
        common = ~rare[sub.col]
        r, c, v = sub.row[common], sub.col[common], sub.data[common]
        order = np.lexsort((df[c], r))
        r, c, v = r[order], c[order], v[order]
        start = np.flatnonzero(np.r_[True, r[1:] != r[:-1]])
        size = np.diff(np.r_[start, len(r)])
        pos = np.arange(len(r)) - np.repeat(start, size)
        # add the rarest common features one by one until the row's retrieval vector carries
        # share_min of its weight, at most n_fallback of them
        cum = np.cumsum(v.astype(np.float64) ** 2)
        before = cum - v.astype(np.float64) ** 2 - np.repeat((cum - v.astype(np.float64) ** 2)[start], size)
        take = (pos < n_always) | ((pos < n_fallback) & (share[fb[r]] + before < share_min))
        add_r, add_c, add_v = fb[r[take]], c[take], v[take]
    if len(add_r):
        extra = sparse.csr_matrix((add_v, (add_r, add_c)), shape=Wq.shape, dtype=np.float32)
        Qr = (Qr + extra).tocsr()
    used = rare.copy()
    if len(add_c):
        used[np.unique(add_c)] = True
    Sr = _keep_cols(Ws, used)
    low = has & (share < share_min)
    stats = {"df_cap": int(cap), "capped_features": int(((df > cap) & (df > 0)).sum()),
             "fallback_rows": int(low.sum()), "fallback_share": float(low.sum() / max(has.sum(), 1)),
             "common_features_per_row": float(len(add_r) / max(has.sum(), 1)),
             "empty_retrieval_rows": int((has & (np.diff(Qr.indptr) == 0)).sum())}
    return Sr.astype(np.float32), Qr.astype(np.float32), stats


# ---------------------------------------------------------------------------
# 3. Rescue blocks (bounded exact / structured inverted indexes)
# ---------------------------------------------------------------------------

def _hash(strs):
    return pd.util.hash_array(np.asarray(strs, dtype=object), categorize=False) if len(strs) \
        else np.empty(0, np.uint64)


def _multi_keys(rows_keys):
    """[(row, [key strings])] -> (rows int64, hashes uint64)."""
    rows = [r for r, ks in rows_keys for _ in ks]
    keys = [k for _, ks in rows_keys for k in ks]
    return np.asarray(rows, np.int64), _hash(keys)


def _rarest(tokens, df, n=2):
    known = [(df.get(t, 0), t) for t in dict.fromkeys(tokens) if df.get(t, 0) > 0]
    return [t for _, t in sorted(known)[:n]]


def _block_keys(tab, side, df_name, df_addr):
    """Keys per record for every rescue block: {block: list of (row, [key, ...])}."""
    g = {f: tab[f].to_pylist() for f in ("name_full", "name_tokens_sorted", "name_phonetic",
                                          "name_base", "name_dba", "address_full", "postcode",
                                          "house_number", "state")}
    out = {b: [] for b in BLOCKS}
    for i in range(tab.num_rows):
        nf, ns, nb, dba = g["name_full"][i], g["name_tokens_sorted"][i], g["name_base"][i], g["name_dba"][i]
        af, pcode, house, st = g["address_full"][i], g["postcode"][i], g["house_number"][i], g["state"][i]
        if nf:
            out["name_full"].append((i, ["F|" + nf]))
        if ns:
            out["name_sorted"].append((i, ["S|" + ns]))
        ph = " ".join(sorted(g["name_phonetic"][i].split()))
        if len(ph) >= 3:
            out["name_phon"].append((i, ["P|" + ph]))
        if af.count(" ") >= 1:
            out["addr_full"].append((i, ["A|" + af]))
        if pcode and house:
            out["pc_house"].append((i, [f"H|{pcode}|{house}"]))
        name_toks = ns.split()
        addr_words = [t for t in af.split() if t.isalpha() and len(t) >= 3]
        if pcode and name_toks:
            toks = name_toks if side == "s" else _rarest(name_toks, df_name)
            out["pc_rare_name"].append((i, [f"R|{pcode}|{t}" for t in toks]))
        if st and house and addr_words:
            toks = addr_words if side == "s" else _rarest(addr_words, df_addr)
            out["st_house_rare"].append((i, [f"T|{st}|{house}|{t}" for t in toks]))
        if side == "s":
            keys = (["D|" + nb] if nb else []) + (["D|" + dba] if dba else [])
        else:
            keys = ["D|" + dba] if dba else []
        if keys:
            out["dba"].append((i, keys))
        if side == "q" and nb:                     # record name_base -> S1 DBA ('E|' side)
            out.setdefault("dba_rev", []).append((i, ["E|" + nb]))
        if side == "s" and dba:
            out.setdefault("dba_rev", []).append((i, ["E|" + dba]))
    return out


def _token_df(tab, field, alpha_min=0):
    toks = [t for x in tab[field].to_pylist() for t in set(x.split())
            if not alpha_min or (t.isalpha() and len(t) >= alpha_min)]
    return pd.Series(toks, dtype=object).value_counts().to_dict() if toks else {}


def _keys_chunk(tab, side, df_name, df_addr):
    return _block_keys(tab, side, df_name, df_addr)


def build_blocks(s_tab, q_tab, n_jobs=1, chunk=200_000):
    """Inverted indexes: per block, S1 rows sorted by key hash with group starts and sizes,
    and query keys in CSR layout (q_ptr, q_keys) so a contiguous row range slices cheaply."""
    df_name = _token_df(s_tab, "name_tokens_sorted")
    df_addr = _token_df(s_tab, "address_full", alpha_min=3)

    def keys_of(tab, side):
        n = tab.num_rows
        if n_jobs <= 1 or n <= 2 * chunk:
            parts = [(0, _keys_chunk(tab, side, df_name, df_addr))]
        else:
            res = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_keys_chunk)(tab.slice(i, chunk), side, df_name, df_addr)
                for i in range(0, n, chunk))
            parts = list(zip(range(0, n, chunk), res))
        merged = {}
        for off, d in parts:
            for b, lst in d.items():
                merged.setdefault(b, []).extend((r + off, ks) for r, ks in lst)
        return merged

    ks, kq = keys_of(s_tab, "s"), keys_of(q_tab, "q")
    blocks = {}
    for b in list(BLOCKS) + ["dba_rev"]:
        s_rows, s_h = _multi_keys(ks.get(b, []))
        q_rows, q_h = _multi_keys(kq.get(b, []))
        order = np.argsort(s_h, kind="stable")
        s_rows, s_h = s_rows[order], s_h[order]
        uk, start, size = np.unique(s_h, return_index=True, return_counts=True)
        q_order = np.argsort(q_rows, kind="stable")
        q_rows, q_h = q_rows[q_order], q_h[q_order]
        q_ptr = np.r_[0, np.cumsum(np.bincount(q_rows, minlength=q_tab.num_rows))].astype(np.int64)
        blocks[b] = {"s_rows": s_rows.astype(np.int32), "keys": uk, "start": start.astype(np.int64),
                     "size": size.astype(np.int64), "q_ptr": q_ptr, "q_keys": q_h}
    return blocks


def block_lookup(blk, lo, hi, cap):
    """(query row, S1 row) pairs of block `blk` for query rows lo..hi-1, skipping keys whose
    S1 group has more than cap members. Returns (q rows, s rows, n keys skipped as too big)."""
    a, b = blk["q_ptr"][lo], blk["q_ptr"][hi]
    if a == b or len(blk["keys"]) == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64), 0
    qk = np.asarray(blk["q_keys"][a:b])
    qrow = np.repeat(np.arange(lo, hi), np.diff(np.asarray(blk["q_ptr"][lo:hi + 1])))
    pos = np.searchsorted(blk["keys"], qk)
    pos[pos >= len(blk["keys"])] = 0
    hit = blk["keys"][pos] == qk
    size = np.where(hit, blk["size"][pos], 0)
    big = int((size > cap).sum())
    use = (size > 0) & (size <= cap)
    qrow, pos, size = qrow[use], pos[use], size[use]
    rep = np.repeat(np.arange(len(pos)), size)
    off = np.arange(len(rep)) - np.repeat(np.r_[0, np.cumsum(size)][:-1], size)
    s_rows = np.asarray(blk["s_rows"])[blk["start"][pos][rep] + off]
    return qrow[rep], s_rows.astype(np.int64), big


# ---------------------------------------------------------------------------
# 4. Context: build, save, load
# ---------------------------------------------------------------------------

def _codes(s_col, q_col):
    s = pc.fill_null(s_col, "").to_numpy(zero_copy_only=False)
    q = pc.fill_null(q_col, "").to_numpy(zero_copy_only=False)
    codes, uniq = pd.factorize(np.concatenate([s, q]))
    codes = codes.astype(np.int32)
    empty = np.flatnonzero(uniq == "")
    if len(empty):
        codes[codes == empty[0]] = -1
    return codes[:len(s)], codes[len(s):]


def _log_count(sc, qc):
    """log1p(number of S1 rows with the same value) for S1 and query rows (NaN if empty)."""
    n = int(max(sc.max(initial=-1), qc.max(initial=-1))) + 1
    cnt = np.bincount(sc[sc >= 0], minlength=n)
    f = lambda c: np.where(c >= 0, np.log1p(cnt[np.maximum(c, 0)]), np.nan).astype(np.float32)
    return f(sc), f(qc)


def build_context(s_tab, q_tab, out_dir, df_cap_frac=0.02, min_df_cap=2000, share_min=0.5,
                  n_fallback=4, n_always=2, n_jobs=1, truth_keys=None, log=print):
    """Computes and saves everything blocking needs for one partition. Returns stats."""
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    n_s, n_q = s_tab.num_rows, q_tab.num_rows
    stats = {"n_s1": n_s, "n_q": n_q, "channels": {}}
    cap = max(int(min_df_cap), int(df_cap_frac * n_s))
    for ch in CHANNELS:
        Cs, Cq = channel_counts(ch, s_tab, q_tab, n_jobs)
        df = np.bincount(Cs.indices, minlength=Cs.shape[1])
        idf = np.zeros(Cs.shape[1], np.float32)
        ok = df > 0
        idf[ok] = (np.log((1.0 + n_s) / (1.0 + df[ok])) + 1.0).astype(np.float32)
        Ws, Wq = _weigh_blocks(Cs, idf, n_jobs), _weigh_blocks(Cq, idf, n_jobs)
        del Cs, Cq
        Sr, Qr, st = soft_cap(Ws, Wq, df, cap, share_min, n_fallback, n_always)
        st.update({"s_nnz_per_row": Ws.nnz / max(n_s, 1), "q_nnz_per_row": Wq.nnz / max(n_q, 1),
                   "q_empty_rows": float((np.diff(Wq.indptr) == 0).mean()) if n_q else 0.0})
        stats["channels"][ch] = st
        save_csr(out_dir, f"{ch}_S", Ws)
        save_csr(out_dir, f"{ch}_Q", Wq)
        save_csr(out_dir, f"{ch}_Qr", Qr)
        save_csr(out_dir, f"{ch}_STr", Sr.T.tocsr())
        log(f"    channel {ch}: df cap {cap:,} ({st['capped_features']:,} features over it), "
            f"rows under half weight {st['fallback_share']:.3f}, common features used per row "
            f"{st['common_features_per_row']:.2f}, S1 nnz/row {st['s_nnz_per_row']:.1f}, "
            f"record nnz/row {st['q_nnz_per_row']:.1f}, empty records {st['q_empty_rows']:.4f}, "
            f"records unsearchable {st['empty_retrieval_rows']:,}")
        del Ws, Wq, Sr, Qr
    for k, f in TOKEN_SETS.items():
        save_csr(out_dir, f"tok_{k}_S", binary_tokens(s_tab[f].to_pylist(), n_jobs))
        save_csr(out_dir, f"tok_{k}_Q", binary_tokens(q_tab[f].to_pylist(), n_jobs))
    arrays = {}
    for f in CODE_FIELDS:
        arrays[f"s_{f}"], arrays[f"q_{f}"] = _codes(s_tab[f], q_tab[f])
    arrays["s_name_freq"], arrays["q_name_freq"] = _log_count(arrays["s_name_base"], arrays["q_name_base"])
    sa, qa = _codes(s_tab["address_full"], q_tab["address_full"])
    arrays["s_addr_freq"], arrays["q_addr_freq"] = _log_count(sa, qa)
    for side, tab in (("s", s_tab), ("q", q_tab)):
        arrays[f"{side}_addr_missing"] = tab["address_missing"].to_numpy(zero_copy_only=False).astype(np.float32)
        arrays[f"{side}_nonlatin"] = tab["name_nonlatin"].to_numpy(zero_copy_only=False).astype(np.float32)
        arrays[f"{side}_gid"] = tab["gid"].to_numpy().astype(np.int64)
    arrays["q_src3"] = (q_tab["src"].to_numpy() == 3).astype(np.float32)
    if truth_keys is not None:
        arrays["truth_keys"] = np.asarray(truth_keys, np.int64)
    blocks = build_blocks(s_tab, q_tab, n_jobs)
    os.makedirs(os.path.join(out_dir, "arr"), exist_ok=True)
    for k, v in arrays.items():
        np.save(os.path.join(out_dir, "arr", k + ".npy"), v)
    for b, d in blocks.items():
        os.makedirs(os.path.join(out_dir, "blk", b), exist_ok=True)
        for k, v in d.items():
            np.save(os.path.join(out_dir, "blk", b, k + ".npy"), v)
    stats["blocks"] = {b: {"s1_keys": int(len(d["keys"])),
                           "records_with_key": int((np.diff(d["q_ptr"]) > 0).sum())}
                       for b, d in blocks.items()}
    stats["build_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(out_dir, "context.json"), "w") as f:
        json.dump(stats, f, indent=1)
    log(f"    context built in {stats['build_seconds']:.0f}s")
    return stats


def save_csr(d, name, M):
    M = M.tocsr()
    M.sort_indices()
    np.save(os.path.join(d, f"{name}.data.npy"), M.data.astype(np.float32))
    np.save(os.path.join(d, f"{name}.indices.npy"), M.indices.astype(np.int32))
    ip = M.indptr.astype(np.int64 if M.nnz >= 2 ** 31 else np.int32)
    np.save(os.path.join(d, f"{name}.indptr.npy"), ip)
    with open(os.path.join(d, f"{name}.shape.json"), "w") as f:
        json.dump(list(M.shape), f)


def load_csr(d, name, mmap=True):
    mode = "c" if mmap else None  # copy-on-write: shared, never written, but writeable for sparse_dot_topn
    with open(os.path.join(d, f"{name}.shape.json")) as f:
        shape = tuple(json.load(f))
    data = np.load(os.path.join(d, f"{name}.data.npy"), mmap_mode=mode)
    indices = np.load(os.path.join(d, f"{name}.indices.npy"), mmap_mode=mode)
    indptr = np.load(os.path.join(d, f"{name}.indptr.npy"), mmap_mode=mode)
    M = sparse.csr_matrix((data, indices, indptr), shape=shape, copy=False)
    M.has_sorted_indices = True
    return M


def load_context(d, mmap=True):
    """Everything build_context saved, memory-mapped."""
    ctx = {"dir": d, "mats": {}, "tok": {}, "arr": {}, "blocks": {}}
    for ch in CHANNELS:
        ctx["mats"][ch] = {k: load_csr(d, f"{ch}_{k}", mmap) for k in ("S", "Q", "Qr", "STr")}
    for k in TOKEN_SETS:
        ctx["tok"][k] = (load_csr(d, f"tok_{k}_S", mmap), load_csr(d, f"tok_{k}_Q", mmap))
    mode = "c" if mmap else None  # copy-on-write: shared, never written, but writeable for sparse_dot_topn
    for f in os.listdir(os.path.join(d, "arr")):
        ctx["arr"][f[:-4]] = np.load(os.path.join(d, "arr", f), mmap_mode=mode)
    for b in os.listdir(os.path.join(d, "blk")):
        bd = os.path.join(d, "blk", b)
        ctx["blocks"][b] = {f[:-4]: np.load(os.path.join(bd, f), mmap_mode=mode)
                            for f in os.listdir(bd)}
    with open(os.path.join(d, "context.json")) as f:
        ctx["stats"] = json.load(f)
    ctx["n_s"], ctx["n_q"] = ctx["stats"]["n_s1"], ctx["stats"]["n_q"]
    return ctx
