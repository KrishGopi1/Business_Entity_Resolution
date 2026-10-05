"""
Pair-model features (v4) for the pairs the stage-2 cut keeps:

  * every stage-2 feature (expand4.S2_FEATS: v3's 44 ranker features over the expanded set,
    stage-1 score and rank, the cheap v4 features) and the stage-2 score, its rank and its gap
    to the record's best (record context; no count that grows with S2/S3 per S1)
  * v3.1's string comparisons (features.EXTRA), computed on v4's string files
  * v4 string comparisons aimed at the real-data loss report:
      lo_ratio, lo_tset     loose romanization (Indic script vs English spelling)
      lo_jw                 Jaro-Winkler of the loose names without spaces
      ds_partial            partial ratio of the despaced names (domains, handles)
      nl_first_same, nl_first_ratio, nl_last_same
                            first / last legal-free name token (a decoy often changes one
                            name token: 'jmtk finance' vs 'mtk finance', 'lancaster, brinkley'
                            vs 'motta, brinkley')
      nl_q_extra, nl_s_extra
                            share of one side's legal-free tokens with no close match (ratio
                            >= 85) on the other side
      ac_tsort              token-sort similarity of the core addresses
      h_eq, h_prefix, h_numdiff
                            door numbers equal, one a prefix / suffix of the other, relative
                            difference of their leading integers
      n4_min_lev, n4_sub1, n4_trunc, n4_unmatched_q, n4_unmatched_s
                            numbers of the address that match exactly, and for the others the
                            closest pair: its edit distance, whether it is a one-digit
                            substitution of equal length (the decoy pattern 'c-32' vs 'c-33'),
                            or a truncation / extension ('2776' vs '776')
"""
import numpy as np
import pyarrow as pa
from rapidfuzz import distance, fuzz, process

import context4 as c4
import expand4 as ex
import features as ft3

CTX2 = ["s2_score", "s2_rank", "s2_gap", "rescued"]
V4X = ["lo_ratio", "lo_tset", "lo_jw", "ds_partial", "nl_first_same", "nl_first_ratio",
       "nl_last_same", "nl_q_extra", "nl_s_extra", "ac_tsort", "h_eq", "h_prefix", "h_numdiff",
       "n4_min_lev", "n4_sub1", "n4_trunc", "n4_unmatched_q", "n4_unmatched_s"]
FEATURES = list(ex.S2_FEATS) + CTX2 + list(ft3.EXTRA) + V4X
PAIR_COLS = ["qi", "si", "qg", "sg"] + list(ex.S2_FEATS) + CTX2


def open_strings(path):
    return pa.ipc.open_file(pa.memory_map(path, "r")).read_all()


def _take(tab, rows):
    idx = pa.array(np.asarray(rows, np.int64))
    return {c: tab.column(c).take(idx).to_numpy(zero_copy_only=False) for c in c4.STR4}


_sim, _eq = ft3._sim, ft3._eq


# The helpers below give the same values as their plain loops (see tests/test_v4.py) but split
# each distinct string once: a record's string recurs in all of its ~25 pairs.

def _splitter():
    cache = {}

    def split(x):
        t = cache.get(x)
        if t is None:
            t = cache[x] = x.split()
        return t
    return split


def _contain(a, b, min_len=3):
    """features.py's _contain: share of the tokens of a (min_len+ characters) found inside b
    with spaces removed."""
    out = np.full(len(a), np.nan, np.float32)
    ca, cb = {}, {}
    for i, (x, y) in enumerate(zip(a, b)):
        toks = ca.get(x)
        if toks is None:
            toks = ca[x] = [t for t in x.split() if len(t) >= min_len]
        if toks and y:
            yy = cb.get(y)
            if yy is None:
                yy = cb[y] = y.replace(" ", "")
            out[i] = sum(t in yy for t in toks) / len(toks)
    return out


def _tok_share(a, b):
    """features.py's _tok_share: share of the tokens of a present among the tokens of b."""
    out = np.full(len(a), np.nan, np.float32)
    split, cs = _splitter(), {}
    for i, (x, y) in enumerate(zip(a, b)):
        tx = split(x)
        if tx and y:
            ty = cs.get(y)
            if ty is None:
                ty = cs[y] = set(y.split())
            out[i] = sum(t in ty for t in tx) / len(tx)
    return out


def _first_last(a, b):
    n = len(a)
    fs, fr, ls = (np.full(n, np.nan, np.float32) for _ in range(3))
    split = _splitter()
    for i, (x, y) in enumerate(zip(a, b)):
        tx, ty = split(x), split(y)
        if tx and ty:
            fs[i] = tx[0] == ty[0]
            fr[i] = fuzz.ratio(tx[0], ty[0]) / 100.0
            ls[i] = tx[-1] == ty[-1]
    return fs, fr, ls


def _extra_share(a, b, cut=85):
    """Share of a's tokens with no token of b at ratio >= cut."""
    out = np.full(len(a), np.nan, np.float32)
    split = _splitter()
    for i, (x, y) in enumerate(zip(a, b)):
        tx, ty = split(x), split(y)
        if tx and ty:
            miss = 0
            for t in tx:
                if t in ty:
                    continue
                if process.extractOne(t, ty, scorer=fuzz.ratio, score_cutoff=cut) is None:
                    miss += 1
            out[i] = miss / len(tx)
    return out


def _lead_int(x):
    d = ""
    for c in x:
        if c.isdigit():
            d += c
        elif d:
            break
    return int(d[:9]) if d else -1


def _house(a, b):
    n = len(a)
    eq, pre, nd = (np.full(n, np.nan, np.float32) for _ in range(3))
    lead = {}
    for i, (x, y) in enumerate(zip(a, b)):
        if x and y:
            eq[i] = x == y
            pre[i] = x != y and (x.startswith(y) or y.startswith(x) or x.endswith(y) or y.endswith(x))
            u = lead.get(x)
            if u is None:
                u = lead[x] = _lead_int(x)
            v = lead.get(y)
            if v is None:
                v = lead[y] = _lead_int(y)
            if u >= 0 and v >= 0:
                nd[i] = abs(u - v) / max(u, v, 1)
    return eq, pre, nd


def _numbers(a, b):
    n = len(a)
    ml, sub, tr, uq, us = (np.full(n, np.nan, np.float32) for _ in range(5))
    lev = distance.Levenshtein.distance
    sets = {}
    for i, (x, y) in enumerate(zip(a, b)):
        if not x or not y:
            continue
        sx = sets.get(x)
        if sx is None:
            sx = sets[x] = set(x.split())
        sy = sets.get(y)
        if sy is None:
            sy = sets[y] = set(y.split())
        if not sx or not sy:
            continue
        ox, oy = sx - sy, sy - sx
        uq[i], us[i] = len(ox), len(oy)
        if not ox or not oy:
            ml[i], sub[i], tr[i] = 0.0, 0.0, 0.0
            continue
        best, bs, bt = 99, 0.0, 0.0
        for p in ox:
            for r in oy:
                d = lev(p, r)
                if d < best:
                    best = d
                    bs = float(d == 1 and len(p) == len(r))
                    bt = float(len(p) != len(r) and (p.startswith(r) or r.startswith(p)
                                                     or p.endswith(r) or r.endswith(p)))
        ml[i], sub[i], tr[i] = best, bs, bt
    return ml, sub, tr, uq, us


def pair_features(pairs, S, Q, workers=1):
    """pairs: dict of arrays (PAIR_COLS). S, Q: the partition's v4 string files.
    Returns a float32 matrix in FEATURES order (NaN = missing)."""
    qi, si = np.asarray(pairs["qi"]), np.asarray(pairs["si"])
    n = len(qi)
    if n == 0:
        return np.empty((0, len(FEATURES)), np.float32)
    s, q = _take(S, si), _take(Q, qi)
    F = {c: np.asarray(pairs[c], np.float32) for c in list(ex.S2_FEATS) + CTX2}
    F["nm_raw_ratio"] = _sim(s["name_raw_clean"], q["name_raw_clean"], fuzz.ratio, workers)
    F["nm_full_ratio"] = _sim(s["name_full"], q["name_full"], fuzz.ratio, workers)
    F["nm_tsort"] = _sim(s["name_base"], q["name_base"], fuzz.token_sort_ratio, workers)
    F["nm_tset"] = _sim(s["name_base"], q["name_base"], fuzz.token_set_ratio, workers)
    F["nm_phon_ratio"] = _sim(s["name_phonetic"], q["name_phonetic"], fuzz.ratio, workers)
    F["nm_nospace_jw"] = _sim(s["name_nospace"], q["name_nospace"],
                              distance.JaroWinkler.normalized_similarity, workers, 1.0)
    F["nm_exact_full"] = _eq(s["name_full"], q["name_full"])
    F["nm_exact_core"] = _eq(s["name_core"], q["name_core"])
    dba = [_sim(s["name_base"], q["name_dba"], fuzz.token_set_ratio, workers)]
    if (s["name_dba"] != "").any():
        dba += [_sim(s["name_dba"], q["name_base"], fuzz.token_set_ratio, workers),
                _sim(s["name_dba"], q["name_dba"], fuzz.token_set_ratio, workers)]
    F["nm_dba_best"] = np.fmax.reduce(dba) if len(dba) > 1 else dba[0]
    F["nm_acronym"] = (((s["name_acronym"] != "") & (q["name_nospace"] == s["name_acronym"]))
                       | ((q["name_acronym"] != "") & (s["name_nospace"] == q["name_acronym"]))
                       ).astype(np.float32)
    ls = np.fromiter((len(x) for x in s["name_base"]), np.float32, n)
    lq = np.fromiter((len(x) for x in q["name_base"]), np.float32, n)
    F["nm_len_diff"] = np.abs(ls - lq)
    F["ad_tset"] = _sim(s["address_full"], q["address_full"], fuzz.token_set_ratio, workers)
    F["ad_tsort"] = _sim(s["address_full"], q["address_full"], fuzz.token_sort_ratio, workers)
    F["floor_same"] = _eq(s["floor_number"], q["floor_number"])
    F["nl_tset"] = _sim(s["name_nolegal"], q["name_nolegal"], fuzz.token_set_ratio, workers)
    F["nl_tsort"] = _sim(s["name_nolegal"], q["name_nolegal"], fuzz.token_sort_ratio, workers)
    F["nl_contain_s"] = _contain(s["name_nolegal"], q["name_nolegal"])
    F["nl_contain_q"] = _contain(q["name_nolegal"], s["name_nolegal"])
    F["ac_tset"] = _sim(s["address_core"], q["address_core"], fuzz.token_set_ratio, workers)
    F["ac_contain_s"] = _tok_share(s["address_core"], q["address_core"])
    F["ac_contain_q"] = _tok_share(q["address_core"], s["address_core"])
    F["house_lev"] = _sim(s["house_number"], q["house_number"], distance.Levenshtein.distance, workers, 1.0)
    F["num_lev"] = _sim(s["numeric_tokens"], q["numeric_tokens"], distance.Levenshtein.distance, workers, 1.0)
    # v4
    F["lo_ratio"] = _sim(s["name_loose"], q["name_loose"], fuzz.ratio, workers)
    F["lo_tset"] = _sim(s["name_loose"], q["name_loose"], fuzz.token_set_ratio, workers)
    sl = np.char.replace(s["name_loose"].astype(str), " ", "").astype(object)
    ql = np.char.replace(q["name_loose"].astype(str), " ", "").astype(object)
    F["lo_jw"] = _sim(sl, ql, distance.JaroWinkler.normalized_similarity, workers, 1.0)
    F["ds_partial"] = _sim(s["name_despaced"], q["name_despaced"], fuzz.partial_ratio, workers)
    F["nl_first_same"], F["nl_first_ratio"], F["nl_last_same"] = _first_last(s["name_nolegal"], q["name_nolegal"])
    F["nl_q_extra"] = _extra_share(q["name_nolegal"], s["name_nolegal"])
    F["nl_s_extra"] = _extra_share(s["name_nolegal"], q["name_nolegal"])
    F["ac_tsort"] = _sim(s["address_core"], q["address_core"], fuzz.token_sort_ratio, workers)
    F["h_eq"], F["h_prefix"], F["h_numdiff"] = _house(s["house_number"], q["house_number"])
    (F["n4_min_lev"], F["n4_sub1"], F["n4_trunc"], F["n4_unmatched_q"],
     F["n4_unmatched_s"]) = _numbers(q["numbers4"], s["numbers4"])
    return np.column_stack([F[c] for c in FEATURES]).astype(np.float32)
