"""Rescue blocks, the soft df cap and the keep rule."""
import numpy as np
import pyarrow as pa
from scipy import sparse

import blocking as bk
import indexing as ix
import normalize as nz


def _tab(names, addrs):
    t = nz.normalize_records(names, addrs, n_jobs=1)
    t = t.append_column("gid", pa.array(np.arange(len(names), dtype=np.int32)))
    return t.append_column("src", pa.array(np.full(len(names), 2, np.int8)))


def test_block_lookup_respects_cap():
    s = _tab(["Sai Enterprises"] * 5 + ["Acme Tools"], ["1 A St"] * 6)
    q = _tab(["Sai Enterprises", "Acme Tools"], ["", ""])
    b = ix.build_blocks(s, q)["name_full"]
    qr, sr, big = ix.block_lookup(b, 0, 2, cap=3)
    assert list(qr) == [1] and list(sr) == [5] and big == 1          # 5 > cap: skipped
    qr, sr, big = ix.block_lookup(b, 0, 2, cap=10)
    assert sorted(sr[qr == 0]) == [0, 1, 2, 3, 4] and big == 0


def test_postcode_rare_name_block():
    s = _tab(["Sai Enterprises", "Sai Traders"], ["1 MG Rd, Pune 411001", "2 MG Rd, Pune 411001"])
    q = _tab(["SAI TRADERS PVT LTD"], ["MG ROAD PUNE 411001"])
    b = ix.build_blocks(s, q)["pc_rare_name"]
    qr, sr, _ = ix.block_lookup(b, 0, 1, cap=10)
    assert 1 in set(sr)


def test_soft_cap_never_leaves_a_record_unsearchable():
    # 3 features: col 0 is common (df 100), cols 1-2 rare
    Ws = sparse.random(200, 3, density=0.3, format="csr", random_state=0, dtype=np.float32)
    df = np.array([100, 5, 5])
    Wq = sparse.csr_matrix(np.array([[1.0, 0, 0], [0.6, 0.8, 0]], np.float32))
    Sr, Qr, st = ix.soft_cap(Ws, Wq, df, cap=50, share_min=0.5, n_fallback=1, n_always=0)
    assert np.diff(Qr.indptr)[0] == 1                # common-only record keeps its common feature
    assert st["empty_retrieval_rows"] == 0
    assert set(Qr[1].indices) == {1}                 # rare share 0.64 >= 0.5: common left out
    assert 0 in set(Sr.indices)                      # S1 side keeps the column a record uses
    _, Qr2, _ = ix.soft_cap(Ws, Wq, df, cap=50, share_min=0.5, n_fallback=1, n_always=1)
    assert set(Qr2[1].indices) == {0, 1}             # n_always: common evidence always kept


def test_keep_rule_has_no_s1_side_cap_and_bounds_records():
    q = np.repeat(np.arange(3), 50)
    score = np.tile(np.linspace(1, 0.001, 50), 3).astype(np.float32)
    keep, rank = bk.keep_rule(q, score, keep_min=2, keep_max=10, floor=0.5)
    per = np.bincount(q[keep])
    assert (per == 10).all()
    keep, _ = bk.keep_rule(q, score, keep_min=2, keep_max=10, floor=2.0)
    assert (np.bincount(q[keep]) == 2).all()


def test_trim_ties_keeps_all_ties_with_kth():
    R = sparse.csr_matrix(np.array([[0.9, 0.5, 0.5, 0.5, 0.1]], np.float32))
    r, c, v = bk.trim_ties(R, 2)
    assert sorted(c) == [0, 1, 2, 3]
