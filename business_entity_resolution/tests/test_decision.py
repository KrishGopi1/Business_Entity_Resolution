import numpy as np

import decision as dc
import model


def test_per_entity_f05_rules():
    # S1 0: 2 true (records 0, 1), predicts 0 only -> P 1, R .5 -> F0.5 = 1.25*.5/(.25+.5)
    # S1 1: no true record, predicts nothing -> 1;  S1 2: no true record, predicts 5 -> 0
    gt_s, gt_q = np.array([0, 0]), np.array([0, 1])
    f, n_true, n_pred, tp = dc.per_entity_f05(np.array([0, 2]), np.array([0, 5]), gt_s, gt_q,
                                              np.array([0, 1, 2]), 10)
    assert np.allclose(f, [1.25 * 0.5 / 0.75, 1.0, 0.0])
    assert n_true.tolist() == [2, 0, 0] and n_pred.tolist() == [1, 0, 1] and tp.tolist() == [1, 0, 0]


def test_expected_f_prefix():
    k, _ = dc.expected_f05_best_k(np.array([[0.9], [0.2]]))
    assert k.tolist() == [1, 0]                   # a lone candidate is kept iff p > 0.5 (u = 0)
    k, _ = dc.expected_f05_best_k(np.array([[0.51], [0.49]]))
    assert k.tolist() == [1, 0]
    k, _ = dc.expected_f05_best_k(np.array([[0.99, 0.98, 0.05]]))
    assert k.tolist() == [2]


def test_exclusive_assignment_and_threshold():
    qg = np.array([0, 0, 1, 2])
    sg = np.array([10, 11, 10, 12])
    p = np.array([0.6, 0.9, 0.8, 0.3], np.float32)
    s, q, pp = dc.decide(qg, sg, p, {"mode": "threshold", "min_prob": 0.5})
    assert sorted(zip(s.tolist(), q.tolist())) == [(10, 1), (11, 0)]   # record 0 goes to 11 only


def test_calibration_monotone():
    rng = np.random.default_rng(0)
    p = rng.random(2000)
    y = rng.random(2000) < p ** 2
    cal = dc.fit_calibration(p, y)
    c = dc.apply_calibration(np.sort(p), cal)
    assert (np.diff(c) >= -1e-7).all() and c[0] < 0.2 and c[-1] > 0.8


def test_context_and_folds():
    q = np.array([0, 0, 0, 1])
    C = model.context(q, np.array([0.2, 0.9, 0.5, 0.7], np.float32))
    assert C[:, 1].tolist() == [3, 1, 2, 1]                      # rank inside the record
    assert np.allclose(C[:, 2], [0.7, 0.0, 0.4, 0.0])           # best - p
    assert np.allclose(C[:, 3], [-0.7, 0.4, -0.4, 0.7])         # p - best other
    f = model.fold_of(np.arange(1000), 3)
    assert set(f.tolist()) == {0, 1, 2} and (model.fold_of(np.arange(1000), 3) == f).all()


def test_post_keep_policy():
    from pipeline import Config, post_keep
    rp = np.array([1, 5, 6, 20, 21, 40])
    rs = np.array([0.9, 1e-6, 0.01, 0.002, 0.5, 0.5])
    assert post_keep(Config(), rp, rs).all()                            # off by default
    keep = post_keep(Config(post_keep_max=20, post_floor=0.001), rp, rs)
    assert keep.tolist() == [True, True, True, True, False, False]      # top 5 always kept
