"""v4: string views, pair features, cache reuse and an end-to-end run on generated data."""
import json
import os
from dataclasses import replace

import numpy as np
import pyarrow as pa
import pytest

import synth
import views4
from pipeline import Config, run


@pytest.mark.parametrize("a,b", [("technology", "teknoloji"), ("private", "praivet"),
                                 ("laxmi", "lakshmi"), ("telangana", "telamgana"),
                                 ("media", "midiya"), ("engineering", "innjiniyaring")])
def test_loose_romanization_joins_spellings(a, b):
    assert views4.loose_token(a) == views4.loose_token(b)


def test_loose_keeps_different_names_apart():
    assert views4.loose_token("kumar") != views4.loose_token("kapoor")


def test_legal_words_dropped_anywhere():
    assert views4.name_nolegal("adams ltd center") == "adams center"
    assert views4.name_nolegal("phyuchar teknoloji elaelapi") == "phyuchar teknoloji"
    assert views4.name_nolegal("supar hai marketimg pra li") == "supar hai marketimg"


def test_indic_state_in_local_script():
    assert views4.indic_state("hatkanangle, महाराष्ट्र")[0] == "mh"
    assert views4.indic_state("office no.759/9, ಕರ್ನಾಟಕ")[0] == "ka"


def test_door_number_features_flag_one_digit_changes():
    import features4 as f4
    q = {"numbers4": np.array(["32 400001"], object)}
    s = {"numbers4": np.array(["33 400001"], object)}
    ml, sub, tr, uq, us = f4._numbers(q["numbers4"], s["numbers4"])
    assert ml[0] == 1 and sub[0] == 1 and tr[0] == 0
    eq, pre, nd = f4._house(np.array(["2776"], object), np.array(["776"], object))
    assert eq[0] == 0 and pre[0] == 1


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    root = tmp_path_factory.mktemp("ber4")
    data = synth.generate(str(root / "data"), scale=0.3, seed=3)
    return Config(data_dir=data, work_dir=str(root / "work"), out_dir=str(root / "out"),
                  n_jobs=2, train_s1=1500, tune_s1=500, holdout_s1=500, min_df_cap=100,
                  k_name=10, k_addr=10, k_joint=10, k_noaddr=30, ranker_sample=20_000,
                  floor=0.0, max_rounds=200, s2_sample=20_000, model_s1=2000)


def test_v4_end_to_end(cfg):
    import pipeline4
    import schema
    run(cfg, ["prepare", "block"])                         # stands in for the v3 work/ cache
    assert pipeline4.v3_cache_state(cfg) == "ok"
    run(cfg, ["prepare", "block", "context4", "dense", "stage2", "cut", "report4",
              "train", "evaluate", "predict"])
    rd = os.path.join(cfg.work_dir, "v4", "reports")
    assert "V4 CANDIDATE GATE" in open(os.path.join(rd, "cut_report.txt")).read()
    assert "V4 SCORE GATES" in open(os.path.join(rd, "final_report.txt")).read()
    issues = schema.check_submission(os.path.join(cfg.out_dir, schema.MATCH_FILE),
                                     os.path.join(cfg.out_dir, schema.CAND_FILE),
                                     os.path.join(cfg.data_dir, "test"))
    assert issues == []


def test_changed_blocking_settings_refuse_to_reblock(cfg):
    c = replace(cfg, k_name=11)
    with pytest.raises(SystemExit, match="v3"):
        run(c, ["block"])
