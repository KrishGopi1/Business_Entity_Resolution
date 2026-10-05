"""Submission files: one row per test S1, lists by score, matches within candidates."""
import os

import numpy as np

import schema


def _write_test_dir(d):
    os.makedirs(d, exist_ok=True)
    rows = {1: [("S1-1", "a", "x", "US"), ("S1-2", "b", "y", "US")],
            2: [("S2-1", "a", "x", "US")], 3: [("S3-1", "a", "x", "US"), ("S3-2", "c", "z", "US")]}
    for k, rs in rows.items():
        with open(os.path.join(d, f"test_source{k}.tsv"), "w") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            f.writelines("\t".join(r) + "\n" for r in rs)


def test_write_and_check(tmp_path):
    td = tmp_path / "test"
    _write_test_dir(td)
    s1 = np.array(["S1-1", "S1-2"], dtype=object)
    q = np.array(["S2-1", "S3-1", "S3-2"], dtype=object)
    m, c = tmp_path / "m.tsv", tmp_path / "c.tsv"
    schema.write_lists(m, schema.MATCH_COL, s1, [0, 0], [0, 1], q, [0.9, 0.95])
    schema.write_lists(c, schema.CAND_COL, s1, [0, 0, 1], [0, 1, 2], q, [0.9, 0.95, 0.1])
    lines = open(m).read().splitlines()
    assert lines == ["source1_entity_id\tmatched_entity_ids", "S1-1\tS3-1,S2-1", "S1-2\t"]
    assert schema.check_submission(m, c, td, log=lambda *a: None) == []
    schema.write_lists(c, schema.CAND_COL, s1, [0], [0], q, [0.9])       # S3-1 not a candidate
    assert any("missing from candidate" in x for x in schema.check_submission(m, c, td, log=lambda *a: None))
