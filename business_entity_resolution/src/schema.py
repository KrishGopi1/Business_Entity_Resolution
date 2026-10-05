"""
Data contracts for the Business Entity Resolution pipeline (v3): input files and columns,
reading, country labels, CPU limits, and the submission files (writing and checking).

Nothing here depends on the country set: labels are an open set, and only unambiguous
spellings of the known countries are merged.
"""
import os

import numpy as np
import pandas as pd
import pyarrow as pa

COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
MATCH_FILE, CAND_FILE = "matching_results.tsv", "candidate_pairs.tsv"
MATCH_COL, CAND_COL = "matched_entity_ids", "candidate_entity_ids"
QUERY_PREFIXES = ("S2-", "S3-")


def source_path(data_dir, split, k):
    """k = 1, 2, 3 for the source files, 'gt' for the ground truth."""
    name = f"{split}_ground_truth.tsv" if k == "gt" else f"{split}_source{k}.tsv"
    return os.path.join(data_dir, split, name)


# ---------------------------------------------------------------------------
# CPU limits
# ---------------------------------------------------------------------------

def usable_cpus():
    """CPUs this process may really use: os.cpu_count() limited by CPU affinity and by a
    container's cgroup quota (the HPC container shows 96 CPUs but allows 25). Running more
    busy threads than the quota makes the kernel throttle the whole container."""
    n = os.cpu_count() or 2
    if hasattr(os, "sched_getaffinity"):
        n = min(n, len(os.sched_getaffinity(0)))
    try:                                                    # cgroup v2
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()[:2]
        if quota != "max":
            n = min(n, max(1, int(int(quota) / int(period))))
    except (OSError, ValueError):
        try:                                                # cgroup v1
            with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
                quota = int(f.read())
            with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
                period = int(f.read())
            if quota > 0:
                n = min(n, max(1, quota // period))
        except (OSError, ValueError):
            pass
    return n


def default_jobs(n_jobs=None):
    return max(1, usable_cpus() - 1) if not n_jobs or n_jobs < 1 else int(n_jobs)


THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")


def pin_native_threads():
    """One thread per native pool (BLAS, OpenMP). The pipeline parallelizes with its own
    processes and passes explicit thread counts to XGBoost and rapidfuzz. Must run before
    numpy/scipy are imported to take effect; run.py calls it first."""
    for v in THREAD_VARS:
        os.environ[v] = "1"


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def read_tsv(path, usecols=None):
    """Tab-separated, every column as text, empty fields as ''. pyarrow's reader first
    (5-10x faster than pandas on these files), pandas if pyarrow cannot parse the file."""
    cols = list(usecols) if usecols else None
    try:
        import pyarrow.csv as pacsv
        t = pacsv.read_csv(
            path,
            parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
            convert_options=pacsv.ConvertOptions(
                include_columns=cols, strings_can_be_null=False,
                column_types={c: pa.string() for c in (cols or COLUMNS + GT_COLUMNS)}))
        df = t.to_pandas()
    except Exception:                                       # noqa: BLE001
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False,
                         usecols=cols, quoting=3)
    for c in df.columns:
        df[c] = df[c].fillna("").astype(str)
    need = set(cols or (GT_COLUMNS if "ground_truth" in os.path.basename(path) else COLUMNS))
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return df[cols] if cols else df


def explode_truth(gt):
    """Ground-truth table -> (S1 id per true pair, record id per true pair)."""
    lists = gt["matched_entity_ids"].astype(str).to_numpy()
    counts = np.array([0 if not x.strip() else x.count(",") + 1 for x in lists], np.int64)
    flat = [t.strip() for x in lists if x.strip() for t in x.split(",")]
    return (np.repeat(gt["source1_entity_id"].astype(str).to_numpy(), counts),
            np.array(flat, dtype=object))


COUNTRY_ALIASES = {
    "usa": "us", "u.s.": "us", "u.s.a.": "us", "united states": "us",
    "united states of america": "us", "ind": "india", "in": "india", "bharat": "india",
    "fr": "france", "fra": "france", "république française": "france",
}


def normalize_country(values):
    s = pd.Series(values, dtype=object).fillna("").astype(str).str.strip().str.lower()
    return s.map(lambda x: COUNTRY_ALIASES.get(x, x)).to_numpy(dtype=object)


# ---------------------------------------------------------------------------
# Submission files
# ---------------------------------------------------------------------------

def write_lists(path, column, s1_ids, sg, qg, q_ids, score):
    """One row per S1 in test-file order; each S1's record ids sorted by score (descending),
    comma-separated. sg / qg are integer row ids into s1_ids / q_ids."""
    sg = np.asarray(sg, np.int64)
    order = np.lexsort((-np.asarray(score, np.float64), sg))
    sg, ids = sg[order], q_ids[np.asarray(qg)[order]]
    bounds = np.searchsorted(sg, np.arange(len(s1_ids) + 1))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{column}\n")
        buf = []
        for i, s in enumerate(s1_ids):
            buf.append(f"{s}\t{','.join(ids[bounds[i]:bounds[i + 1]])}\n")
            if len(buf) >= 100_000:
                f.writelines(buf)
                buf = []
        f.writelines(buf)


def _iter_rows(path, batch=200_000):
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n")
        rows = []
        for line in f:
            rows.append(line.rstrip("\r\n").split("\t"))
            if len(rows) >= batch:
                yield header, rows
                rows = []
        yield header, rows


def check_submission(matching_path, candidate_path, test_dir, smoke=False, log=print):
    """Checks both files against the submission rules. Returns the list of problems
    (empty = pass). Run the official utils/validate_submission.py as well."""
    test_s1 = set(read_tsv(os.path.join(test_dir, "test_source1.tsv"), ["entity_id"])["entity_id"])
    test_q = pd.Index(pd.concat([read_tsv(os.path.join(test_dir, f"test_source{k}.tsv"),
                                          ["entity_id"])["entity_id"] for k in (2, 3)]))
    issues, matched = [], {}
    for path, col in ((matching_path, MATCH_COL), (candidate_path, CAND_COL)):
        name = os.path.basename(path)
        seen = set()
        bad_cols = dup_rows = dup_in_list = bad_prefix = unknown = stray = 0
        for header, rows in _iter_rows(path):
            if header != f"source1_entity_id\t{col}" and not seen:
                issues.append(f"{name}: header is {header!r}")
            flat = []
            for r in rows:
                if len(r) != 2:
                    bad_cols += 1
                    continue
                dup_rows += r[0] in seen
                seen.add(r[0])
                ids = r[1].split(",") if r[1] else []
                dup_in_list += len(set(ids)) != len(ids)
                flat.extend(ids)
                if col == MATCH_COL:
                    matched[r[0]] = ids
                elif r[0] in matched:
                    stray += len(set(matched[r[0]]) - set(ids))
            bad_prefix += sum(not x.startswith(QUERY_PREFIXES) for x in flat)
            unknown += int((~pd.Index(flat).isin(test_q)).sum()) if flat else 0
        for n, msg in ((bad_cols, "rows without exactly 2 columns"),
                       (dup_rows, "duplicate source1_entity_id rows"),
                       (0 if smoke else len(test_s1 - seen), "test S1 entities missing"),
                       (len(seen - test_s1), "rows whose id is not a test S1 entity"),
                       (dup_in_list, "rows with a duplicate id inside the list"),
                       (bad_prefix, "listed ids that are not S2-/S3- ids"),
                       (unknown, "listed ids not found in the test S2/S3 files"),
                       (stray, "matched ids missing from candidate_pairs.tsv for their S1")):
            if n:
                issues.append(f"{name}: {n:,} {msg}")
    if issues:
        log("SELF-CHECK FOUND ISSUES:\n  " + "\n  ".join(issues))
    else:
        log("self-check: PASS (both files follow the submission rules)"
            + (" [smoke test: only a sample of test entities]" if smoke else ""))
    return issues
