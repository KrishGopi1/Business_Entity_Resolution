"""
Command-line entry point of the v4 pipeline. From code/business_entity_resolution/:

    python src/run.py --data DATASET --work V3_WORK                  # every v4 stage in order
    python src/run.py --data DATASET --work V3_WORK --stage context4 # or one stage at a time:
        context4, dense, stage2, cut, report4, train, cross, evaluate, predict
    python src/run.py --data DATASET --work V3_WORK --zip TEAM       # package the submission

--work is v3's work folder: v4 reuses its prepare / block output (checked by fingerprint, never
rebuilt without --allow-reblock) and writes everything new to V3_WORK/v4.
On a fresh dataset the same command also runs v3's prepare and block first.
A stage whose settings or code changed since it was cached is recomputed; --force recomputes.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import schema  # noqa: E402

schema.pin_native_threads()       # before numpy / scipy load their thread pools

import argparse  # noqa: E402
from dataclasses import fields  # noqa: E402

from pipeline import STAGES, V4_STAGES, Config, run  # noqa: E402

ALL = ["prepare", "block"] + V4_STAGES

SKIP = {"data_dir", "work_dir", "out_dir", "n_jobs", "force", "channels", "blocks"}


def main():
    p = argparse.ArgumentParser(description="Business Entity Resolution pipeline (v4)")
    p.add_argument("--data", default="dataset", help="folder holding train/ and test/")
    p.add_argument("--work", default=None, help="cache folder (default work, or work_quick)")
    p.add_argument("--out", default=None, help="output folder (default output, or output_quick)")
    p.add_argument("--stage", default="all", choices=["all"] + STAGES)
    p.add_argument("--quick", action="store_true", help="small smoke test (not a valid submission)")
    p.add_argument("--force", action="store_true", help="recompute the selected stages")
    p.add_argument("--jobs", type=int, default=0, help="worker processes (0 = CPU quota - 1)")
    p.add_argument("--channels", default=None, help="comma list of retrieval channels (ablation)")
    p.add_argument("--blocks", default=None, help="comma list of rescue blocks, or 'none' (ablation)")
    p.add_argument("--zip", metavar="TEAM", default=None, help="only build TEAM_submission.zip")
    for f in fields(Config):
        if f.name in SKIP:
            continue
        flag = "--" + f.name.replace("_", "-")
        if f.type in (bool, "bool"):
            p.add_argument(flag, default=None, action=argparse.BooleanOptionalAction)
        else:
            typ = {"int": int, "float": float, "str": str}.get(getattr(f.type, "__name__", f.type), str)
            p.add_argument(flag, type=typ, default=None)
    a = p.parse_args()
    kw = {"data_dir": a.data, "n_jobs": a.jobs, "force": a.force}
    if a.work:
        kw["work_dir"] = a.work
    if a.out:
        kw["out_dir"] = a.out
    if a.channels:
        kw["channels"] = tuple(x for x in a.channels.split(",") if x)
    if a.blocks:
        kw["blocks"] = () if a.blocks == "none" else tuple(x for x in a.blocks.split(",") if x)
    for f in fields(Config):
        v = getattr(a, f.name, None)
        if f.name not in SKIP and v is not None:
            kw[f.name] = v
    cfg = Config.quick(**kw) if a.quick else Config(**kw)
    import pyarrow as pa
    pa.set_cpu_count(cfg.jobs)
    pa.set_io_thread_count(min(cfg.jobs, 8))
    if a.zip:
        import model
        model.make_zip(cfg, a.zip)
        return
    else:
        run(cfg, ALL if a.stage == "all" else [a.stage])


if __name__ == "__main__":
    main()
