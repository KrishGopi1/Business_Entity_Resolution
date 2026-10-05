"""
Cross-encoder on a Kaggle GPU, for machines without one (--cross kaggle).

On the Kaggle notebook (GPU on, Internet on for the first download of the model):

    !pip install -q transformers==5.17.0
    !python kaggle_cross.py --export /kaggle/input/<your-dataset> --out /kaggle/working

Input: the folder written by `run.py --stage cross --cross kaggle` (pairs.parquet,
q_train / s_train / q_test / s_test .parquet, export.json). Output in --out:
cross_scores.parquet (split, qg, sg, logit) and model/ (the fine-tuned weights, for the
submission zip). Needs only torch, transformers, pandas and pyarrow; it does not import the
pipeline.

The model starts from export.json's model (default intfloat/multilingual-e5-small, MIT) with a
fresh one-logit head and is fine-tuned on split 0 (training entities only, labelled) with a
binary cross-entropy loss; then splits 1 and 2 are scored. Checkpoints are written every 10% of
training, and a restarted run skips finished work.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def texts(folder, side, split):
    t = pd.read_parquet(os.path.join(folder, f"{side}_{split}.parquet"))
    return pd.Series(t.text.to_numpy(), index=t.gid.to_numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--model", default=None, help="override export.json's model (a local path works)")
    ap.add_argument("--score-batch", type=int, default=512)
    ap.add_argument("--max-train", type=int, default=0, help="use at most this many fine-tuning pairs (smoke tests)")
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    ex = json.load(open(os.path.join(a.export, "export.json")))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device {device}; export counts {ex['counts']}")
    pairs = pd.read_parquet(os.path.join(a.export, "pairs.parquet"))
    tq, ts = texts(a.export, "q", "train"), texts(a.export, "s", "train")
    L = ex["max_len"]
    mdir = os.path.join(a.out, "model")
    amp = device == "cuda"
    if os.path.exists(os.path.join(mdir, "done")):
        log("fine-tuned model found; skipping training")
    else:
        name = a.model or ex["model"]
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=1).to(device)
        tr = pairs[pairs.split == 0]
        if a.max_train:
            tr = tr.iloc[:a.max_train]
        qg, sg, y = tr.qg.to_numpy(), tr.sg.to_numpy(), tr.y.to_numpy().astype(np.float32)
        log(f"fine-tuning on {len(y):,} pairs ({y.mean():.0%} true), {ex['epochs']} epoch(s)")
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
        B = ex["batch"]
        steps = max(1, ex["epochs"] * (len(y) // B))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda i: min(1.0, (i + 1) / max(1, steps // 20)) * max(0.0, 1 - i / steps))
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        bce = torch.nn.BCEWithLogitsLoss()
        rng = np.random.default_rng(53)
        t0, step = time.time(), 0
        for _ in range(ex["epochs"]):
            order = rng.permutation(len(y))
            for i in range(0, len(order) - B + 1, B):
                b = order[i:i + B]
                f = tok(list(tq.loc[qg[b]]), list(ts.loc[sg[b]]), truncation=True, max_length=L,
                        padding=True, return_tensors="pt").to(device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                    loss = bce(model(**f).logits.float().squeeze(-1), torch.from_numpy(y[b]).to(device))
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                step += 1
                if step % max(1, steps // 20) == 0:
                    el = time.time() - t0
                    log(f"  step {step}/{steps}, loss {loss.item():.4f}, {el / 60:.1f} min, "
                        f"~{el / step * (steps - step) / 60:.0f} min left")
        model.save_pretrained(mdir)
        tok.save_pretrained(mdir)
        open(os.path.join(mdir, "done"), "w").close()
    tok = AutoTokenizer.from_pretrained(mdir)
    model = AutoModelForSequenceClassification.from_pretrained(mdir, num_labels=1).to(device).eval()
    if amp:
        model = model.half()
    outs = []
    for split, name in ((1, "train"), (2, "test")):
        part = os.path.join(a.out, f"scores_{split}.parquet")
        p = pairs[pairs.split == split]
        if os.path.exists(part):
            outs.append(pd.read_parquet(part))
            continue
        q, s = texts(a.export, "q", name), texts(a.export, "s", name)
        qt, st = q.loc[p.qg.to_numpy()].to_numpy(), s.loc[p.sg.to_numpy()].to_numpy()
        # longest first keeps padding small
        order = np.argsort([-(len(x) + len(z)) for x, z in zip(qt, st)], kind="stable")
        logit = np.empty(len(p), np.float32)
        t0 = time.time()
        with torch.no_grad():
            for i in range(0, len(p), a.score_batch):
                o = order[i:i + a.score_batch]
                f = tok(list(qt[o]), list(st[o]), truncation=True, max_length=L, padding=True,
                        return_tensors="pt").to(device)
                logit[o] = model(**f).logits.float().squeeze(-1).cpu().numpy()
                if (i // a.score_batch) % 200 == 0 and i:
                    el = time.time() - t0
                    log(f"  split {split}: {i:,}/{len(p):,} pairs, ~{el / i * (len(p) - i) / 60:.0f} min left")
        df = pd.DataFrame({"split": np.int8(split), "qg": p.qg.to_numpy(), "sg": p.sg.to_numpy(),
                           "logit": logit})
        df.to_parquet(part, index=False)
        log(f"split {split}: {len(p):,} pairs scored in {(time.time() - t0) / 60:.1f} min")
        outs.append(df)
    pd.concat(outs, ignore_index=True).to_parquet(os.path.join(a.out, "cross_scores.parquet"), index=False)
    log(f"wrote {os.path.join(a.out, 'cross_scores.parquet')}")


if __name__ == "__main__":
    main()
