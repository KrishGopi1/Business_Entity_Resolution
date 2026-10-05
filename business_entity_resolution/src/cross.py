"""
Optional cross-encoder on contested records (v4 architecture, sections 6 and 7.2, Phase 3).

A record is contested when the pair model is unsure of it: its best probability is above
cross_lo and either below cross_hi or less than cross_margin above its second S1. Only the record's top
three S1s are read by the cross-encoder, so the cost stays at a few million pairs instead of the
1B pairs of the whole candidate set.

--cross finetune
    1. fine-tune a transformer (default intfloat/multilingual-e5-small, MIT, used as a
       sequence-pair classifier) on pairs of TRAINING entities only: every true pair, and false
       pairs from the kept candidates at a 1:3 mix of hard (stage-2 rank <= 2) and ordinary
       ones. Input is 'name | address' of each side in the original script.
    2. score the top three S1s of every contested record with a tune or holdout S1
    3. stacker: a logistic regression of the true label on the pair model's logit, the
       cross-encoder's logit and the S1's rank, fitted on records whose S1s include no holdout
       entity (tune records); the holdout stays untouched for final_report.txt
    4. the stacked probabilities choose each contested record's S1 again; the calibration and
       the decision rule are then re-fitted on tune at the same density as before
predict: the same scoring, stacker and re-choice on contested test records before the rule.

The weights are saved in <v4>/cross/model and bundled with the code: nothing is looked up at
prediction time. Needs torch and transformers and, in practice, a GPU.
"""
import os
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pipeline import load_json, log
from pipeline4 import fp_ok, v4dir, write_fp

MAX_LEN = 96


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class Texts:
    """'name | address' by global id, read once per split."""

    def __init__(self, cfg, split):
        nd = os.path.join(cfg.work_dir, split, "norm")
        self.q = self._read(os.path.join(nd, "q_all.parquet"))
        self.s = self._read(os.path.join(nd, "s1_all.parquet"))

    @staticmethod
    def _read(path):
        t = pq.read_table(path, columns=["name_raw_clean", "address_raw_clean"])
        n = t["name_raw_clean"].to_numpy(zero_copy_only=False)
        a = t["address_raw_clean"].to_numpy(zero_copy_only=False)
        return np.array([(x or "") + (" | " + y if y else "") for x, y in zip(n, a)], dtype=object)


def contested(a, cfg):
    p1 = a["p1"].to_numpy()
    margin = p1 - a["second1"].to_numpy()
    return (p1 > cfg.cross_lo) & ((p1 < cfg.cross_hi) | (margin < cfg.cross_margin))


def _pairs_of(a, mask):
    """Long table of (row in a, rank, qg, sg, p) for the top three S1s of the masked records."""
    rows = np.flatnonzero(mask)
    out = []
    for r, (sc, pc) in enumerate((("sg1", "p1"), ("sg2", "p2"), ("sg3", "p3")), start=1):
        sg = a[sc].to_numpy()[rows]
        ok = sg >= 0
        out.append(pd.DataFrame({"row": rows[ok], "rank": r, "qg": a["qg"].to_numpy()[rows[ok]],
                                 "sg": sg[ok], "p": a[pc].to_numpy()[rows[ok]]}))
    return pd.concat(out, ignore_index=True)


def _logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _design(p, c, rank):
    lp = _logit(p)
    return np.column_stack([np.ones(len(p)), lp, c, lp * c, rank == 1, rank == 2])


def fit_stacker(p, c, rank, y, l2=1.0, iters=50):
    """Logistic regression by Newton steps (a handful of features)."""
    X = _design(p, c, rank)
    w = np.zeros(X.shape[1])
    w[1] = 1.0
    for _ in range(iters):
        z = X @ w
        mu = 1 / (1 + np.exp(-z))
        g = X.T @ (mu - y) + l2 * np.r_[0, w[1:]]
        H = (X * (mu * (1 - mu))[:, None]).T @ X + l2 * np.diag(np.r_[0, np.ones(len(w) - 1)])
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w.tolist()


def apply_stacker(w, p, c, rank):
    return (1 / (1 + np.exp(-_design(p, c, rank) @ np.asarray(w)))).astype(np.float32)


def rechoose(a, pairs, p_new):
    """Contested records take the S1 with the best stacked probability; p1 / second1 become the
    stacked values. Other records are unchanged."""
    a = a.copy()
    t = pairs.assign(pn=p_new).sort_values(["row", "pn"], ascending=[True, False], kind="stable")
    first = t.groupby("row", sort=False).head(1)
    second = t.groupby("row", sort=False).nth(1)
    rows = first["row"].to_numpy()
    a.loc[rows, "sg1"] = first["sg"].to_numpy()
    a.loc[rows, "p1"] = first["pn"].to_numpy()
    a.loc[rows, "second1"] = 0.0
    a.loc[second["row"].to_numpy(), "second1"] = second["pn"].to_numpy()
    a["crossed"] = False
    a.loc[rows, "crossed"] = True
    return a


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _load(path, device):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForSequenceClassification.from_pretrained(path, num_labels=1)
    return tok, m.to(device)


def score(tok, model, left, right, device, batch=512):
    import torch
    model.eval()
    out = np.empty(len(left), np.float32)
    with torch.no_grad():
        for i in range(0, len(left), batch):
            f = tok(list(left[i:i + batch]), list(right[i:i + batch]), truncation=True,
                    max_length=MAX_LEN, padding=True, return_tensors="pt").to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                out[i:i + batch] = model(**f).logits.float().squeeze(-1).cpu().numpy()
    return out


def training_pairs(cfg, rng):
    """(qg, sg, y) from the pair model's training rows (training entities only)."""
    import model4
    t = model4.read_outputs(cfg, "train", "feat", "train", ["qg", "sg", "y", "s2_rank"])
    y = t["y"].to_numpy()
    pos = np.flatnonzero(y == 1)
    n_pos = min(len(pos), cfg.cross_pairs // 2)
    pos = rng.choice(pos, n_pos, replace=False)
    neg = np.flatnonzero(y == 0)
    hard = neg[t["s2_rank"].to_numpy()[neg] <= 2]
    n_neg = cfg.cross_pairs - n_pos
    n_hard = min(len(hard), n_neg // 4)
    pick = np.r_[pos, rng.choice(hard, n_hard, replace=False),
                 rng.choice(neg, min(len(neg), n_neg - n_hard), replace=False)]
    rng.shuffle(pick)
    return t["qg"].to_numpy()[pick], t["sg"].to_numpy()[pick], y[pick].astype(np.float32)


def finetune(cfg, device, out_dir):
    import torch
    rng = np.random.default_rng(cfg.seed + 53)
    qg, sg, y = training_pairs(cfg, rng)
    log(f"cross: fine-tuning on {len(y):,} pairs of training entities ({y.mean():.0%} true)")
    tx = Texts(cfg, "train")
    tok, model = _load(cfg.cross_model, device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    B = cfg.cross_batch
    steps = max(1, cfg.cross_epochs * (len(y) // B))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / max(1, steps // 20)) * max(0.0, 1 - i / steps))
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    bce = torch.nn.BCEWithLogitsLoss()
    t0, step = time.time(), 0
    for _ in range(cfg.cross_epochs):
        order = rng.permutation(len(y))
        for i in range(0, len(order) - B + 1, B):
            b = order[i:i + B]
            f = tok(list(tx.q[qg[b]]), list(tx.s[sg[b]]), truncation=True, max_length=MAX_LEN,
                    padding=True, return_tensors="pt").to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = model(**f).logits.float().squeeze(-1)
                loss = bce(logits, torch.from_numpy(y[b]).to(device))
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % max(1, steps // 10) == 0:
                log(f"    step {step}/{steps}, loss {loss.item():.4f} ({(time.time() - t0) / 60:.1f} min)")
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return out_dir


def cross_dir(cfg):
    return v4dir(cfg, "cross")


def model_path(cfg):
    return os.path.join(cross_dir(cfg), "model")


def _eval_pairs(cfg):
    import model4
    a = model4.read_outputs(cfg, "train", "score", "asg")
    m = contested(a, cfg)
    return a, m, _pairs_of(a, m)


def _test_asg(cfg):
    import model4
    a = model4.read_outputs(cfg, "test", "pred", "asg")
    if a is None:
        raise SystemExit("no test predictions: run --stage predict first (with --cross off)")
    return a.reset_index(drop=True)


def _finish(cfg, a, m, pairs, c, model_name):
    """Stacker on tune records, re-choice, decision rule re-fitted."""
    import model4
    d = cross_dir(cfg)
    _, _, roles, _ = model4.eval_inputs(cfg)
    y = model4.label_pairs(cfg, pairs.qg.to_numpy(), pairs.sg.to_numpy())
    # tune records: no holdout S1 among the record's top three
    hold = pd.Series(roles[pairs.sg.to_numpy()] == 3).groupby(pairs.row.to_numpy()).transform("any").to_numpy()
    fit = ~hold
    w = fit_stacker(pairs.p.to_numpy()[fit], c[fit], pairs["rank"].to_numpy()[fit], y[fit])
    pn = apply_stacker(w, pairs.p.to_numpy(), c, pairs["rank"].to_numpy())
    ax = rechoose(a, pairs, pn)
    ax["y1"] = model4.label_pairs(cfg, ax.qg.to_numpy(), ax.sg1.to_numpy())
    ax.to_parquet(os.path.join(d, "train_asg.parquet"), index=False)
    info = {"model": model_name, "stacker": w, "contested_share": float(m.mean()),
            "pairs": int(len(pairs)), "fit_pairs": int(fit.sum()),
            "changed_records": int((ax.sg1 != a.sg1).sum())}
    write_fp(os.path.join(d, "_fp.json"), cfg, "cross", **info)
    log(f"cross: stacker weights {np.round(w, 3).tolist()}; best S1 changed for "
        f"{info['changed_records']:,} records")


def build(cfg):
    """Stage 'cross': fine-tune, score contested tune / holdout records, fit the stacker, re-fit
    the decision rule. Needs the train stage. With --cross kaggle: export the pairs for
    kaggle_cross.py, or import its scores (--cross-scores)."""
    import model4
    if not fp_ok(cfg, os.path.join(model4.model_dir(cfg), "model.json"), "train"):
        raise SystemExit("no pair model for the current settings: run --stage train first")
    if cfg.cross == "kaggle":
        return build_kaggle(cfg)
    d = cross_dir(cfg)
    fpp = os.path.join(d, "_fp.json")
    device = _device()
    if device != "cuda":
        from pipeline import meta
        n = meta(cfg, "test")["n_q"]
        if n > 200_000:
            raise SystemExit("cross: no GPU visible on this machine. Use --cross kaggle instead: it "
                             "exports the pairs for a Kaggle GPU (RUN_AWS.md section 3).")
        log("cross: no GPU visible; running on the CPU (small data only)")
    if not fp_ok(cfg, fpp, "cross"):
        mp = finetune(cfg, device, model_path(cfg)) if cfg.cross == "finetune" else cfg.cross_model
        a, m, pairs = _eval_pairs(cfg)
        tx = Texts(cfg, "train")
        tok, model = _load(mp, device)
        t0 = time.time()
        c = score(tok, model, tx.q[pairs.qg.to_numpy()], tx.s[pairs.sg.to_numpy()], device)
        log(f"cross: {m.mean():.1%} of evaluation records contested; {len(pairs):,} pairs scored in "
            f"{(time.time() - t0) / 60:.1f} min")
        _finish(cfg, a, m, pairs, c, "model" if cfg.cross == "finetune" else cfg.cross_model)
    model4.tune_decision(cfg)


# ---------------------------------------------------------------------------
# Kaggle split: export here, fine-tune and score on a Kaggle GPU, import the scores
# ---------------------------------------------------------------------------

def export_dir(cfg):
    return v4dir(cfg, "cross", "kaggle_export")


def export(cfg):
    """Pairs and texts for kaggle_cross.py: split 0 = fine-tuning pairs (training entities,
    labelled), 1 = contested tune / holdout pairs, 2 = contested test pairs."""
    out = export_dir(cfg)
    rng = np.random.default_rng(cfg.seed + 53)
    qg0, sg0, y0 = training_pairs(cfg, rng)
    _, m1, p1 = _eval_pairs(cfg)
    at = _test_asg(cfg)
    m2 = contested(at, cfg)
    p2 = _pairs_of(at, m2)
    pairs = pd.DataFrame({
        "split": np.r_[np.zeros(len(qg0)), np.ones(len(p1)), np.full(len(p2), 2)].astype(np.int8),
        "qg": np.r_[qg0, p1.qg.to_numpy(), p2.qg.to_numpy()].astype(np.int32),
        "sg": np.r_[sg0, p1.sg.to_numpy(), p2.sg.to_numpy()].astype(np.int32),
        "y": np.r_[y0, np.full(len(p1) + len(p2), -1)].astype(np.float32)})
    pairs.to_parquet(os.path.join(out, "pairs.parquet"), index=False)
    for split, sel in (("train", pairs.split < 2), ("test", pairs.split == 2)):
        tx = Texts(cfg, split)
        for side, arr in (("q", tx.q), ("s", tx.s)):
            g = np.unique(pairs[f"{side}g"].to_numpy()[sel.to_numpy()])
            pd.DataFrame({"gid": g.astype(np.int32), "text": arr[g]}).to_parquet(
                os.path.join(out, f"{side}_{split}.parquet"), index=False)
    bk_save = {"fp": fp4_cross(cfg), "model": cfg.cross_model, "epochs": cfg.cross_epochs,
               "batch": cfg.cross_batch, "max_len": MAX_LEN,
               "counts": {"finetune": int(len(qg0)), "eval": int(len(p1)), "test": int(len(p2))}}
    import json
    with open(os.path.join(out, "export.json"), "w") as f:
        json.dump(bk_save, f, indent=1)
    size = sum(os.path.getsize(os.path.join(out, x)) for x in os.listdir(out))
    log(f"cross: exported {len(qg0):,} fine-tuning, {len(p1):,} evaluation ({m1.mean():.1%} of records) "
        f"and {len(p2):,} test pairs ({m2.mean():.1%} of records) to {out} ({size / 1e6:.0f} MB).\n"
        "  Next: upload that folder to Kaggle as a dataset, run src/kaggle_cross.py on a GPU there,\n"
        "  download cross_scores.parquet and run --stage cross again with --cross-scores <file>.")


def fp4_cross(cfg):
    from pipeline4 import fp4
    return fp4(cfg, "cross")


def _read_scores(cfg):
    sc = pd.read_parquet(cfg.cross_scores)
    ex = load_json(os.path.join(export_dir(cfg), "export.json"))
    if ex["fp"] != fp4_cross(cfg):
        raise SystemExit("the Kaggle export was made with other settings or code: run --stage cross "
                         "without --cross-scores to export again")
    return sc


def _lookup(sc, split, qg, sg):
    t = sc[sc.split == split]
    key = pd.MultiIndex.from_arrays([t.qg.to_numpy(), t.sg.to_numpy()])
    idx = key.get_indexer(pd.MultiIndex.from_arrays([qg, sg]))
    if (idx < 0).any():
        raise SystemExit(f"{int((idx < 0).sum()):,} pairs have no Kaggle score: the scores do not "
                         "belong to this export")
    return t.logit.to_numpy()[idx].astype(np.float64)


def build_kaggle(cfg):
    import model4
    fpp = os.path.join(cross_dir(cfg), "_fp.json")
    if not cfg.cross_scores:
        export(cfg)
        return
    if not fp_ok(cfg, fpp, "cross"):
        sc = _read_scores(cfg)
        a, m, pairs = _eval_pairs(cfg)
        c = _lookup(sc, 1, pairs.qg.to_numpy(), pairs.sg.to_numpy())
        log(f"cross: {len(pairs):,} Kaggle scores read for {m.mean():.1%} of evaluation records")
        _finish(cfg, a, m, pairs, c, "kaggle")
    model4.tune_decision(cfg)


def active(cfg):
    return cfg.cross != "off"


def train_asg(cfg):
    """The assigned table after the cross-encoder, or None when it is off / not built."""
    if not active(cfg):
        return None
    fpp = os.path.join(cross_dir(cfg), "_fp.json")
    if not fp_ok(cfg, fpp, "cross"):
        raise SystemExit("--cross is on but the cross stage has not run for these settings: "
                         "run --stage cross first")
    return pd.read_parquet(os.path.join(cross_dir(cfg), "train_asg.parquet"))


def apply_test(cfg, a):
    """Contested test records: cross-encoder, stacker, re-choice."""
    info = load_json(os.path.join(cross_dir(cfg), "_fp.json"))
    m = contested(a, cfg)
    pairs = _pairs_of(a.reset_index(drop=True), m)
    if info["model"] == "kaggle":
        c = _lookup(_read_scores(cfg), 2, pairs.qg.to_numpy(), pairs.sg.to_numpy())
        log(f"cross: {len(pairs):,} Kaggle scores read for {m.mean():.1%} of test records")
        pn = apply_stacker(info["stacker"], pairs.p.to_numpy(), c, pairs["rank"].to_numpy())
        return rechoose(a.reset_index(drop=True), pairs, pn)
    mp = model_path(cfg) if info["model"] == "model" else info["model"]
    device = _device()
    tx = Texts(cfg, "test")
    tok, model = _load(mp, device)
    t0 = time.time()
    c = score(tok, model, tx.q[pairs.qg.to_numpy()], tx.s[pairs.sg.to_numpy()], device)
    log(f"cross: {m.mean():.1%} of test records contested; {len(pairs):,} pairs scored in "
        f"{(time.time() - t0) / 60:.1f} min")
    pn = apply_stacker(info["stacker"], pairs.p.to_numpy(), c, pairs["rank"].to_numpy())
    return rechoose(a.reset_index(drop=True), pairs, pn)
