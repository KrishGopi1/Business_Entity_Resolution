"""
Optional bi-encoder (v4 architecture, section 6): a multilingual sentence encoder that reads
'name | address' in the ORIGINAL script (no transliteration; the model learns the mapping), used
two ways:
    * a retrieval source: each record's top dense_k S1s of its partition by cosine ('dense'
      bit in expand4), aimed at non-Latin names and names that share no token with the S1;
    * a feature on every candidate pair: cos_dense, with its rank and gap inside the record.

--dense pretrained   the model as downloaded (default intfloat/multilingual-e5-small, MIT, 118M)
--dense finetune     first fine-tuned on true pairs of training entities with v3's top-ranked
                     false candidate of the same record as a hard negative (in-batch negatives
                     plus the hard one, contrastive loss), dense_pairs pairs, dense_epochs epochs

Needs torch and sentence-transformers (Apache-2.0) and, in practice, a GPU: encoding ~25M texts
and the exact top-k (a matrix product per partition) are GPU work. The model weights are saved
in <v4>/dense/model and bundled with the code, so nothing is looked up at prediction time.
"""
import os
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pipeline import load_json, log, meta
from pipeline4 import excluded_records, fp_ok, v4dir, write_fp

PREFIX = "query: "          # e5 models expect a prefix; the same one on both sides (symmetric)
MAX_LEN = 64


def _texts(norm_path, rows=None):
    t = pq.read_table(norm_path, columns=["name_raw_clean", "address_raw_clean"])
    n = t["name_raw_clean"].to_numpy(zero_copy_only=False)
    a = t["address_raw_clean"].to_numpy(zero_copy_only=False)
    if rows is not None:
        n, a = n[rows], a[rows]
    return [PREFIX + (x or "") + (" | " + y if y else "") for x, y in zip(n, a)]


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load(path_or_name, device):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(path_or_name, device=device)
    m.max_seq_length = MAX_LEN
    return m


def encode(model, texts, out_path, batch, block=200_000):
    """Normalized float16 embeddings written to a .npy file block by block."""
    dim = (model.get_embedding_dimension() if hasattr(model, "get_embedding_dimension")
           else model.get_sentence_embedding_dimension())
    E = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=(len(texts), dim))
    for i in range(0, len(texts), block):
        v = model.encode(texts[i:i + block], batch_size=batch, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=False)
        E[i:i + len(v)] = v.astype(np.float16)
    E.flush()
    del E
    return np.load(out_path, mmap_mode="r")


def topk(Eq, Es, k, device, qb=8192):
    """Exact top-k S1 per record by cosine: CSR layout (q_ptr, s)."""
    import torch
    k = min(k, len(Es))
    n_q = len(Eq)
    s_out = np.empty((n_q, k), np.int32)
    if k == 0 or n_q == 0:
        return np.zeros(n_q + 1, np.int64), np.empty(0, np.int32)
    dt = torch.float16 if device == "cuda" else torch.float32
    S = torch.from_numpy(np.asarray(Es, np.float32)).to(device=device, dtype=dt)
    qb = qb if device == "cuda" else 1024
    with torch.no_grad():
        for i in range(0, n_q, qb):
            Q = torch.from_numpy(np.asarray(Eq[i:i + qb], np.float32)).to(device=device, dtype=dt)
            idx = torch.topk(Q @ S.T, k, dim=1).indices
            s_out[i:i + len(Q)] = idx.cpu().numpy().astype(np.int32)
    return np.arange(0, n_q * k + 1, k, dtype=np.int64), s_out.ravel()


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

def training_triplets(cfg, rng):
    """(record gid, true S1 gid, hard negative S1 gid or -1) for true pairs of non-tune/holdout
    entities, from records that train nothing else."""
    import expand4 as ex
    w = os.path.join(cfg.work_dir, "train")
    gt = np.load(os.path.join(w, "gt.npz"))
    roles = np.load(os.path.join(w, "roles.npy"))
    excl = np.load(excluded_records(cfg))
    ok = (roles[gt["s"]] <= 1) & ~excl[gt["q"]]
    s, q = gt["s"][ok].astype(np.int64), gt["q"][ok].astype(np.int64)
    if len(q) > cfg.dense_pairs:
        pick = rng.choice(len(q), cfg.dense_pairs, replace=False)
        s, q = s[pick], q[pick]
    n_q = len(excl)
    truth = np.sort(gt["s"].astype(np.int64) * n_q + gt["q"])
    want = np.zeros(n_q, bool)
    want[q] = True
    hard = np.full(n_q, -1, np.int64)
    got = []
    for part in meta(cfg, "train")["partitions"]:
        cand = os.path.join(cfg.work_dir, "train", "cand", part["name"])
        for pairs, _ in ex.chunks_of(cand):
            if not os.path.exists(pairs):
                continue
            t = pq.read_table(pairs, columns=["qg", "sg", "rank_pos"],
                              filters=[("rank_pos", "<=", 3)])
            qg, sg, rp = (t[c].to_numpy() for c in ("qg", "sg", "rank_pos"))
            m = want[qg]
            qg, sg, rp = qg[m].astype(np.int64), sg[m].astype(np.int64), rp[m]
            neg = ~np.isin(sg * n_q + qg, truth)
            got.append((qg[neg], sg[neg], rp[neg]))
    if got:
        qg, sg, rp = (np.concatenate([g[i] for g in got]) for i in range(3))
        o = np.lexsort((rp, qg))
        qg, sg = qg[o], sg[o]
        first = np.r_[True, qg[1:] != qg[:-1]] if len(qg) else np.empty(0, bool)
        hard[qg[first]] = sg[first]
    return q, s, hard[q]


def finetune(cfg, model, device, out_dir):
    import torch
    rng = np.random.default_rng(cfg.seed + 41)
    q, s, h = training_triplets(cfg, rng)
    log(f"dense: fine-tuning on {len(q):,} true pairs ({(h >= 0).mean():.0%} with a hard negative)")
    nd = os.path.join(cfg.work_dir, "train", "norm")
    need_q = np.unique(q)
    need_s = np.unique(np.r_[s, h[h >= 0]])
    tq = dict(zip(need_q.tolist(), _texts(os.path.join(nd, "q_all.parquet"), need_q)))
    ts = dict(zip(need_s.tolist(), _texts(os.path.join(nd, "s1_all.parquet"), need_s)))
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    B = cfg.dense_batch
    steps = cfg.dense_epochs * (len(q) // B)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / max(1, steps // 20)) * max(0.0, 1 - i / max(steps, 1)))
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    ce = torch.nn.CrossEntropyLoss()

    def emb(texts):
        f = model.tokenize(texts)
        f = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in f.items()}
        return torch.nn.functional.normalize(model(f)["sentence_embedding"], dim=-1)

    t0, step = time.time(), 0
    for ep in range(cfg.dense_epochs):
        order = rng.permutation(len(q))
        for i in range(0, len(order) - B + 1, B):
            b = order[i:i + B]
            anchors = [tq[x] for x in q[b]]
            pos = [ts[x] for x in s[b]]
            hb = h[b]
            neg = [ts[x] for x in hb[hb >= 0]]
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                a = emb(anchors)
                c = emb(pos + neg)
                logits = a @ c.T * 20.0
                loss = ce(logits.float(), torch.arange(len(b), device=device))
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % max(1, steps // 10) == 0:
                log(f"    step {step}/{steps}, loss {loss.item():.4f} ({(time.time() - t0) / 60:.1f} min)")
    model.eval()
    model.save(out_dir)
    return out_dir


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------

def build(cfg):
    d = v4dir(cfg, "dense")
    fpp = os.path.join(d, "_fp.json")
    if fp_ok(cfg, fpp, "dense"):
        log("dense: cached")
        return
    device = _device()
    if device != "cuda":
        n = meta(cfg, "train")["n_q"] + meta(cfg, "test")["n_q"]
        if n > 200_000:
            raise SystemExit("dense: no GPU visible, and encoding %s records on the CPU would take many "
                             "hours. Run without --dense on this machine (the default)." % f"{n:,}")
        log("dense: no GPU visible; running on the CPU (small data only)")
    model = _load(cfg.dense_model, device)
    if cfg.dense == "finetune":
        path = finetune(cfg, model, device, os.path.join(d, "model"))
        model = _load(path, device)
    model.eval()
    for split in ("train", "test"):
        nd = os.path.join(cfg.work_dir, split, "norm")
        for part in meta(cfg, split)["partitions"]:
            out = v4dir(cfg, split, "dense", part["name"])
            t0 = time.time()
            Es = encode(model, _texts(os.path.join(nd, f"s1_{part['name']}.parquet")),
                        os.path.join(out, "emb_s.npy"), cfg.dense_batch)
            Eq = encode(model, _texts(os.path.join(nd, f"q_{part['name']}.parquet")),
                        os.path.join(out, "emb_q.npy"), cfg.dense_batch)
            t1 = time.time()
            q_ptr, s = topk(Eq, Es, cfg.dense_k, device)
            np.save(os.path.join(out, "q_ptr.npy"), q_ptr)
            np.save(os.path.join(out, "s.npy"), s)
            log(f"dense {split}/{part['name']}: {len(Es):,} S1 + {len(Eq):,} records encoded in "
                f"{(t1 - t0) / 60:.1f} min, top-{cfg.dense_k} in {(time.time() - t1) / 60:.1f} min")
    write_fp(fpp, cfg, "dense", device=device)
