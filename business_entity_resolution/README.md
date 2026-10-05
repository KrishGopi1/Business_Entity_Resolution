# Business Entity Resolution: pipeline v4

v4 implements `V4_ARCHITECTURE.md` on top of v3. It does **not** re-block: it reads v3's cached
`work/` folder (normalization, retrieval contexts, the stage-1 ranker and the stored candidates,
up to 100 per record) and adds everything else incrementally.

**Status (2026-09-26): every stage runs end to end on generated data (`tests/synth.py`).
Nothing has been measured on the real data yet.** The bi-encoder and cross-encoder paths were
tested only with tiny random models built offline, because model downloads are blocked in the
build environment. Real-data numbers come from your first run.

```
v3 work/ (cached, reused)            v4 (new, in work/v4/)
 prepare  -> norm, roles, gt          context4  views + matrices: legal-free name, loose
 block    -> ctx, ranker, cand                  romanization, despaced name, core address,
            (<=100 per record)                  Indic-script state, numbers
                                      dense     optional bi-encoder: top-20 per record + cosine
                                      stage2    add new sources to the stored candidates, fit
                                                the stage-2 ranker (XGBoost rank:pairwise)
                                      cut       top 5 + floor, <= 25 per record, address rescue
                                      report4   recall / ceiling by layer and source, v4 gates
                                      train     pair features, 3-fold XGBoost, rule at test density
                                      cross     optional cross-encoder on contested records
                                      evaluate  holdout report, v4 score gates
                                      predict   test: expand -> cut -> features -> score, fused
```

```
src/
  (v3, unchanged) schema.py normalize.py indexing.py blocking.py features.py decision.py
                  model.py evaluate.py
  pipeline.py   v3 stages + v4 config; refuses to re-block when the v3 cache does not match
  pipeline4.py  v4 stages, fingerprints chained on v3's, cache check, stage-2 fit, cut
  views4.py     legal-free name, loose romanization, despaced name, core address, Indic state
  context4.py   v4 TF-IDF channels, token matrices and exact keys per partition
  expand4.py    new sources, incremental union, stage-1 rescoring, stage-2 features and cut
  stage2.py     stage-2 ranker: fit, keep curve, floor choice
  features4.py  pair-model features (v3.1's + v4's)
  model4.py     pair model, calibration and rule at test density, fused predict
  dense.py      optional bi-encoder (sentence-transformers)
  cross.py      optional cross-encoder (transformers) and stacker; Kaggle export / import
  kaggle_cross.py  standalone cross-encoder fine-tune + scoring for a Kaggle GPU notebook
  evaluate4.py  cut_report.txt, final_report.txt, test diagnostics
tests/          pytest (35 tests, ~75 s, includes a v4 end-to-end run on generated data)
```

Defaults fit an 8-vCPU machine without a GPU (tested target: AWS r8i.2xlarge): workers =
CPU quota - 1, XGBoost on the CPU when no GPU is visible, bi-encoder off, cross-encoder off.
The cross-encoder runs on a Kaggle GPU with `--cross kaggle` (RUN_AWS.md section 3); `--dense`
and `--cross finetune` stop with an explanation when no GPU is visible and the data is large.

## Setup

```
pip install -r requirements.txt
pip install -r requirements-gpu.txt      # only on a GPU machine (--dense / --cross finetune)
python -m pytest -q tests                # 35 passed
```

## Regenerating the submission from scratch

Everything needed is in this folder: the code reads only the provided `train/` and `test/`
files (no external data, no lookups). One command rebuilds every stage, from reading the TSVs to
the two output files:

```
bash reproduce.sh DATA_DIR work output          # GPU machine: cross-encoder fine-tuned here
CROSS=kaggle bash reproduce.sh DATA_DIR work output   # CPU-only machine: cross-encoder on a Kaggle GPU
CROSS=off bash reproduce.sh DATA_DIR work output      # pair model only, no GPU
```

`reproduce.sh` runs `src/run.py` with the settings of the submitted run: prepare (normalization),
block (stage-1 retrieval and ranker), context4, stage2, cut, report4, train, cross, evaluate,
predict. With a new `work` folder nothing is reused; every stage writes its result there, so an
interrupted run resumes where it stopped.

- The cross-encoder starts from the public `intfloat/multilingual-e5-small` weights (MIT), which
  `transformers` downloads once. For a machine without internet, put a copy of them in a
  folder and set `CROSS_MODEL=/that/folder`. It is then fine-tuned on training pairs only.
- `--s2-keep-noaddr 0` in `reproduce.sh`: this package adds an optional rule that keeps the best
  25 candidates of records without an address; the submitted run predates it, so it is off.
- Time on an 8-vCPU, 64 GB machine (CPU): about 6.5 h for prepare and block, then about 4.7 h for
  the v4 stages; the cross-encoder takes about 1-1.5 h on a T4 [est]. Fixed seeds; GPU
  fine-tuning can differ in the last digits between runs.

## Running

`--data` must be the **same absolute path** you used for v3: v3's fingerprints include it.
`--work` points at v3's work folder. The first v4 stage checks the cache and stops at once with
an explanation if it does not match, instead of re-blocking for 6 hours.

```
python src/run.py --data D --work W                 # every stage in order (v3 ones are skipped)
python src/run.py --data D --work W --stage cut     # or one stage at a time
python src/run.py --data D --work W --zip TEAMNAME  # submission zip (code, outputs, weights)
```

Outputs: `W/v4/reports/cut_report.txt`, `final_report.txt`, `test_diagnostics.txt`,
`audit_france.tsv`, and the two submission files in `--out` (default `output/`).

| option | default | meaning |
|---|---|---|
| `--s2-keep-min / --s2-keep-max` | 5 / 25 | stage-2 cut per record |
| `--s2-floor` | -1 (auto) | stage-2 score floor; auto keeps `--s2-target` 0.998 of held-out true pairs |
| `--big-cap / --big-top` | 5000 / 20 | ranked big-key blocks: keys shared by up to 5,000 S1s, best 20 by address |
| `--no-bigkey --no-src-desp --no-src-loose --no-src-nlchan --no-src-dschan` | on | ablate a new source |
| `--dense off\|pretrained\|finetune` | off | bi-encoder channel and cosine feature |
| `--cross off\|finetune\|kaggle` | off | cross-encoder on contested records; `kaggle` exports the pairs for `src/kaggle_cross.py` on a Kaggle GPU and imports its `--cross-scores` file (RUN_AWS.md section 3) |
| `--model-s1` | 300000 | S1 entities whose pairs train the pair model |
| `--tune-density` | stress | calibration and rule fitted at test density |
| `--v4-dir` | W/v4 | keep two v4 runs side by side on one v3 cache |
| `--trust-v3-cache` | off | use v3's files even if their fingerprints differ (e.g. moved data folder) |
| `--strict-gates` | off | refuse to train when the v4 candidate gate fails |

Every v4 stage is cached with a fingerprint of its settings and code, chained on v3's block
fingerprint; `cut` and `predict` resume from the last finished chunk.
