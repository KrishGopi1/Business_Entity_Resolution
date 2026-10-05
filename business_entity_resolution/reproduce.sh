#!/usr/bin/env bash
# Regenerates matching_results.tsv and candidate_pairs.tsv from the provided data, from scratch,
# with the settings of the submitted run (v4 Run A + cross-encoder).
#
#   bash reproduce.sh DATA_DIR [WORK_DIR] [OUT_DIR]
#
# DATA_DIR holds train/ and test/ as provided. WORK_DIR (default ./work) must be new or a
# folder made by this code. Every stage is cached and resumes after an interruption.
#
# CROSS (default: finetune) chooses how the cross-encoder runs:
#   finetune  on this machine; needs a CUDA GPU, requirements-gpu.txt, and the public
#             intfloat/multilingual-e5-small weights (downloaded once by transformers, or set
#             CROSS_MODEL to a local copy of them)
#   kaggle    on a CPU-only machine: stops after exporting the pairs; run src/kaggle_cross.py
#             on a GPU, then rerun with CROSS_SCORES=<its cross_scores.parquet>
#   off       pair model only (the 0.975 variant)
set -euo pipefail
DATA=$(realpath -m "${1:?usage: bash reproduce.sh DATA_DIR [WORK_DIR] [OUT_DIR]}")
WORK=$(realpath -m "${2:-work}")
OUT=$(realpath -m "${3:-output}")
CROSS=${CROSS:-finetune}
cd "$(dirname "$0")"

# --s2-keep-noaddr 0: the submitted run predates this optional setting (see README)
X=(--data "$DATA" --work "$WORK" --s2-keep-noaddr 0)
if [ -n "${CROSS_MODEL:-}" ]; then X+=(--cross-model "$CROSS_MODEL"); fi
if [ -n "${EXTRA_ARGS:-}" ]; then read -ra E <<< "$EXTRA_ARGS"; X+=("${E[@]}"); fi   # e.g. --jobs 6

if [ "$CROSS" = "kaggle" ] && [ -z "${CROSS_SCORES:-}" ]; then
  # prepare, block, every v4 stage, and the plain prediction; then export for the GPU
  python src/run.py "${X[@]}" --out "$OUT"
  python src/run.py "${X[@]}" --cross kaggle --stage cross
  echo "Now run src/kaggle_cross.py on a GPU (RUN_AWS.md section 3), then:"
  echo "  CROSS=kaggle CROSS_SCORES=/path/cross_scores.parquet bash reproduce.sh $DATA $WORK $OUT"
  exit 0
fi
if [ "$CROSS" = "kaggle" ]; then
  X+=(--cross kaggle --cross-scores "$CROSS_SCORES")
  for s in cross evaluate predict; do python src/run.py "${X[@]}" --stage $s --out "$OUT"; done
else
  # one pass: prepare, block, context4, stage2, cut, report4, train, cross, evaluate, predict
  python src/run.py "${X[@]}" --cross "$CROSS" --out "$OUT"
fi
echo "submission files: $OUT/matching_results.tsv $OUT/candidate_pairs.tsv"
