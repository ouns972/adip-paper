#!/usr/bin/env bash
# Gap / improvement sweep for one (n, warm-solver) pair.
# Writes under runs/ so paper files in results/ are never overwritten.
# Usage: scripts/repro_gap_sweep.sh <n> <solver> [num_runs] [base_seed_start]
#   n      : 400 | 1000 | 5000 | 10000
#   solver : scip | highs | cbc | cpsat
set -euo pipefail
cd "$(dirname "$0")/.."

N="${1:?n required}"
SOLVER="${2:?warm-solver required}"
NUM_RUNS="${3:-100}"
SEED_START="${4:-40}"

case "$N" in
  400)
    PHASE2_LR=""
    LAMBDA=100
    ;;
  1000|5000|10000)
    PHASE2_LR="--phase2-lr 0.0001"
    LAMBDA=500
    ;;
  *)
    echo "Unsupported n=$N (expected 400, 1000, 5000, or 10000)" >&2
    exit 2
    ;;
esac

OUT_DIR="runs/gap_${SOLVER}_n${N}_s${SEED_START}_k${NUM_RUNS}"
mkdir -p "$OUT_DIR"
# shellcheck disable=SC2086
python -m adip.benchmark_gap_sweep \
  --warm-solver "$SOLVER" \
  --base-seed-start "$SEED_START" \
  --num-runs "$NUM_RUNS" \
  --n "$N" \
  --exp-id 0 \
  --q-kind indefinite \
  --device cpu \
  --scip-warm-limit 500 \
  --scip-warm-start-seed-shift 424267 \
  --scip-param timing/clocktype=wall \
  --ar-lr 0.001 \
  $PHASE2_LR \
  --ar-d-model 64 --ar-nhead 4 --ar-nlayers 2 \
  --ar-sparse-attn-prev-fraction 0.1 --ar-sparse-attn-pattern band \
  --phase1-ce-target-prob 0.999 --phase1-ce-max-steps 150 \
  --phase1-ce-plateau-patience 12 --phase1-ce-plateau-min-delta 1e-5 \
  --phase2-w-soft 1.0 --phase2-tau 0.5 \
  --phase2-lambda "$LAMBDA" --phase2-q-weight 1.0 \
  --phase2-max-steps 2000 --phase2-plateau-patience 45 --phase2-plateau-min-delta 1e-5 \
  --phase2-time-limit 500 --gumbel-ste \
  --output-dir "$OUT_DIR"
