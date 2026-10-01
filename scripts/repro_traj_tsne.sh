#!/usr/bin/env bash
# Trajectory t-SNE + distinct knapsack-boundary endpoints (basin_analysis).
# Writes under runs/ so paper files in results/ are never overwritten.
# Usage: scripts/repro_traj_tsne.sh [n] [base_seed]
# Default seed 57: large ADIP improvement in the shipped n=400 SCIP gap sweep.
set -euo pipefail
cd "$(dirname "$0")/.."

N="${1:-400}"
SEED="${2:-57}"

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
    echo "Unsupported n=$N" >&2
    exit 2
    ;;
esac

OUT_DIR="runs/traj_n${N}_s${SEED}"
OUT_PNG="${OUT_DIR}/traj_tsne.png"
mkdir -p "$OUT_DIR"
# shellcheck disable=SC2086
python -m adip.benchmark_continuous_relax \
  --warm-solver scip \
  --base-seed "$SEED" --n "$N" --exp-id 0 \
  --q-kind indefinite --device cpu --no-progress \
  --scip-warm-limit 500 --phase2-time-limit 500 \
  --scip-warm-start-seed-shift 424267 --scip-param timing/clocktype=wall \
  --ar-lr 0.001 $PHASE2_LR \
  --ar-d-model 64 --ar-nhead 4 --ar-nlayers 2 \
  --ar-sparse-attn-prev-fraction 0.1 --ar-sparse-attn-pattern band \
  --phase1-ce-target-prob 0.999 --phase1-ce-max-steps 150 \
  --phase1-ce-plateau-patience 12 --phase1-ce-plateau-min-delta 1e-5 \
  --phase2-w-soft 1.0 --phase2-tau 0.5 \
  --phase2-lambda "$LAMBDA" --phase2-q-weight 1.0 \
  --phase2-max-steps 2000 --phase2-plateau-patience 45 --phase2-plateau-min-delta 1e-5 \
  --gumbel-ste \
  --plot-trajectories "$OUT_PNG" \
  --tsne-phase2-stride 5 --tsne-perplexity 30 --tsne-seed 0 \
  --tsne-max-rows 500 --tsne-max-phase2 220 --tsne-max-rays 96 \
  --endpoint-l2-threshold 1e-3 \
  --output-dir "$OUT_DIR"

echo "Wrote $OUT_PNG (paper figure remains results/traj_tsne.png)."
