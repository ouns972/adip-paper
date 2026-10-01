#!/usr/bin/env bash
# Anytime objective plot (matched-time SCIP vs SCIP+ADIP; CE excluded).
# Regenerations write under runs/; paper files in results/ are left untouched.
#
# Fast replot from shipped data:
#   scripts/repro_obj_time.sh
#
# Full regenerate (re-solves matched-time SCIP; needs cont_relax JSON with obj_time):
#   scripts/repro_obj_time.sh --full
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/obj_time

if [[ "${1:-}" == "--full" ]]; then
  python -m adip.benchmark_obj_time \
    --adip-json results/cont_relax_scip_n400_k50_seed102.json \
    --scip-time-limit 1000 --warm-budget 500 \
    --out-png runs/obj_time/obj_vs_time.png \
    --out-json runs/obj_time/obj_vs_time.json \
    --y-break=-20,60,1500,1800 \
    --xlim 0,1000 \
    --invlog-remaining-x
else
  python -m adip.benchmark_obj_time \
    --from-json results/obj_vs_time_seed102.json \
    --out-png runs/obj_time/obj_vs_time.png \
    --y-break=-20,60,1500,1800 \
    --xlim 0,1000 \
    --invlog-remaining-x
fi

echo "Wrote runs/obj_time/ (paper figure remains results/obj_vs_time_seed102.png)."
