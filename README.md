# ADIP — Autoregressive Differentiable Method for Integer Programming

Code and reproduction scripts for the paper
*Autoregressive Differentiable Method for Integer Programming*,
accompanying the arXiv release.

ADIP solves the Quadratic Knapsack Problem (QKP) with a three-stage pipeline:

1. **MIP warm-start** — obtain a feasible incumbent from SCIP or an open-source MIP/CP solver  
2. **Teacher-forcing CE** — fit an autoregressive bit model to that incumbent  
3. **Differentiable phase-2** — refine with a Gumbel–Softmax / STE relaxation and a knapsack Lagrangian  

The reported ADIP solution is the rounded soft \(\tilde x\) at the best phase-2 loss.

| Warm solver | Backend | QKP model |
|-------------|---------|-----------|
| `scip` | SCIP (`pyscipopt`) | native quadratic epigraph |
| `highs` | HiGHS | Fortet linearization |
| `cbc` | COIN-OR CBC (OR-Tools) | Fortet MILP |
| `cpsat` | OR-Tools CP-SAT | Fortet-equivalent boolean products + knapsack |

Default instances use **indefinite** \(Q\) (entries in \([-5,5]\)), so HiGHS / CBC / CP-SAT see the linearized binary program. Warm solvers share the same wall budget (`--scip-warm-limit`, default 500s); `--warm-threads` defaults to 1.

## Setup

```bash
poetry install
```

Requires a working SCIP installation (`pyscipopt`). Poetry also installs HiGHS (`highspy`) and OR-Tools (CBC + CP-SAT).

## Paper figures

Shipped artifacts under `results/`:

| File | Figure |
|------|--------|
| `gap_improvement_sweep_scip_n400_s40_k100.{json,png}` | Gap vs % improvement (SCIP, \(n=400\), 100 seeds) |
| `traj_tsne.png` | t-SNE of phase-2 trajectories (\(n=400\); red = knapsack-boundary endpoints) |
| `obj_vs_time_seed102.{json,png}` | Anytime objective; ADIP source run in `cont_relax_scip_n400_k50_seed102.json` |

Checksums of the Python package sources used for these figures are in `results/SOURCE_SHA256.txt`.

## Reproduction recipes

Shared ADIP knobs (all sizes): `--ar-lr 0.001`, Transformer `64/4/2`, band sparse attention, phase-1 CE 150 steps, phase-2 Gumbel–STE, warm + phase-2 budgets 500s each.

| \(n\) | Phase-2 LR | \(\lambda\) (`--phase2-lambda`) |
|------|------------|----------------------------------|
| 400 | same as `--ar-lr` (omit `--phase2-lr`) | 100 |
| 1000, 5000, 10000 | `--phase2-lr 0.0001` | 500 |

### Gap / improvement sweeps

100 seeds × warm solvers × problem sizes:

```bash
# n=400
for s in scip highs cbc cpsat; do
  scripts/repro_gap_sweep.sh 400 "$s" 100 40
done

# n=1000 / 5000 / 10000  (phase2-lr=1e-4, λ=500)
for n in 1000 5000 10000; do
  for s in scip highs cbc cpsat; do
    scripts/repro_gap_sweep.sh "$n" "$s" 100 40
  done
done
```

Equivalent SCIP \(n=400\) command (paper gap figure):

```bash
python -m adip.benchmark_gap_sweep \
  --warm-solver scip --base-seed-start 40 --num-runs 100 --n 400 --exp-id 0 \
  --q-kind indefinite --device cpu --scip-warm-limit 500 \
  --scip-warm-start-seed-shift 424267 --scip-param timing/clocktype=wall \
  --ar-lr 0.001 --ar-d-model 64 --ar-nhead 4 --ar-nlayers 2 \
  --ar-sparse-attn-prev-fraction 0.1 --ar-sparse-attn-pattern band \
  --phase1-ce-target-prob 0.999 --phase1-ce-max-steps 150 \
  --phase1-ce-plateau-patience 12 --phase1-ce-plateau-min-delta 1e-5 \
  --phase2-w-soft 1.0 --phase2-tau 0.5 --phase2-lambda 100 --phase2-q-weight 1.0 \
  --phase2-max-steps 2000 --phase2-plateau-patience 45 --phase2-plateau-min-delta 1e-5 \
  --phase2-time-limit 500 --gumbel-ste
```

For \(n\in\{1000,5000,10000\}\) add `--phase2-lr 0.0001` and set `--phase2-lambda 500`.

### Trajectory t-SNE

Records a phase-2 \(\tilde x\) trace, fits 3D t-SNE, highlights knapsack-boundary endpoints in red, and prints `basin_analysis`:

```bash
scripts/repro_traj_tsne.sh 400 57
```

### Anytime objective plot

Broken \(y\)-axis (keep 0 + zoom on late improvement) and late-time expansion via \(-\log(T-t)\):

```bash
# Replot from shipped JSON (no solver required)
scripts/repro_obj_time.sh

# Full regenerate: rematch SCIP for 1000s using the shipped ADIP cont_relax JSON
scripts/repro_obj_time.sh --full
```

## Package layout

| Module | Role |
|--------|------|
| `ar_model` | Causal autoregressive Transformer over bits |
| `phase1_ce` / `teacher_ce` | Teacher-forcing CE on the warm-start incumbent |
| `phase2_train` / `soft_lagrangian` | Soft objective + knapsack penalty |
| `pipeline` | End-to-end MIP warm-start → train |
| `warm_solvers` / `qkp_linearize` | HiGHS, CBC, CP-SAT (Fortet) |
| `benchmark_*` | Figure / table reproduction entrypoints |
| `trajectory_tsne` | t-SNE plot + `basin_analysis` |
| `obj_time_plot` | Anytime objective curves |

## Citation

If you use this code, please cite the accompanying paper (arXiv link to be added upon release).
