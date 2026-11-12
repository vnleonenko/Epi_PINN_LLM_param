# R3.7 high-budget protocol

Written before the rerun. This file freezes the computation limits. It does not change the scientific model.

## What stays the same as R3.7

- Data: `data/covid-19_Kouprianov.csv`, sha256 `bde98003e5fd10331bcd0adf2a2f7c8ccb75398b34296de47355b54b060f3aab`.
- Window: days 0 through 368 (369 daily points). Day 0 is 5 July 2020.
- State: observed `(S, I, R, D)` at t = 0. `N = 6,000,000`. No R or D offset.
- Objective: `RSS = Σ (I_sim − I_obs)^2` on those 369 days. `S`, `R`, and `D` are not in the objective.
- Solver: segment-wise `solve_ivp` RK45, `rtol = 1e-6`, `atol = 1e-8`.
- Bounds: each `β ∈ [0.01, 0.99]`, `γ ∈ [0.001, 0.5]`, `μ ∈ [0.0001, 0.1]`.
- Breakpoint counts: 0, 2, and 3 (1, 3, and 4 segments). Minimum segment length 14 days. Softmax length parameterization unchanged. `z ∈ [−6, 6]`.
- Optimizer: L-BFGS-B. `ftol = 1e-12`, `gtol = 1e-8`.
- The original 16 starts, seeds 0..15, Generator PCG64(`20261108 + seed`), including start 0 at the published Phase-1 point. These 16 are the matched comparison.

## What changes: budget only

Chosen before this run, because the previous piecewise points stopped at `maxfun = 500` without an L-BFGS convergence message:

- `maxiter = 5000` (was 200)
- `maxfun = 20000` (was 500)

A `success` flag is a local termination code. It is not a claim of a global optimum.

## Extra runs, labelled and not mixed into the 16-start table

1. `continuation_previous_best`: L-BFGS-B started at the best point already stored in the original `out_r37` for that model, with this larger budget. The L-BFGS history is reset. This is not one of the 16 fresh starts.
2. `nested_bp2_embed_eval`: the best fresh 2-breakpoint solution of this run is embedded in the 3-breakpoint parameterization by splitting its longest segment into two equal pieces, both at least 14 days, and copying that segment’s β onto both pieces. The RSS is evaluated and not optimized. It checks that the 3-breakpoint model can represent that 2-breakpoint curve.
3. `nested_bp2_embed_optimized`: the same embedding is then optimized with the same budget. Its result is reported separately from the best of the 16 fresh 3-breakpoint starts.

The comparison figure uses only the best of the 16 fresh starts for each model, plus the published 120-day Phase-1 curve as context. That published curve is not the matched refit.

## Metrics and checks

Unchanged: `g_early` on increments t = 0..109, `g_accel` on t = 110..149, `I(150)−I(110)`, full-window RSS/RMSE/MAE and the I peak. Checks: positivity, conservation of N, one-day state continuity, constant-β nested limit, and re-integration at `rtol = 1e-8`, `atol = 1e-10`.

No confidence intervals and no significance tests. The run does not alter the manuscript or the original R3.7 output directory.
