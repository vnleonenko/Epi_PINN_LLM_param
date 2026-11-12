# Frozen protocol: constrained optimizer vs Parameter Generator (R3.2 / R1.2)

Do not change this protocol after seeing results. Do not edit the manuscript, existing figures, or published tables. Implement the experiment in an isolated directory such as `experiments/r3_constrained_optimizer/`. A branch that contains only this experiment's code and outputs is acceptable; do not merge to the default branch.

This is another Phase 2 configuration: Intent Parser + optimizer instead of the Generator. The optimizer receives the parsed intent, minimizes the relative change of β, γ, and μ, and is checked by the same Critic conditions. Any scientific outcome is acceptable (optimizer no worse than the Generator, or worse / infeasible). Do not retune windows, thresholds, starts, or the objective to make one side look better.

## Model and data

- Same SIRD surrogate and same simulation horizon as the Critic on the first wave of Saint Petersburg.
- Phase-1 starting point (Table 2): β0 = 0.1219, γ0 = 0.0990, μ0 = 0.0099, N = 6.0e6.
- Phase-1 fit is on the first 120 days from 5 July 2020. Use the same initial state (S, I, R, D) and the same ODE integrator as the Critic / Table 2 surrogate.
- Prefer calling the Critic's peak-day and peak-height function directly so those quantities are computed the same way.
- Sanity check before any optimization (stop and report if this fails; do not retune): with (β0, γ0, μ0) the interpolated peak must match Table 2 to the published rounding, namely peak day 264.3 and peak height 39 121.
- r = β − γ − μ (from the parameters, not from a trajectory slope).
- CFR = μ / (γ + μ).
- R0 = β / (γ + μ).
- Peak day is discrete on a numerical grid: use a time step of at most 0.1 day and parabolic interpolation of the maximum, as for the published 264.3.

## Requests (all six from Table 2)

1. higher — “Peak should be higher” — named quantity: peak height I
2. lower — “Peak should be lower” — named quantity: peak height I
3. earlier — “Peak should be earlier” — named quantity: peak day
4. later — “Peak should be later” — named quantity: peak day
5. growth — “faster initial growth” — named quantity: r
6. mortality — “higher mortality” — named quantity: CFR

Intent: use Intent Parser logs from the Table 2 runs if they are in the repository, so the configuration is truly “LLM parser + optimizer” (LLM called once, Generator never called). If those logs are absent, the mapping above is unambiguous from Table 2; record that the logs were not found and proceed with this mapping. Do not call an LLM.

## Decision variables and objective

- Variables: β, γ, μ inside the Generator window of the setting (below).
- Minimize J = ((β−β0)/β0)² + ((γ−γ0)/γ0)² + ((μ−μ0)/μ0)².
- Also report max_i |(p_i − p_i0) / p_i0|.

## Critic constraints (inequalities, not equalities)

Thresholds from Phase-1 values (day0 = 264.3, height0 = 39121, r0 = 0.0130, CFR0 = 0.0909):

- higher: height ≥ height0 × 1.03
- lower: height ≤ height0 × 0.97
- earlier: day ≤ day0 − 1
- later: day ≥ day0 + 1
- growth: r ≥ r0 × 1.03
- mortality: CFR ≥ CFR0 × 1.03

When the unnamed / second axis is held:

- if the named quantity is height (higher/lower): peak day ∈ [day0 − 8, day0 + 8]
- if the named quantity is day (earlier/later): peak height ∈ [height0 × 0.97, height0 × 1.03]
- setting G holds both peak axes: day ∈ [day0 − 8, day0 + 8] and height ∈ [height0 × 0.97, height0 × 1.03]

A point is feasible only if it also lies inside the setting's parameter window.

## Settings

### D (default) — like pass A; compare with Table 2

- Window: β ∈ [0.1119, 0.1319] (β0 ± 0.01), γ ∈ [0.0985, 0.0995] (γ0 ± 0.0005), μ ∈ [0.00985, 0.00995] (μ0 ± 0.00005).
- Exception: for mortality only, μ ∈ [0.0089, 0.0109] (μ0 ± 0.001).
- Unnamed axis free (no hold).
- All six requests.

### S (strict) — like pass B; compare with Table 3 starter

- Only the four peak requests: higher, lower, earlier, later.
- Window: β ∈ [0.1119, 0.1319], γ ∈ [0.0890, 0.1090] (γ0 ± 0.01), μ ∈ [0.00985, 0.00995].
- Hold the unnamed axis as above.

### G (growth and mortality with the peak held)

- Requests: growth and mortality only.
- Same window as D (including the wider μ window for mortality).
- Hold both peak axes: day ± 8 days and height ± 3%.

## Target levels

T1 — minimal admissible change: the Critic threshold inequalities above.

T2 — equal-effect comparison: the named quantity must move at least as far as in the accepted reference run, in the requested direction. The hold constraints of the setting still apply.

Reference named-quantity values (do not require the optimizer to match the reference parameters, only the named effect):

Table 2 (Generator, pass A) — use for D and G:

| request   | β      | γ      | μ      | r      | CFR    | peak day | peak height | T2 named target        |
|-----------|--------|--------|--------|--------|--------|----------|-------------|------------------------|
| higher    | 0.1319 | 0.0990 | 0.0099 | 0.0230 | 0.0909 | 197.8    | 100455      | height ≥ 100455 |
| lower     | 0.1169 | 0.0990 | 0.0099 | 0.0080 | 0.0909 | 305.9    | 18273       | height ≤ 18273         |
| earlier   | 0.1319 | 0.0990 | 0.0099 | 0.0230 | 0.0909 | 197.8    | 100455      | day ≤ 197.8            |
| later     | 0.1150 | 0.0990 | 0.0099 | 0.0061 | 0.0909 | 312.7    | 12555       | day ≥ 312.7            |
| growth    | 0.1319 | 0.0985 | 0.0099 | 0.0235 | 0.0913 | 195.4    | 104848      | r ≥ 0.0235             |
| mortality | 0.1219 | 0.0990 | 0.0105 | 0.0124 | 0.0959 | 268.3    | 35894       | CFR ≥ 0.0959           |

Table 3 (starter, pass B) — use for S:

| request | β      | γ      | μ      | r      | CFR    | peak day | peak height | T2 named target   |
|---------|--------|--------|--------|--------|--------|----------|-------------|-------------------|
| higher  | 0.1169 | 0.0940 | 0.0099 | 0.0130 | 0.0953 | 271.5    | 42257       | height ≥ 42257    |
| lower   | 0.1269 | 0.1040 | 0.0099 | 0.0130 | 0.0869 | 257.1    | 36356       | height ≤ 36356    |
| earlier | 0.1319 | 0.1079 | 0.0099 | 0.0141 | 0.0840 | 244.2    | 39121       | day ≤ 244.2       |
| later   | 0.1119 | 0.0901 | 0.0099 | 0.0119 | 0.0990 | 287.9    | 39121       | day ≥ 287.9       |

There are 24 problems: D×6×2 + S×4×2 + G×2×2.

## Numerics

- Primary solver: SLSQP (`scipy.optimize.minimize`, method `SLSQP`).
- If a start fails or returns a worse point, retry that start with `trust-constr`.
- At least 20 starts from a Latin hypercube over the setting's window, plus the Phase-1 point itself if it lies in the window (it does).
- Plus a dense grid of 41³ points in the same window (41 values per coordinate, inclusive of both bounds, so the edges of the window are included).
- If no grid point is feasible, report `infeasible` for that problem (the optimizer may still be recorded, but the grid is the feasibility oracle).
- The reported solution is the feasible point with smallest J among: all successful optimizer starts and all feasible grid points. Optionally polish that point with SLSQP; keep the polished point only if it remains feasible and J does not increase.
- Efficiency (does not change the protocol): the default D window can be gridded once and reused for every non-mortality D and G problem; the mortality D/G window is a second grid; the S window is a third grid. Evaluate each unique window's 41³ grid once, then score every relevant request/target on those stored simulations.
- Record the number of simulator calls and wall time for each problem and in total.
- Do not stop early to favor one method. Do not drop infeasible cases.

## Outputs (required)

1. A reproducible script (Python) that implements this protocol.
2. `results.csv` — one row per request × setting × target, with:
   - request, setting (D/S/G), target (T1/T2)
   - β, γ, μ, R0, r, CFR
   - peak day, peak height
   - J and max relative parameter change
   - change of the named quantity and of the unnamed / peak axes (days, %)
   - feasible (yes/no), critic_accept (yes/no)
   - n_simulations, wall_time_s
   - solver_used, n_starts_converged, grid_n_feasible
   - reference Generator or starter β, γ, μ, named-quantity value, and the corresponding J of that reference point (recomputed with the same J formula) for comparison
3. A short English report (`results.md`) that:
   - restates that the protocol was frozen
   - reports the Phase-1 sanity check
   - lists every one of the 24 outcomes, including infeasible
   - compares each optimizer row with the matching Table 2 or Table 3 row
   - states clearly, without overclaim, whether the Generator ever needed a smaller J, a smaller side effect, or found a feasible point when the optimizer did not
   - notes that this is not a forecast test
4. Save artifacts under the agent's artifacts directory as a zip that contains the script, CSV, report, and any logs needed to reproduce the 24 rows.
5. Do not modify the manuscript.

## Checks

- Recompute Table 2 / Table 3 parameter sets through the same simulator and confirm peak day/height match the published values to the published rounding. If a reference point does not match, report the discrepancy and still optimize against the protocol targets above; do not silently shift targets.
- Population conservation and non-negativity of compartments on reported solutions.
- J of a reported point equals the formula above.
