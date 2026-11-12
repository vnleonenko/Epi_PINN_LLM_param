# R3.7 protocol (frozen before optimization)

Retrospective description on the first-wave Saint Petersburg series. Not a forecast test and not a revision of the manuscript.

## Data and time origin

File `data/covid-19_Kouprianov.csv`, sha256 `bde98003e5fd10331bcd0adf2a2f7c8ccb75398b34296de47355b54b060f3aab`. 369 rows, `t = 0,1,...,368`. Row 0 is 5 July 2020 and the last row is 8 July 2021 (Table A14, Figure A3). `S+I+R+D = 6,000,000` on every row. Day-0 state is the observed `(S,I,R,D) = (5995513, 4197, 248, 42)`. No subtraction of `R(0)` or `D(0)`.

This is the series in Section 4.1. The observed active-case peak is day 199 at 104,932. Log-growth of `I` averages 1.248% per day on increments `t = 0..109` and 2.890% per day on `t = 110..149`, which are the manuscript’s “1.2–1.3% over the first 110 days” and “2.9% between days 110 and 150”.

The Phase-1 fit used only the first 120 days and is not this window. The 185-day window is the PINN training window. The direct fit that states the acceleration and reports the SIRD curve in Table A13 (peak day 196, height 77,790) is the full 369-day series. The common estimation window is therefore days 0 through 368 inclusive.

## Models

Same SIRD compartments. `γ` and `μ` are constant in time. `β` is either constant, or piecewise constant with exactly 2 breakpoints (3 segments), or exactly 3 breakpoints (4 segments). The counts are fixed here. The dates are fitted. They are not taken from a clinical calendar and they are not placed by hand on days 110–150.

Minimum segment length is 14 days. That minimum is not in the source; it is the declared default so a segment cannot collapse onto a point. Lengths are `14 + excess × softmax(z)` with each `z` in `[-6, 6]` and excess `= 368 − 14 × (number of segments)`.

## Objective and solver

Minimize the daily residual sum of squares of active cases only:

`RSS = Σ_{t=0}^{368} (I_sim(t) − I_obs(t))^2`.

This is the Table A13 fit to `I(t)` (`n = 369`), not the Phase-1 loss (MSE on every 7th training point). `S`, `R`, and `D` are simulated and reported, and they are not in the objective. A one-variance fit on raw `S,I,R,D` would be dominated by `S` and is not the curve in Table A13.

Each segment is integrated with `solve_ivp`, method RK45, `rtol=1e-6`, `atol=1e-8`. The state at a breakpoint is the initial state of the next segment. A single adaptive step is not allowed to cross a breakpoint.

Parameter bounds, from `SIRD_calibration.ipynb` cell 0: each `β ∈ [0.01, 0.99]`, `γ ∈ [0.001, 0.5]`, `μ ∈ [0.0001, 0.1]`.

## Optimization

L-BFGS-B, 16 starts, seeds 0..15, `numpy` Generator PCG64(`20261108 + seed`). Same `maxiter=200`, `maxfun=500`, `ftol=1e-12`, `gtol=1e-8` for every start of every model.

Start 0, all models: every segment `β = 0.1219`, `γ = 0.0990`, `μ = 0.0099`, equal segments (`z = 0`). Starts 1..15: log-uniform draws inside the bounds and, for piecewise models, independent log-normal segment multipliers (σ = 0.35) and `z` uniform on `[-2, 2]`. The best finite RSS is kept. Starts that stop on the evaluation budget are kept in the table and can win on RSS. No start is deleted.

The published `(0.1219, 0.0990, 0.0099)` trajectory is plotted only as a contextual 120-day Phase-1 curve. It is not the matched constant refit.

## Metrics fixed before fitting

On observed and fitted `I`, with the same increments:

- `g_early`: mean `log(I_{t+1}/I_t)` for `t = 0..109`
- `g_accel`: mean `log(I_{t+1}/I_t)` for `t = 110..149`
- `accel = g_accel − g_early`
- `delta_I = I(150) − I(110)`

Also the mean of the model per-capita rate `β(t) S(t)/N − γ − μ` on integer days 0..110 and 110..150. Full window: RSS, RMSE, and MAE of `I`; RMSE of `S`, `R`, and `D`; day and height of the maximum of fitted `I`.

No numerical cutoff is declared for “reproduced”. The result is the comparison of these quantities. No confidence intervals and no significance tests.

## Checks

Positivity of `S,I,R,D`, conservation of `N`, continuity of the state at breakpoints, equality of the piecewise simulator to a one-segment solve when all `β` are equal, and re-integration of each best solution at `rtol=1e-8`, `atol=1e-10`. A NaN is a failed start and is retained in the start table.
