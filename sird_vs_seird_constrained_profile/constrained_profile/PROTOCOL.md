
# Protocol: constrained SEIR-D E(0) profile (latent period 2–14 days)

Frozen protocol for reviewer point R3.6 / plan Task 4. The runner is
`run_constrained_e0_profile.py` in this folder. It does not train a PINN,
does not edit the manuscript, and does not write into
`supplementary_model_comparison/`.

`Phase 1 (model calibration)/full_wave_calibration.py` is the script that
wrote `supplementary_model_comparison/` (SHA-256 recorded in
`run_config.json`) but that file is not in this repository. This runner
reproduces the direct least-squares path that
`deferred_tasks/code_audit/check_s_vs_se.py` already checked against
`ode_lsq_reference.csv`: fixed-step RK4, the same initial conditions, and
the S channel of SEIR-D matched to S+E. Parameters are optimized in log
space by multi-start `scipy.optimize.least_squares` (TRF), which is the
procedure described in `supplementary_model_comparison/README.md`.

The E(0)/I(0) = 0.5 rows already stored for the 2–14 day constraint are a
sanity check, not a seed. The optimizer is not started from those published
parameter vectors.

## Data

- File: `data/covid-19_Kouprianov.csv` (in this folder)
- SHA-256: `bde98003e5fd10331bcd0adf2a2f7c8ccb75398b34296de47355b54b060f3aab`
- The original run used the workspace path `src/Epi_PINN_LLM_param/Phase 1 (model calibration)/real_datasets/covid-19_Kouprianov.csv`; the bytes are identical (same SHA-256). The same bytes were also at `src/PINN_LLM_dinara/tester/NEW_PINN/real_datasets/covid-19_Kouprianov.csv`.
- Window: rows of that file, days 0 through 368 inclusive (n = 369). No smoothing, no downsampling.
- Population N = 6_000_000. The file satisfies S + I + R + D = N on every row.

## Models

SIRD compartments (S, I, R, D), initial state (S(0), I(0), R(0), D(0)) taken from day 0.

SEIR-D compartments (S, E, I, R, D):

- E(0) = (E(0)/I(0)) · I(0), with I(0), R(0), D(0) from day 0
- S(0) = S_data(0) − E(0)
- Rates: dS/dt = −β S I / N, dE/dt = β S I / N − σ E, dI/dt = σ E − (γ + μ) I, dR/dt = γ I, dD/dt = μ I

Only the fixed-ratio mode is run. E(0)/I(0) ∈ {0, 0.25, 0.5, 1, 2, 4, 8, 12, 16, 24}, the `e0_profile` grid in `supplementary_model_comparison/run_config.json`. Self-consistent, growth-consistent, and E-stationary modes are not re-run.

Integrator: classical RK4, step 0.02 day, state recorded at each integer day (369 samples). This is the integrator in `check_s_vs_se.py`, which reproduces the stored direct-fit RSS.

## Error models (separate fits)

1. **I only.** Residual I_model(t) − I_data(t). n = 369. k = 3 (SIRD) or 4 (SEIR-D).
2. **Four compartments, one variance.** Residual vector (S, I, R, D), summed as one RSS. n = 1476 = 4 × 369. For SEIR-D the S residual is (S_model + E_model) − S_data, which is the match already used by the stored fits (`check_s_vs_se.py`). SIRD uses S_model − S_data.

## Parameters and bounds

Log-parameters, box constraints from `run_config.json`, with σ restricted to the existing `incubation_2_14_sigma` interval:

| Parameter | Lower | Upper |
|---|---|---|
| β | 1e-4 | 5 |
| σ | 1/14 | 1/2 |
| γ | 1e-4 | 2 |
| μ | 1e-7 | 0.5 |

σ is not allowed outside [1/14, 1/2]. A fitted value is reported as being on a bound when it lies within about 0.1% of that edge on a log scale (|log(value/edge)| ≤ 10⁻³). For the latent period, the σ lower edge is 14 days and the σ upper edge is 2 days.

## Multi-start

`scipy.optimize.least_squares`, method `trf`, on z = log(parameter), bounds log(lower) and log(upper). The kept point is the smallest RSS. Ties within a relative RSS of 10⁻⁹ keep the earliest start.

SEIR-D, 26 starts, matching the start count in `ode_lsq_reference.csv`:

1. Notebook initial guess used by the released comparison: β = 0.1133380423496375 (the early-growth rule in `SIRD_calibration.ipynb`: mean of log(I_t/I_{t−1}) over the first four steps, plus 0.1), γ = 0.1, μ = 0.01, σ = 1/5.2.
2. Mean of the 10 main `SEIR-D-PINN` seeds at E(0)/I(0) = 0.5 in `supplementary_model_comparison/aic_runs.csv`, with each coordinate clipped into the box above. The mean σ lies below 1/14 and is clipped to 1/14.
3. A 24-point grid: β, γ, and μ at the 25% and 75% points of their log-bound interval (2 × 2 × 2), and σ at the lower edge, the geometric centre, and the upper edge (3).

SIRD, 10 starts: the same 2 × 2 × 2 log-quantile grid for (β, γ, μ), plus the notebook guess (β, γ, μ), plus the mean of the 10 main `SIRD-PINN` seeds. The released CSV records 8 SIRD starts, all reaching one optimum; the two documented extra starts are included here and do not change the kept fit when they reach that same optimum.

## AIC

The formula written by `supplementary_model_comparison` (SIBIRCON 2022, Eq. 6):

AIC = n ln(RSS) + 2k

ΔAIC = AIC_SEIR-D − AIC_SIRD = n ln(RSS_SEIR-D / RSS_SIRD) + 2

The baseline is the SIRD fit from this same runner, same window, same N, same error model. The paper’s n ln(RSS/n) + 2k differs by n ln n and gives the same ΔAIC.

## Metrics

For every profile point: E(0)/I(0), β, σ, γ, μ, latent period 1/σ, whether 1/σ is on the 2-day edge, the 14-day edge, or neither, RSS, ΔAIC against the matching SIRD baseline, ever-infected share 1 − S(end)/N (S is the susceptible compartment), dip_ratio = min over days 0..59 of (fitted I / observed I), dip_day, peak height and peak day of fitted I on the integer grid. Also recorded, because the released CSV has them: R0 = β/(γ+μ), R(end), D(end), analytic dI/dt at t = 0, start counts.

On the I-only criterion the trajectory of I depends on γ and μ only through γ+μ, so the split between γ and μ is not identified. The reported split is the one returned by the winning start.

## What this run does not do

- No PINN training and no use of `supplementary_aic_bic/aic_bic_sird_seird.py` (that file uses a different window and N).
- No self-consistent, growth-consistent, or E-stationary E(0).
- No overwrite of `supplementary_model_comparison/*`.
- SIRD is the formal limit of SEIR-D as 1/σ → 0 with E(0) → 0. That limit is outside the 2–14 day box, so a constrained fit is not required to recover SIRD.

