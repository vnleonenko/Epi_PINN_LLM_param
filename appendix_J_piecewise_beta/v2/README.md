# Appendix G: piecewise-constant β (Table A14, Figure A4)

This folder supports manuscript Appendix G, "Piecewise-Constant Transmission Rate". The folder name, and the appendix letters and table numbers inside `run_r37.py`, `PROTOCOL*.md` and the archived reports in `results/`, follow an earlier draft of the manuscript; the root `README.md` maps them to the published appendices.

Retrospective SIRD fits to the first-wave active cases in Saint Petersburg (days 0–368). The models are a constant β, β with exactly two breakpoints, and β with exactly three breakpoints. The dates of the breakpoints are estimated from the data.

## Contents

- `run_r37.py`: the script. `config.json`, `PROTOCOL.md` and `PROTOCOL_HIGHBUDGET.md` fix the settings before the fit.
- `data/covid-19_Kouprianov.csv`: the input series (SHA-256 bde98003…0f3aab).
- `prior_low_budget/starts.csv`: the starts of the earlier low-budget run (maxiter = 200, maxfun = 500). The script uses them only for its continuation starts.
- `results/`: the archived high-budget run (maxiter = 5000, maxfun = 20000) behind Table A14 and Figure A4.

## Reproduce

Run from the repository root:

```bash
python appendix_J_piecewise_beta/run_r37.py
```

The script needs numpy, scipy, pandas and matplotlib, and takes about 12 minutes. It writes its output to `appendix_J_piecewise_beta/results_rerun/`, so the archived `results/` folder is left untouched. Set the `R37_OUT` environment variable to write somewhere else.

## Four-series refit (Appendix G, last paragraph)

The same models, data, window, bounds, solver and L-BFGS-B settings as above, but
fitted to S, I, R and D with one variance for all four series (SIRD4; n = 1476),
the first error model of Appendix F, and, as a second error model, to I, R and D
with a separate variance per series (IRDsep). Both error models were fitted for
all three models. The I-only archive in `results/` is not changed.

- `fit_piecewise_allcomp.py`: the script. Per model and error model, 64 seeded
  starts plus labelled starts (the best I-only point from `results/summary.json`,
  the best point of an earlier 16-start search in
  `data/allcomp_prev16_best_starts.csv`, and the best simpler model of the same
  run embedded); the 5 best end points are polished (L-BFGS-B on the log of the
  objective, then bounded Nelder-Mead).
- `allcomp_core.py`: the objectives, starts and metrics used by the script.
- `make_figure_allcomp.py`: observed I, R, D against the four-series fits (top)
  and the archived I-only fits (bottom).
- `results_allcomp/`: the archived run behind the manuscript text:
  `summary.csv` (one row per error model and model, plus the re-integrated
  I-only fits as `Ionly_archived`), `starts.csv` (every start and polish),
  `trajectories.csv`, `run_meta.json` (observed values, settings, versions),
  `fig_IRD_allcomp_vs_Ionly.png`, `results_key_table.csv` (the key columns of
  `summary.csv`) and `run.log` (the log of the run).

From the repository root (about 35 minutes with 7 workers):

```bash
python appendix_J_piecewise_beta/fit_piecewise_allcomp.py
python appendix_J_piecewise_beta/make_figure_allcomp.py appendix_J_piecewise_beta/results_allcomp_rerun
```

The fit writes to `appendix_J_piecewise_beta/results_allcomp_rerun/`, so the
archived `results_allcomp/` is left untouched; pass `--out` to write elsewhere.

Numbers in the manuscript (`results_allcomp/summary.csv`; SIRD4 rows unless noted):

| Manuscript | Value | Column |
|---|---|---|
| γ + μ ≈ 0.033 day⁻¹; 1/(γ+μ) ≈ 30 days | 0.0326 (bp2), 0.0329 (bp3); 30.7, 30.4 days | `gamma` + `mu`; `infectious_days_1_over_gamma_mu` is 1/(γ+μ) |
| 7.3–7.5% of the population left S by day 368 | 7.29% (bp2), 7.49% (bp3) | `left_S_frac_368` |
| about 13 100 deaths by day 368 | 13 134 (bp2), 13 092 (bp3) | `D_368` |
| 7.7% registered cases, 15 880 registered deaths | 7.72%, 15 880 | `run_meta.json`, `observed` |
| growth over days 110–150: 2.75% (bp2), 2.60% (bp3), 2.89% observed, 0.53% constant β | 2.75, 2.60, 2.89, 0.53 %/day | `g_accel`; observed in `run_meta.json` |
| ΔAIC piecewise vs constant: −5 285 (bp2), −5 575 (bp3) | −5 284.9, −5 575.0 | `AIC_SIRD4` differences |
| separate variances, bp3: ΔAIC −2 366 | −2 366.3 (bp2: −2 190.9) | IRDsep rows, `AIC_IRDsep` differences |
| RMSE of I about 6 100 vs 1 500–1 600 for the fits to I alone | 6 084 (bp2), 6 063 (bp3); I-only 1 602 (bp2), 1 504 (bp3) | `rmse_I`; `Ionly_archived` rows |
| breakpoints: days 55 and 182 (bp2); 44, 186, 285 (bp3) | 55.38, 181.68; 44.24, 186.27, 284.70 | `breaks` |
| decline of I over the first 45–55 days | first-segment β/(γ+μ) 0.88 (bp2, to day 55), 0.73 (bp3, to day 44) | `R0_segments`, `breaks` |
| 64 starts per model, best points polished | 64 | `run_meta.json` `n_starts`; `starts.csv` |

