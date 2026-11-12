# R3.1 provenance

Isolated experiment. Repository files were not modified. No training used observed series as targets.

## Source of the Phase-3 step

- Training function: `EINN_PINN.train_model` in `src/Epi_PINN_LLM_param/agents/PINN_const.py`.
- sha256 of that file equals the viruses-main snapshot: `e4161cfab53eca5e47f5be7790ea47e4891bb1f08e429b846e0f667d96d59c64`.
- Surrogate solver file sha256 also matches the snapshot: `1bd03d437c9f7cb51a43085071632f3c0819858fef719991ead172fbfa8ebc9f`.
- Unmodified copies: `PINN_const_source.py`, `SurrogateModel_source.py`.
- Modified copy used for the 40 runs: `PINN_const_r31.py`. The only change is that `lambda_ode = 0` still logs `L_ode` but drops it from the loss and from the backward graph.
- Caller used for the manuscript Phase-3 step: `PINNVerificationNode` in `main_test.py`, which passes `n_epoch=10000`, `lambda_data=1`, `lambda_ode=0.1`, `lambda_ic=0.1`, `lambda_bc=0`, `train_size=len(trajectory)`, and `dropout_rate=0.005` for 100 passes. Those values match Appendix G. The function defaults (`lambda_bc=0.1`, dropout 0.05) are not the Phase-3 call. `main()` sets `lambda_ode=1` for the separate PINN agent, which this experiment does not train.

## Accepted simulations

Solver: `scipy.integrate.solve_ivp`, method RK45, `rtol=1e-6`, `atol=1e-8`, 1000 uniform points on days 0–400. Same settings as `SIRDSurrogate.simulate`. Initial conditions are the day-0 CSV row with no R or D shift. Population is 6,000,000.

- Case 1, first wave, higher peak: β=0.1295, γ=0.0990, μ=0.0099. CSV `covid-19_Kouprianov.csv` row 0: S=5995513, I=4197, R=248, D=42. Regenerated peak day 210.6106, height 83777.33. Table β=0.1319 is not this target. The same solver reproduces the manuscript Phase-1 peak (day 264.3, height 39121) and the Table 2 β=0.1319 peak (day 197.8, height 100455).
- Case 2, O2 pass B, lower peak: β=0.2331, γ=0.1823, μ=0.00704. CSV `covid-19_SPb_autumn2021.csv` row 0: S=5996610, I=2911, R=440, D=39. Peak day 116.51651651651652, height 115320.96257297913, matching `verification_20261023_193508.json`. The run_config β=0.2281, γ=0.1773 are the Phase-1 values, not the training target. Verification predictions in that JSON are network output, not the training target.

## Training details taken from the code

- Last-epoch weights. There is no best-checkpoint reload.
- Collocation points are the 1000 training times.
- Time is not normalized. Compartments are z-scored on all 1000 points (`std` with `unbiased=False`), then softplus is applied after the inverse transform.
- ODE derivatives in the loss and in the metrics are autograd of the physical outputs with respect to time in days.
- Dropout probability is 0 during training. Inference uses p=0.005, 100 passes, seed+1000, unsmoothed 2.5 and 97.5 percentiles. The manuscript figure additionally draws a Gaussian-smoothed edge (sigma 10). Metrics follow Appendix E.3 and use the unsmoothed percentiles.
- Case-2 `run_config.json` records git commit `3886babc693c650546aea9fdf365188f3ecad23a` with a dirty tree. That verification JSON stores `pinn_deterministic`, which the current node does not write. The training call above is the current node, which matches the snapshot file for `PINN_const.py`.

## Post-hoc correction of derived files

No training was repeated and no repository or manuscript file was edited. Weights, prediction archives, logs, and `results_per_run.csv` are unchanged. `paired_diff.csv` no longer treats `band_w_pct` as a win: `n_A_smaller` for that metric is blank. A smaller unsmoothed width occurred in 6/10 seeds on case 1 and 5/10 on case 2; that is a count, not a quality score. The earlier 0 was an unset winner written as zero. `sim_peak_in_band` already had a blank winner field. `fig_I_case1.png` and `fig_I_case2.png` were redrawn from the saved Monte Carlo percentile arrays: the fill remains those unsmoothed percentiles, and the edge lines are `gaussian_filter1d(..., sigma=10)` as in `create_comparison_plot` when called with `ci_smoothing_sigma=10`. Appendix E.3 widths and coverage stay unsmoothed. The source plot text box uses the smoothed width and is not the E.3 metric. S1 remains -0.158%, below the published +0.2% floor by 0.36 percentage points, and was not adjusted. The pre-correction copies of these derived files are not included in this folder.

## Outputs

`appendix_H_ode_term/results/` (the archived run). A rerun writes to `appendix_H_ode_term/results_rerun/` by default.

Resume, which skips runs already listed in `progress_manifest.json`:

`python3 appendix_H_ode_term/run_r31.py --out <output folder>`
