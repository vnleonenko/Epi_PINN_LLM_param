# R3.1 report

Ran Phase-3 trainings on the accepted SIRD simulations (days 0–400, 1000 points), configurations A (λ_ode = 0.1) and B (λ_ode = 0), seeds 0–9. Completed runs: 40/40. Device: cpu, torch 2.15.0+cpu, 4 CPU threads. S1 case2 A seed0 dropout-off peak-height error: -0.1581%.

## Code versus Appendix G

Phase-3 call matches Appendix G: 128-256-128 tanh, softplus after rescaling, 10000 epochs, Adam 1e-4, ReduceLROnPlateau patience 1000 factor 0.5, λ_data = 1, λ_ode = 0.1, λ_ic = 0.1, λ_bc = 0, I weighted ×10, β, γ, μ frozen, dropout p = 0.005 only at inference (100 passes). The function default λ_bc is 0.1 and the function default dropout rate is 0.05; the Phase-3 caller passes 0 and 0.005. main() sets λ_ode = 1 for the unused PINN agent; the Phase-3 node hardcodes 0.1. Weights are the last epoch. Collocation is the 1000 data times. Time is not normalized. B logs L_ode and omits it from the loss and from the backward graph. The scheduler sees total loss, so B can follow a different learning-rate path. Appendix E.3 and the columns `sim_peak_in_band` and `band_w_pct` use the unsmoothed 2.5–97.5% percentiles from `predict_with_uncertainty`. Those metric values were not changed. `create_comparison_plot` fills the same raw percentiles and then draws `gaussian_filter1d` edges; the manuscript call uses sigma 10 (the function default is 2). The width printed in that plot’s text box is the smoothed edge difference, so it is not the E.3 band width and was not used here. Repo PINN_const.py matches the viruses-main snapshot (sha256 e4161cfab53eca5e47f5be7790ea47e4891bb1f08e429b846e0f667d96d59c64). Case-2 run_config records commit 3886bab with a dirty tree and stores a deterministic peak the current node does not write; the training call used here is the current node.

## Checks

- S2: passed
- S3: passed
- S4: passed
- S5: passed
- S6: passed
- S7: passed
- A_matches_source_20_epochs: passed
- S1: passed as near the published band, not as inside it. Case 2 A seed 0 dropout-off peak_h_err_pct = -0.1581, which is 0.36 percentage points below the Appendix E.3 floor of +0.2% (published runs were +0.2% to +3.1%). It is not a several-percent deviation, and it was not retuned. Torch seed 1 on the same case was +0.60% and 0.0 days, near Appendix E.3 seed 1 (+0.55%, about +0.4 day) and not identical.
- S8: a NaN run is kept and flagged; none are deleted.

## Table 1. Dropout off, mean ± SD (min–max)

| Case | Cfg | peak_h_err_pct | peak_day_err | relrmse_I | odeRes_I | dI_err |
|---|---|---|---|---|---|---|
| case1 | A | -0.8393 ± 2.286 (-5.757–1.668) | -4.164 ± 2.298 (-7.608–-0.8008) | 0.01647 ± 0.013 (0.003188–0.04298) | 0.4083 ± 0.1983 (0.2117–0.8387) | 0.4022 ± 0.1976 (0.2101–0.8342) |
| case1 | B | -0.9811 ± 2.664 (-5.883–2.302) | -4.204 ± 2.314 (-8.008–-1.201) | 0.01713 ± 0.01311 (0.003515–0.04365) | 0.3833 ± 0.2065 (0.1702–0.8053) | 0.3767 ± 0.2062 (0.1683–0.8024) |
| case2 | A | -0.3753 ± 2.331 (-5.292–3.119) | -1.001 ± 2.513 (-5.606–2.002) | 0.01172 ± 0.008885 (0.005047–0.03108) | 0.4943 ± 0.2166 (0.2366–0.7372) | 0.4897 ± 0.2204 (0.2329–0.7365) |
| case2 | B | -0.3318 ± 2.094 (-3.978–3.177) | -1.161 ± 2.705 (-6.006–2.402) | 0.01311 ± 0.01075 (0.005015–0.03315) | 0.529 ± 0.2818 (0.2252–0.9783) | 0.5231 ± 0.2843 (0.222–0.9705) |

## Table 2. Dropout on, mean ± SD (min–max); band count is yes/n

| Case | Cfg | mean_peak_h_err_pct | mean_peak_day_err | sim_peak_in_band | band_w_pct |
|---|---|---|---|---|---|
| case1 | A | 5.095 ± 4.475 (-2.539–14.55) | -3.964 ± 5.847 (-14.01–3.203) | 10/10 | 78.97 ± 58.99 (23.65–212.6) |
| case1 | B | 4.968 ± 4.622 (-2.64–15.17) | -3.964 ± 5.847 (-14.01–3.203) | 10/10 | 79.34 ± 59.52 (23.66–215.2) |
| case2 | A | 6.031 ± 2.664 (2.008–10.38) | 0.2402 ± 3.676 (-7.608–4.004) | 10/10 | 90.37 ± 62.21 (21–241.2) |
| case2 | B | 6.055 ± 2.987 (2.007–11.93) | 0.1602 ± 3.719 (-7.608–4.004) | 10/10 | 87.85 ± 61.65 (21.03–237) |

## Flags

- case1: range_overlap_peak_h = True; range_overlap_relrmse_I = True; A_lower_odeRes (S 9/10, I 3/10, R 6/10, D 6/10).
- case2: range_overlap_peak_h = True; range_overlap_relrmse_I = True; A_lower_odeRes (S 10/10, I 6/10, R 6/10, D 10/10).

## Anomalies

No NaN runs.
Dropout-off peak-height errors with absolute value above 3 percentage points (kept): case1_A_seed3 peak_h_err_pct=-4.18 final_loss=3.58e+08; case1_A_seed4 peak_h_err_pct=-5.76 final_loss=2.39e+08; case1_B_seed3 peak_h_err_pct=-5.55 final_loss=2.49e+08; case1_B_seed4 peak_h_err_pct=-5.88 final_loss=2.41e+08; case2_A_seed4 peak_h_err_pct=-3.20 final_loss=4.36e+08; case2_A_seed5 peak_h_err_pct=-5.29 final_loss=4.61e+08; case2_A_seed7 peak_h_err_pct=3.12 final_loss=4.87e+07; case2_B_seed4 peak_h_err_pct=-3.98 final_loss=4.87e+08; case2_B_seed5 peak_h_err_pct=-3.61 final_loss=4.88e+08; case2_B_seed7 peak_h_err_pct=3.18 final_loss=1.68e+09.
The learning rate stayed at 1e-4 in every run. Seed-0 loss curves fall by several orders of magnitude and then oscillate; case2 B seed 7 ended at a higher total loss (1.68e9) than the other case-2 runs.

## Measured difference

On the accepted simulations, the paired Phase-3 runs differed as follows: case1: mean dropout-off peak-height error -0.839% at λ_ode = 0.1 and -0.981% at λ_ode = 0, with absolute-error ranges overlapping = True, and odeRes_I lower at λ_ode = 0.1 in 3/10 seeds; case2: mean dropout-off peak-height error -0.375% at λ_ode = 0.1 and -0.332% at λ_ode = 0, with absolute-error ranges overlapping = True, and odeRes_I lower at λ_ode = 0.1 in 6/10 seeds.

## Post-hoc correction (no retraining)

Derived files only. Weights, `preds/*.npz`, logs, and `results_per_run.csv` are the original 40 runs. `band_w_pct` is a width, not an error, so its paired winner fields `A_smaller` and `n_A_smaller` are blank. The previous summary stored 0 because the metric was outside the winner rule and the unset count was written as 0. Numerically, the unsmoothed width is smaller for λ_ode = 0.1 in 6/10 seeds (case 1) and 5/10 (case 2), with no exact ties; that count is not a quality result. `sim_peak_in_band` was already blank in the winner field (10/10 yes for A and B). Other winner counts were checked against the seed rows. The only false zero was `band_w_pct`. Case 1 `mean_peak_day_err` has `n_A_smaller` 0 because all 10 seeds tie, and that 0 is kept with `n_tie` 10. Panel 2 of `fig_I_case1.png` and `fig_I_case2.png` still fills the saved unsmoothed percentiles and now overplots Gaussian edges with sigma 10, the same `gaussian_filter1d` call the manuscript plot uses, applied identically to A and B. S1 was not recomputed.

## Artifacts

- Outputs: `appendix_H_ode_term/results/`
- Source: `appendix_H_ode_term/`
- Resume: `python3 appendix_H_ode_term/run_r31.py --out <output folder>`

The comparison is the contribution of the ODE term to Phase-3 simulator-surrogate fidelity and to the Monte Carlo outputs. It is not a Phase-1 score, an observed-data forecast score, or an interval-calibration result.
