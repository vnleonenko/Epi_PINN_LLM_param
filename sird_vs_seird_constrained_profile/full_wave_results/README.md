# SIRD vs SEIR-D on the full wave: results

Written by `../full_wave_calibration.py` (run from the repository root). Every number
lives in the CSV files; this note explains what they contain and how to read them.
A new run deletes and rewrites the CSV files, `run_config.json`, `figures/`, `logs/` and
`pinn_agent_plots/`; this README and the Word documents are written by hand and left alone.

## Files

| File | Content |
|---|---|
| `aic_comparison.csv` | One row per model: the specified table (`model, k, n, RSS, AIC, beta, sigma, gamma, mu`) followed by cross-checks |
| `aic_runs.csv` | Every PINN run (model x seed x E0), all criteria and diagnostics |
| `ode_lsq_reference.csv` | Direct least-squares fits of the ODE models themselves, four error models |
| `e0_sensitivity.csv` | SEIR-D PINN runs with other assumed E(0) / I(0), paired with SIRD on the same seeds |
| `figures/` | Every figure as `.png` (300 dpi), `.pdf` (vector) and `.tiff` (600 dpi, LZW); names below without extension |
| `figures/i_overlay` | Main figure: (a) I(t) of both PINNs with their AIC and the seed-paired ΔAIC, (b) ΔAIC of the direct ODE fits against the assumed E(0) |
| `figures/panel_A_pinn_output`, `figures/panel_D_delta_aic_vs_E0` | Panels (a) and (b) of the main figure on their own |
| `figures/panel_B_ode_fit_to_I` | Direct ODE fits to I(t) alone, AIC and ΔAIC in the legend |
| `figures/delta_aic_summary` | ΔAIC (SEIR-D − SIRD) of every approach side by side, for AIC on S, I, R, D and on I(t) alone |
| `figures/spec_criterion_fit` | The fits to all four compartments on their own: neither model reproduces the wave |
| `figures/full_wave_fit` | PINN diagnostics: I, latent E, R, D, S-channel error, ODE with recovered parameters |
| `figures/parameters_by_seed`, `figures/parameter_convergence` | PINN-recovered parameters per seed, and their trajectories during training |
| `SIRD_vs_SEIR-D_figures_and_summary.docx` | English notes: a three-paragraph summary, AIC tables, a caption and description for every figure |
| `Отчёт_…docx`, `Обновление_расчёта_…docx` | Earlier reports in Russian (the full report and the update after E(0) tied to the fitted rates) |
| `run_config.json` | Settings, versions, git commit, uncommitted files, SHA-256 of the code and data used |
| `pinn_agent_plots/`, `logs/` | PINNAgent's own plots and the training printout of every run |

## Criteria

AIC = n ln(RSS) + 2k (Huaman, Plesovskaya, Leonenko, IEEE SIBIRCON 2022, Eq. 6),
k = 3 for SIRD and 4 for SEIR-D. Only differences between models on the same data and
the same error model mean anything; a difference of 2 is the price of one parameter.

| Column suffix | Residuals | n | Why |
|---|---|---|---|
| (none) | S, I, R, D, one variance | 1476 | the specified RSS |
| `_IRD` | I, R, D, one variance | 1107 | the data's S is N - I - R - D, so its residual repeats the other three |
| `_IRDsep` | I, R, D, one variance each: sum 369 ln RSS_c + 2k | 1107 | the compartments differ in scale by two orders of magnitude |
| `_I` | I only | 369 | the shape of the infected curve |

The pooled S, I, R, D criterion is dominated by S and R (millions and hundreds of
thousands of people), so it mostly measures the cumulative flows, not the shape of I(t).

## aic_comparison.csv

* `RSS` is the geometric mean over seeds and `AIC` the arithmetic mean, so each row
  satisfies AIC = n ln(RSS) + 2k. The same holds for `RSS_I` / `AIC_I` and the rest.
* `delta_<criterion>` is the seed-paired difference from SIRD-PINN (SEIR-D minus SIRD;
  negative favours SEIR-D), with `_sd` and `_se`. `delta_<criterion>_resolved` is true
  when |delta| > 2 SE. `<criterion>_lower_in_seeds` counts the seeds in which the model
  has the lower value.
* `AIC_ode*` score the ODE solution with the PINN-recovered parameters.
* `converged_seeds` counts runs whose parameters changed by less than 1 % over the last
  1000 epochs; `ode_rel_residual_<X>` is RMS(ODE residual) / RMS(dX/dt) of the network.

## ode_lsq_reference.csv

Maximum-likelihood fit of each ODE model under each error model: log-parameters within
bounds, a multi-start grid plus the notebook's initial guess and the mean PINN estimates,
best AIC kept (`starts_within_1` = starts that reached it within one unit). SEIR-D is
fitted with sigma free (`sigma_range = free`) and with the incubation period held to
2-14 days, for several assumed E(0) / I(0). `delta_AIC_vs_SIRD` is against SIRD under the
same error model; `at_bound` names parameters stopped by a bound; `ever_infected_share`
= 1 - S(end)/N and `R_end`, `D_end` compare the implied epidemic size with the data
(observed: 7.7 % of the population, R(end) = 444,328, D(end) = 15,880).

Fitting I alone identifies beta and gamma + mu, not gamma and mu separately.

## What the numbers show (seeds 0-9, E(0) = 0.5 I(0))

1. **Model-agnostic PINN.** The same `PINNAgent.train` call calibrates both classes, and both
   PINN outputs follow the data (peak on day 197 for both; observed day 199). The base run does
   not use EINN_PINN itself: there beta, gamma and mu are created with `requires_grad=False` and
   the lines that enable gradients in PINNAgent are commented out, so a literal run of Step 2.3
   of the specification would calibrate no epidemiological parameter at all.
2. **The specified AIC does not separate the models.** PINN-output AIC: SIRD 32,704, SEIR-D
   32,711; the seed-paired difference (SEIR-D minus SIRD on the same seed) is +8 with SE 321,
   and SEIR-D is lower in 7 of 10 seeds. Medians are 32,369 and 32,372. This AIC measures how
   well a neural network fits the data, not which set of equations is right.
3. **SEIR-D-PINN parameters do not converge with these training settings.** mu drifts to about
   1e-6, where the (tanh(z)+1)/2 parameterization saturates, and sigma keeps moving: the latent
   period ranges from 5.7 to 33.3 days across seeds (0 of 10 runs converged; SIRD 9 of 10). A
   3,000-epoch warm-up with frozen parameters did not change this. It is a property of the
   training, not non-identifiability of the model: mu is pinned by the observed D (direct ODE
   fit 0.0047, moment estimate from the data 0.0012). Do not report these sigma and mu as
   estimates.
4. **The PINN-recovered parameters do not reproduce the wave.** The ODE solved with them is
   flat (peak on the last day, about 1 % infected) for both models, as R0 of 1.05-1.07 implies.
   The ODE residual of the trained networks is of the same size as the derivatives
   (`ode_rel_residual_*` about 1). The paired ODE-level difference, +142 with SE 29, is the only
   difference in the PINN part that exceeds two standard errors.
5. **Direct ODE fits answer the shape question, and the answer depends on the error model.**
   * S, I, R, D (specified): SIRD is preferred. SEIR-D drives sigma to its upper bound, i.e. the
     latent stage vanishes. With E(0) = 0 the two fits coincide and the difference is +2.3, the
     cost of one parameter; +25 with E(0) = 0.5 I(0) is the price of that assumed initial
     condition, +70 with the latent period held to 2-14 days. Under this criterion neither model
     describes the wave: the fitted peaks are 15.6k on day 288 and 22.7k on day 328 against
     104,932 observed on day 199.
   * I alone (shape of the infected curve): SEIR-D is preferred by 289-413 units, but only with
     a 14-30-day latent period against a 2.6-day infectious period, and 60-83 % of the city
     infected against 7.7 % recorded as cases. The fitted curve does not rise slowly: dI/dt at
     t = 0 is negative, I falls from 4,197 to 1,038 by day 9 against 4,906 observed, and stays
     below the data on 83 of the first 100 days. The advantage also depends on the assumed
     E(0): it shrinks monotonically and changes sign beyond about 8 I(0), because the fitted
     trajectory keeps E/I at its quasi-stationary 11.4.
   * Only beta and gamma + mu are identified by the I-only fit, not gamma and mu separately.
   * Multi-start: all 8 SIRD starts reach one optimum under every error model; for SEIR-D only
     4 to 26 of 26 do, so its optima are an upper bound and the numbers are conservative
     against it.
   * Neither constant-parameter model reproduces both the shape and the size of this wave: it
     turned over at 7.7 % recorded, so not through susceptible depletion. That needs a
     time-varying transmission rate. The caveat is ascertainment: the model equates I with
     recorded active cases and takes S as N - I - R - D, i.e. assumes complete ascertainment.

## E(0) tied to the fitted rates

E(0) is not observed. Assuming a value (the `e0_mode = fixed` rows, E(0) = 0.5 I(0) by default)
leaves the model inconsistent at t = 0: the inflow sigma E(0) does not match the outflow
(gamma + mu) I(0), so the fitted curve drops below the data for the first weeks and then rises
steeply. Three further `e0_mode` values compute E(0) from the parameters being fitted instead,
each reading "consistent" differently:

* `self-consistent`: E(0) = I(0)(gamma + mu)/sigma, so dI/dt(0) = 0 and the curve starts flat;
* `growth-consistent`: E(0) = I(0)(r + gamma + mu)/sigma, the E/I ratio of a wave already growing at
  the observed early rate r (first 10 days);
* `E-stationary`: E(0) = beta S(0) I(0)/(N sigma), so dE/dt(0) = 0 instead.

Fitted to I alone, SEIR-D still wins under all three, but by tens of units rather than hundreds:
-35, -21 and -10 against -370 with the assumed E(0). The size depends on the rule, so quote the
order of magnitude, not a single number. The parameters become plausible under every rule (latent
period 6.1, 5.2 and 2.6 days against 30; R0 1.22-1.28 against 2.16; 33-40 % infected against 83 %),
but the curve now looks like SIRD's: the dI/dt(0) = 0 fit peaks at 79.4k against the observed 104.9k
(SIRD 77.8k), runs above the data in mid-rise (2.2 times on day 90) and improves the RSS on I over
SIRD by only 9.5 %.

On the multi-series criteria the dI/dt(0) = 0 and growth-consistent rules reduce the difference to
the parameter penalty (+2.2 to +2.4): sigma runs to its bound and the latent stage vanishes, and the
earlier -29 on I, R, D with one variance disappears with the inconsistent start. The E-stationary
rule degenerates there: it does not bound E(0) from above, and the fit puts 11-15 % of the city into
E with a latent period of 840-1000 days, R0 about 5 and the peak on the last day. SEIR-D then wins
formally (-37.6 and -5.7), but only one start in 26 found that optimum.

The rules differ in the slope I starts with: flat (dI/dt(0) = 0), the observed 1.7 % a day (growth),
or about 3 % a day (E-stationary). The steeper the imposed start, the smaller the advantage on I;
the growth rule is the one consistent with the data. The E-stationary rule on I alone reproduces the
revision memo's quasi-stationary variant: delta AIC -9.6, i.e. delta BIC -5.7.

The 11.4 quoted in the Word report as the consistent E(0)/I(0) is (gamma + mu)/sigma evaluated at the
parameters fitted with the assumed E(0), where 1/sigma = 30 days. When E(0) and the rates are fitted
together, sigma rises to 0.164 and gamma + mu falls to 0.169, and the consistent point is 1.03.

`dip_ratio` (the minimum over the first 60 days of the ratio of the fitted I to the data), `dip_day`
and `dI_dt0` in ode_lsq_reference.csv quantify the early drop.

## Caveats on reading the differences

Residuals here are systematic misfit, not noise: the lag-1 autocorrelation of the residuals of
the fitted ODEs is 0.992-0.999 per compartment (0.996 for I in the I-only fit). The usual AR(1) correction gives an effective
sample size of order one, i.e. it degenerates. Read the sign of a difference, its robustness to
the error model and the plausibility of the parameters, not the number of units as strength of
evidence.

AIC convention: this folder uses n ln(RSS) + 2k (SIBIRCON 2022, Eq. 6). The paper draft and the
revision memo use n ln(RSS/n) + 2k, which differs by the constant n ln n at fixed n: differences
are identical, absolute values are not. The SIRD I-only fit here (RSS 5.40e10, AIC 9125) is the
same fit as SIRD in the memo's tab:aicbic, where it is 6943.8.

## Deviations from the specification

* `n` is the number of residuals (4 x 369 = 1476), as in the SIBIRCON 2022 derivation; the
  example table in the specification shows 369.
* The S channel of SEIR-D is matched to S + E (the data's S is N - I - R - D and includes the
  not-yet-infectious), which the specification left open in its section 3.3.
* Both models run with three optional switches enabled: trainable parameters, a separate Adam
  rate for them, and returning the lowest-loss state. Their defaults reproduce EINN_PINN.
* The experiment is a script (`full_wave_calibration.py`), which the specification allows. The
  optional Italy dataset was not run. BIC is not computed.

## Review history

Both versions of this experiment were reviewed by independent agents, each finding re-checked
separately. Confirmed and fixed in the first version:

* the epidemic parameters shared the network's Adam rate of 1e-4 and had not converged in
  10,000 epochs (they now have their own rate, and a convergence diagnostic is reported);
* the PINN-output AIC was presented as a comparison of the two models (it is now labelled a
  goodness-of-fit diagnostic, and the model comparison is made on the ODE fits);
* RSS was averaged arithmetically over seeds, so AIC = n ln(RSS) + 2k did not hold row by row
  (it is now the geometric mean);
* a failed run stopped the whole experiment; `--quick` overwrote the real results; the run
  configuration did not record the hashes of the code and data.

The second version was reviewed again, including the text of the Word report in this folder;
the errors it found in that text (an inverted statement about how AIC differences depend on n,
a degenerate autocorrelation range, the direction of the sigma bound in the 2-14 day variant,
and the claim that only beta and gamma + mu are identified by the I curve, which ignores that
sigma is identified for SEIR-D) were corrected.
