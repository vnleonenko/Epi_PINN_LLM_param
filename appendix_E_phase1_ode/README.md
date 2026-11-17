# Appendix E (Table A10b): Phase-1 PINN with and without the ODE term

This folder supports the Phase-1 part of manuscript Appendix E, "PINN with and
without the ODE Term" (second paragraph of the appendix and Table A10, part b).
The Phase-3 part of the same appendix (Table A10, part a) is in
`appendix_H_ode_term/`.

The folder repeats the Phase-1 PINN calibration of Section 4.1.1 on the first
wave in Saint Petersburg and compares it with the same network trained without
the ODE term of the loss.

## Setup

- Data: `data/covid-19_Kouprianov.csv`, active cases in Saint Petersburg from
  5 July 2020 to 8 July 2021 (369 days), N = S + I + R + D on day 0.
- Training window: days 0-184 (first 185 days).
- Network and loss: as in the Phase-1 calibration code of the paper
  (`EINN_PINN`, `NEW_PINN/PINN_const.py`, repository blabla9111/PINN_LLM,
  branch `tester`): fully connected network 1-128-256-128-4 with tanh,
  standardized outputs with softplus, beta, gamma, mu trainable through
  (tanh z + 1) / 2 with initial values 0.1, 0.1, 0.001; data term with the
  error of I weighted 10 times; ODE residual on all 369 days.
- Training: 10 000 epochs, Adam (learning rate 1e-4) with ReduceLROnPlateau
  (patience 1 000, factor 0.5), lambda_data = 1, lambda_ic = 0.1,
  lambda_bc = 0.1.
- Configurations: A, lambda_ode = 1 (the value of the published runs);
  B, lambda_ode = 0; additional values 0.01, 0.1 and 10.
- Seeds 42-51, shared by all configurations (the seeds of the published runs).

With lambda_ode = 0 the parameters receive no gradient, so beta, gamma, mu stay
at their initial values and are not estimates.

## Metrics

For each run, on the network output for I:

- `rmse_I`, `mae_I`: errors on days 185-214 against the observed active cases;
- `peak_day`, `peak_height`: maximum over days 0-368;
- `peak_day_window`, `peak_height_window`: maximum over days 185-214, the
  definition used for the predicted peak day in Section 4.1.1 (the observed
  peak, day 199, lies in this window);
- `beta`, `gamma`, `mu`: parameter values at the end of training.

`data/published_phase1_runs.csv` holds the per-run results of the published
calibration (seeds 42-51) for comparison with configuration A.

## Reproduce

Requires Python 3.10+, PyTorch, NumPy and pandas. On a CPU one run takes
about 75 s with one thread.

```
python run_phase1_ode.py --lambda-ode 1.0 0.0 --seeds 42-51
python run_phase1_ode.py --lambda-ode 0.01 0.1 10 --seeds 42-51
python summarize.py --runs results_rerun/runs.csv
```

Runs are written to `results_rerun/` (or `--out`); runs already present in
`runs.csv` are skipped. The network output for I of each run is saved in
`curves/` as a NumPy array over days 0-368.

## Files

- `pinn_sird.py`: the network, the loss and the training loop.
- `run_phase1_ode.py`: runs the configurations in parallel processes.
- `summarize.py`: means, SDs and paired comparisons with lambda_ode = 0.
- `results/runs.csv`, `results/summary.csv`, `results/summary.md`,
  `results/curves/`: results of the runs reported in Appendix E
  (Table A10b and the lambda_ode = 0.01, 0.1, 10 comparison in the text).

The reported runs were made on a CPU (Linux, Python 3.13, PyTorch 2.15,
NumPy 2.5, pandas 3.0). With the same seeds, configuration A reproduces the
published per-run values closely (beta equal to three or four decimals, RMSE
within a few percent; compare `results/runs.csv` with
`data/published_phase1_runs.csv`), but not bit for bit, because the published
runs were made on a different machine.

## Numbers in the manuscript (Appendix E, Table A10b)

All from `results/summary.md` (`results/summary.csv`), architecture 128-256-128:

| Manuscript | Value | Source row |
|---|---|---|
| RMSE of I, A / B | 4 149 ± 882 / 5 613 ± 1 252 | lambda_ode 1 / 0 |
| MAE of I, A / B | 3 652 ± 764 / 4 692 ± 707 | lambda_ode 1 / 0 |
| Peak day, A / B | 198.1 ± 0.7 / 204.8 ± 3.6 (days 197-199 / 201-210) | lambda_ode 1 / 0 |
| Peak height, A / B | 98 743 ± 1 486 / 102 472 ± 3 302 | lambda_ode 1 / 0 |
| A better (RMSE / MAE / peak day / peak height) | 7 / 8 / 10 / 0 of 10 | lambda_ode 1 |
| RMSE range, A / B | 2 897-5 666 / 4 297-8 008 | lambda_ode 1 / 0 |
| beta, A | 0.0459 ± 0.0010 | lambda_ode 1 |
| RMSE lower than without the term | 7 (0.01), 9 (0.1), 1 (10) of 10 | lambda_ode 0.01 / 0.1 / 10 |
| Peak days with lambda_ode = 10 | 193-195 | lambda_ode 10 |
| Published calibration | RMSE 4 162 ± 887, beta 0.0458 ± 0.0011 | `data/published_phase1_runs.csv` |
