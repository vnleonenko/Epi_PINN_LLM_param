# Appendix H: Phase-3 network with and without the ODE term (Table A16)

Configuration A is the published one (λ_ode = 0.1) and configuration B is the same network without the ODE term (λ_ode = 0). Each is trained on two accepted SIRD simulations, case 1 and case 2, with seeds 0–9, which gives 40 trainings.

## Contents

- `run_r31.py`: the script.
- `PINN_const_r31.py`: a copy of `agents/PINN_const.py` that differs in one place. With λ_ode = 0 the ODE term is still logged but is dropped from the loss.
- `PINN_const_source.py` and `SurrogateModel_source.py`: unmodified copies of the repository files. The script checks that `agents/PINN_const.py` and `agents/SurrogateModel.py` match them.
- `covid-19_Kouprianov.csv` and `covid-19_SPb_autumn2021.csv`: these files are used only for the day-0 initial conditions.
- `PROVENANCE.md` and `spec_current.md`: the provenance of the inputs and the fixed protocol.
- `results/`: the archived run.
  - `results_summary.csv` and `paired_diff.csv` are the source of Table A16.
  - `s1.json` is the control run.
  - `config_dump.json` records the CPU and torch 2.15.0.
  - The folder also holds the per-seed logs, `preds/*.npz` with the network predictions, and the figures.

## Reproduce

Run from the repository root:

```bash
python appendix_H_ode_term/run_r31.py --checks-only   # pre-training checks only, a few seconds
python appendix_H_ode_term/run_r31.py                 # all 40 trainings, several hours on CPU
```

Output goes to `appendix_H_ode_term/results_rerun/`. Use `--out` to write somewhere else. Network weights are not stored here.
