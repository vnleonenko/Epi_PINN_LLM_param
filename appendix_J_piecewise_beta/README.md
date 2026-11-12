# Appendix J: piecewise-constant β (Table A18)

Retrospective SIRD fits to the first-wave active cases in Saint Petersburg (days 0–368). The models are a constant β, β with exactly two breakpoints, and β with exactly three breakpoints. The dates of the breakpoints are estimated from the data.

## Contents

- `run_r37.py`: the script. `config.json`, `PROTOCOL.md` and `PROTOCOL_HIGHBUDGET.md` fix the settings before the fit.
- `data/covid-19_Kouprianov.csv`: the input series (SHA-256 bde98003…0f3aab).
- `prior_low_budget/starts.csv`: the starts of the earlier low-budget run (maxiter = 200, maxfun = 500). The script uses them only for its continuation starts.
- `results/`: the archived high-budget run (maxiter = 5000, maxfun = 20000) behind Table A18.

## Reproduce

Run from the repository root:

```bash
python appendix_J_piecewise_beta/run_r37.py
```

The script needs numpy, scipy, pandas and matplotlib, and takes about 12 minutes. It writes its output to `appendix_J_piecewise_beta/results_rerun/`, so the archived `results/` folder is left untouched. Set the `R37_OUT` environment variable to write somewhere else.
