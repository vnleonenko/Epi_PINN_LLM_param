# Appendix I: constrained optimizer in place of the Parameter Generator (Table A17)

For each of the six requests in Table 2, the script finds the smallest relative change of β, γ and μ that the Critic accepts. It covers three settings (D, S and G), with two targets each (T1 and T2), so 24 problems in total. The protocol was fixed before the run, see `PROTOCOL.md`. No LLM is called.

## Contents

- `run_experiment.py` is the script. It takes `SIRDSurrogate` from `agents/SurrogateModel.py` in the repository and the initial state from `Phase 1 (model calibration)/real_datasets/covid-19_Kouprianov.csv`.
- `PROTOCOL.md` is the protocol that was fixed before the run. It is kept unchanged, and its MD5 is recorded in `results/logs/provenance.json`.
- `results/` holds the archived run:
  - `results.csv` and `results.md` are the source of Table A17.
  - `logs/` holds the starts, the grids `grid_*.npz`, the sanity checks and the run log.
  - In the files, setting G is called `M`.

## Reproduce

Run from the repository root:

```bash
python appendix_I_constrained_optimizer/run_experiment.py           # about 5 minutes
python appendix_I_constrained_optimizer/run_experiment.py --smoke   # quick check, not the reported run
```

The output goes to `appendix_I_constrained_optimizer/results_rerun/`. Use `--outdir` to write it somewhere else.

In this copy of the script, only the repository and output paths were changed. Because of that, its MD5 differs from the `script_md5` recorded in `results/logs/provenance.json`.
