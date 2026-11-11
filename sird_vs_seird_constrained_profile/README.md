# Appendix F: SIRD vs SEIR-D (Tables A14 and A15)

This folder is self-contained. Put it in the repository root and run the scripts from the repository root.
The scripts use `agents/PINN_const.py` and `agents/PINNAgent.py` from the repository. The other code is in this folder.

## Contents

- `data/covid-19_Kouprianov.csv`: the input data (SHA-256 bde98003…0f3aab, N = 6e6, days 0–368).
- `full_wave_calibration.py`, `PINN_seird.py`, `model_selection.py`: the SIRD and SEIR-D comparison (PINN with 10 seeds plus direct least-squares fits, ΔAIC, and the E(0)/I(0) profile).
- `full_wave_results/`: the results of that run behind Table A14 (`aic_runs.csv`, `ode_lsq_reference.csv`, `aic_comparison.csv`, `e0_sensitivity.csv`, `run_config.json`).
- `constrained_profile/`: the E(0)/I(0) profile with the latent period limited to σ ∈ [1/14, 1/2] (Table A15). It includes the script, `PROTOCOL.md`, `results.md`, `profile.csv`, `summary.json`, `starts.jsonl` and `run.log`.

## Reproduce

```bash
# Table A14 (full run with 10 PINN seeds; add --quick for a short smoke test)
python "appendix_F_seird_vs_sird/full_wave_calibration.py" --out appendix_F_seird_vs_sird/full_wave_results_rerun

# Table A15 (reads full_wave_results/aic_runs.csv)
python appendix_F_seird_vs_sird/constrained_profile/run_constrained_e0_profile.py
```

## Note on hashes

`full_wave_results/run_config.json` was written by the original script, which lived at `Phase 1 (model calibration)/full_wave_calibration.py` with SHA-256 f414be91… and imported `utils/model_selection.py` and `agents/PINN_seird.py`.
In this folder, only the import paths, the data path and the `TRACKED_FILES` list were changed. The computations are unchanged.
`PINN_seird.py` (389d7799…) and `model_selection.py` (e2fa8cf4…) are byte-identical to the originals.
The paths inside the archived `PROTOCOL.md` and `run.log` refer to the original layout, where `supplementary_model_comparison/` is now `full_wave_results/`.
