# Additional R3 computations (17 Nov 2026)

Build the integrator first in each folder: `gcc -O2 -shared -fPIC -o librk4.so rk4.c`.

- `aic_error_models/` — SIRD vs SEIR-D on the four first-wave series under three error models
  (one pooled variance, separate variances, log(1+x) scale); `python3 fit.py one,log`, `python3 fit.py sep`,
  `python3 summarize.py fits_*.json`. Data: covid-19_Kouprianov.csv (first wave, days 0–368) as `data.csv`.
  Reproduces the published pooled values (+55.7 at E(0)=2I(0) restricted; +2.3 unrestricted at E(0)=0).
- `bootstrap_band_O2/` — moving-block bootstrap (B=1000, L=7) of the second-wave Phase-1 fit at O2,
  shifted by the accepted change (beta −0.01), Appendix K.3 and Figure 4b. Data: SPb.COVID-19.united.csv.
- `intent_trained_baseline/` — leave-one-out TF-IDF classifiers (logistic regression, 1-NN) on the
  111-request corpus of Appendix J.


## Software versions
These computations do not use PyTorch (the PINN runs of Appendix E used PyTorch 2.15.0, CPU).
Python 3.13.17; NumPy 2.5.3; SciPy 1.18.1; scikit-learn 1.9.1; Matplotlib 3.11.2; gcc 13.3.0 (Ubuntu 24.04).
