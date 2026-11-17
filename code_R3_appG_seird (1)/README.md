# Appendix G: SIRD and SEIR-D with constant and piecewise-constant beta (first wave)

- `pw.c` -> `libpw.so` (gcc -O3 -shared -fPIC -o libpw.so pw.c): RK4 (dt = 0.02 day) for SIRD/SEIR-D with piecewise-constant beta.
- `core.py`: data (`data.csv`, Saint Petersburg first wave, S = N - I - R - D, N = 6e6) and the simulator.
- `fit.py I` / `fit.py four`: least-squares fits to I alone or to S, I, R, D with one variance; 0, 2, 3 breakpoints
  (segments >= 14 days); SEIR-D latent period 2-14 days, E(0) in {0, 0.5, 1, 2, 4} x I(0); 1296 starts per objective.
  Output: `fits_I.json`, `fits_four.json`. AIC = m log(RSS/m) + 2k (k counts betas, breakpoints, gamma, mu, and for SEIR-D sigma and E(0)).
- `summ.py I|four`: best fit per (breakpoints, model, E(0)) and dAIC against SIRD with the same breakpoints.
- `fig_pw.py`: Figure A4.
The SIRD fits reproduce Appendix G (RSS 5.40e10, 9.47e8, 8.36e8 on I; growth 0.53/2.74/2.62 %/day and dAIC 5285/5575 on four series).


## Software versions
These computations do not use PyTorch (the PINN runs of Appendix E used PyTorch 2.15.0, CPU).
Python 3.13.17; NumPy 2.5.3; SciPy 1.18.1; scikit-learn 1.9.1; Matplotlib 3.11.2; gcc 13.3.0 (Ubuntu 24.04).
