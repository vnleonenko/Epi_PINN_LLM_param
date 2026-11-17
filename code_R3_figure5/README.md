# Figure 5: Monte Carlo Dropout band vs. bootstrap band (O2, pass A, lower-peak request)

- `mcd_band.npy` — unsmoothed pointwise 2.5–97.5% Monte Carlo Dropout band of the Phase-3 run of Figure 4
  (columns: day, lower, upper). Recovered from the vector paths of `fig_autumn2021_lower_regen_mcd.pdf`
  (`band_svg.npy`, `paths.json`); axis calibration checked against the caption of Figure 4
  (width at the peak of the mean 91 580 vs. 92 592; peak of the mean on day 133.4 vs. 133.3).
- The bootstrap band is taken from `boot/band.json` (see the bootstrap code in this archive).
- `fig_bands.py` — computes coverage of the observations after the origin (days 51–133):
  dropout band 71% (100% up to the observed peak on day 85, 50% after), bootstrap band 29% (17% / 38%),
  and draws Figure 5.
