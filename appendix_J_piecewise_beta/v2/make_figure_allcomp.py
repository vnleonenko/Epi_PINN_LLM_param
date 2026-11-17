#!/usr/bin/env python3
"""Figure: observed I, R, D vs SIRD fits (constant beta, 2 and 3 breakpoints);
top row: fits to all four compartments (SIRD4); bottom row: archived I-only fits.

Usage, from the repository root:
  python appendix_J_piecewise_beta/make_figure_allcomp.py
  python appendix_J_piecewise_beta/make_figure_allcomp.py appendix_J_piecewise_beta/results_allcomp_rerun
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
out = Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "results_allcomp")
tr = pd.read_csv(out / "trajectories.csv")
t = tr["t"]
colors = {"constant": "#0072B2", "bp2": "#D55E00", "bp3": "#CC79A7"}
labels = {"constant": "constant β", "bp2": "2 breakpoints", "bp3": "3 breakpoints"}
rows = [("SIRD4", "Fit to S, I, R, D"), ("Ionly_archived", "Fit to I only (Table A14)")]
fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
for r, (obj, title) in enumerate(rows):
    for c, comp in enumerate("IRD"):
        ax = axes[r, c]
        ax.plot(t, tr[f"{comp}_obs"], color="k", lw=1.4, label="observed")
        for m in ("constant", "bp2", "bp3"):
            ax.plot(t, tr[f"{comp}_{obj}_{m}"], color=colors[m], lw=1.6, label=labels[m])
        ax.set_title(f"{title}: {comp}")
        ax.set_xlabel("Day (0 = 5 July 2020)")
        ax.grid(alpha=0.3)
        if comp != "I":
            ax.set_yscale("log")
        if c == 0:
            ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(out / "fig_IRD_allcomp_vs_Ionly.png", dpi=140, bbox_inches="tight")
print(out / "fig_IRD_allcomp_vs_Ionly.png")
