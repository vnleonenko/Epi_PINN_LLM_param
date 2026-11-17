"""Summarize runs.csv: mean and SD per configuration and the paired comparison
of each configuration with lambda_ode = 0 (same architecture, same seed)."""

import argparse
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OBS_DAY, OBS_PEAK = 199, 104932
METRICS = ["rmse_I", "mae_I", "peak_day", "peak_height", "peak_day_window",
           "peak_height_window", "beta", "gamma", "mu"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default=os.path.join(HERE, "results", "runs.csv"))
    args = p.parse_args()
    df = pd.read_csv(args.runs)
    out_dir = os.path.dirname(args.runs)

    rows = []
    for (arch, lam), g in df.groupby(["arch", "lambda_ode"]):
        row = {"arch": arch, "lambda_ode": lam, "n": len(g)}
        for m in METRICS:
            row[f"{m}_mean"] = g[m].mean()
            row[f"{m}_sd"] = g[m].std()
        row["peak_day_min"] = g["peak_day"].min()
        row["peak_day_max"] = g["peak_day"].max()
        row["rmse_min"] = g["rmse_I"].min()
        row["rmse_max"] = g["rmse_I"].max()
        base = df[(df.arch == arch) & (df.lambda_ode == 0)].set_index("seed")
        if lam != 0 and len(base):
            pair = g.merge(base.reset_index(), on="seed", suffixes=("", "_b"))
            row["pairs"] = len(pair)
            row["pairs_lower_rmse"] = int((pair.rmse_I < pair.rmse_I_b).sum())
            row["pairs_lower_mae"] = int((pair.mae_I < pair.mae_I_b).sum())
            err = (pair.peak_day - OBS_DAY).abs()
            err_b = (pair.peak_day_b - OBS_DAY).abs()
            row["pairs_closer_peak_day"] = int((err < err_b).sum())
            row["pairs_tied_peak_day"] = int((err == err_b).sum())
            h_err = (pair.peak_height - OBS_PEAK).abs()
            h_err_b = (pair.peak_height_b - OBS_PEAK).abs()
            row["pairs_closer_peak_height"] = int((h_err < h_err_b).sum())
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    lines = ["# Phase-1 PINN with and without the ODE term", "",
             "Observed: peak on day 199 (104 932 active cases). Errors are for the network output "
             "for I on days 185-214. Peak day: argmax of the network output over days 0-368; "
             "window peak: argmax over days 185-214 (the definition used in Section 4.1.1).", "",
             "| arch | lambda_ode | n | RMSE | RMSE range | MAE | peak day | peak day range | peak height | "
             "window peak day | beta | gamma | mu | pairs better than lambda_ode=0: RMSE / MAE / peak day / peak height |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        pairs = (f"{r['pairs_lower_rmse']} / {r['pairs_lower_mae']} / {r['pairs_closer_peak_day']} "
                 f"(ties {r['pairs_tied_peak_day']}) / {r['pairs_closer_peak_height']} of {r['pairs']}"
                 ) if "pairs" in r else ""
        lines.append(
            f"| {r['arch']} | {r['lambda_ode']:g} | {r['n']} "
            f"| {r['rmse_I_mean']:.0f} ± {r['rmse_I_sd']:.0f} | {r['rmse_min']:.0f}-{r['rmse_max']:.0f} "
            f"| {r['mae_I_mean']:.0f} ± {r['mae_I_sd']:.0f} "
            f"| {r['peak_day_mean']:.1f} ± {r['peak_day_sd']:.1f} | {r['peak_day_min']}-{r['peak_day_max']} "
            f"| {r['peak_height_mean']:.0f} ± {r['peak_height_sd']:.0f} "
            f"| {r['peak_day_window_mean']:.1f} ± {r['peak_day_window_sd']:.1f} "
            f"| {r['beta_mean']:.4f} ± {r['beta_sd']:.4f} | {r['gamma_mean']:.4f} ± {r['gamma_sd']:.4f} "
            f"| {r['mu_mean']:.5f} ± {r['mu_sd']:.5f} | {pairs} |")
    lines += ["", "With lambda_ode = 0 the parameters receive no gradient and stay at their "
              "initial values (0.1, 0.1, 0.001); they are listed only for completeness."]
    text = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
