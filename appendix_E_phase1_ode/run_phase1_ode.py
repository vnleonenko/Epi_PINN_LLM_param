"""Phase-1 PINN calibration on the first wave with and without the ODE term.

Each run trains the network on the first 185 days of the Saint Petersburg
series and evaluates the network output for I on days 185-214.

Example:
    python run_phase1_ode.py --lambda-ode 1.0 0.0 --seeds 42-51
"""

import argparse
import csv
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch

from pinn_sird import ARCHITECTURES, SIRDPINN

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "covid-19_Kouprianov.csv")
TRAIN_SIZE = 185
HORIZON = 30
INIT_PARAMS = {"beta": 0.1, "gamma": 0.1, "mu": 0.001}

FIELDS = ["arch", "lambda_ode", "seed", "epochs", "params_trained",
          "peak_day", "peak_height", "peak_day_window", "peak_height_window",
          "rmse_I", "mae_I", "beta", "gamma", "mu",
          "loss_total", "loss_data", "loss_ode", "seconds"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_one(job):
    arch, lam, seed, epochs = job
    torch.set_num_threads(1)
    df = pd.read_csv(DATA)
    S, I, R, D = (df[c].values.astype(float) for c in ("S", "I", "R", "D"))
    t = np.arange(len(S), dtype=float)
    N = S[0] + I[0] + R[0] + D[0]

    set_seed(seed)
    model = SIRDPINN(t, S, I, R, D, N, TRAIN_SIZE, INIT_PARAMS, ARCHITECTURES[arch])
    start = time.time()
    loss, l_data, l_ode = model.fit(epochs, lambda_ode=lam)
    seconds = time.time() - start

    I_pred = model.predict()[1]
    a, b = TRAIN_SIZE, TRAIN_SIZE + HORIZON
    err = I_pred[a:b] - I[a:b]
    beta, gamma, mu = (x.item() for x in model.rates())
    row = {
        "arch": arch, "lambda_ode": lam, "seed": seed, "epochs": epochs,
        "params_trained": lam > 0,
        "peak_day": int(np.argmax(I_pred)), "peak_height": float(I_pred.max()),
        "peak_day_window": int(a + np.argmax(I_pred[a:b])),
        "peak_height_window": float(I_pred[a:b].max()),
        "rmse_I": float(np.sqrt(np.mean(err ** 2))), "mae_I": float(np.mean(np.abs(err))),
        "beta": beta, "gamma": gamma, "mu": mu,
        "loss_total": loss, "loss_data": l_data, "loss_ode": l_ode,
        "seconds": round(seconds, 1),
    }
    return row, I_pred


def parse_seeds(text):
    if "-" in text:
        lo, hi = text.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in text.split(",")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lambda-ode", type=float, nargs="+", default=[1.0, 0.0])
    p.add_argument("--seeds", default="42-51")
    p.add_argument("--arch", default="128-256-128", choices=sorted(ARCHITECTURES))
    p.add_argument("--epochs", type=int, default=10_000)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--out", default=os.path.join(HERE, "results_rerun"))
    args = p.parse_args()

    curves_dir = os.path.join(args.out, "curves")
    os.makedirs(curves_dir, exist_ok=True)
    runs_path = os.path.join(args.out, "runs.csv")
    done = set()
    if os.path.exists(runs_path):
        prev = pd.read_csv(runs_path)
        done = {(r.arch, float(r.lambda_ode), int(r.seed), int(r.epochs)) for r in prev.itertuples()}
    jobs = [(args.arch, lam, s, args.epochs) for lam in args.lambda_ode
            for s in parse_seeds(args.seeds)
            if (args.arch, lam, s, args.epochs) not in done]

    new_file = not os.path.exists(runs_path)
    with open(runs_path, "a", newline="") as f, ProcessPoolExecutor(args.workers) as pool:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for row, I_pred in pool.map(run_one, jobs):
            writer.writerow(row)
            f.flush()
            name = f"I_{row['arch']}_lode{row['lambda_ode']:g}_seed{row['seed']}.npy"
            np.save(os.path.join(curves_dir, name), I_pred.astype(np.float32))
            print(f"{row['arch']} lambda_ode={row['lambda_ode']:g} seed={row['seed']}: "
                  f"peak day {row['peak_day']}, RMSE {row['rmse_I']:.0f}, "
                  f"beta {row['beta']:.4f} ({row['seconds']} s)", flush=True)


if __name__ == "__main__":
    main()
