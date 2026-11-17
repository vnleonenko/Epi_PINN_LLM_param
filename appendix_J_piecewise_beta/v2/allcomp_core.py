#!/usr/bin/env python3
"""Shared functions of the four-series piecewise-beta refit (manuscript Appendix G).

Same SIRD model, data, window (days 0-368), N, initial state, piecewise-constant
beta parameterization (0, 2, 3 breakpoints; softmax lengths, 14-day minimum),
bounds, solver and L-BFGS-B settings as run_r37.py (the I-only fits of
Table A14). Only the objective changes:

  primary  "SIRD4":  RSS over S, I, R, D with one variance (n = 4 x 369 = 1476),
                     AIC = n ln(RSS) + 2k, the first error model of Appendix F
                     (Table A12).
  secondary "IRDsep": I, R, D with one variance per compartment,
                     objective sum_c n_c ln(RSS_c), AIC = that + 2k.

This module only holds the model, objective, start and metric functions.
The fit reported in the manuscript is run by fit_piecewise_allcomp.py.
It reuses run_r37.py, config.json, data/covid-19_Kouprianov.csv and the
archived I-only results/ in the same folder.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
# Same folder as run_r37.py, config.json, data/ and the I-only results/.
DEFAULT_J = HERE

J = None  # run_r37 module, loaded in load_j()


def load_j(jdir: Path):
    global J
    spec = importlib.util.spec_from_file_location("run_r37", jdir / "run_r37.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    J = mod
    return mod


MODELS = [("constant", 1), ("bp2", 3), ("bp3", 4)]
OBJECTIVES = {
    "SIRD4": {"channels": [0, 1, 2, 3], "separate": False,
              "label": "S, I, R, D, one variance (as SEIR-D, Table A12)"},
    "IRDsep": {"channels": [1, 2, 3], "separate": True,
               "label": "I, R, D, one variance per compartment"},
}


def n_params(n_seg):
    return n_seg + 2 + (n_seg - 1)  # betas, gamma, mu, breakpoint dates


def score(traj, obs_mat, objective):
    spec = OBJECTIVES[objective]
    res = traj[spec["channels"]] - obs_mat[spec["channels"]]
    if spec["separate"]:
        rss_c = np.sum(res ** 2, axis=1)
        return float(np.sum(res.shape[1] * np.log(rss_c)))
    return float(np.sum(res ** 2))


def obj_fn(x, n_seg, y0, t_obs, obs_mat, population, settings, objective):
    betas, gamma, mu, z = J.unpack(x, n_seg)
    breaks = J.breaks_from_z(z, n_seg, settings["t_max"], settings["min_len"])
    traj = J.integrate(betas, breaks, gamma, mu, y0, t_obs, population,
                       settings["rtol"], settings["atol"])
    if traj is None or not np.isfinite(traj).all():
        return 1e30
    return score(traj, obs_mat, objective)


def job(args):
    (jdir, objective, name, n_seg, run_set, seed, x0) = args
    if J is None:
        load_j(Path(jdir))
    cfg = J.CFG
    t_obs, obs, y0 = J.load_observations()
    obs_mat = np.vstack([obs[c] for c in "SIRD"])
    population = float(cfg["series"]["N"])
    settings = J.settings_of(cfg)
    box = J.bounds_for(n_seg, cfg)
    x0 = np.clip(np.asarray(x0, float), [b[0] for b in box], [b[1] for b in box])
    t0 = time.perf_counter()
    result = minimize(
        obj_fn, x0,
        args=(n_seg, y0, t_obs, obs_mat, population, settings, objective),
        method=cfg["optimizer"]["method"], bounds=box,
        options={k: cfg["optimizer"][k] for k in ("maxiter", "maxfun", "ftol", "gtol")},
    )
    row = J.row_from_result(name, run_set, seed, n_seg, result, box,
                            settings["t_max"], settings["min_len"])
    row["objective"] = objective
    row["obj_value"] = row.pop("rss")
    row["seconds"] = time.perf_counter() - t0
    row["x"] = json.dumps([float(v) for v in result.x])
    row.pop("jac", None)
    print(f"[{objective}] {run_set} {name} seed {seed}: obj={row['obj_value']:.6e} "
          f"status={row['status']} nit={row['nit']} nfev={row['nfev']} {row['seconds']:.0f}s", flush=True)
    return row


def pieces(row, n_seg):
    betas = [float(row[f"beta{i}"]) for i in range(1, n_seg + 1)]
    breaks = [float(row[f"break{i}"]) for i in range(1, n_seg)
              if f"break{i}" in row and pd.notna(row[f"break{i}"])]
    return betas, breaks


def vec(betas, breaks, gamma, mu, cfg):
    return J.vector_from_pieces(betas, breaks, gamma, mu, cfg["series"]["t_max"],
                                cfg["breakpoints"]["min_segment_days"]).tolist()


def embed(betas, breaks, gamma, mu, n_target, cfg):
    """Split the longest segment(s) in half until n_target segments; copy beta."""
    t_max = cfg["series"]["t_max"]
    edges = [0.0] + list(breaks) + [t_max]
    betas = list(betas)
    while len(betas) < n_target:
        lengths = np.diff(edges)
        k = int(np.argmax(lengths))
        mid = edges[k] + lengths[k] / 2.0
        edges = edges[:k + 1] + [mid] + edges[k + 1:]
        betas = betas[:k] + [betas[k], betas[k]] + betas[k + 1:]
    return vec(betas, edges[1:-1], gamma, mu, cfg)


def metrics(name, row, n_seg, objective, t_obs, obs, y0, population, cfg):
    betas, breaks = pieces(row, n_seg)
    g, m = float(row["gamma"]), float(row["mu"])
    traj = J.integrate(betas, breaks, g, m, y0, t_obs, population,
                       cfg["solver"]["rtol"], cfg["solver"]["atol"])
    tight = J.integrate(betas, breaks, g, m, y0, t_obs, population, 1e-8, 1e-10)
    obs_mat = np.vstack([obs[c] for c in "SIRD"])
    res = traj - obs_mat
    rss = {c: float(np.sum(res[i] ** 2)) for i, c in enumerate("SIRD")}
    n = len(t_obs)
    k = n_params(n_seg)
    I = traj[1]
    peak = int(np.argmax(I))
    sst_I = float(np.sum((obs["I"] - obs["I"].mean()) ** 2))
    out = {
        "objective": objective, "model": name, "k": k,
        "start_won": f"{row['run_set']}:{row['seed']}", "status": row["status"],
        "message": row["message"], "nit": row["nit"], "nfev": row["nfev"],
        "n_at_bound": row["n_at_bound"], "bound_hits": row["bound_hits"],
        "gamma": g, "mu": m, "gamma_plus_mu": g + m,
        "infectious_days_1_over_gamma_mu": 1.0 / (g + m),
        "betas": json.dumps([round(b, 6) for b in betas]),
        "breaks": json.dumps([round(b, 2) for b in breaks]),
        "R0_segments": json.dumps([round(b / (g + m), 4) for b in betas]),
        "Reff_at_segment_start": json.dumps([
            round(b * traj[0, int(np.floor(s))] / population / (g + m), 4)
            for b, s in zip(betas, [0.0] + breaks)]),
        "rss_S": rss["S"], "rss_I": rss["I"], "rss_R": rss["R"], "rss_D": rss["D"],
        "rss_SIRD": sum(rss.values()),
        "rmse_I": float(np.sqrt(rss["I"] / n)), "rmse_R": float(np.sqrt(rss["R"] / n)),
        "rmse_D": float(np.sqrt(rss["D"] / n)), "rmse_S": float(np.sqrt(rss["S"] / n)),
        "R2_I": 1.0 - rss["I"] / sst_I,
        "AIC_SIRD4": 4 * n * np.log(sum(rss.values())) + 2 * k,
        "AIC_Ionly": n * np.log(rss["I"]) + 2 * k,
        "AIC_IRDsep": n * (np.log(rss["I"]) + np.log(rss["R"]) + np.log(rss["D"])) + 2 * k,
        "peak_day_I": peak, "peak_I": float(I[peak]),
        "g_early": J.growth(I, 0, 111), "g_accel": J.growth(I, 110, 151),
        "delta_I_110_150": float(I[150] - I[110]),
        "left_S_frac_368": float(1.0 - traj[0, -1] / population),
        "R_368": float(traj[2, -1]), "D_368": float(traj[3, -1]),
        "max_abs_N_error": float(np.max(np.abs(traj.sum(axis=0) - population))),
        "min_compartment": float(np.min(traj)),
        "tight_rel_rmse_I": None if tight is None else float(
            np.sqrt(np.mean((tight[1] - I) ** 2)) / np.sqrt(np.mean(I ** 2))),
    }
    out["accel"] = out["g_accel"] - out["g_early"]
    return out, traj


