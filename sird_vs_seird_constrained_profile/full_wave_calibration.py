"""Preliminary experiment: calibration of SIRD and SEIR-D on the full epidemic curve, compared by AIC.

Question
--------
SIRD has no latent period: an infected person is infectious at once, so outbreaks rise
sharply. SEIR-D inserts the exposed compartment E, a delay that smooths growth; COVID-19
and influenza both have a non-zero incubation period. Does adding E improve the fit to
the shape of the real curve enough to justify its extra parameter, sigma, by AIC?

What the script computes
------------------------
1. PINN calibration, as specified. One ``PINNAgent.train(...)`` call per model, the two
   calls differing only in ``pinn_class`` (agents/PINN_seird.py: EINN_PINN_SIRD,
   EINN_PINN_SEIRD). PINNAgent itself is unchanged, which is the model-agnostic claim.
   AIC = n ln(RSS) + 2k of the PINN output, RSS unweighted over S, I, R, D, k = 3 / 4
   (utils/model_selection.py) -> aic_comparison.csv, specified columns first.
2. The same criteria for the ODE solution with the PINN-recovered parameters (*_ode*).
3. Direct ODE least squares -> ode_lsq_reference.csv: the maximum-likelihood fit of each
   compartmental model under four error models: S, I, R, D with one variance (the
   specified RSS); I, R, D with one variance (the data's S = N - I - R - D repeats
   them); I, R, D with a variance each; I alone (the shape of the infected curve).
   SEIR-D is fitted with sigma free and with the incubation period held to 2-14 days,
   and E(0) is profiled over several E0 / I0 ratios.

Why items 2 and 3
-----------------
The PINN output is a neural network fitted to the data with the ODE as a penalty. Here
the data term dominates: both PINNs follow the data whatever their equations, and the
seed-to-seed spread of the PINN-output AIC is far larger than the 2 units one parameter
costs. That AIC does not by itself compare the curve shapes of the two models; the
ODE-based criteria do. Column meanings and caveats: model_selection_results/README.md.

Setup
-----
* Data: real_datasets/covid-19_Kouprianov.csv - Saint Petersburg, active cases
  (ACTIVE.sk), 5 July 2020 - 8 July 2021, 369 days, N = S + I + R + D = 6,000,000.
* Full curve: train_size = 369, no held-out part (calibration, not forecasting).
* PINN: 10,000 epochs; lambda_data = 1, lambda_ode = 1, lambda_ic = 0.1, lambda_bc = 0
  (the data reach the end of the wave, so I(end) = 0 is not imposed). Epidemic
  parameters are trainable with their own Adam rate 1e-2 (network 1e-4), and the state
  with the lowest loss is returned. Seeds 0..9 per model; a seed initializes the hidden
  layers of both networks identically, so the models are also compared seed by seed.
* Initial guess as in SIRD_calibration.ipynb: beta0 from the early growth rate,
  gamma0 = 0.1, mu0 = 0.01; SEIR-D adds sigma0 = 1/5.2 per day and E(0) = 0.5 I0.

Outputs in model_selection_results/ (model_selection_results_quick/ with --quick):
aic_comparison.csv, aic_runs.csv, e0_sensitivity.csv, ode_lsq_reference.csv,
run_config.json, pinn_agent_plots/, logs/, and figures/ with each figure as .png (300 dpi),
.pdf and .tiff (600 dpi): i_overlay (main figure: PINN output and Delta AIC against E(0)),
its panels on their own (panel_A_pinn_output, panel_D_delta_aic_vs_E0) with the ODE fits to
I(t) (panel_B_ode_fit_to_I), delta_aic_summary, spec_criterion_fit, full_wave_fit,
parameters_by_seed, parameter_convergence.

Run from the repository root:
    ./venv/bin/python "Phase 1 (model calibration)/full_wave_calibration.py"
    ./venv/bin/python "Phase 1 (model calibration)/full_wave_calibration.py" --quick   # smoke test
"""
import argparse
import contextlib
import hashlib
import io
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent  # repository root: agents/PINN_const.py, agents/PINNAgent.py
for _p in (REPO_ROOT, HERE):  # HERE first: model_selection.py, PINN_seird.py
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from model_selection import (  # noqa: E402
    compute_aic,
    compute_aic_separate_variances,
    compute_aicc,
    compute_rss,
    count_residuals,
)

DATASETS = {"kouprianov": HERE / "data" / "covid-19_Kouprianov.csv"}
SIRD, SEIRD = "SIRD-PINN", "SEIR-D-PINN"
K = {SIRD: 3, SEIRD: 4}
PARAMS = {SIRD: ("beta", "gamma", "mu"), SEIRD: ("beta", "sigma", "gamma", "mu")}
SLUG = {SIRD: "sird", SEIRD: "seird"}
ODE_NAME = {SIRD: "SIRD", SEIRD: "SEIR-D"}
SIGMA0 = 1 / 5.2  # mean incubation period 5.2 days
PARAM_LR = 1e-2   # Adam rate of the epidemic parameters; the network keeps 1e-4
LAMBDAS = {"lambda_data": 1.0, "lambda_ode": 1.0, "lambda_ic": 0.1, "lambda_bc": 0.0}
OBSERVED = ("S", "I", "R", "D")
# error models: column suffix -> (compartments, one variance per compartment, description)
CRITERIA = {
    "": (("S", "I", "R", "D"), False, "S, I, R, D, one variance (specified)"),
    "_IRD": (("I", "R", "D"), False, "I, R, D, one variance"),
    "_IRDsep": (("I", "R", "D"), True, "I, R, D, one variance per compartment"),
    "_I": (("I",), False, "I only (shape of the infected curve)"),
}
CONVERGENCE_WINDOW, CONVERGENCE_TOL = 1000, 0.01  # epochs; largest relative parameter change
# direct ODE least squares
LSQ_BOUNDS = {"beta": (1e-4, 5.0), "sigma": (1e-3, 1e2), "gamma": (1e-4, 2.0), "mu": (1e-7, 0.5)}
SIGMA_FREE, SIGMA_2_14 = "free", "incubation 2-14 d"
E0_FIXED, E0_QUASI = "fixed", "self-consistent"          # dI/dt(0) = 0
E0_GROWTH, E0_ESTAT = "growth-consistent", "E-stationary"  # the growing wave; dE/dt(0) = 0
E0_RULES = (E0_QUASI, E0_GROWTH, E0_ESTAT)
GROWTH_WINDOW = 10  # days of data used for the early growth rate
INCUBATION_2_14 = (1 / 14, 1 / 2)  # sigma for a mean incubation period of 2 to 14 days
LSQ_E0_PROFILE = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0)
ODE_OPTIONS = {"rtol": 1e-8, "atol": 1e-4, "mxstep": 20_000}
FIGURES_DIR = "figures"
FIGURES = ("i_overlay", "panel_A_pinn_output", "panel_B_ode_fit_to_I", "panel_D_delta_aic_vs_E0",
           "delta_aic_summary", "spec_criterion_fit", "full_wave_fit", "parameters_by_seed", "parameter_convergence")
# PNG for reports, vector PDF, and TIFF for journal submission (RGB, lossless LZW; journals ask for >= 300 dpi)
FIGURE_FORMATS = ("png", "pdf", "tiff")
PNG_DPI, TIFF_DPI = 300, 600
# what a run writes; deleted at the start of the next run into the same directory
GENERATED = ("aic_comparison.csv", "aic_runs.csv", "e0_sensitivity.csv", "ode_lsq_reference.csv", "run_config.json")
GENERATED_DIRS = ("logs", "pinn_agent_plots", FIGURES_DIR)
TRACKED_FILES = ("appendix_F_seird_vs_sird/full_wave_calibration.py", "appendix_F_seird_vs_sird/PINN_seird.py",
                 "agents/PINN_const.py", "agents/PINNAgent.py", "appendix_F_seird_vs_sird/model_selection.py")


# ----------------------------------------------------------------------------- data and models

def load_series(path):
    """t, S, I, R, D arrays and N = S + I + R + D of the first row."""
    df = pd.read_csv(path)
    S, I, R, D = (df[c].to_numpy(dtype=float) for c in OBSERVED)
    return {"t": np.arange(len(S), dtype=float), "S": S, "I": I, "R": R, "D": D,
            "N": float(S[0] + I[0] + R[0] + D[0])}


def initial_guess(I):
    """beta0, gamma0, mu0 exactly as calibrate_sird() in SIRD_calibration.ipynb."""
    growth = [np.log(I[i] / I[i - 1]) for i in range(1, min(5, len(I))) if I[i - 1] > 0]
    beta0 = float(np.clip(np.mean(growth) + 0.1, 0.1, 0.8)) if growth else 0.3
    return {"beta": beta0, "gamma": 0.1, "mu": 0.01}


def sird_rhs(y, t, beta, gamma, mu, N):
    S, I, R, D = y
    infection = beta * S * I / N
    return [-infection, infection - (gamma + mu) * I, gamma * I, mu * I]


def seird_rhs(y, t, beta, sigma, gamma, mu, N):
    S, E, I, R, D = y
    infection = beta * S * I / N
    return [-infection, infection - sigma * E, sigma * E - (gamma + mu) * I, gamma * I, mu * I]


def early_growth(data, window=GROWTH_WINDOW):
    """Growth rate of the observed I over the first ``window`` days, per day."""
    return float(np.log(data["I"][window] / data["I"][0]) / window)


def resolve_e0(params, data, e0):
    """E(0) as a number, or computed from the fitted rates by one of the E0_RULES.

    E(0) is not observed, and an assumed value leaves the model inconsistent at t = 0. Three
    rules remove that freedom, each by a different reading of "consistent":

    * E0_QUASI  : E(0) = I(0)(gamma + mu)/sigma, so dI/dt(0) = 0 and the curve starts flat;
    * E0_GROWTH : E(0) = I(0)(r + gamma + mu)/sigma with r the observed early growth rate,
                  the E/I ratio of a wave already growing at rate r;
    * E0_ESTAT  : E(0) = beta S(0) I(0) / (N sigma), so dE/dt(0) = 0 instead.

    The three give different answers, which is itself a result: see the delta report.
    """
    if e0 == E0_QUASI:
        return float(data["I"][0] * (params["gamma"] + params["mu"]) / params["sigma"])
    if e0 == E0_GROWTH:
        rate = early_growth(data)
        return float(data["I"][0] * (rate + params["gamma"] + params["mu"]) / params["sigma"])
    if e0 == E0_ESTAT:
        return float(params["beta"] * data["S"][0] * data["I"][0] / (data["N"] * params["sigma"]))
    return float(e0)


def integrate(model, params, data, E0):
    """ODE solution with the given parameters from the data's first state, in the data layout.

    For SEIR-D the initial state is S0 - E0, E0, I0, R0, D0 (the data's S is S + E, see
    agents/PINN_seird.py) and the returned S channel is S + E; E is returned separately.
    ``E0`` may be E0_QUASI, in which case it is computed from the parameters (see resolve_e0).
    """
    t, N = data["t"], data["N"]
    S0, I0, R0, D0 = (data[c][0] for c in OBSERVED)
    if model == SIRD:
        S, I, R, D = odeint(sird_rhs, [S0, I0, R0, D0], t,
                            args=(params["beta"], params["gamma"], params["mu"], N), **ODE_OPTIONS).T
        return {"S": S, "I": I, "R": R, "D": D}
    E0 = resolve_e0(params, data, E0)
    S, E, I, R, D = odeint(seird_rhs, [S0 - E0, E0, I0, R0, D0], t,
                           args=(params["beta"], params["sigma"], params["gamma"], params["mu"], N), **ODE_OPTIONS).T
    return {"S": S + E, "I": I, "R": R, "D": D, "E": E}


def ever_infected_share(solution, N):
    """1 - S(end) / N: the share of the population infected by the last day (for SEIR-D, S excludes E)."""
    return float(1 - (solution["S"][-1] - solution.get("E", np.zeros(1))[-1]) / N)


def criteria(data, pred, k, prefix=""):
    """RSS and AIC of a prediction under every error model in CRITERIA, and AICc of the specified one.

    Columns: rss / aic (S, I, R, D, the specified criterion), rss_IRD / aic_IRD,
    aic_IRDsep, rss_I / aic_I, rss_S, rss_R, rss_D and aicc, each with ``prefix``.
    """
    T = count_residuals(data, ("I",))
    rss = {c: compute_rss(data, pred, (c,)) for c in OBSERVED}
    row = {"n_residuals": count_residuals(data, OBSERVED), "n_points": T}
    for suffix, (comps, separate, _) in CRITERIA.items():
        if separate:
            row[f"{prefix}aic{suffix}"] = compute_aic_separate_variances([rss[c] for c in comps], [T] * len(comps), k)
        else:
            total = sum(rss[c] for c in comps)
            row[f"{prefix}rss{suffix}"] = total
            row[f"{prefix}aic{suffix}"] = compute_aic(total, T * len(comps), k)
    row.update({f"{prefix}rss_{c}": rss[c] for c in ("S", "R", "D")})
    row[f"{prefix}aicc"] = compute_aicc(row[f"{prefix}aic"], row["n_residuals"], k)
    return row


# ----------------------------------------------------------------------------- direct ODE least squares

def lsq_starts(model, extra=(), quick=False):
    """Starting points for the direct ODE fit: a grid, then ``extra`` (missing or non-finite sigma -> sigma0)."""
    betas, gammas, sigmas = (((0.1,), (0.1,), (0.2, 10.0)) if quick
                             else ((0.04, 0.1, 0.3), (0.03, 0.1), (0.05, 0.2, 1.0, 10.0)))
    grid = [{"beta": b, "gamma": g, "mu": 0.002, "sigma": s}
            for b in betas for g in gammas for s in (sigmas if model == SEIRD else (SIGMA0,))]
    extra = [{"sigma": SIGMA0, **{p: float(v) for p, v in e.items() if np.isfinite(v) and v > 0}} for e in extra]
    return grid + extra


def fit_ode(model, data, E0, suffix, starts, sigma_bounds=None):
    """Least-squares fit of one compartmental model under the error model CRITERIA[suffix].

    Parameters are fitted in log space within LSQ_BOUNDS (sigma within ``sigma_bounds``
    if given) from every start, and the fit with the lowest AIC is kept. With one
    variance per compartment the weights 1 / variance are re-estimated from the fit
    (iteratively reweighted least squares; its fixed point maximizes that likelihood).
    """
    comps, separate, _ = CRITERIA[suffix]
    names = PARAMS[model]
    bounds = dict(LSQ_BOUNDS, **({"sigma": tuple(sigma_bounds)} if sigma_bounds else {}))
    lo = np.log([bounds[p][0] for p in names])
    hi = np.log([bounds[p][1] for p in names])
    T = len(data["t"])

    n_residuals = len(data["t"]) * len(comps)

    def feasible(x):
        """E(0) above S(0) is not a state: the optimizer must be pushed away from it."""
        return model == SIRD or resolve_e0(dict(zip(names, np.exp(x))), data, E0) < data["S"][0]

    def solve(x):
        return integrate(model, dict(zip(names, np.exp(x))), data, E0)

    fits = []
    for start in starts:
        x = np.clip(np.log([start[p] for p in names]), lo + 1e-9, hi - 1e-9)
        weights = dict.fromkeys(comps, 1.0)
        for _ in range(8 if separate else 1):
            def residuals(z, w=weights):
                if not feasible(z):
                    return np.full(n_residuals, 1e7)
                sol = solve(z)
                return np.concatenate([np.sqrt(w[c]) * (sol[c] - data[c]) for c in comps])

            x = least_squares(residuals, x, bounds=(lo, hi), xtol=1e-12, ftol=1e-12, max_nfev=3000).x
            if separate:
                sol = solve(x)
                weights = {c: T / max(float(np.sum((sol[c] - data[c]) ** 2)), 1e-300) for c in comps}
        fits.append((criteria(data, solve(x), len(names))[f"aic{suffix}"], x))
    aic, x = min(fits, key=lambda f: f[0])
    solution = solve(x)
    return {"params": dict(zip(names, np.exp(x))), "aic": aic, "solution": solution,
            "criteria": criteria(data, solution, len(names)),
            "at_bound": [p for p, v, a, b in zip(names, x, lo, hi) if min(v - a, b - v) < 1e-3],
            "starts": len(starts), "starts_within_1": sum(a <= aic + 1 for a, _ in fits)}


def lsq_jobs(data_path, e0_factor, e0_profile, extra_starts, quick=False):
    """Direct fits to run: SIRD; SEIR-D with sigma free over ``e0_profile``; with a 2-14 d latent
    period; and with E(0) computed from the fitted rates instead of assumed (E0_QUASI)."""
    jobs = []
    for suffix in CRITERIA:
        jobs.append({"model": SIRD, "suffix": suffix, "sigma_range": "", "sigma_bounds": None,
                     "e0_factor": None, "e0_mode": E0_FIXED})
        jobs += [{"model": SEIRD, "suffix": suffix, "sigma_range": SIGMA_FREE, "sigma_bounds": None,
                  "e0_factor": f, "e0_mode": E0_FIXED} for f in e0_profile]
        jobs.append({"model": SEIRD, "suffix": suffix, "sigma_range": SIGMA_2_14, "sigma_bounds": INCUBATION_2_14,
                     "e0_factor": e0_factor, "e0_mode": E0_FIXED})
        jobs += [{"model": SEIRD, "suffix": suffix, "sigma_range": SIGMA_FREE, "sigma_bounds": None,
                  "e0_factor": None, "e0_mode": rule} for rule in E0_RULES]
        jobs.append({"model": SEIRD, "suffix": suffix, "sigma_range": SIGMA_2_14, "sigma_bounds": INCUBATION_2_14,
                     "e0_factor": None, "e0_mode": E0_QUASI})
    for job in jobs:
        job.update(data_path=str(data_path), starts=lsq_starts(job["model"], extra_starts.get(job["model"], ()), quick))
    return jobs


def run_lsq_job(job):
    """One direct fit -> {"key", "row", "solution"}."""
    data = load_series(Path(job["data_path"]))
    model, suffix = job["model"], job["suffix"]
    if model == SIRD:
        E0 = 0.0
    elif job["e0_mode"] in E0_RULES:
        E0 = job["e0_mode"]
    else:
        E0 = job["e0_factor"] * data["I"][0]
    fit = fit_ode(model, data, E0, suffix, job["starts"], job["sigma_bounds"])
    comps, separate, description = CRITERIA[suffix]
    p, sol = fit["params"], fit["solution"]
    I = sol["I"]
    e0_value = np.nan if model == SIRD else resolve_e0(p, data, E0)
    # how far the fitted curve drops below the data in the first weeks, and the slope it starts with
    window = min(60, len(I))
    ratio = I[:window] / data["I"][:window]
    dip_day = int(np.argmin(ratio))
    outflow = (p["gamma"] + p["mu"]) * data["I"][0]
    inflow = p["sigma"] * e0_value if model == SEIRD else p["beta"] * data["S"][0] * data["I"][0] / data["N"]
    row = {"criterion": description, "compartments": "+".join(comps),
           "error_model": "one variance per compartment" if separate else "one variance",
           "model": ODE_NAME[model], "sigma_range": job["sigma_range"], "e0_mode": job["e0_mode"],
           "E0_over_I0": np.nan if model == SIRD else e0_value / data["I"][0],
           "E0": np.nan if model == SIRD else e0_value,
           "k": K[model], "n": len(data["t"]) * len(comps),
           "RSS": fit["criteria"]["rss_IRD" if separate else f"rss{suffix}"], "AIC": fit["aic"],
           **{name: p.get(name, np.nan) for name in ("beta", "sigma", "gamma", "mu")},
           "R0": p["beta"] / (p["gamma"] + p["mu"]),
           "incubation_days": 1 / p["sigma"] if "sigma" in p else np.nan,
           "at_bound": ",".join(fit["at_bound"]), "peak_day": int(np.argmax(I)), "peak_I": float(I.max()),
           "R_end": float(sol["R"][-1]), "D_end": float(sol["D"][-1]),
           "ever_infected_share": ever_infected_share(sol, data["N"]),
           "dip_day": dip_day, "dip_ratio": float(ratio[dip_day]),
           "dI_dt0": float(inflow - outflow),
           "starts": fit["starts"], "starts_within_1": fit["starts_within_1"]}
    return {"key": (model, suffix, job["sigma_range"], job["e0_factor"], job["e0_mode"]),
            "row": row, "solution": fit["solution"]}


def lsq_reference(jobs, workers):
    """Run the direct fits -> (table with delta_AIC_vs_SIRD per criterion, {key: {"row", "solution"}})."""
    if workers <= 1:
        results = [run_lsq_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            results = list(pool.map(run_lsq_job, jobs))
    table = pd.DataFrame([r["row"] for r in results])
    sird_aic = table[table["model"] == ODE_NAME[SIRD]].set_index("criterion")["AIC"]
    table.insert(table.columns.get_loc("AIC") + 1, "delta_AIC_vs_SIRD", table["AIC"] - table["criterion"].map(sird_aic))
    fits = {r["key"]: {"row": table.iloc[i].to_dict(), "solution": r["solution"]} for i, r in enumerate(results)}
    return table, fits


# ----------------------------------------------------------------------------- one PINN calibration run

class CaptureModel:
    """PINNAgent ``pinn_class`` that keeps a reference to the model it builds.

    PINNAgent.train returns parameters, losses and plot paths, not the network; the
    reference lets this script evaluate the trained trajectories without changing
    PINNAgent.
    """

    def __init__(self, factory):
        self.factory = factory
        self.models = []

    def __call__(self, **kwargs):
        model = self.factory(**kwargs)
        self.models.append(model)
        return model


def _param_drift(history, names, window=CONVERGENCE_WINDOW):
    """Largest relative change of any parameter over the last ``window`` epochs of ``params_history``."""
    if len(history) < 2:
        return np.nan
    last = history[-1]
    earlier = [h for h in history if h["epoch"] <= last["epoch"] - window]
    ref = earlier[-1] if earlier else history[0]
    return max(abs(last[p] - ref[p]) / max(abs(last[p]), 1e-12) for p in names)


def ode_consistency(model, t):
    """RMS of each ODE residual over the RMS of the network's own derivative (0 = the ODE holds)."""
    import torch

    tb = torch.tensor(t, dtype=torch.float32).reshape(-1, 1).requires_grad_(True)
    out = model.compute_ode_residual(tb)
    residuals, states = (out[0], out[1]) if isinstance(out[0], tuple) else (out[:4], out[4:])
    names = "SEIRD" if len(states) == 5 else "SIRD"
    result = {}
    for name, f, x in zip(names, residuals, states):
        dx = torch.autograd.grad(x.sum(), tb, retain_graph=True)[0].reshape(-1)
        rms_f, rms_dx = f.detach().pow(2).mean().sqrt(), dx.detach().pow(2).mean().sqrt()
        result[f"ode_rel_residual_{name}"] = float(rms_f / rms_dx.clamp_min(1e-12))
    return result


def _calibrate(spec, row, started):
    import torch

    from PINN_seird import EINN_PINN_SEIRD, EINN_PINN_SIRD
    from agents.PINNAgent import PINNAgent

    torch.set_num_threads(spec["threads"])
    torch.manual_seed(spec["seed"])
    np.random.seed(spec["seed"])
    data = load_series(Path(spec["data_path"]))
    name, init = spec["model"], spec["init"]
    E0 = spec["e0_factor"] * data["I"][0]
    row["E0"] = E0 if name == SEIRD else np.nan

    switches = {"trainable_params": True, "param_lr": spec["param_lr"], "restore_best": spec["restore_best"]}
    if name == SIRD:
        factory = partial(EINN_PINN_SIRD, **switches)
    else:
        factory = partial(EINN_PINN_SEIRD, E0_assumed=E0, init_params={"sigma": spec["sigma0"]}, **switches)
    capture = CaptureModel(factory)
    agent = PINNAgent(pinn_class=capture, n_epoch=spec["epochs"], device="cpu",
                      results_dir=spec["plots_dir"], verbose=False, **spec["lambdas"])

    with open(Path(spec["log_dir"]) / f"{spec['tag']}.log", "w", encoding="utf-8") as log, \
            contextlib.redirect_stdout(log):
        result = agent.train(beta=init["beta"], gamma=init["gamma"], mu=init["mu"],
                             t=data["t"], S=data["S"], I=data["I"], R=data["R"], D=data["D"],
                             population=data["N"], train_size=len(data["t"]), tag=spec["tag"])
    row["seconds"] = round(time.time() - started, 1)
    if not result.get("success"):
        row["error"] = result.get("error")
        return {"row": row, "curves": None}

    model = capture.models[-1]
    params = model.params.get_params_dict()
    S_obs, I, R, D = (x.numpy().astype(float) for x in model.predict(data["t"]))
    pinn = {"S": S_obs, "I": I, "R": R, "D": D}
    if not (all(np.isfinite(v) for v in params.values()) and all(np.all(np.isfinite(x)) for x in pinn.values())):
        row["error"] = "non-finite parameters or predictions"
        return {"row": row, "curves": None}

    ode = integrate(name, params, data, E0)
    best = model.best_epoch
    drift = _param_drift(model.params_history, PARAMS[name])
    row.update(success=True, best_epoch=best, param_drift=drift, converged=bool(drift < CONVERGENCE_TOL))
    row.update({p: params.get(p, np.nan) for p in ("beta", "sigma", "gamma", "mu")})
    row["R0"] = params["beta"] / (params["gamma"] + params["mu"])
    row["incubation_days"] = 1 / params["sigma"] if name == SEIRD else np.nan
    row.update(criteria(data, pinn, K[name]))
    row.update(criteria(data, ode, K[name], prefix="ode_"))
    row.update(ode_consistency(model, data["t"]))
    row.update(final_loss=model.losses[best], final_loss_data=model.losses_data[best],
               final_loss_ode=model.losses_ode[best],
               peak_day_pinn=int(np.argmax(I)), peak_I_pinn=float(I.max()),
               peak_day_ode=int(np.argmax(ode["I"])), peak_I_ode=float(ode["I"].max()),
               ode_ever_infected_share=ever_infected_share(ode, data["N"]))
    curves = {"pinn": pinn, "ode": ode, "history": model.params_history}
    if name == SEIRD:
        curves["E"] = model.predict_seird(data["t"])[1].numpy().astype(float)
    return {"row": row, "curves": curves}


def run_one(spec):
    """Calibrate one model with one seed through PINNAgent -> {"row", "curves"}; a failure becomes a row."""
    started = time.time()
    row = {"model": spec["model"], "seed": spec["seed"], "main": spec["main"], "e0_factor": spec["e0_factor"],
           "k": K[spec["model"]], "success": False}
    try:
        Path(spec["log_dir"]).mkdir(parents=True, exist_ok=True)
        return _calibrate(spec, row, started)
    except Exception as exc:  # noqa: BLE001 - one failed run must not stop the others
        row.update(error=f"{type(exc).__name__}: {exc}", seconds=round(time.time() - started, 1))
        with contextlib.suppress(OSError):
            with open(Path(spec["log_dir"]) / f"{spec['tag']}.log", "a", encoding="utf-8") as log:
                log.write("\n" + traceback.format_exc())
        return {"row": row, "curves": None}


def _failed(spec, exc):
    return {"row": {"model": spec["model"], "seed": spec["seed"], "main": spec["main"],
                    "e0_factor": spec["e0_factor"], "k": K[spec["model"]], "success": False,
                    "error": f"{type(exc).__name__}: {exc}"}, "curves": None}


def run_all(specs, workers):
    """Run every spec, in parallel processes when workers > 1; print progress; never raises."""
    results = []

    def report(result):
        r = result["row"]
        tail = (f"AIC {r['aic']:.1f}, AIC_ode {r['ode_aic']:.1f}, beta {r['beta']:.4f}, gamma {r['gamma']:.4f}, "
                f"mu {r['mu']:.5f}, R0 {r['R0']:.3f}, drift {r['param_drift']:.1e}" if r["success"]
                else f"FAILED: {r.get('error')}")
        print(f"[{len(results)}/{len(specs)}] {r['model']} seed {r['seed']} E0/I0={r['e0_factor']:g} "
              f"({r.get('seconds', '?')} s): {tail}", flush=True)

    if workers <= 1:
        for spec in specs:
            results.append(run_one(spec))
            report(results[-1])
        return results
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(run_one, spec): spec for spec in specs}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - the worker process itself died
                result = _failed(futures[future], exc)
            results.append(result)
            report(result)
    return results


# ----------------------------------------------------------------------------- tables

# PINN-run columns summarized per model -> label in aic_comparison.csv
SUMMARY_CRITERIA = {"aic": "AIC", "aic_IRD": "AIC_IRD", "aic_IRDsep": "AIC_IRDsep", "aic_I": "AIC_I",
                    "ode_aic": "AIC_ode", "ode_aic_IRD": "AIC_ode_IRD", "ode_aic_IRDsep": "AIC_ode_IRDsep",
                    "ode_aic_I": "AIC_ode_I"}
SUMMARY_RSS = {"rss_IRD": "RSS_IRD", "rss_I": "RSS_I", "ode_rss": "RSS_ode", "ode_rss_IRD": "RSS_ode_IRD",
               "ode_rss_I": "RSS_ode_I"}


def geometric_mean(values):
    values = np.asarray(values, dtype=float)
    return float(np.exp(np.mean(np.log(values)))) if values.size else np.nan


def paired_difference(g, ref, col):
    """Seed-paired differences g[col] - ref[col] -> mean, SD, SE, number of pairs."""
    common = g.index.intersection(ref.index)
    d = (g.loc[common, col] - ref.loc[common, col]).to_numpy(dtype=float)
    if d.size == 0:
        return np.nan, np.nan, np.nan, 0
    sd = float(d.std(ddof=1)) if d.size > 1 else np.nan
    return float(d.mean()), sd, sd / np.sqrt(d.size) if d.size > 1 else np.nan, int(d.size)


def summarize(runs):
    """One row per model over the main runs, the specified columns first.

    RSS is the geometric mean over seeds, so that AIC (the mean over seeds) equals
    n ln(RSS) + 2k in every row. delta_* are seed-paired differences from SIRD-PINN with
    their SD and SE; delta_*_resolved is |mean| > 2 SE.
    """
    main = runs[runs["main"] & runs["success"]]
    groups = {m: main[main["model"] == m].set_index("seed") for m in (SIRD, SEIRD)}
    rows = []
    for model, g in groups.items():
        if g.empty:
            continue
        rival = groups[SEIRD if model == SIRD else SIRD]
        row = {"model": model, "k": K[model], "n": int(g["n_residuals"].iloc[0]),
               "RSS": geometric_mean(g["rss"]), "AIC": float(g["aic"].mean())}
        row.update({p: float(g[p].mean()) for p in ("beta", "sigma", "gamma", "mu")})
        row.update(n_points=int(g["n_points"].iloc[0]), seeds=len(g),
                   converged_seeds=int(g["converged"].astype(bool).sum()),
                   AIC_sd=float(g["aic"].std(ddof=1)) if len(g) > 1 else np.nan, AICc=float(g["aicc"].mean()))
        for col, label in SUMMARY_CRITERIA.items():
            if col != "aic":
                row[label] = float(g[col].mean())
            mean, sd, se, _ = paired_difference(g, groups[SIRD], col)
            row.update({f"delta_{label}": mean, f"delta_{label}_sd": sd, f"delta_{label}_se": se})
            row[f"delta_{label}_resolved"] = (bool(abs(mean) > 2 * se)
                                              if model != SIRD and np.isfinite(se) and se > 0 else np.nan)
            common = g.index.intersection(rival.index)
            row[f"{label}_lower_in_seeds"] = f"{int((g.loc[common, col] < rival.loc[common, col]).sum())}/{len(common)}"
        row.update({label: geometric_mean(g[col]) for col, label in SUMMARY_RSS.items()})
        row.update({f"{p}_sd": float(g[p].std(ddof=1)) if len(g) > 1 else np.nan
                    for p in ("beta", "sigma", "gamma", "mu", "R0")})
        row.update(R0=float(g["R0"].mean()), incubation_days=float(g["incubation_days"].mean()),
                   peak_day_pinn=float(g["peak_day_pinn"].mean()), peak_day_ode=float(g["peak_day_ode"].mean()),
                   final_loss_data=float(g["final_loss_data"].mean()), final_loss_ode=float(g["final_loss_ode"].mean()))
        row.update({c: float(g[c].mean()) for c in g.columns if c.startswith("ode_rel_residual_")})
        rows.append(row)
    return pd.DataFrame(rows)


def e0_sensitivity(runs, seeds):
    """SEIR-D PINN runs grouped by E0/I0 (seeds < ``seeds``), paired with SIRD on the same seeds."""
    ok = runs[runs["success"] & (runs["seed"] < seeds)]
    sird = ok[(ok["model"] == SIRD) & ok["main"]].set_index("seed")
    rows = []
    for factor, g in ok[ok["model"] == SEIRD].groupby("e0_factor"):
        g = g.set_index("seed")
        row = {"E0_over_I0": factor, "E0": g["E0"].iloc[0], "seeds": len(g)}
        for col, label in (("aic", "AIC"), ("aic_I", "AIC_I"), ("ode_aic", "AIC_ode"), ("ode_aic_I", "AIC_ode_I")):
            mean, _, se, _ = paired_difference(g, sird, col)
            row.update({label: float(g[col].mean()), f"delta_{label}_vs_SIRD": mean, f"delta_{label}_se": se})
        row.update({p: float(g[p].mean()) for p in ("beta", "sigma", "gamma", "mu", "R0")})
        row["peak_day_ode"] = float(g["peak_day_ode"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figures

COLOR = {SIRD: "#D32F2F", SEIRD: "#1565C0"}
XLABEL = "Day (day 0 = 5 July 2020)"
MINUS = "−"


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # The figures go into a report at page width (17 cm = 6.7 in), so the canvas is sized in
    # inches close to that and the fonts below are what the reader actually sees on paper.
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                         "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "figure.titlesize": 10, "lines.markersize": 2.5})
    return plt


def save_figure(fig, out, name):
    """Write ``fig`` to out/figures/<name>.png, .pdf and .tiff (FIGURE_FORMATS), then close it."""
    import matplotlib.pyplot as plt
    from PIL import Image

    folder = Path(out) / FIGURES_DIR
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.png", dpi=PNG_DPI)
    fig.savefig(folder / f"{name}.pdf")
    # matplotlib writes TIFF as RGBA; submission systems often reject an alpha channel, so render and drop it
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=TIFF_DPI)
    buffer.seek(0)
    with Image.open(buffer) as image:
        image.convert("RGB").save(folder / f"{name}.tiff", dpi=(TIFF_DPI, TIFF_DPI), compression="tiff_lzw")
    plt.close(fig)


def _signed(value):
    """ΔAIC for a label: explicit sign, true minus, thousands separator (+8, −370, +1,015)."""
    return ("+" if value >= 0 else MINUS) + f"{abs(value):,.0f}"


def _stack(curves, model, kind, key):
    return np.array([c[kind][key] for (m, _), c in curves.items() if m == model])


def _observed_I(ax, data):
    ax.plot(data["t"][::2], data["I"][::2], "o", color="black", alpha=0.5, label="Observed active cases")


def _finish_I_axes(ax):
    ax.set_xlabel(XLABEL)
    ax.set_ylabel("Active cases I(t)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")  # the early part of the wave is low, so the box stays clear of the data
    ax.set_ylim(top=ax.get_ylim()[1] * 1.35)  # room for the legend above the data


def panel_pinn(ax, summary, curves, data):
    """I(t) of both PINNs, mean and range over seeds; AIC of the PINN output and the seed-paired ΔAIC."""
    t = data["t"]
    by_model = summary.set_index("model")
    _observed_I(ax, data)
    for model in (SIRD, SEIRD):
        values = _stack(curves, model, "pinn", "I")
        if not values.size or model not in by_model.index:
            continue
        row = by_model.loc[model]
        spread = f" (SD {row['AIC_sd']:,.0f})" if np.isfinite(row["AIC_sd"]) else ""
        # the two mean curves nearly coincide: SEIR-D dashed on top of a wider SIRD line keeps both visible
        style = {"lw": 2.6} if model == SIRD else {"lw": 1.4, "ls": "--"}
        ax.fill_between(t, values.min(0), values.max(0), color=COLOR[model], alpha=0.2, lw=0)
        ax.plot(t, values.mean(0), color=COLOR[model], label=f"{model}: AIC {row['AIC']:,.0f}{spread}", **style)
    if SEIRD in by_model.index and SIRD in by_model.index:
        row = by_model.loc[SEIRD]
        se = (f", SE {row['delta_AIC_se']:,.0f}\n{int(row['seeds'])} paired seeds"
              if np.isfinite(row["delta_AIC_se"]) else "")
        # right-hand side, below the legend, right of the peak and above the tail of the wave
        # the panel plots I(t), but this AIC, like the legend's, is computed on all four compartments
        ax.text(0.98, 0.72, f"ΔAIC (SEIR-D {MINUS} SIRD)\n= {_signed(row['delta_AIC'])}{se}\nAIC on S, I, R, D",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.5,
                bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "0.7"})
    _finish_I_axes(ax)


def _ode_fits(ax, fits, data, suffix, e0_factor):
    """SIRD and SEIR-D fitted under one criterion; AIC of SIRD and ΔAIC of each SEIR-D variant in the legend."""
    for key, style, name in (((SIRD, suffix, "", None, E0_FIXED), {"color": COLOR[SIRD]}, "SIRD"),
                             ((SEIRD, suffix, SIGMA_FREE, e0_factor, E0_FIXED), {"color": COLOR[SEIRD]},
                              f"SEIR-D, E(0) = {e0_factor:g} I(0) assumed"),
                             ((SEIRD, suffix, SIGMA_FREE, None, E0_QUASI), {"color": COLOR[SEIRD], "ls": "--"},
                              "SEIR-D, E(0) from fitted rates")):
        if key not in fits:
            continue
        row = fits[key]["row"]
        score = f"AIC {row['AIC']:,.0f}" if key[0] == SIRD else f"ΔAIC {_signed(row['delta_AIC_vs_SIRD'])}"
        ax.plot(data["t"], fits[key]["solution"]["I"], lw=2, label=f"{name}: {score}", **style)


def panel_ode_fit_I(ax, fits, data, e0_factor):
    """Direct ODE fits to I(t) alone."""
    _observed_I(ax, data)
    _ode_fits(ax, fits, data, "_I", e0_factor)
    _finish_I_axes(ax)


def panel_delta_aic_e0(ax, fits, e0_factor):
    """ΔAIC (SEIR-D − SIRD) of the direct fits against the assumed E(0)/I(0), for AIC on S, I, R, D and on
    I(t) alone; a star marks E(0) computed from the fitted rates instead of assumed."""
    # label offsets: grey values above their curve, blue below (the blue profile climbs steeply to the right)
    for suffix, color, label, dy in (("", "#555555", "S, I, R, D", 1), ("_I", COLOR[SEIRD], "I(t) alone", -1)):
        points = sorted((f, fits[(m, sfx, rng, f, mode)]["row"]["delta_AIC_vs_SIRD"])
                        for (m, sfx, rng, f, mode) in fits
                        if m == SEIRD and sfx == suffix and rng == SIGMA_FREE and mode == E0_FIXED and f is not None)
        if points:
            ax.plot([x for x, _ in points], [y for _, y in points], "o-", ms=3, lw=1.5, color=color,
                    label=f"AIC on {label}, E(0) assumed")
        consistent = fits.get((SEIRD, suffix, SIGMA_FREE, None, E0_QUASI))
        if consistent:
            x, y = consistent["row"]["E0_over_I0"], consistent["row"]["delta_AIC_vs_SIRD"]
            ax.plot([x], [y], "*", ms=12, color=color, markeredgecolor="black", markeredgewidth=0.7,
                    label=f"AIC on {label}, E(0) from fitted rates")
            ax.annotate(_signed(y), (x, y), xytext=(7, -3), textcoords="offset points", fontsize=7,
                        color=color, va="top", fontweight="bold")
        assumed = fits.get((SEIRD, suffix, SIGMA_FREE, e0_factor, E0_FIXED))
        if assumed:
            y = assumed["row"]["delta_AIC_vs_SIRD"]
            ax.annotate(_signed(y), (e0_factor, y), xytext=(4, 4 * dy), textcoords="offset points", fontsize=7,
                        color=color, va="bottom" if dy > 0 else "top")
    ax.axvline(e0_factor, color="gray", ls=":", lw=1, label=f"E(0) = {e0_factor:g} I(0), as in the PINN runs")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xscale("symlog", linthresh=0.25)  # the profile is spaced almost logarithmically
    ax.set_xlim(-0.03, 30)                   # E(0) >= 0: no negative side of the symlog axis
    ticks = [0, 0.5, 1, 2, 4, 8, 16, 24]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{x:g}" for x in ticks])
    ax.minorticks_off()
    low, high = ax.get_ylim()
    ax.set_ylim(low - 0.05 * (high - low), high + 0.55 * (high - low))  # headroom for the legend
    ax.set_xlabel("E(0) / I(0)")
    ax.set_ylabel(f"ΔAIC (SEIR-D {MINUS} SIRD)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", framealpha=1)


def plot_overlay(summary, curves, fits, data, e0_factor, out):
    """Main figure: (a) I(t) of both PINNs with their AIC, (b) ΔAIC of the direct fits against E(0).

    The two panels, and the direct ODE fits to I(t) (panel B of the four-panel layout this replaces),
    are also written as figures of their own; the names keep the letters of that layout.
    """
    plt = _pyplot()
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.5, 3.3))
    panel_pinn(left, summary, curves, data)
    left.set_title("(a) PINN output: mean and range over seeds", loc="left")
    panel_delta_aic_e0(right, fits, e0_factor)
    right.set_title("(b) ODE fits: ΔAIC against the assumed E(0)", loc="left")
    fig.tight_layout()
    save_figure(fig, out, "i_overlay")

    for name, title, draw in (
            ("panel_A_pinn_output", "PINN output: mean and range over seeds",
             lambda ax: panel_pinn(ax, summary, curves, data)),
            ("panel_B_ode_fit_to_I", "ODE fitted to I(t) alone",
             lambda ax: panel_ode_fit_I(ax, fits, data, e0_factor)),
            ("panel_D_delta_aic_vs_E0", "ODE fits: ΔAIC against the assumed E(0)",
             lambda ax: panel_delta_aic_e0(ax, fits, e0_factor))):
        fig, ax = plt.subplots(figsize=(3.9, 3.3))
        draw(ax)
        ax.set_title(title, loc="left")
        fig.tight_layout()
        save_figure(fig, out, name)


def plot_delta_aic(summary, fits, e0_factor, out):
    """ΔAIC (SEIR-D − SIRD) of every approach on one chart, for AIC on S, I, R, D and on I(t) alone.

    PINN rows are seed-paired means with a bar of ±2 SE (the resolved_* test of aic_comparison.csv);
    the direct fits are single maximum-likelihood fits and have no bar.
    """
    plt = _pyplot()
    by_model = summary.set_index("model")
    seird = by_model.loc[SEIRD] if SEIRD in by_model.index else None
    rows = (("PINN output", "AIC"), ("ODE with the PINN\nparameters", "AIC_ode"),
            (f"Direct ODE fit,\nE(0) = {e0_factor:g} I(0) assumed", (SIGMA_FREE, e0_factor, E0_FIXED)),
            ("Direct ODE fit,\nE(0) from fitted rates", (SIGMA_FREE, None, E0_QUASI)))
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.9), sharey=True)
    for ax, (suffix, title) in zip(axes, (("", "AIC on S, I, R, D"), ("_I", "AIC on I(t) alone"))):
        sird = fits.get((SIRD, suffix, "", None, E0_FIXED))
        if sird:
            title += f" (n = {int(sird['row']['n']):,})"
        for i, (label, source) in enumerate(rows):
            y = len(rows) - 1 - i
            if isinstance(source, str):
                if seird is None:
                    continue
                mean, se = seird[f"delta_{source}{suffix}"], seird[f"delta_{source}{suffix}_se"]
            elif (SEIRD, suffix, *source) in fits:
                mean, se = fits[(SEIRD, suffix, *source)]["row"]["delta_AIC_vs_SIRD"], np.nan
            else:
                continue
            if np.isfinite(se):
                ax.errorbar([mean], [y], xerr=[2 * se], fmt="o", color="black", ecolor="0.45", capsize=3, ms=5)
                text = f"{_signed(mean)} (SE {se:,.0f})"
            else:
                ax.plot([mean], [y], "D", color=COLOR[SEIRD], ms=5)
                text = _signed(mean)
            ax.annotate(text, (mean, y), xytext=(0, 6), textcoords="offset points", ha="center", va="bottom",
                        fontsize=7, bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6})
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([label for label, _ in reversed(rows)])
        ax.set_ylim(-0.6, len(rows) - 0.3)
        low, high = ax.get_xlim()
        ax.set_xlim(low - 0.08 * (high - low), high + 0.08 * (high - low))
        ax.set_title(title, loc="left")
        ax.set_xlabel(f"ΔAIC = AIC(SEIR-D) {MINUS} AIC(SIRD);  < 0 favours SEIR-D")
        ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    save_figure(fig, out, "delta_aic_summary")


def plot_spec_fit(fits, data, e0_factor, out):
    """Separate figure: fitted to all four compartments, neither model reproduces the wave."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    _observed_I(ax, data)
    _ode_fits(ax, fits, data, "", e0_factor)
    ax.set_title("ODE fitted to S, I, R, D: neither constant-parameter model reproduces the wave", loc="left")
    _finish_I_axes(ax)
    fig.tight_layout()
    save_figure(fig, out, "spec_criterion_fit")


def plot_fit(runs, curves, data, out):
    """Diagnostics, six panels: PINN and ODE I(t), latent E(t), R(t), D(t), S-channel deviation."""
    plt = _pyplot()
    t = data["t"]
    main = runs[runs["main"] & runs["success"]]
    aic = main.groupby("model")["aic"].mean()
    aic_ode = main.groupby("model")["ode_aic"].mean()

    def band(ax, model, values, label):
        if values.size:
            ax.fill_between(t, values.min(0), values.max(0), color=COLOR[model], alpha=0.2, lw=0)
            ax.plot(t, values.mean(0), color=COLOR[model], lw=2, label=label)

    def observed(ax, key):
        ax.plot(t[::3], data[key][::3], "o", ms=3, color="black", alpha=0.6, label="Observed")

    fig, axes = plt.subplots(3, 2, figsize=(7.0, 8.6))
    ax = axes[0, 0]
    observed(ax, "I")
    for model in (SIRD, SEIRD):
        if model in aic:
            band(ax, model, _stack(curves, model, "pinn", "I"), f"{model}: AIC {aic[model]:.0f}")
    ax.set_title("A. I(t), PINN output (mean and range over seeds)", loc="left")

    ax = axes[0, 1]
    observed(ax, "I")
    for model in (SIRD, SEIRD):
        for i, series in enumerate(_stack(curves, model, "ode", "I")):
            ax.plot(t, series, color=COLOR[model], lw=1, alpha=0.6,
                    label=f"{model}: ODE, AIC {aic_ode[model]:.0f}" if i == 0 else None)
    ax.set_title("B. ODE with the PINN-recovered parameters", loc="left")

    ax = axes[1, 0]
    E = np.array([c["E"] for (m, _), c in curves.items() if m == SEIRD])
    band(ax, SEIRD, E, "SEIR-D PINN, latent E")
    for i, series in enumerate(_stack(curves, SEIRD, "ode", "E")):
        ax.plot(t, series, "--", color=COLOR[SEIRD], lw=1, alpha=0.5, label="ODE E" if i == 0 else None)
    if E.size:
        e0 = main.loc[main["model"] == SEIRD, "E0"].iloc[0]
        ax.scatter([0], [e0], color="black", zorder=5, label=f"E(0) = {e0:.0f} (assumed)")
    ax.set_title("C. Exposed compartment E(t) (no data)", loc="left")

    for ax, key, title in ((axes[1, 1], "R", "D. Recovered R(t)"), (axes[2, 0], "D", "E. Deceased D(t)")):
        observed(ax, key)
        for model in (SIRD, SEIRD):
            band(ax, model, _stack(curves, model, "pinn", key), model)
        ax.set_title(title, loc="left")

    ax = axes[2, 1]
    for model in (SIRD, SEIRD):
        values = _stack(curves, model, "pinn", "S")
        if values.size:
            band(ax, model, values - data["S"], f"{model}: {'S + E' if model == SEIRD else 'S'}")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("F. S channel: PINN minus observed S", loc="left")

    for ax in axes.flat:
        ax.set_xlabel(XLABEL)
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
    fig.suptitle("PINN calibration of SIRD and SEIR-D on the full curve: diagnostics")
    fig.tight_layout()
    save_figure(fig, out, "full_wave_fit")


def plot_parameters(runs, curves, init, out):
    """Recovered parameters by seed, and their trajectories during training."""
    plt = _pyplot()
    from matplotlib.lines import Line2D

    main = runs[runs["main"] & runs["success"]]
    fig, axes = plt.subplots(1, 5, figsize=(7.5, 2.4))
    for ax, p in zip(axes, ("beta", "gamma", "mu", "sigma", "R0")):
        for j, model in enumerate((SIRD, SEIRD)):
            values = main.loc[main["model"] == model, p].dropna().to_numpy()
            if values.size:
                ax.scatter(j + np.linspace(-0.15, 0.15, values.size), values, color=COLOR[model], s=25)
        if p in init:
            ax.axhline(init[p], color="gray", ls="--", lw=1, label="initial guess")
            ax.legend(fontsize=6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["SIRD", "SEIR-D"])
        ax.set_title(p)
        ax.grid(alpha=0.3)
    fig.suptitle("PINN-recovered parameters by seed")
    fig.tight_layout()
    save_figure(fig, out, "parameters_by_seed")

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6))
    greek = {"beta": "β", "sigma": "σ (SEIR-D only)", "gamma": "γ", "mu": "μ"}
    for ax, p in zip(axes.flat, ("beta", "sigma", "gamma", "mu")):
        for (model, _), c in curves.items():
            history = c.get("history") or []
            if history and p in history[0]:
                ax.plot([h["epoch"] for h in history], [h[p] for h in history], color=COLOR[model], lw=0.8, alpha=0.7)
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("per day")
        ax.set_title(greek[p], loc="left")
        ax.grid(alpha=0.3, which="both")
    axes.flat[0].legend(handles=[Line2D([], [], color=COLOR[m], label=m) for m in (SIRD, SEIRD)])
    fig.suptitle("Parameter trajectories during PINN training (every seed); y axis logarithmic, values per day")
    fig.tight_layout()
    save_figure(fig, out, "parameter_convergence")


# ----------------------------------------------------------------------------- provenance and main

def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance(data_path):
    """Versions, git commit, uncommitted changes and SHA-256 of every file the results depend on."""
    import scipy
    import torch

    files = [REPO_ROOT / f for f in TRACKED_FILES] + [Path(data_path)]
    relative = [str(p.relative_to(REPO_ROOT)) for p in files]
    try:
        git = partial(subprocess.run, cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
        commit = git(["git", "rev-parse", "HEAD"]).stdout.strip() or None
        uncommitted = git(["git", "status", "--porcelain", "--", *relative]).stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        commit, uncommitted = None, None
    return {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
            "scipy": scipy.__version__, "pandas": pd.__version__, "git_commit": commit,
            "uncommitted_files": uncommitted, "files_sha256": {r: _sha256(p) for r, p in zip(relative, files)}}


def clear_outputs(out):
    """Delete what an earlier run of this script wrote to ``out``, and nothing else."""
    for name in GENERATED:
        (out / name).unlink(missing_ok=True)
    for name in GENERATED_DIRS:
        shutil.rmtree(out / name, ignore_errors=True)


def parse_args(argv):
    cpus = os.cpu_count() or 2
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="kouprianov")
    parser.add_argument("--epochs", type=int, default=10_000)
    parser.add_argument("--seeds", type=int, default=10, help="seeds per model in the main comparison")
    parser.add_argument("--param-lr", type=float, default=PARAM_LR,
                        help="Adam rate of the epidemic parameters; 0 = EINN_PINN's single rate of 1e-4")
    parser.add_argument("--lambda-ode", type=float, default=LAMBDAS["lambda_ode"])
    parser.add_argument("--last-iterate", action="store_true", help="keep the last iterate, not the lowest-loss state")
    parser.add_argument("--e0-factor", type=float, default=0.5, help="E(0) / I(0) for SEIR-D in the main comparison")
    parser.add_argument("--e0-sensitivity", default="0.25,1.0", help="extra E(0)/I(0) ratios for PINN runs; empty to skip")
    parser.add_argument("--sensitivity-seeds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=max(1, min(6, cpus // 2 - 1)))
    parser.add_argument("--threads", type=int, default=2, help="torch threads per worker")
    parser.add_argument("--out", default=None,
                        help="output directory (default model_selection_results, with --quick model_selection_results_quick)")
    parser.add_argument("--quick", action="store_true",
                        help="smoke test: 2 epochs, 1 seed, no E0 runs, small least-squares grid, 1 worker")
    args = parser.parse_args(argv)
    args.e0_sensitivity = [float(x) for x in args.e0_sensitivity.split(",") if x.strip()]
    args.param_lr = args.param_lr or None
    if args.quick:
        args.epochs, args.seeds, args.e0_sensitivity, args.workers = 2, 1, [], 1
    if args.out is None:
        args.out = str(HERE / ("model_selection_results_quick" if args.quick else "model_selection_results"))
    return args


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clear_outputs(out)
    data_path = DATASETS[args.dataset]
    data = load_series(data_path)
    init = initial_guess(data["I"])
    lambdas = {**LAMBDAS, "lambda_ode": args.lambda_ode}

    base = {"data_path": str(data_path), "init": init, "sigma0": SIGMA0, "epochs": args.epochs, "lambdas": lambdas,
            "param_lr": args.param_lr, "restore_best": not args.last_iterate, "threads": args.threads,
            "plots_dir": str(out / "pinn_agent_plots"), "log_dir": str(out / "logs")}
    specs = [{**base, "model": m, "seed": s, "main": True, "e0_factor": args.e0_factor, "tag": f"{SLUG[m]}_seed{s}"}
             for m in (SIRD, SEIRD) for s in range(args.seeds)]
    specs += [{**base, "model": SEIRD, "seed": s, "main": False, "e0_factor": f, "tag": f"seird_e0x{f:g}_seed{s}"}
              for f in args.e0_sensitivity if f != args.e0_factor for s in range(args.sensitivity_seeds)]
    e0_profile = [args.e0_factor] if args.quick else sorted({*LSQ_E0_PROFILE, args.e0_factor})

    config = {"dataset": str(data_path.relative_to(REPO_ROOT)), "N": data["N"], "n_points": len(data["t"]),
              "train_size": len(data["t"]), "initial_guess": {**init, "sigma": SIGMA0}, "e0_factor": args.e0_factor,
              "E0": args.e0_factor * data["I"][0], "epochs": args.epochs, "lambdas": lambdas,
              "param_lr": args.param_lr, "network_lr": 1e-4, "restore_best": not args.last_iterate,
              "seeds": args.seeds, "e0_sensitivity": args.e0_sensitivity, "sensitivity_seeds": args.sensitivity_seeds,
              "convergence": {"window_epochs": CONVERGENCE_WINDOW, "max_relative_change": CONVERGENCE_TOL},
              "lsq": {"bounds": LSQ_BOUNDS, "incubation_2_14_sigma": INCUBATION_2_14, "e0_profile": e0_profile,
                      "quick_grid": args.quick, "e0_modes": [E0_FIXED, *E0_RULES],
                      "e0_rules": {E0_QUASI: "E(0) = I(0) (gamma + mu) / sigma, so dI/dt(0) = 0",
                                   E0_GROWTH: f"E(0) = I(0) (r + gamma + mu) / sigma, r = observed growth "
                                              f"over the first {GROWTH_WINDOW} days",
                                   E0_ESTAT: "E(0) = beta S(0) I(0) / (N sigma), so dE/dt(0) = 0"},
                      "E0_note": "E0 and e0_factor above apply to the PINN runs and the e0_mode = fixed rows"},
              "criteria": {suffix or "(specified)": description for suffix, (_, _, description) in CRITERIA.items()},
              "aic": "n ln(RSS) + 2k, SIBIRCON 2022 Eq. 6; k = 3 (SIRD) / 4 (SEIR-D)",
              "workers": args.workers, "threads_per_worker": args.threads, "quick": args.quick,
              "provenance": provenance(data_path)}
    (out / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"{len(specs)} PINN runs, {args.epochs} epochs each, param lr {args.param_lr}, "
          f"{args.workers} worker(s) x {args.threads} thread(s); initial guess {init}, sigma0 = {SIGMA0:.4f}", flush=True)

    started = time.time()
    results = run_all(specs, args.workers)
    runs = pd.DataFrame([r["row"] for r in results]).sort_values(
        ["main", "model", "e0_factor", "seed"], ascending=[False, False, True, True]).reset_index(drop=True)
    curves = {(r["row"]["model"], r["row"]["seed"]): r["curves"]
              for r in results if r["row"]["main"] and r["row"]["success"]}
    summary = summarize(runs)
    sensitivity = e0_sensitivity(runs, args.sensitivity_seeds) if args.e0_sensitivity else pd.DataFrame()

    # direct ODE least squares; the PINN estimates are among the starting points
    ok = runs[runs["main"] & runs["success"]]
    extra = {m: [dict(init)] + ([ok.loc[ok["model"] == m, list(PARAMS[m])].mean().to_dict()]
                                if (ok["model"] == m).any() else []) for m in (SIRD, SEIRD)}
    lsq_table, fits = lsq_reference(lsq_jobs(data_path, args.e0_factor, e0_profile, extra, args.quick), args.workers)

    runs.to_csv(out / "aic_runs.csv", index=False)
    summary.to_csv(out / "aic_comparison.csv", index=False)
    lsq_table.to_csv(out / "ode_lsq_reference.csv", index=False)
    if not sensitivity.empty:
        sensitivity.to_csv(out / "e0_sensitivity.csv", index=False)
    if curves:
        plot_overlay(summary, curves, fits, data, args.e0_factor, out)
        plot_delta_aic(summary, fits, args.e0_factor, out)
        plot_spec_fit(fits, data, args.e0_factor, out)
        plot_fit(runs, curves, data, out)
        plot_parameters(runs, curves, config["initial_guess"], out)

    shown = ["model", "k", "n", "RSS", "AIC", "delta_AIC", "delta_AIC_se", "AIC_I", "delta_AIC_I", "delta_AIC_I_se",
             "AIC_ode", "delta_AIC_ode", "beta", "sigma", "gamma", "mu", "R0", "converged_seeds"]
    lsq_shown = ["criterion", "model", "sigma_range", "e0_mode", "E0_over_I0", "n", "AIC", "delta_AIC_vs_SIRD",
                 "beta", "sigma", "gamma", "mu", "incubation_days", "peak_day", "ever_infected_share", "at_bound"]
    headline = lsq_table[(lsq_table["model"] == ODE_NAME[SIRD]) | (lsq_table["e0_mode"] == E0_QUASI)
                         | (lsq_table["E0_over_I0"] == args.e0_factor)]
    headline = headline[headline["sigma_range"] != SIGMA_2_14]  # the console table would not fit otherwise
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print("\nPINN calibration (aic_comparison.csv):\n" + summary[[c for c in shown if c in summary]].to_string(index=False))
        print("\nDirect ODE least squares (ode_lsq_reference.csv, main E0):\n" + headline[lsq_shown].to_string(index=False))
        if not sensitivity.empty:
            print("\nE0 sensitivity of the SEIR-D PINN (e0_sensitivity.csv):\n" + sensitivity.to_string(index=False))
    failed = int((~runs["success"].astype(bool)).sum())
    print(f"\n{len(runs)} PINN runs and {len(lsq_table)} direct fits in {time.time() - started:.0f} s, "
          f"{failed} failed; results in {out}")
    return summary


if __name__ == "__main__":
    main()
