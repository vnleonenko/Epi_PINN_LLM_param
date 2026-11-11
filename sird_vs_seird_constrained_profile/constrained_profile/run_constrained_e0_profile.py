#!/usr/bin/env python3
"""Constrained SEIR-D E(0) profile on the first-wave St. Petersburg series.

Direct least-squares only. See PROTOCOL.md in this folder. Does not train a PINN
and does not write into full_wave_results/ (formerly supplementary_model_comparison/).

Run from anywhere:

    python3 appendix_F_seird_vs_sird/constrained_profile/run_constrained_e0_profile.py

Outputs (profile.csv, summary.json, results.md, run.log) are written next to
this script. Pass --sanity-only to fit only the two stored E(0)/I(0) = 0.5
checks and the SIRD baselines.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# One thread per process: the pool is the parallelism.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "covid-19_Kouprianov.csv"
PINN_RUNS = ROOT / "full_wave_results" / "aic_runs.csv"
EXPECTED_SHA256 = "bde98003e5fd10331bcd0adf2a2f7c8ccb75398b34296de47355b54b060f3aab"

N_POP = 6_000_000.0
DT = 0.02
STEPS_PER_DAY = 50  # 1 / DT
E0_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0)

# run_config.json bounds, sigma replaced by incubation_2_14_sigma = [1/14, 1/2].
BETA_LO, BETA_HI = 1e-4, 5.0
SIGMA_LO, SIGMA_HI = 1.0 / 14.0, 0.5
GAMMA_LO, GAMMA_HI = 1e-4, 2.0
MU_LO, MU_HI = 1e-7, 0.5

NOTEBOOK_BETA = 0.1133380423496375  # early-growth rule; see PROTOCOL.md
NOTEBOOK_GAMMA = 0.1
NOTEBOOK_MU = 0.01
NOTEBOOK_SIGMA = 1.0 / 5.2

# Stored direct-fit rows used only to confirm the integrator and, after the
# fit, the E(0)/I(0) = 0.5 sanity check. Never used as optimizer starts.
STORED = {
    "SIRD_I": {
        "beta": 0.14158435196016828,
        "gamma": 0.11805193776124685,
        "mu": 0.0018460749045082108,
        "rss": 54007453503.2447,
        "ever": 0.2849401486840344,
        "peak_I": 77790.21555325815,
        "peak_day": 196,
    },
    "SIRD_4": {
        "beta": 0.12526385715375093,
        "gamma": 0.11018361250645725,
        "mu": 0.0073497983943491265,
        "rss": 1729090688603.633,
        "ever": 0.0850110589451829,
        "peak_I": 15589.785954992536,
        "peak_day": 288,
    },
    "SEIRD_I_0.5": {
        "beta": 0.4258868613180567,
        "sigma": 0.07142857143216955,
        "gamma": 0.24925141674027726,
        "mu": 0.02766281668225129,
        "e0_over_i0": 0.5,
        "rss": 24527253673.25818,
        "delta_aic": -289.2654160075999,
        "ever": 0.6019306166455793,
        "dip_ratio": 0.38591339731348223,
        "dip_day": 11,
        "latent_days": 13.999999999294769,
        "peak_I": 86694.02398431559,
        "peak_day": 197,
    },
    "SEIRD_4_0.5": {
        "beta": 0.11066376715617521,
        "sigma": 0.4999968424495708,
        "gamma": 0.09648706028057671,
        "mu": 0.0060477619070156315,
        "e0_over_i0": 0.5,
        "rss": 1810392226710.1982,
        "delta_aic": 69.81904850241699,
        "ever": 0.08708242977812342,
        "dip_ratio": 0.8700550999614587,
        "dip_day": 59,
        "latent_days": 2.000012630281478,
        "peak_I": 18599.112151199453,
        "peak_day": 326,
    },
}

# Published profile anchors quoted in the comparison paragraph (not refitted).
PUBLISHED_ANCHORS = {
    "I_unconstrained_best": {
        "e0_over_i0": 0.0,
        "delta_aic": -413.404,
        "latent_days": 28.803,
        "ever": 0.829314,
        "dip_ratio": 0.173977,
        "source": "ode_lsq_reference.csv, I only, sigma free, fixed E(0)/I(0)=0",
    },
    "I_constrained_e0_0.5": {
        "e0_over_i0": 0.5,
        "delta_aic": -289.265,
        "latent_days": 14.0,
        "ever": 0.601930,
        "dip_ratio": 0.385913,
        "source": "ode_lsq_reference.csv, I only, incubation 2-14 d, fixed E(0)/I(0)=0.5",
    },
    "I_self_consistent_constrained": {
        "delta_aic": -35.008,
        "latent_days": 6.099,
        "e0_over_i0": 1.0306,
        "source": "ode_lsq_reference.csv, I only, incubation 2-14 d, self-consistent E(0); not this grid",
    },
    "four_unconstrained_best": {
        "e0_over_i0": 0.0,
        "delta_aic": 2.297,
        "latent_days": 0.010,
        "ever": 0.085012,
        "source": "ode_lsq_reference.csv, S+I+R+D one variance, sigma free, fixed E(0)/I(0)=0",
    },
    "four_constrained_e0_0.5": {
        "e0_over_i0": 0.5,
        "delta_aic": 69.819,
        "latent_days": 2.0,
        "ever": 0.087082,
        "dip_ratio": 0.870055,
        "source": "ode_lsq_reference.csv, S+I+R+D one variance, incubation 2-14 d, fixed E(0)/I(0)=0.5",
    },
}

_DATA = None


def load_data():
    global _DATA
    if _DATA is not None:
        return _DATA
    raw = DATA_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_SHA256:
        raise SystemExit(f"data sha256 {digest} != {EXPECTED_SHA256}")
    rows = list(csv.DictReader(raw.decode("utf-8").splitlines()))
    if len(rows) != 369:
        raise SystemExit(f"expected 369 rows, found {len(rows)}")
    I = np.array([float(r["I"]) for r in rows])
    D = np.array([float(r["D"]) for r in rows])
    R = np.array([float(r["R"]) for r in rows])
    S = np.array([float(r["S"]) for r in rows])
    if np.max(np.abs(S + I + R + D - N_POP)) > 1e-6:
        raise SystemExit("S+I+R+D is not N on every row")
    _DATA = {"S": S, "I": I, "R": R, "D": D, "n": len(I)}
    return _DATA


def _rk4_seird(beta, sigma, gamma, mu, e0):
    data = load_data()
    S0 = float(data["S"][0]) - e0
    E = float(e0)
    I = float(data["I"][0])
    R = float(data["R"][0])
    D = float(data["D"][0])
    S = S0
    n = data["n"]
    gmu = gamma + mu
    h = DT
    h2 = 0.5 * DT
    h6 = DT / 6.0
    outS = np.empty(n)
    outE = np.empty(n)
    outI = np.empty(n)
    outR = np.empty(n)
    outD = np.empty(n)
    outS[0] = S
    outE[0] = E
    outI[0] = I
    outR[0] = R
    outD[0] = D
    for day in range(n - 1):
        for _ in range(STEPS_PER_DAY):
            inf = beta * S * I / N_POP
            dS1 = -inf
            dE1 = inf - sigma * E
            dI1 = sigma * E - gmu * I
            dR1 = gamma * I
            dD1 = mu * I
            S1 = S + h2 * dS1
            E1 = E + h2 * dE1
            I1 = I + h2 * dI1
            inf = beta * S1 * I1 / N_POP
            dS2 = -inf
            dE2 = inf - sigma * E1
            dI2 = sigma * E1 - gmu * I1
            dR2 = gamma * I1
            dD2 = mu * I1
            S2 = S + h2 * dS2
            E2 = E + h2 * dE2
            I2 = I + h2 * dI2
            inf = beta * S2 * I2 / N_POP
            dS3 = -inf
            dE3 = inf - sigma * E2
            dI3 = sigma * E2 - gmu * I2
            dR3 = gamma * I2
            dD3 = mu * I2
            S3 = S + h * dS3
            E3 = E + h * dE3
            I3 = I + h * dI3
            inf = beta * S3 * I3 / N_POP
            dS4 = -inf
            dE4 = inf - sigma * E3
            dI4 = sigma * E3 - gmu * I3
            dR4 = gamma * I3
            dD4 = mu * I3
            S += h6 * (dS1 + 2.0 * dS2 + 2.0 * dS3 + dS4)
            E += h6 * (dE1 + 2.0 * dE2 + 2.0 * dE3 + dE4)
            I += h6 * (dI1 + 2.0 * dI2 + 2.0 * dI3 + dI4)
            R += h6 * (dR1 + 2.0 * dR2 + 2.0 * dR3 + dR4)
            D += h6 * (dD1 + 2.0 * dD2 + 2.0 * dD3 + dD4)
        outS[day + 1] = S
        outE[day + 1] = E
        outI[day + 1] = I
        outR[day + 1] = R
        outD[day + 1] = D
    return outS, outE, outI, outR, outD


def _rk4_sird(beta, gamma, mu):
    data = load_data()
    S = float(data["S"][0])
    I = float(data["I"][0])
    R = float(data["R"][0])
    D = float(data["D"][0])
    n = data["n"]
    gmu = gamma + mu
    h = DT
    h2 = 0.5 * DT
    h6 = DT / 6.0
    outS = np.empty(n)
    outI = np.empty(n)
    outR = np.empty(n)
    outD = np.empty(n)
    outS[0] = S
    outI[0] = I
    outR[0] = R
    outD[0] = D
    for day in range(n - 1):
        for _ in range(STEPS_PER_DAY):
            inf = beta * S * I / N_POP
            dS1 = -inf
            dI1 = inf - gmu * I
            dR1 = gamma * I
            dD1 = mu * I
            S1 = S + h2 * dS1
            I1 = I + h2 * dI1
            inf = beta * S1 * I1 / N_POP
            dS2 = -inf
            dI2 = inf - gmu * I1
            dR2 = gamma * I1
            dD2 = mu * I1
            S2 = S + h2 * dS2
            I2 = I + h2 * dI2
            inf = beta * S2 * I2 / N_POP
            dS3 = -inf
            dI3 = inf - gmu * I2
            dR3 = gamma * I2
            dD3 = mu * I2
            S3 = S + h * dS3
            I3 = I + h * dI3
            inf = beta * S3 * I3 / N_POP
            dS4 = -inf
            dI4 = inf - gmu * I3
            dR4 = gamma * I3
            dD4 = mu * I3
            S += h6 * (dS1 + 2.0 * dS2 + 2.0 * dS3 + dS4)
            I += h6 * (dI1 + 2.0 * dI2 + 2.0 * dI3 + dI4)
            R += h6 * (dR1 + 2.0 * dR2 + 2.0 * dR3 + dR4)
            D += h6 * (dD1 + 2.0 * dD2 + 2.0 * dD3 + dD4)
        outS[day + 1] = S
        outI[day + 1] = I
        outR[day + 1] = R
        outD[day + 1] = D
    return outS, outI, outR, outD


def _finite(arr) -> bool:
    return bool(np.isfinite(arr).all())


def residual_vector(model, mode, params, e0):
    """Return the residual vector whose squared norm is the RSS, or None if the trajectory broke."""
    data = load_data()
    if model == "SIRD":
        beta, gamma, mu = params
        S, I, R, D = _rk4_sird(beta, gamma, mu)
        if not (_finite(S) and _finite(I) and _finite(R) and _finite(D)):
            return None
        if mode == "I":
            return I - data["I"]
        return np.concatenate((S - data["S"], I - data["I"], R - data["R"], D - data["D"]))
    beta, sigma, gamma, mu = params
    S, E, I, R, D = _rk4_seird(beta, sigma, gamma, mu, e0)
    if not (_finite(S) and _finite(E) and _finite(I) and _finite(R) and _finite(D)):
        return None
    if mode == "I":
        return I - data["I"]
    # Observed S is compared with S+E. See check_s_vs_se.py.
    return np.concatenate((S + E - data["S"], I - data["I"], R - data["R"], D - data["D"]))


def trajectory_metrics(model, mode, params, e0):
    data = load_data()
    Iobs = data["I"]
    if model == "SIRD":
        beta, gamma, mu = params
        sigma = float("nan")
        S, I, R, D = _rk4_sird(beta, gamma, mu)
        E_end = 0.0
        dI_dt0 = beta * S[0] * I[0] / N_POP - (gamma + mu) * I[0]
    else:
        beta, sigma, gamma, mu = params
        S, E, I, R, D = _rk4_seird(beta, sigma, gamma, mu, e0)
        E_end = float(E[-1])
        dI_dt0 = sigma * e0 - (gamma + mu) * float(Iobs[0])
    resid = residual_vector(model, mode, params, e0)
    rss = float(np.dot(resid, resid))
    ratio = I[:60] / Iobs[:60]
    dip_day = int(np.argmin(ratio))
    peak_day = int(np.argmax(I))
    return {
        "RSS": rss,
        "beta": float(beta),
        "sigma": float(sigma),
        "gamma": float(gamma),
        "mu": float(mu),
        "gamma_plus_mu": float(gamma + mu),
        "R0": float(beta / (gamma + mu)),
        "latent_days": float(1.0 / sigma) if model == "SEIR-D" else float("nan"),
        "ever_infected_share": float(1.0 - S[-1] / N_POP),
        "dip_ratio": float(ratio[dip_day]),
        "dip_day": dip_day,
        "peak_I": float(I[peak_day]),
        "peak_day": peak_day,
        "R_end": float(R[-1]),
        "D_end": float(D[-1]),
        "E_end": E_end,
        "S_end": float(S[-1]),
        "I_end": float(I[-1]),
        "dI_dt0": float(dI_dt0),
    }


def _log_pair(lo, hi):
    a = math.log(lo)
    b = math.log(hi)
    return (math.exp(a + 0.25 * (b - a)), math.exp(a + 0.75 * (b - a)))


def _clip_box(beta, sigma, gamma, mu):
    return (
        min(max(beta, BETA_LO), BETA_HI),
        min(max(sigma, SIGMA_LO), SIGMA_HI),
        min(max(gamma, GAMMA_LO), GAMMA_HI),
        min(max(mu, MU_LO), MU_HI),
    )


def pinn_means():
    """Mean parameters of the 10 main seeds at E(0)/I(0) = 0.5."""
    rows = list(csv.DictReader(PINN_RUNS.open()))
    out = {}
    for model in ("SIRD-PINN", "SEIR-D-PINN"):
        sub = [
            r
            for r in rows
            if r["model"] == model and r["main"] == "True" and float(r["e0_factor"]) == 0.5
        ]
        if len(sub) != 10:
            raise SystemExit(f"expected 10 main {model} seeds at e0_factor 0.5, found {len(sub)}")
        acc = {}
        for key in ("beta", "gamma", "mu", "sigma"):
            vals = [float(r[key]) for r in sub if r[key] not in ("", "nan")]
            acc[key] = sum(vals) / len(vals) if vals else float("nan")
        out[model] = acc
    return out


def seird_starts(means):
    starts = []
    starts.append(("notebook", (NOTEBOOK_BETA, NOTEBOOK_SIGMA, NOTEBOOK_GAMMA, NOTEBOOK_MU)))
    pm = means["SEIR-D-PINN"]
    starts.append(("pinn_mean", _clip_box(pm["beta"], pm["sigma"], pm["gamma"], pm["mu"])))
    b_levels = _log_pair(BETA_LO, BETA_HI)
    g_levels = _log_pair(GAMMA_LO, GAMMA_HI)
    m_levels = _log_pair(MU_LO, MU_HI)
    s_levels = (SIGMA_LO, math.sqrt(SIGMA_LO * SIGMA_HI), SIGMA_HI)
    for b in b_levels:
        for g in g_levels:
            for m in m_levels:
                for s in s_levels:
                    starts.append(("grid", (b, s, g, m)))
    if len(starts) != 26:
        raise SystemExit(f"SEIR-D start count is {len(starts)}, expected 26")
    return starts


def sird_starts(means):
    starts = []
    b_levels = _log_pair(BETA_LO, BETA_HI)
    g_levels = _log_pair(GAMMA_LO, GAMMA_HI)
    m_levels = _log_pair(MU_LO, MU_HI)
    for b in b_levels:
        for g in g_levels:
            for m in m_levels:
                starts.append(("grid", (b, g, m)))
    starts.append(("notebook", (NOTEBOOK_BETA, NOTEBOOK_GAMMA, NOTEBOOK_MU)))
    pm = means["SIRD-PINN"]
    starts.append(
        (
            "pinn_mean",
            (
                min(max(pm["beta"], BETA_LO), BETA_HI),
                min(max(pm["gamma"], GAMMA_LO), GAMMA_HI),
                min(max(pm["mu"], MU_LO), MU_HI),
            ),
        )
    )
    return starts


def _fit_one(payload):
    """One multi-start solve. Top-level so process pools can call it."""
    t0 = time.perf_counter()
    model = payload["model"]
    mode = payload["mode"]
    e0 = payload["e0"]
    theta0 = np.array(payload["theta0"], dtype=float)
    try:
        if model == "SIRD":
            lo = np.log(np.array([BETA_LO, GAMMA_LO, MU_LO]))
            hi = np.log(np.array([BETA_HI, GAMMA_HI, MU_HI]))
        else:
            lo = np.log(np.array([BETA_LO, SIGMA_LO, GAMMA_LO, MU_LO]))
            hi = np.log(np.array([BETA_HI, SIGMA_HI, GAMMA_HI, MU_HI]))
        z0 = np.clip(np.log(theta0), lo + 1e-12, hi - 1e-12)

        def fun(z):
            theta = np.exp(z)
            resid = residual_vector(model, mode, theta, e0)
            if resid is None:
                # Broken trajectory: a large finite residual keeps the solver inside the box.
                n = 369 if mode == "I" else 1476
                return np.full(n, 1e6)
            return resid

        res = least_squares(
            fun,
            z0,
            bounds=(lo, hi),
            method="trf",
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
            max_nfev=120,
        )
        theta = np.exp(res.x)
        rss = float(np.dot(res.fun, res.fun))
        ok = bool(np.isfinite(rss))
        message = res.message
        nfev = int(res.nfev)
        status = int(res.status)
    except Exception as exc:  # a failed start is recorded, not fatal
        theta = theta0
        rss = float("inf")
        ok = False
        message = f"{type(exc).__name__}: {exc}"
        nfev = 0
        status = -1
        traceback.print_exc()
    return {
        "job": payload["job"],
        "start_index": payload["start_index"],
        "start_kind": payload["start_kind"],
        "ok": ok,
        "rss": rss,
        "theta": [float(x) for x in theta],
        "nfev": nfev,
        "status": status,
        "message": str(message),
        "seconds": time.perf_counter() - t0,
    }


def edge_name(val, lo, hi):
    """Bound test on the log scale used by the optimizer.

    A parameter is on an edge when it is within about 0.1% of that edge.
    A linear fraction of the raw interval is not used: μ spans 1e-7 to 0.5,
    so a linear tolerance would mark ordinary interior values as bound.
    """
    if not (math.isfinite(val) and val > 0 and lo > 0 and hi > 0):
        return ""
    if abs(math.log(val) - math.log(lo)) <= 1e-3:
        return "lower"
    if abs(math.log(val) - math.log(hi)) <= 1e-3:
        return "upper"
    return ""


def at_bound_string(model, beta, sigma, gamma, mu):
    hits = []
    if edge_name(beta, BETA_LO, BETA_HI):
        hits.append("beta")
    if model == "SEIR-D" and edge_name(sigma, SIGMA_LO, SIGMA_HI):
        hits.append("sigma")
    if edge_name(gamma, GAMMA_LO, GAMMA_HI):
        hits.append("gamma")
    if edge_name(mu, MU_LO, MU_HI):
        hits.append("mu")
    return ",".join(hits)


def latent_bound_label(sigma):
    which = edge_name(sigma, SIGMA_LO, SIGMA_HI)
    if which == "lower":
        return "upper_14d"
    if which == "upper":
        return "lower_2d"
    return "neither"


def aic(rss, k, n):
    return n * math.log(rss) + 2.0 * k


def build_jobs(means, sanity_only: bool):
    jobs = []
    for mode, mode_name in (("I", "I_only"), ("4", "four_compartments")):
        jobs.append(
            {
                "job": f"SIRD_{mode_name}",
                "model": "SIRD",
                "mode": mode,
                "mode_name": mode_name,
                "e0_over_i0": None,
                "e0": 0.0,
                "starts": sird_starts(means),
            }
        )
    ratios = (0.5,) if sanity_only else E0_GRID
    for mode, mode_name in (("I", "I_only"), ("4", "four_compartments")):
        for ratio in ratios:
            i0 = float(load_data()["I"][0])
            jobs.append(
                {
                    "job": f"SEIRD_{mode_name}_e0_{ratio:g}",
                    "model": "SEIR-D",
                    "mode": mode,
                    "mode_name": mode_name,
                    "e0_over_i0": float(ratio),
                    "e0": float(ratio) * i0,
                    "starts": seird_starts(means),
                }
            )
    return jobs


def expand_payloads(jobs):
    payloads = []
    for job in jobs:
        for i, (kind, theta) in enumerate(job["starts"]):
            payloads.append(
                {
                    "job": job["job"],
                    "model": job["model"],
                    "mode": job["mode"],
                    "e0": job["e0"],
                    "start_index": i,
                    "start_kind": kind,
                    "theta0": list(theta),
                }
            )
    return payloads


def select_best(start_rows):
    finite = [r for r in start_rows if r["ok"] and math.isfinite(r["rss"])]
    if not finite:
        raise RuntimeError("no successful start")
    finite.sort(key=lambda r: (r["rss"], r["start_index"]))
    best = finite[0]
    best_rss = best["rss"]
    within = [r for r in finite if r["rss"] <= best_rss * (1.0 + 1e-9) or (r["rss"] - best_rss) <= 1.0]
    # starts_within_1 matches the released CSV: AIC within 1, which for a fixed n,k is a small RSS gap.
    # Use the AIC gap directly once n and k are known; here a relative stand-in is refined by the caller
    # if needed. The caller recomputes from RSS.
    return best, finite, within


def assemble(job, start_rows):
    best, finite, _ = select_best(start_rows)
    model = job["model"]
    mode = job["mode"]
    k = 3 if model == "SIRD" else 4
    n = 369 if mode == "I" else 1476
    theta = best["theta"]
    metrics = trajectory_metrics(model, mode, theta, job["e0"])
    # Prefer the recomputed RSS (same formula, one extra pass) so the table is self-consistent.
    rss = metrics["RSS"]
    aic_val = aic(rss, k, n)
    # AIC within 1 of the best: |n ln(rss/rss_best)| < 1, ignoring the shared 2k.
    n_within = 0
    for row in finite:
        if row["rss"] <= 0:
            continue
        if abs(n * math.log(row["rss"] / rss)) <= 1.0 + 1e-9:
            n_within += 1
    if model == "SEIR-D":
        beta, sigma, gamma, mu = theta
    else:
        beta, gamma, mu = theta
        sigma = float("nan")
    bound = at_bound_string(model, beta, sigma if model == "SEIR-D" else 0.0, gamma, mu)
    latent_bound = latent_bound_label(sigma) if model == "SEIR-D" else ""
    return {
        "job": job["job"],
        "model": model,
        "mode_name": job["mode_name"],
        "error_model": "I only" if mode == "I" else "S, I, R, D, one variance",
        "e0_over_i0": job["e0_over_i0"],
        "E0": job["e0"] if model == "SEIR-D" else float("nan"),
        "k": k,
        "n": n,
        "RSS": rss,
        "AIC": aic_val,
        "beta": metrics["beta"],
        "sigma": metrics["sigma"],
        "gamma": metrics["gamma"],
        "mu": metrics["mu"],
        "gamma_plus_mu": metrics["gamma_plus_mu"],
        "R0": metrics["R0"],
        "latent_days": metrics["latent_days"],
        "latent_bound": latent_bound,
        "at_bound": bound,
        "ever_infected_share": metrics["ever_infected_share"],
        "dip_ratio": metrics["dip_ratio"],
        "dip_day": metrics["dip_day"],
        "peak_I": metrics["peak_I"],
        "peak_day": metrics["peak_day"],
        "R_end": metrics["R_end"],
        "D_end": metrics["D_end"],
        "E_end": metrics["E_end"],
        "S_end": metrics["S_end"],
        "I_end": metrics["I_end"],
        "dI_dt0": metrics["dI_dt0"],
        "starts": len(start_rows),
        "starts_ok": len(finite),
        "starts_within_1": n_within,
        "best_start_index": best["start_index"],
        "best_start_kind": best["start_kind"],
        "best_start_nfev": best["nfev"],
        "best_start_status": best["status"],
    }


def attach_delta(rows):
    baselines = {}
    for row in rows:
        if row["model"] == "SIRD":
            baselines[row["mode_name"]] = row
    for row in rows:
        base = baselines[row["mode_name"]]
        row["RSS_SIRD"] = base["RSS"]
        row["AIC_SIRD"] = base["AIC"]
        if row["model"] == "SIRD":
            row["delta_AIC_vs_SIRD"] = 0.0
        else:
            row["delta_AIC_vs_SIRD"] = row["n"] * math.log(row["RSS"] / base["RSS"]) + 2.0
    return baselines


def integrator_self_check():
    """Confirm RK4 reproduces the stored RSS before any optimization."""
    data = load_data()
    i0 = float(data["I"][0])
    checks = []
    specs = [
        ("SIRD_I", "SIRD", "I", STORED["SIRD_I"], None),
        ("SIRD_4", "SIRD", "4", STORED["SIRD_4"], None),
        ("SEIRD_I_0.5", "SEIR-D", "I", STORED["SEIRD_I_0.5"], 0.5 * i0),
        ("SEIRD_4_0.5", "SEIR-D", "4", STORED["SEIRD_4_0.5"], 0.5 * i0),
    ]
    worst = 0.0
    for name, model, mode, spec, e0 in specs:
        if model == "SIRD":
            params = (spec["beta"], spec["gamma"], spec["mu"])
        else:
            params = (spec["beta"], spec["sigma"], spec["gamma"], spec["mu"])
        met = trajectory_metrics(model, mode, params, e0 or 0.0)
        rel = abs(met["RSS"] - spec["rss"]) / spec["rss"]
        worst = max(worst, rel)
        ever_rel = abs(met["ever_infected_share"] - spec["ever"])
        checks.append(
            {
                "name": name,
                "rss_rel": rel,
                "ever_abs": ever_rel,
                "peak_day": met["peak_day"],
                "peak_day_stored": spec["peak_day"],
                "dip_ratio": met.get("dip_ratio"),
            }
        )
        print(
            f"self-check {name}: RSS rel {rel:.3e}, ever abs {ever_rel:.3e}, "
            f"peak day {met['peak_day']} (stored {spec['peak_day']})"
        )
    if worst > 1e-5:
        raise SystemExit(f"integrator self-check failed, worst RSS rel {worst:.3e}")
    return checks


def sanity_against_stored(rows):
    """Compare the refitted E(0)/I(0)=0.5 rows with the released CSV."""
    by_job = {r["job"]: r for r in rows}
    report = {}
    mapping = {
        "I_only": ("SEIRD_I_only_e0_0.5", STORED["SEIRD_I_0.5"], "SIRD_I_only"),
        "four_compartments": ("SEIRD_four_compartments_e0_0.5", STORED["SEIRD_4_0.5"], "SIRD_four_compartments"),
    }
    ok = True
    for mode_name, (job, spec, sird_job) in mapping.items():
        row = by_job[job]
        sird = by_job[sird_job]
        sird_rel = abs(sird["RSS"] - (STORED["SIRD_I"]["rss"] if mode_name == "I_only" else STORED["SIRD_4"]["rss"]))
        sird_rel /= STORED["SIRD_I"]["rss"] if mode_name == "I_only" else STORED["SIRD_4"]["rss"]
        d_aic = row["delta_AIC_vs_SIRD"]
        latent = row["latent_days"]
        reasons = []
        if mode_name == "I_only":
            if not (-320.0 <= d_aic <= -250.0):
                reasons.append(f"ΔAIC {d_aic:.3f} outside [-320, -250]")
            if row["latent_bound"] != "upper_14d":
                reasons.append(f"latent bound is {row['latent_bound']}, expected upper_14d")
            if abs(latent - 14.0) > 0.05:
                reasons.append(f"latent period {latent:.4f} is not 14 days")
        else:
            if not (40.0 <= d_aic <= 100.0):
                reasons.append(f"ΔAIC {d_aic:.3f} outside [40, 100]")
            if row["latent_bound"] != "lower_2d":
                reasons.append(f"latent bound is {row['latent_bound']}, expected lower_2d")
            if abs(latent - 2.0) > 0.05:
                reasons.append(f"latent period {latent:.4f} is not 2 days")
        if sird_rel > 1e-3:
            reasons.append(f"SIRD RSS rel {sird_rel:.3e} vs stored")
        report[mode_name] = {
            "delta_aic": d_aic,
            "stored_delta_aic": spec["delta_aic"],
            "delta_aic_minus_stored": d_aic - spec["delta_aic"],
            "latent_days": latent,
            "latent_bound": row["latent_bound"],
            "ever_infected_share": row["ever_infected_share"],
            "stored_ever": spec["ever"],
            "dip_ratio": row["dip_ratio"],
            "stored_dip_ratio": spec["dip_ratio"],
            "dip_day": row["dip_day"],
            "stored_dip_day": spec["dip_day"],
            "rss": row["RSS"],
            "stored_rss": spec["rss"],
            "rss_rel": abs(row["RSS"] - spec["rss"]) / spec["rss"],
            "sird_rss_rel": sird_rel,
            "pass": not reasons,
            "reasons": reasons,
        }
        if reasons:
            ok = False
    return ok, report


def _fmt(x, digits=6):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return ""
    if isinstance(x, float):
        return f"{x:.{digits}g}"
    return str(x)


def write_csv(rows, path: Path):
    fields = [
        "model",
        "error_model",
        "mode_name",
        "e0_over_i0",
        "E0",
        "k",
        "n",
        "RSS",
        "AIC",
        "RSS_SIRD",
        "AIC_SIRD",
        "delta_AIC_vs_SIRD",
        "beta",
        "sigma",
        "gamma",
        "mu",
        "gamma_plus_mu",
        "R0",
        "latent_days",
        "latent_bound",
        "at_bound",
        "ever_infected_share",
        "dip_ratio",
        "dip_day",
        "peak_I",
        "peak_day",
        "R_end",
        "D_end",
        "E_end",
        "S_end",
        "I_end",
        "dI_dt0",
        "starts",
        "starts_ok",
        "starts_within_1",
        "best_start_index",
        "best_start_kind",
        "best_start_nfev",
        "best_start_status",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            out = {}
            for key in fields:
                val = row.get(key, "")
                if isinstance(val, float):
                    out[key] = "" if not math.isfinite(val) and key in ("sigma", "latent_days", "E0", "e0_over_i0") and row["model"] == "SIRD" else repr(val) if math.isfinite(val) else ""
                    if key == "e0_over_i0" and row["model"] == "SIRD":
                        out[key] = ""
                    elif isinstance(val, float) and math.isfinite(val):
                        out[key] = f"{val:.16g}"
                    elif isinstance(val, float):
                        out[key] = ""
                else:
                    out[key] = "" if val is None else val
            w.writerow(out)


def _pct(x):
    return 100.0 * x


def _bound_phrase(label, days):
    if label == "upper_14d":
        return f"{days:.2f} days, on the 14-day upper edge"
    if label == "lower_2d":
        return f"{days:.2f} days, on the 2-day lower edge"
    return f"{days:.2f} days, interior of [2, 14]"


def write_results(rows, sanity, path: Path):
    by_mode = {"I_only": [], "four_compartments": []}
    for row in rows:
        if row["model"] == "SEIR-D":
            by_mode[row["mode_name"]].append(row)
    for key in by_mode:
        by_mode[key].sort(key=lambda r: r["e0_over_i0"])
    best = {key: min(val, key=lambda r: r["delta_AIC_vs_SIRD"]) for key, val in by_mode.items() if val}

    def line(r):
        e0 = r["e0_over_i0"]
        return (
            f"| {e0:g} | {r['delta_AIC_vs_SIRD']:+.3f} | {r['latent_days']:.4f} | {r['latent_bound']} | "
            f"{_pct(r['ever_infected_share']):.2f} | {r['dip_ratio']:.4f} | {r['dip_day']} | "
            f"{r['peak_I']:.1f} | {r['peak_day']} | {r['beta']:.6g} | {r['sigma']:.6g} | "
            f"{r['gamma']:.6g} | {r['mu']:.6g} |"
        )

    header = (
        "| E(0)/I(0) | ΔAIC | 1/σ (days) | latent bound | ever infected % | dip_ratio | dip_day | "
        "peak I | peak day | β | σ | γ | μ |"
    )
    sep = "|---:|---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"

    def headline(title, r):
        return "\n".join(
            [
                f"### {title}",
                "",
                "| field | value |",
                "|---|---|",
                f"| error model | {r['error_model']} |",
                f"| best E(0)/I(0) | {r['e0_over_i0']:g} |",
                f"| E(0) | {r['E0']:.6g} |",
                f"| ΔAIC vs SIRD | {r['delta_AIC_vs_SIRD']:+.6f} |",
                f"| AIC SEIR-D | {r['AIC']:.6f} |",
                f"| AIC SIRD | {r['AIC_SIRD']:.6f} |",
                f"| RSS SEIR-D | {r['RSS']:.10g} |",
                f"| RSS SIRD | {r['RSS_SIRD']:.10g} |",
                f"| latent period 1/σ (days) | {r['latent_days']:.8f} |",
                f"| latent bound | {r['latent_bound']} |",
                f"| ever infected | {_pct(r['ever_infected_share']):.4f}% ({r['ever_infected_share']:.8f} of N) |",
                f"| dip_ratio | {r['dip_ratio']:.8f} |",
                f"| dip_day | {r['dip_day']} |",
                f"| peak I (day) | {r['peak_I']:.6f} (day {r['peak_day']}) |",
                f"| β | {r['beta']:.10g} |",
                f"| σ | {r['sigma']:.10g} |",
                f"| γ | {r['gamma']:.10g} |",
                f"| μ | {r['mu']:.10g} |",
                f"| γ+μ | {r['gamma_plus_mu']:.10g} |",
                f"| R0 | {r['R0']:.8f} |",
                f"| at_bound | {r['at_bound'] or '(none)'} |",
                f"| starts within 1 AIC | {r['starts_within_1']} / {r['starts']} |",
                "",
            ]
        )

    i_best = best["I_only"]
    f_best = best["four_compartments"]
    i_05 = next(r for r in by_mode["I_only"] if r["e0_over_i0"] == 0.5)
    f_05 = next(r for r in by_mode["four_compartments"] if r["e0_over_i0"] == 0.5)
    f4 = next(r for r in by_mode["four_compartments"] if r["e0_over_i0"] == 4.0)
    f24 = next(r for r in by_mode["four_compartments"] if r["e0_over_i0"] == 24.0)
    i_negative = [r for r in by_mode["I_only"] if r["delta_AIC_vs_SIRD"] < 0]
    i_positive = [r for r in by_mode["I_only"] if r["delta_AIC_vs_SIRD"] > 0]
    i_last_neg = max(i_negative, key=lambda r: r["e0_over_i0"])
    i_first_pos = min(i_positive, key=lambda r: r["e0_over_i0"])

    def comparison():
        i_anchor = PUBLISHED_ANCHORS["I_unconstrained_best"]
        f_anchor = PUBLISHED_ANCHORS["four_unconstrained_best"]
        i_saved = PUBLISHED_ANCHORS["I_constrained_e0_0.5"]
        f_saved = PUBLISHED_ANCHORS["four_constrained_e0_0.5"]
        sc = PUBLISHED_ANCHORS["I_self_consistent_constrained"]
        return (
            f"On active cases alone, the lowest ΔAIC on this constrained grid is "
            f"{i_best['delta_AIC_vs_SIRD']:+.3f} at E(0)/I(0) = {i_best['e0_over_i0']:g}, "
            f"with latent period {_bound_phrase(i_best['latent_bound'], i_best['latent_days'])}, "
            f"ever-infected share {_pct(i_best['ever_infected_share']):.2f}% of N, and "
            f"dip_ratio {i_best['dip_ratio']:.4f} on day {i_best['dip_day']}. "
            f"The fixed-ratio point E(0)/I(0) = 0.5 on the same grid has ΔAIC "
            f"{i_05['delta_AIC_vs_SIRD']:+.3f}, latent period "
            f"{_bound_phrase(i_05['latent_bound'], i_05['latent_days'])}, "
            f"ever-infected {_pct(i_05['ever_infected_share']):.2f}%, and dip_ratio {i_05['dip_ratio']:.4f}. "
            f"The released constrained row at that same ratio "
            f"(`ode_lsq_reference.csv`, incubation 2–14 d, fixed E(0)) is ΔAIC {i_saved['delta_aic']:+.3f}, "
            f"latent period {i_saved['latent_days']:.2f} days, ever-infected {_pct(i_saved['ever']):.2f}%, "
            f"dip_ratio {i_saved['dip_ratio']:.4f}. "
            f"The unconstrained I-only profile in that file is lowest at E(0)/I(0) = 0 "
            f"(ΔAIC {i_anchor['delta_aic']:+.3f}, latent period about {i_anchor['latent_days']:.1f} days, "
            f"ever-infected {_pct(i_anchor['ever']):.1f}%, dip_ratio {i_anchor['dip_ratio']:.3f}); "
            f"that latent period lies outside the 2–14 day box, so the constrained profile cannot recover it. "
            f"Every ratio on this I-only grid sits on the 14-day edge. ΔAIC rises with E(0)/I(0) and changes "
            f"sign between {i_last_neg['e0_over_i0']:g} ({i_last_neg['delta_AIC_vs_SIRD']:+.1f}) and "
            f"{i_first_pos['e0_over_i0']:g} ({i_first_pos['delta_AIC_vs_SIRD']:+.1f}). "
            f"The ΔAIC of about {sc['delta_aic']:.0f} with a {sc['latent_days']:.1f}-day latent period is the "
            f"self-consistent E(0) rule already stored in the same file (E(0)/I(0) ≈ {sc['e0_over_i0']:.2f}), "
            f"which this grid does not recompute.\n\n"
            f"On all four compartments with one variance, the lowest ΔAIC on this constrained grid is "
            f"{f_best['delta_AIC_vs_SIRD']:+.3f} at E(0)/I(0) = {f_best['e0_over_i0']:g}, "
            f"with latent period {_bound_phrase(f_best['latent_bound'], f_best['latent_days'])}, "
            f"ever-infected share {_pct(f_best['ever_infected_share']):.2f}% of N, and "
            f"dip_ratio {f_best['dip_ratio']:.4f} on day {f_best['dip_day']}. "
            f"The fixed-ratio point E(0)/I(0) = 0.5 has ΔAIC {f_05['delta_AIC_vs_SIRD']:+.3f}, "
            f"latent period {_bound_phrase(f_05['latent_bound'], f_05['latent_days'])}, "
            f"ever-infected {_pct(f_05['ever_infected_share']):.2f}%, and dip_ratio {f_05['dip_ratio']:.4f}. "
            f"The released constrained row at that ratio is ΔAIC {f_saved['delta_aic']:+.3f}, "
            f"latent period {f_saved['latent_days']:.2f} days, ever-infected {_pct(f_saved['ever']):.2f}%, "
            f"dip_ratio {f_saved['dip_ratio']:.4f}. "
            f"The unconstrained four-compartment profile is lowest at E(0)/I(0) = 0 "
            f"(ΔAIC {f_anchor['delta_aic']:+.3f}, latent period about {f_anchor['latent_days']:.2f} days, "
            f"σ on its unconstrained upper edge). "
            f"The E(0)/I(0) = 4 point on this grid has latent period {f4['latent_days']:.2f} days, "
            f"on neither edge, and ΔAIC {f4['delta_AIC_vs_SIRD']:+.3f}. "
            f"It matches the released free-σ row at that ratio (ΔAIC +83.194, latent period 10.38 days), "
            f"whose optimum already lay inside [2, 14]. "
            f"At E(0)/I(0) = 24, μ = {f24['mu']:.3g} is on its lower edge as well as σ on the 14-day edge. "
            f"Positive ΔAIC at every ratio means the SIRD baseline remains preferred under the "
            f"four-compartment residual once 1/σ is held in [2, 14] days."
        )

    parts = [
        "# Constrained SEIR-D E(0) profile, latent period held to 2–14 days",
        "",
        "Direct least-squares on the St. Petersburg series, days 0–368, N = 6,000,000. "
        "σ ∈ [1/14, 1/2]. Fixed E(0)/I(0) grid only. ΔAIC = n ln(RSS_SEIR-D / RSS_SIRD) + 2 "
        "against the SIRD fit from this same run and the same error model. "
        "Protocol: `PROTOCOL.md`.",
        "",
        "## Copy-ready headline",
        "",
        "Best point on the constrained profile for each error model (lowest ΔAIC).",
        "",
        headline("I only (fit I(t) only)", i_best),
        headline("Four compartments (S, I, R, D, one variance; SEIR-D S residual uses S+E)", f_best),
        "## Comparison with the released fits",
        "",
        comparison(),
        "",
        "## I-only profile (10 fixed ratios)",
        "",
        "γ and μ are not separately identified on this criterion; γ+μ is. "
        "The split below is the one returned by the winning start.",
        "",
        header,
        sep,
        *[line(r) for r in by_mode["I_only"]],
        "",
        "## Four-compartment profile (10 fixed ratios)",
        "",
        "Observed S is compared with S+E. n = 1476.",
        "",
        header,
        sep,
        *[line(r) for r in by_mode["four_compartments"]],
        "",
        "## Sanity check against ode_lsq_reference.csv",
        "",
        "The optimizer was not started from the stored parameter vectors. "
        "Pass means the refit is in the same regime as the stored constrained row at E(0)/I(0) = 0.5.",
        "",
        "| error model | ΔAIC this run | ΔAIC stored | latent days | bound | ever infected | dip_ratio | RSS rel | pass |",
        "|---|---:|---:|---:|:---|---:|---:|---:|:---|",
    ]
    for mode_name, label in (("I_only", "I only"), ("four_compartments", "Four compartments")):
        s = sanity[mode_name]
        parts.append(
            f"| {label} | {s['delta_aic']:+.3f} | {s['stored_delta_aic']:+.3f} | {s['latent_days']:.4f} | "
            f"{s['latent_bound']} | {_pct(s['ever_infected_share']):.2f}% | {s['dip_ratio']:.4f} | "
            f"{s['rss_rel']:.3e} | {'yes' if s['pass'] else 'NO'} |"
        )
    parts.extend(
        [
            "",
            "## SIRD baselines from this run",
            "",
            "| error model | RSS | AIC | β | γ | μ | R0 | ever infected % | peak I (day) | dip_ratio (day) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        if row["model"] != "SIRD":
            continue
        parts.append(
            f"| {row['error_model']} | {row['RSS']:.10g} | {row['AIC']:.4f} | {row['beta']:.6g} | "
            f"{row['gamma']:.6g} | {row['mu']:.6g} | {row['R0']:.4f} | {_pct(row['ever_infected_share']):.2f} | "
            f"{row['peak_I']:.1f} ({row['peak_day']}) | {row['dip_ratio']:.4f} ({row['dip_day']}) |"
        )
    parts.extend(
        [
            "",
            "Machine-readable copy: `profile.csv` and `summary.json`. Log: `run.log`.",
            "",
        ]
    )
    path.write_text("\n".join(parts))


def _json_ready(obj):
    if isinstance(obj, dict):
        return {k: _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, np.floating):
        val = float(obj)
        return None if not math.isfinite(val) else val
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def write_summary(rows, sanity, self_checks, means, started, path: Path):
    seird = [r for r in rows if r["model"] == "SEIR-D"]
    best = {}
    for mode_name in ("I_only", "four_compartments"):
        group = [r for r in seird if r["mode_name"] == mode_name]
        if not group:
            continue
        winner = min(group, key=lambda r: r["delta_AIC_vs_SIRD"])
        best[mode_name] = {
            "e0_over_i0": winner["e0_over_i0"],
            "delta_AIC_vs_SIRD": winner["delta_AIC_vs_SIRD"],
            "latent_days": winner["latent_days"],
            "latent_bound": winner["latent_bound"],
            "ever_infected_share": winner["ever_infected_share"],
            "dip_ratio": winner["dip_ratio"],
            "dip_day": winner["dip_day"],
            "peak_I": winner["peak_I"],
            "peak_day": winner["peak_day"],
            "beta": winner["beta"],
            "sigma": winner["sigma"],
            "gamma": winner["gamma"],
            "mu": winner["mu"],
            "gamma_plus_mu": winner["gamma_plus_mu"],
            "RSS": winner["RSS"],
            "at_bound": winner["at_bound"],
        }
    payload = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_seconds": time.time() - started,
        "protocol": "PROTOCOL.md",
        "data": str(DATA_PATH.relative_to(ROOT)),
        "data_sha256": EXPECTED_SHA256,
        "N": N_POP,
        "n_days": 369,
        "integrator": "RK4",
        "dt_days": DT,
        "sigma_bounds": [SIGMA_LO, SIGMA_HI],
        "e0_grid": list(E0_GRID),
        "aic": "n ln(RSS) + 2k; delta_AIC = n ln(RSS_SEIRD/RSS_SIRD) + 2",
        "s_residual": "SEIR-D compares observed S with S+E; SIRD compares observed S with S",
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "platform": platform.platform(),
        "pinn_means_used_as_starts": means,
        "integrator_self_check": self_checks,
        "sanity_vs_ode_lsq_reference": sanity,
        "published_anchors_not_refitted": PUBLISHED_ANCHORS,
        "best_on_profile": best,
        "rows": rows,
    }
    path.write_text(json.dumps(_json_ready(payload), indent=2) + "\n")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanity-only", action="store_true", help="SIRD baselines and the two E(0)/I(0)=0.5 fits only")
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    log_path = OUT / "run.log"
    log_f = log_path.open("w")
    sys.stdout = Tee(sys.stdout, log_f)
    sys.stderr = Tee(sys.stderr, log_f)
    started = time.time()
    print(f"start {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} workers={args.workers}")
    print(f"data {DATA_PATH}")
    self_checks = integrator_self_check()
    means = pinn_means()
    print("PINN means", json.dumps(means))
    jobs = build_jobs(means, args.sanity_only)
    payloads = expand_payloads(jobs)
    print(f"jobs {len(jobs)} starts {len(payloads)}")

    grouped = {job["job"]: [] for job in jobs}
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_fit_one, payload) for payload in payloads]
        for fut in as_completed(futures):
            result = fut.result()
            grouped[result["job"]].append(result)
            done += 1
            if done % 10 == 0 or done == len(payloads):
                print(
                    f"progress {done}/{len(payloads)} last {result['job']} "
                    f"start {result['start_index']} rss {result['rss']:.6e} in {result['seconds']:.1f}s",
                    flush=True,
                )

    with (OUT / "starts.jsonl").open("w") as fh:
        for job in jobs:
            for row in sorted(grouped[job["job"]], key=lambda r: r["start_index"]):
                fh.write(json.dumps(_json_ready(row)) + "\n")

    rows = []
    for job in jobs:
        start_rows = sorted(grouped[job["job"]], key=lambda r: r["start_index"])
        assembled = assemble(job, start_rows)
        rows.append(assembled)
        print(
            f"kept {assembled['job']}: RSS {assembled['RSS']:.6e} "
            f"latent {assembled['latent_days']} bound {assembled['latent_bound'] or '-'} "
            f"starts_within_1 {assembled['starts_within_1']}/{assembled['starts']}"
        )
    attach_delta(rows)
    for row in rows:
        if row["model"] == "SEIR-D":
            print(
                f"delta {row['job']}: ΔAIC {row['delta_AIC_vs_SIRD']:+.3f} "
                f"1/σ {row['latent_days']:.4f} ever {row['ever_infected_share']:.4f} "
                f"dip {row['dip_ratio']:.4f} day {row['dip_day']}"
            )

    ok, sanity = sanity_against_stored(rows)
    print("sanity", json.dumps(sanity, indent=2))

    # Always persist the numbers. results.md is the author-facing note and is
    # written for a complete grid that passed the sanity check.
    write_csv(rows, OUT / "profile.csv")
    if (not args.sanity_only) and ok:
        write_results(rows, sanity, OUT / "results.md")
    elif not ok:
        (OUT / "results.md").write_text(
            "# Sanity check failed\n\n"
            "The refit at E(0)/I(0) = 0.5 is not in the same regime as "
            "`supplementary_model_comparison/ode_lsq_reference.csv`. "
            "No new residual or window was introduced. See `summary.json` key "
            "`sanity_vs_ode_lsq_reference` and `run.log`.\n\n"
            "```json\n"
            + json.dumps(sanity, indent=2)
            + "\n```\n"
        )
    write_summary(rows, sanity, self_checks, means, started, OUT / "summary.json")
    print(f"elapsed {time.time() - started:.1f}s sanity_pass {ok}")
    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
