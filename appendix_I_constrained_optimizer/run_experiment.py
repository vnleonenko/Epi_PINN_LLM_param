#!/usr/bin/env python3
"""Frozen constrained optimizer vs the Parameter Generator (protocol R3.2 / R1.2).

Runs the 24 problems in PROTOCOL.md: settings D, S and G, the six Table 2
requests (four of them under S), and targets T1 and T2.

Knobs below are fixed before optimization results are read. Do not edit them
to change a feasible/infeasible call. Smoke mode (--smoke) only checks that
the code path runs; it is not the reported experiment.

Peak day and peak height are taken from the Critic surrogate
(agents.SurrogateModel.SIRDSurrogate.simulate: RK45, t_max=400, 1000 output
points, peak = argmax of I). That is the function that reproduces Table 2.
A dt=0.1 parabolic interpolant is computed once as a diagnostic and is not
used, because it does not reproduce published day 264.3.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import sys
import time
import traceback
import warnings
from pathlib import Path

# One thread per process. The grid and the multistart are already parallel.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube

warnings.filterwarnings(
    "ignore",
    message="delta_grad == 0.0.*",
    category=UserWarning,
)

# ---------------------------------------------------------------------------
# Frozen protocol knobs. Set before any optimization result is inspected.
# ---------------------------------------------------------------------------

BETA0 = 0.1219
GAMMA0 = 0.0990
MU0 = 0.0099
POPULATION_EXPECTED = 6_000_000

# Published Phase-1 anchors used in the inequalities (PROTOCOL.md).
DAY0 = 264.3
HEIGHT0 = 39121.0
R_GROWTH0 = 0.0130
CFR0 = 0.0909

T_MAX = 400
N_OUT = 1000  # Critic / SIRDSurrogate output grid
GRID_N = 41
N_LHS = 20
LHS_SEED = 0
N_WORKERS = 4

# SLSQP / trust-constr. eps is the finite-difference step in the unit-cube
# coordinates the solver sees (the window is mapped onto [0, 1]^3).
SLSQP_OPTIONS = {"ftol": 1e-12, "eps": 1e-6, "maxiter": 200, "disp": False}
TRUST_OPTIONS = {"gtol": 1e-8, "xtol": 1e-10, "maxiter": 200, "verbose": 0}

# Floating-point comparison for r and CFR only. 1e-12 is far below the printed
# fourth decimal of r and below the ~1e-5 gap between CFR 0.095890 and the
# printed T2 level 0.0959. It exists so a decimal parameter set whose exact
# decimal r equals the printed target is not rejected by binary rounding.
# Peak day and peak height are compared with no added slack.
ALGEBRAIC_ATOL = 1e-12

# "J does not increase" for the optional polish, and "worse than the start".
J_COMPARE_ATOL = 1e-12

# Unit-cube bound violation above this is a failed start (then trust-constr).
BOUND_VIOLATION_ATOL = 1e-8

# Critic thresholds from main_test.py (position_threshold, height_threshold)
# and DeterministicCriticAgent defaults (growth, mortality). Holds use the
# pipeline values ±8 days and ±3% height.
POSITION_THRESHOLD = 1.0
HEIGHT_THRESHOLD = 0.03
GROWTH_THRESHOLD = 0.03
MORTALITY_THRESHOLD = 0.03
POSITION_TOLERANCE = 8.0
HEIGHT_TOLERANCE = 0.03

HERE = Path(__file__).resolve().parent
REPO = HERE.parent  # repository root (agents/, "Phase 1 (model calibration)/")
SRC = REPO
DATA_CSV = SRC / "Phase 1 (model calibration)" / "real_datasets" / "covid-19_Kouprianov.csv"
sys.path.insert(0, str(SRC))

from agents.SurrogateModel import SIRDSurrogate  # noqa: E402

# Initial state, filled by load_initial_state() before any solve.
S0 = I0 = R0_COMP = D0 = POPULATION = None
_SUR = None


def load_initial_state():
    """First row of the Saint Petersburg series, the same row the Critic uses."""
    global S0, I0, R0_COMP, D0, POPULATION, _SUR
    with DATA_CSV.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    S0 = int(float(row["S"]))
    I0 = int(float(row["I"]))
    R0_COMP = int(float(row["R"]))
    D0 = int(float(row["D"]))
    POPULATION = S0 + I0 + R0_COMP + D0
    if (S0, I0, R0_COMP, D0, POPULATION) != (5995513, 4197, 248, 42, POPULATION_EXPECTED):
        raise SystemExit(
            f"Initial state {S0, I0, R0_COMP, D0, POPULATION} is not the Table 2 state."
        )
    _SUR = SIRDSurrogate(POPULATION, S0, I0, R0_COMP, D0)


def _init_worker():
    global _SUR
    _SUR = SIRDSurrogate(POPULATION, S0, I0, R0_COMP, D0)


# Printed windows. Endpoints are the decimals in PROTOCOL.md (closed).
WINDOWS = {
    "D": ((0.1119, 0.1319), (0.0985, 0.0995), (0.00985, 0.00995)),
    "M": ((0.1119, 0.1319), (0.0985, 0.0995), (0.0089, 0.0109)),
    "S": ((0.1119, 0.1319), (0.0890, 0.1090), (0.00985, 0.00995)),
}

REQUESTS = {
    "D": ("higher", "lower", "earlier", "later", "growth", "mortality"),
    "S": ("higher", "lower", "earlier", "later"),
    "G": ("growth", "mortality"),
}

# Published reference rows. r and CFR in the tables are recomputed; the
# printed numbers are kept for the discrepancy check.
TABLE2 = {
    "higher": dict(beta=0.1319, gamma=0.0990, mu=0.0099, r=0.0230, cfr=0.0909, day=197.8, height=100455.0),
    "lower": dict(beta=0.1169, gamma=0.0990, mu=0.0099, r=0.0080, cfr=0.0909, day=305.9, height=18273.0),
    "earlier": dict(beta=0.1319, gamma=0.0990, mu=0.0099, r=0.0230, cfr=0.0909, day=197.8, height=100455.0),
    "later": dict(beta=0.1150, gamma=0.0990, mu=0.0099, r=0.0061, cfr=0.0909, day=312.7, height=12555.0),
    "growth": dict(beta=0.1319, gamma=0.0985, mu=0.0099, r=0.0235, cfr=0.0913, day=195.4, height=104848.0),
    "mortality": dict(beta=0.1219, gamma=0.0990, mu=0.0105, r=0.0124, cfr=0.0959, day=268.3, height=35894.0),
}
TABLE3 = {
    "higher": dict(beta=0.1169, gamma=0.0940, mu=0.0099, r=0.0130, cfr=0.0953, day=271.5, height=42257.0),
    "lower": dict(beta=0.1269, gamma=0.1040, mu=0.0099, r=0.0130, cfr=0.0869, day=257.1, height=36356.0),
    "earlier": dict(beta=0.1319, gamma=0.1079, mu=0.0099, r=0.0141, cfr=0.0840, day=244.2, height=39121.0),
    "later": dict(beta=0.1119, gamma=0.0901, mu=0.0099, r=0.0119, cfr=0.0990, day=287.9, height=39121.0),
}

# T2 named levels, copied from the protocol tables. D and G use Table 2.
T2_LEVEL = {
    ("D", "higher"): ("height", "ge", 100455.0),
    ("D", "lower"): ("height", "le", 18273.0),
    ("D", "earlier"): ("day", "le", 197.8),
    ("D", "later"): ("day", "ge", 312.7),
    ("D", "growth"): ("r", "ge", 0.0235),
    ("D", "mortality"): ("cfr", "ge", 0.0959),
    ("G", "growth"): ("r", "ge", 0.0235),
    ("G", "mortality"): ("cfr", "ge", 0.0959),
    ("S", "higher"): ("height", "ge", 42257.0),
    ("S", "lower"): ("height", "le", 36356.0),
    ("S", "earlier"): ("day", "le", 244.2),
    ("S", "later"): ("day", "ge", 287.9),
}

PHRASES = {
    "higher": "Peak should be higher",
    "lower": "Peak should be lower",
    "earlier": "Peak should be earlier",
    "later": "Peak should be later",
    "growth": "faster initial growth",
    "mortality": "higher mortality",
}


def window_id(setting, request):
    if setting == "S":
        return "S"
    if request == "mortality":
        return "M"
    return "D"


def holds(setting, request):
    """Return (hold_day, hold_height) for the unnamed / peak axes."""
    if setting == "D":
        return False, False
    if setting == "S":
        if request in ("higher", "lower"):
            return True, False
        return False, True
    return True, True  # G


def expectations(setting, request):
    """Directions passed to the Critic checks. Unmentioned axes are 'any'."""
    pos, height, growth, mort = "any", "any", "any", "any"
    if request == "higher":
        height = "higher"
    elif request == "lower":
        height = "lower"
    elif request == "earlier":
        pos = "earlier"
    elif request == "later":
        pos = "later"
    elif request == "growth":
        growth = "faster"
    elif request == "mortality":
        mort = "higher"
    hold_day, hold_height = holds(setting, request)
    if hold_day and pos == "any":
        pos = "unchanged"
    if hold_height and height == "any":
        height = "unchanged"
    return pos, height, growth, mort


def t1_level(request):
    if request == "higher":
        return "height", "ge", HEIGHT0 * 1.03
    if request == "lower":
        return "height", "le", HEIGHT0 * 0.97
    if request == "earlier":
        return "day", "le", DAY0 - 1.0
    if request == "later":
        return "day", "ge", DAY0 + 1.0
    if request == "growth":
        return "r", "ge", R_GROWTH0 * 1.03
    if request == "mortality":
        return "cfr", "ge", CFR0 * 1.03
    raise KeyError(request)


def named_level(setting, request, target):
    if target == "T1":
        return t1_level(request)
    return T2_LEVEL[(setting, request)]


def ref_row(setting, request):
    table = TABLE3 if setting == "S" else TABLE2
    return table[request], ("Table 3" if setting == "S" else "Table 2")


def r_of(beta, gamma, mu):
    return beta - gamma - mu


def cfr_of(gamma, mu):
    return mu / (gamma + mu)


def r0_of(beta, gamma, mu):
    return beta / (gamma + mu)


def j_of(beta, gamma, mu):
    beta = np.asarray(beta, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    mu = np.asarray(mu, dtype=float)
    return ((beta - BETA0) / BETA0) ** 2 + ((gamma - GAMMA0) / GAMMA0) ** 2 + ((mu - MU0) / MU0) ** 2


def max_rel_of(beta, gamma, mu):
    beta = np.asarray(beta, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    mu = np.asarray(mu, dtype=float)
    return np.maximum(
        np.maximum(np.abs(beta - BETA0) / BETA0, np.abs(gamma - GAMMA0) / GAMMA0),
        np.abs(mu - MU0) / MU0,
    )


def as_float(value):
    return float(np.asarray(value, dtype=float))


def unpack(u, bounds):
    u = np.asarray(u, dtype=float)
    beta = bounds[0][0] + float(u[0]) * (bounds[0][1] - bounds[0][0])
    gamma = bounds[1][0] + float(u[1]) * (bounds[1][1] - bounds[1][0])
    mu = bounds[2][0] + float(u[2]) * (bounds[2][1] - bounds[2][0])
    return beta, gamma, mu


def pack(beta, gamma, mu, bounds):
    vals = (beta, gamma, mu)
    out = []
    for value, (lo, hi) in zip(vals, bounds):
        out.append((value - lo) / (hi - lo))
    return np.asarray(out, dtype=float)


def in_window(beta, gamma, mu, bounds):
    beta = np.asarray(beta, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    mu = np.asarray(mu, dtype=float)
    return (
        (beta >= bounds[0][0]) & (beta <= bounds[0][1])
        & (gamma >= bounds[1][0]) & (gamma <= bounds[1][1])
        & (mu >= bounds[2][0]) & (mu <= bounds[2][1])
    )


def _compare(value, op, level, kind):
    atol = ALGEBRAIC_ATOL if kind in ("r", "cfr") else 0.0
    if op == "ge":
        return value >= (level - atol)
    if op == "le":
        return value <= (level + atol)
    raise ValueError(op)


def protocol_mask(setting, request, target, day, height, beta, gamma, mu, bounds):
    """Protocol inequalities and the closed parameter window. True = feasible."""
    day = np.asarray(day, dtype=float)
    height = np.asarray(height, dtype=float)
    beta = np.asarray(beta, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    mu = np.asarray(mu, dtype=float)
    mask = in_window(beta, gamma, mu, bounds) & np.isfinite(day) & np.isfinite(height)
    kind, op, level = named_level(setting, request, target)
    if kind == "height":
        value = height
    elif kind == "day":
        value = day
    elif kind == "r":
        value = r_of(beta, gamma, mu)
    elif kind == "cfr":
        value = cfr_of(gamma, mu)
    else:
        raise KeyError(kind)
    mask = mask & _compare(value, op, level, kind)
    hold_day, hold_height = holds(setting, request)
    if hold_day:
        mask = mask & (day >= DAY0 - 8.0) & (day <= DAY0 + 8.0)
    if hold_height:
        mask = mask & (height >= HEIGHT0 * 0.97) & (height <= HEIGHT0 * 1.03)
    return mask


def protocol_feasible(setting, request, target, day, height, beta, gamma, mu, bounds):
    return bool(np.asarray(protocol_mask(
        setting, request, target, day, height, beta, gamma, mu, bounds
    )).reshape(()))


# --- Critic checks, same decisions as DeterministicCriticAgent ---------------
# Source: agents/DeterministicCriticAgent.py
# _check_position, _check_height, _check_relative, _make_decision.
# The agent module imports langchain, which is not required for these checks.
# Thresholds: main_test.py OptimizationPipeline (position 1 day, height 3%,
# hold ±8 days / ±3% when the axis is held). Growth and mortality stay at the
# class defaults of 3%. Expectations are the Table 2 mapping: no Intent Parser
# log is in the repository, and no LLM is called.

def check_position(baseline, new, expected):
    change = new - baseline
    is_significant = abs(change) >= POSITION_THRESHOLD if POSITION_THRESHOLD > 0 else abs(change) > 1e-6
    if expected == "any":
        return True
    if expected == "later":
        return bool(change > 0 and is_significant)
    if expected == "earlier":
        return bool(change < 0 and is_significant)
    return bool(abs(change) <= POSITION_TOLERANCE)


def check_height(baseline, new, expected):
    change = new - baseline
    relative_change = abs(change) / baseline if baseline > 0 else 0.0
    is_significant = relative_change >= HEIGHT_THRESHOLD if HEIGHT_THRESHOLD > 0 else abs(change) > 1e-6
    if expected == "any":
        return True
    if expected == "higher":
        return bool(change > 0 and is_significant)
    if expected == "lower":
        return bool(change < 0 and is_significant)
    return bool(relative_change <= HEIGHT_TOLERANCE)


def check_relative(baseline, new, expected, threshold):
    change = new - baseline
    rel = change / abs(baseline) if baseline else 0.0
    if expected in (None, "any"):
        return True
    if expected in ("higher", "faster", "increase"):
        return bool(rel >= threshold)
    if expected in ("lower", "slower", "decrease"):
        return bool(rel <= -threshold)
    return bool(abs(rel) < threshold)


def critic_accepts(setting, request, day, height, beta, gamma, mu, baseline):
    pos_e, height_e, growth_e, mort_e = expectations(setting, request)
    pos_ok = check_position(baseline["day"], day, pos_e)
    height_ok = check_height(baseline["height"], height, height_e)
    growth_ok = check_relative(baseline["r"], as_float(r_of(beta, gamma, mu)), growth_e, GROWTH_THRESHOLD)
    mort_ok = check_relative(baseline["cfr"], as_float(cfr_of(gamma, mu)), mort_e, MORTALITY_THRESHOLD)
    return bool(pos_ok and height_ok and growth_ok and mort_ok)


def simulate_peak(surrogate, beta, gamma, mu):
    res = surrogate.simulate(beta, gamma, mu, t_max=T_MAX, num_points=N_OUT)
    if res is None:
        return math.nan, math.nan, None
    return float(res["peak_position"]), float(res["peak_height"]), res


def parabolic_peak(t, y):
    """Three-point parabolic interpolation of a discrete maximum. Diagnostic only."""
    k = int(np.argmax(y))
    if k <= 0 or k >= len(y) - 1:
        return float(t[k]), float(y[k]), "edge"
    y0, y1, y2 = float(y[k - 1]), float(y[k]), float(y[k + 1])
    denom = y0 - 2.0 * y1 + y2
    if denom == 0.0:
        return float(t[k]), y1, "flat"
    dt = float(t[k] - t[k - 1])
    delta = 0.5 * (y0 - y2) / denom
    t_peak = float(t[k] + delta * dt)
    h_peak = y1 + ((y2 - y0) / 2.0) * delta + (denom / 2.0) * delta * delta
    return t_peak, h_peak, "ok"


def parabolic_diagnostic(surrogate):
    n = int(round(T_MAX / 0.1)) + 1
    t_eval = np.linspace(0.0, T_MAX, n)
    sol = solve_ivp(
        surrogate.ode_system,
        (0.0, T_MAX),
        [S0, I0, R0_COMP, D0],
        args=(BETA0, GAMMA0, MU0),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
    )
    if not sol.success:
        raise RuntimeError(f"parabolic diagnostic failed: {sol.message}")
    t_peak, h_peak, flag = parabolic_peak(sol.t, sol.y[1])
    return {
        "dt_day": float(t_eval[1] - t_eval[0]),
        "n_points": int(n),
        "flag": flag,
        "peak_day": t_peak,
        "peak_height": h_peak,
        "rounded_day": round(t_peak, 1),
        "rounded_height": round(h_peak),
        "nfev": int(sol.nfev),
    }


def conservation_report(res):
    S, I, R, D = (np.asarray(res[k], dtype=float) for k in ("S", "I", "R", "D"))
    total = S + I + R + D
    return {
        "min_S": float(np.min(S)),
        "min_I": float(np.min(I)),
        "min_R": float(np.min(R)),
        "min_D": float(np.min(D)),
        "max_abs_conservation_error": float(np.max(np.abs(total - POPULATION))),
        "nonnegative": bool(np.min(S) >= -1e-6 and np.min(I) >= -1e-6 and np.min(R) >= -1e-6 and np.min(D) >= -1e-6),
        "conserved": bool(np.max(np.abs(total - POPULATION)) <= 1e-4),
    }


def constraint_text(setting, request, target):
    kind, op, level = named_level(setting, request, target)
    symbol = {"height": "peak_height", "day": "peak_day", "r": "r", "cfr": "CFR"}[kind]
    rel = ">=" if op == "ge" else "<="
    text = f"{symbol} {rel} {level:.6g}"
    hold_day, hold_height = holds(setting, request)
    if hold_day:
        text += f"; peak_day in [{DAY0 - 8:.1f}, {DAY0 + 8:.1f}]"
    if hold_height:
        text += f"; peak_height in [{HEIGHT0 * 0.97:.2f}, {HEIGHT0 * 1.03:.2f}]"
    return text


def value_of(kind, day, height, beta, gamma, mu):
    if kind == "height":
        return height
    if kind == "day":
        return day
    if kind == "r":
        return as_float(r_of(beta, gamma, mu))
    if kind == "cfr":
        return as_float(cfr_of(gamma, mu))
    raise KeyError(kind)


def build_constraints(bounds, setting, request, target, peak):
    """Inequality callables c(u) >= 0 in unit-cube coordinates."""
    constraints = []

    def add_peak(kind, op, level):
        atol = ALGEBRAIC_ATOL if kind in ("r", "cfr") else 0.0

        def fun(u, kind=kind, op=op, level=level, atol=atol):
            beta, gamma, mu = unpack(u, bounds)
            day, height = peak(beta, gamma, mu)
            if not (math.isfinite(day) and math.isfinite(height)):
                return -1e6
            value = value_of(kind, day, height, beta, gamma, mu)
            if op == "ge":
                return value - (level - atol)
            return (level + atol) - value

        constraints.append({"type": "ineq", "fun": fun})

    def add_algebraic(kind, op, level):
        atol = ALGEBRAIC_ATOL

        def fun(u, kind=kind, op=op, level=level):
            beta, gamma, mu = unpack(u, bounds)
            value = value_of(kind, math.nan, math.nan, beta, gamma, mu)
            if op == "ge":
                return value - (level - atol)
            return (level + atol) - value

        def jac(u, kind=kind, op=op):
            beta, gamma, mu = unpack(u, bounds)
            widths = np.array([hi - lo for lo, hi in bounds], dtype=float)
            grad = np.zeros(3, dtype=float)
            if kind == "r":
                grad[:] = (1.0, -1.0, -1.0)
            else:
                denom = (gamma + mu) ** 2
                grad[1] = -mu / denom
                grad[2] = gamma / denom
            if op == "le":
                grad *= -1.0
            return grad * widths

        constraints.append({"type": "ineq", "fun": fun, "jac": jac})

    kind, op, level = named_level(setting, request, target)
    if kind in ("r", "cfr"):
        add_algebraic(kind, op, level)
    else:
        add_peak(kind, op, level)
    hold_day, hold_height = holds(setting, request)
    if hold_day:
        add_peak("day", "ge", DAY0 - 8.0)
        add_peak("day", "le", DAY0 + 8.0)
    if hold_height:
        add_peak("height", "ge", HEIGHT0 * 0.97)
        add_peak("height", "le", HEIGHT0 * 1.03)
    return constraints


def j_grad_u(beta, gamma, mu, bounds):
    widths = np.array([hi - lo for lo, hi in bounds], dtype=float)
    grad_p = np.array([
        2.0 * (beta - BETA0) / BETA0 ** 2,
        2.0 * (gamma - GAMMA0) / GAMMA0 ** 2,
        2.0 * (mu - MU0) / MU0 ** 2,
    ], dtype=float)
    return grad_p * widths


def sort_key(point):
    return (point["J"], point["max_rel"], point["gamma"], point["mu"], point["beta"])


def optimize_from(surrogate, u0, bounds, setting, request, target, slsqp_options, trust_options, allow_trust):
    """One start. SLSQP first; trust-constr if that start fails or is worse."""
    cache = {}
    n_sim = [0]

    def peak(beta, gamma, mu):
        key = (float(beta), float(gamma), float(mu))
        if key not in cache:
            n_sim[0] += 1
            day, height, _res = simulate_peak(surrogate, beta, gamma, mu)
            cache[key] = (day, height)
        return cache[key]

    def fun(u):
        beta, gamma, mu = unpack(u, bounds)
        return as_float(j_of(beta, gamma, mu))

    def jac(u):
        beta, gamma, mu = unpack(u, bounds)
        return j_grad_u(beta, gamma, mu, bounds)

    constraints = build_constraints(bounds, setting, request, target, peak)
    u0 = np.asarray(u0, dtype=float)
    j_start = fun(u0)

    def run_one(method, options):
        try:
            res = minimize(
                fun,
                u0,
                method=method,
                jac=jac,
                bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
                constraints=constraints,
                options=options,
            )
            u = np.asarray(res.x, dtype=float)
            success = bool(res.success)
            message = str(res.message)
        except Exception as exc:  # solver failure is a failed start, not a crash of the study
            return {
                "ok_call": False,
                "success": False,
                "feasible": False,
                "solver": method,
                "message": repr(exc),
                "J": None,
            }
        if u.shape != (3,) or not np.all(np.isfinite(u)):
            return {
                "ok_call": True,
                "success": False,
                "feasible": False,
                "solver": method,
                "message": message,
                "J": None,
            }
        violation = float(max(0.0, np.max(np.maximum(-u, u - 1.0))))
        if violation > BOUND_VIOLATION_ATOL:
            return {
                "ok_call": True,
                "success": success,
                "feasible": False,
                "solver": method,
                "message": message,
                "bound_violation": violation,
                "J": None,
            }
        u = np.clip(u, 0.0, 1.0)
        beta, gamma, mu = unpack(u, bounds)
        day, height = peak(beta, gamma, mu)
        feasible = protocol_feasible(setting, request, target, day, height, beta, gamma, mu, bounds)
        jay = as_float(j_of(beta, gamma, mu))
        return {
            "ok_call": True,
            "success": success,
            "feasible": bool(feasible),
            "solver": method,
            "message": message,
            "bound_violation": violation,
            "beta": beta,
            "gamma": gamma,
            "mu": mu,
            "day": day,
            "height": height,
            "J": jay,
            "max_rel": as_float(max_rel_of(beta, gamma, mu)),
        }

    slsqp = run_one("SLSQP", slsqp_options)
    needs_trust = allow_trust and (
        (not slsqp.get("ok_call", False))
        or (not slsqp.get("success", False))
        or (not slsqp.get("feasible", False))
        or (slsqp.get("J") is None)
        or (slsqp["J"] > j_start + J_COMPARE_ATOL)
    )
    trust = run_one("trust-constr", trust_options) if needs_trust else None
    feasible_runs = [item for item in (slsqp, trust) if item is not None and item.get("feasible")]
    if feasible_runs:
        kept = min(feasible_runs, key=sort_key)
    else:
        kept = {
            "feasible": False,
            "solver": None,
            "J": None,
            "beta": None,
            "gamma": None,
            "mu": None,
            "day": None,
            "height": None,
            "max_rel": None,
        }
    kept = dict(kept)
    kept["n_sim"] = n_sim[0]
    kept["j_start"] = j_start
    kept["trust_called"] = trust is not None
    kept["slsqp_success"] = bool(slsqp.get("success", False))
    kept["slsqp_feasible"] = bool(slsqp.get("feasible", False))
    kept["slsqp_J"] = slsqp.get("J")
    kept["slsqp_message"] = slsqp.get("message")
    kept["trust_success"] = None if trust is None else bool(trust.get("success", False))
    kept["trust_feasible"] = None if trust is None else bool(trust.get("feasible", False))
    kept["trust_message"] = None if trust is None else trust.get("message")
    return kept


def _optimize_start(payload):
    u0, bounds, setting, request, target, slsqp_options, trust_options = payload
    try:
        return optimize_from(
            _SUR, u0, bounds, setting, request, target, slsqp_options, trust_options, True
        )
    except Exception:
        return {
            "feasible": False,
            "solver": None,
            "J": None,
            "n_sim": 0,
            "trust_called": False,
            "slsqp_success": False,
            "slsqp_feasible": False,
            "error": traceback.format_exc(),
        }


def _peaks_batch(triples):
    n = len(triples)
    days = np.empty(n, dtype=float)
    heights = np.empty(n, dtype=float)
    for i, row in enumerate(triples):
        day, height, _res = simulate_peak(_SUR, float(row[0]), float(row[1]), float(row[2]))
        days[i] = day
        heights[i] = height
    return days, heights


def evaluate_grid(pool, axes, log):
    params = np.column_stack([
        grid.ravel() for grid in np.meshgrid(*axes, indexing="ij")
    ])
    batch = 1000
    batches = [params[i:i + batch] for i in range(0, len(params), batch)]
    days = np.empty(len(params), dtype=float)
    heights = np.empty(len(params), dtype=float)
    offset = 0
    t0 = time.perf_counter()
    for i, (days_b, heights_b) in enumerate(pool.imap(_peaks_batch, batches, chunksize=1)):
        n = len(days_b)
        days[offset:offset + n] = days_b
        heights[offset:offset + n] = heights_b
        offset += n
        if i % 10 == 0 or i == len(batches) - 1:
            log(f"  grid batch {i + 1}/{len(batches)}")
    elapsed = time.perf_counter() - t0
    if offset != len(params):
        raise RuntimeError("grid evaluation did not cover every point")
    return params, days, heights, elapsed


def best_mask_point(params, days, heights, mask):
    idx = np.flatnonzero(np.asarray(mask))
    if len(idx) == 0:
        return None
    beta = params[idx, 0]
    gamma = params[idx, 1]
    mu = params[idx, 2]
    jay = np.asarray(j_of(beta, gamma, mu), dtype=float)
    rel = np.asarray(max_rel_of(beta, gamma, mu), dtype=float)
    order = np.lexsort((beta, mu, gamma, rel, jay))
    i = int(idx[int(order[0])])
    return {
        "beta": float(params[i, 0]),
        "gamma": float(params[i, 1]),
        "mu": float(params[i, 2]),
        "day": float(days[i]),
        "height": float(heights[i]),
        "J": float(jay[int(order[0])]),
        "max_rel": float(rel[int(order[0])]),
        "solver": "grid",
        "feasible": True,
    }


def named_change(request, day, height, beta, gamma, mu):
    if request in ("higher", "lower"):
        return (height - HEIGHT0) / HEIGHT0 * 100.0, "percent_height"
    if request in ("earlier", "later"):
        return day - DAY0, "days"
    if request == "growth":
        r_value = as_float(r_of(beta, gamma, mu))
        return (r_value - R_GROWTH0) / R_GROWTH0 * 100.0, "percent_r"
    r_cfr = as_float(cfr_of(gamma, mu))
    return (r_cfr - CFR0) / CFR0 * 100.0, "percent_CFR"


def side_effect(request, day, height):
    """Scalar side effect, frozen with the protocol.

    Peak-height requests: absolute peak-day shift in days.
    Peak-day requests: absolute peak-height shift in percent of height0.
    Growth and mortality: the larger of |day shift|/8 and |height shift|/3%,
    the two hold half-widths, so the axes share a scale.
    """
    day_shift = abs(day - DAY0)
    height_pct = abs(height - HEIGHT0) / HEIGHT0 * 100.0
    if request in ("higher", "lower"):
        return day_shift
    if request in ("earlier", "later"):
        return height_pct
    return max(day_shift / 8.0, height_pct / 3.0)


def find_intent_logs(repo):
    hits = []
    parser_hits = []
    skip = {".git", "__pycache__", "node_modules", ".venv"}
    for path in repo.rglob("*"):
        if not path.is_file():
            continue
        if any(part in skip for part in path.parts):
            continue
        if path.suffix.lower() not in {".json", ".log", ".txt", ".csv"}:
            continue
        if "intent" not in path.name.lower():
            continue
        rel = str(path.relative_to(repo))
        hits.append(rel)
        try:
            if path.stat().st_size > 2_000_000:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")[:100_000]
        except OSError:
            continue
        if "position_direction" in text or "expected_position" in text:
            parser_hits.append(rel)
    return hits, parser_hits


def point_record(beta, gamma, mu, day, height):
    return {
        "beta": beta,
        "gamma": gamma,
        "mu": mu,
        "R0": as_float(r0_of(beta, gamma, mu)),
        "r": as_float(r_of(beta, gamma, mu)),
        "CFR": as_float(cfr_of(gamma, mu)),
        "peak_day": day,
        "peak_height": height,
        "J": as_float(j_of(beta, gamma, mu)),
        "max_rel_change": as_float(max_rel_of(beta, gamma, mu)),
    }


def fmt_cell(value):
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return ""
        return format(float(value), ".16g")
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    return str(value)


def write_csv(path, rows):
    columns = [
        "request", "setting", "target", "phrase", "window", "constraint",
        "beta", "gamma", "mu", "R0", "r", "CFR",
        "peak_day", "peak_height", "J", "max_rel_change",
        "named_quantity", "named_value", "named_change", "named_change_unit",
        "unnamed_peak_day_change_days", "unnamed_peak_height_change_pct",
        "side_effect", "feasible", "critic_accept",
        "n_simulations", "wall_time_s", "solver_used", "n_starts", "n_starts_converged",
        "grid_n_points", "grid_n_feasible",
        "ref_table", "ref_beta", "ref_gamma", "ref_mu", "ref_named_value", "ref_J",
        "ref_r", "ref_CFR", "ref_peak_day_simulated", "ref_peak_height_simulated",
        "ref_peak_day_published", "ref_peak_height_published", "ref_peak_matches_published",
        "ref_feasible", "ref_side_effect",
        "generator_smaller_J", "generator_smaller_side_effect",
        "generator_feasible_when_optimizer_not",
        "ref_J_smaller_ignoring_feasibility",
        "optimizer_feasible", "optimizer_beta", "optimizer_gamma", "optimizer_mu",
        "optimizer_J", "optimizer_peak_day", "optimizer_peak_height", "optimizer_solver",
        "optimizer_n_simulations", "polish_kept",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt_cell(row.get(key)) for key in columns})


def yn(flag):
    if flag is None:
        return "n/a"
    return "yes" if flag else "no"


def write_report(path, rows, meta):
    lines = []
    lines.append("# Constrained optimizer versus the Parameter Generator")
    lines.append("")
    lines.append("The protocol in `PROTOCOL.md` was frozen before this run. Windows, Critic thresholds, the objective, the Latin-hypercube seed, the number of starts, the 41³ grids, and the solver options were not changed after the results were known. This comparison is not a forecast test: it asks whether a constrained minimizer of the relative parameter change can meet the same Critic inequalities as the published Generator and starter points.")
    lines.append("")
    lines.append("## Phase-1 sanity check")
    lines.append("")
    san = meta["sanity"]
    lines.append(
        f"The Critic surrogate `SIRDSurrogate.simulate` (RK45, rtol 1e-6, atol 1e-8, horizon {T_MAX} days, {N_OUT} output times, peak = argmax of I) at β = {BETA0}, γ = {GAMMA0}, μ = {MU0} gives peak day {san['critic_day']:.10f} and peak height {san['critic_height']:.6f}. Rounded to the published precision these are day {san['critic_day_rounded']:.1f} and height {san['critic_height_rounded']:.0f}. The published Table 2 anchors are day 264.3 and height 39121. The check passes."
    )
    lines.append("")
    para = san["parabolic"]
    lines.append(
        f"A diagnostic that the protocol also names, a time step of {para['dt_day']:.1f} day with three-point parabolic interpolation of the same ODE, gives day {para['peak_day']:.6f} and height {para['peak_height']:.4f}, which round to {para['rounded_day']:.1f} and {para['rounded_height']:.0f}. The day does not match 264.3. That interpolant was not used for the objective, the constraints, or the grid, and it was not retuned. The 24 problems use the Critic argmax, which is the definition that matches Table 2."
    )
    lines.append("")
    lines.append(
        f"Initial state from `{meta['data_relpath']}`, first row: S = {S0}, I = {I0}, R = {R0_COMP}, D = {D0}, N = {POPULATION}. Intent Parser logs for the Table 2 runs were not found ({meta['intent_note']}). No LLM was called. The six requests use the protocol mapping: {', '.join(PHRASES[k] + ' → ' + k for k in ('higher', 'lower', 'earlier', 'later', 'growth', 'mortality'))}."
    )
    lines.append("")
    lines.append("## Reference peaks")
    lines.append("")
    lines.append("Each published parameter vector was integrated with the same surrogate. A mismatch is reported and the T2 target is left at the printed number.")
    lines.append("")
    lines.append("| Table | Request | Published day, height | Simulated day, height | Rounded match |")
    lines.append("| --- | --- | --- | --- | --- |")
    for item in meta["references"]:
        lines.append(
            f"| {item['table']} | {item['request']} | {item['published_day']:.1f}, {item['published_height']:.0f} | {item['day']:.4f}, {item['height']:.2f} | {yn(item['match'])} |"
        )
    if meta["reference_mismatches"]:
        lines.append("")
        lines.append("Mismatches at the published rounding:")
        for item in meta["reference_mismatches"]:
            lines.append(
                f"- {item['table']} {item['request']}: simulated day {item['day']:.4f} (rounds to {item['rounded_day']:.1f}) and height {item['height']:.2f} (rounds to {item['rounded_height']:.0f}); published day {item['published_day']:.1f}, height {item['published_height']:.0f}."
            )
    else:
        lines.append("")
        lines.append("Every reference row matches the published day and height at the published rounding.")
    lines.append("")
    lines.append("## All 24 problems")
    lines.append("")
    lines.append(
        "J = ((β−β0)/β0)² + ((γ−γ0)/γ0)² + ((μ−μ0)/μ0)² with the Phase-1 point (0.1219, 0.0990, 0.0099). "
        f"A problem is feasible only when at least one point of its {meta['grid_n']}³ grid meets the protocol inequalities inside the window. "
        "The reported point is then the feasible point with the smallest J among the successful optimizer starts and the feasible grid points, polished with SLSQP when that polish stays feasible and does not increase J. "
        "If the grid has no feasible point, the row is infeasible; an optimizer point may still be listed in the CSV and is not the reported solution. "
        "Algebraic inequalities on r and CFR use an absolute tolerance of 1e-12. Peak day and peak height do not. "
        "`critic_accept` applies `DeterministicCriticAgent`'s checks to the simulated Phase-1 baseline (day "
        f"{san['critic_day']:.4f}, height {san['critic_height']:.2f}, r = {san['critic_r']:.6g}, CFR = {san['critic_cfr']:.6g}), which is not identical to the printed anchors 264.3, 39121, 0.0130 and 0.0909."
    )
    lines.append("")
    lines.append("| Setting | Request | Target | Feasible | Critic | J | Named change | Unnamed / peak axes | Reference |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        if row["feasible"] != "yes":
            jay = "infeasible"
            named = ""
            unnamed = ""
        else:
            jay = f"{row['J']:.6e}"
            unit_label = {
                "percent_height": "% height",
                "days": "days",
                "percent_r": "% r",
                "percent_CFR": "% CFR",
            }[row["named_change_unit"]]
            named = f"{row['named_change']:+.4g} {unit_label}"
            unnamed = (
                f"day {row['unnamed_peak_day_change_days']:+.2f} d, "
                f"height {row['unnamed_peak_height_change_pct']:+.2f}%"
            )
        ref = (
            f"{row['ref_table']} J {row['ref_J']:.6e}, feasible {row['ref_feasible']}"
        )
        lines.append(
            f"| {row['setting']} | {row['request']} | {row['target']} | {row['feasible']} | {row['critic_accept']} | {jay} | {named} | {unnamed} | {ref} |"
        )
    lines.append("")
    n_feas = sum(1 for row in rows if row["feasible"] == "yes")
    lines.append(f"{n_feas} of the 24 problems are feasible on the grid oracle. {24 - n_feas} are infeasible.")
    lines.append("")
    lines.append("Per-problem detail (solver, grid count, and the comparison flags) is in `results.csv`.")
    lines.append("")
    for row in rows:
        if row["feasible"] != "yes":
            extra = ""
            if row.get("optimizer_feasible") == "yes":
                extra = (
                    f" The best optimizer start was feasible under the same inequalities "
                    f"(J {row['optimizer_J']:.6e}, {row['optimizer_solver']}) but the grid oracle is empty, so the row stays infeasible."
                )
            lines.append(
                f"- {row['setting']} {row['request']} {row['target']}: infeasible. "
                f"Grid feasible points: {row['grid_n_feasible']}. "
                f"Starts that reached a feasible point: {row['n_starts_converged']} of {row['n_starts']}.{extra}"
            )
        else:
            lines.append(
                f"- {row['setting']} {row['request']} {row['target']}: "
                f"β {row['beta']:.6f}, γ {row['gamma']:.6f}, μ {row['mu']:.6f}, "
                f"R0 {row['R0']:.4f}, r {row['r']:.4f}, CFR {row['CFR']:.4f}, "
                f"peak day {row['peak_day']:.4f}, height {row['peak_height']:.2f}, "
                f"J {row['J']:.6e}, max relative change {row['max_rel_change']:.4g}, "
                f"solver {row['solver_used']}, grid feasible {row['grid_n_feasible']}, "
                f"converged starts {row['n_starts_converged']}/{row['n_starts']}, "
                f"critic {row['critic_accept']}."
            )
    lines.append("")
    lines.append("## Comparison with the Generator and the starter")
    lines.append("")
    lines.append("Settings D and G are compared with the Table 2 Generator parameters. Setting S is compared with the Table 3 starter. Reference J uses the same formula at those published parameters. A reference point is feasible only when the same protocol inequalities accept it. Side effects use the simulated peak, not the printed peak: absolute day shift for higher/lower, absolute height percent for earlier/later, and the larger of |Δday|/8 and |Δheight|/3% for growth and mortality.")
    lines.append("")
    smaller_j = [row for row in rows if row["generator_smaller_J"] == "yes"]
    smaller_side = [row for row in rows if row["generator_smaller_side_effect"] == "yes"]
    gen_only = [row for row in rows if row["generator_feasible_when_optimizer_not"] == "yes"]
    raw_j = [row for row in rows if row["ref_J_smaller_ignoring_feasibility"] == "yes"]
    lines.append(
        f"On J, among problems where both the reported point and the reference point are feasible, the reference has the smaller J in {len(smaller_j)} problems"
        + (": " + ", ".join(f"{r['setting']} {r['request']} {r['target']}" for r in smaller_j) if smaller_j else "")
        + "."
    )
    lines.append(
        f"On the side-effect scalar, the same both-feasible rule gives the reference the smaller side effect in {len(smaller_side)} problems"
        + (": " + ", ".join(f"{r['setting']} {r['request']} {r['target']}" for r in smaller_side) if smaller_side else "")
        + "."
    )
    lines.append(
        f"On feasibility, the reference point meets the protocol inequalities in {len(gen_only)} problem{'s' if len(gen_only) != 1 else ''} whose grid oracle is empty"
        + (": " + ", ".join(f"{r['setting']} {r['request']} {r['target']}" for r in gen_only) if gen_only else "")
        + "."
    )
    lines.append(
        f"Ignoring feasibility, the reference parameter vector has a smaller J than the reported point in {len(raw_j)} feasible optimizer rows. That count includes reference vectors that miss the printed constraint, so it is not by itself evidence that the Generator solved the problem with a smaller change."
    )
    if not smaller_j and not smaller_side and not gen_only:
        lines.append("Under the both-feasible rule, the Generator and the starter do not record a smaller J, a smaller side effect, or a feasible point where the optimizer's grid is empty.")
    opt_only = [
        row for row in rows
        if row["feasible"] == "yes" and row["ref_feasible"] == "no"
    ]
    lines.append(
        f"The optimizer reports a feasible point in {len(opt_only)} problems where the published reference parameters do not meet the same inequalities"
        + (": " + ", ".join(f"{r['setting']} {r['request']} {r['target']}" for r in opt_only) if opt_only else "")
        + "."
    )
    disagree = [row for row in rows if row["feasible"] == "yes" and row["critic_accept"] != "yes"]
    lines.append(
        f"Protocol-feasible reported points that the code Critic rejects: {len(disagree)}"
        + (": " + ", ".join(f"{r['setting']} {r['request']} {r['target']}" for r in disagree) if disagree else "")
        + "."
    )
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append(meta["check_paragraph"])
    lines.append("")
    lines.append("## Reproducibility")
    lines.append("")
    lines.append(f"Command: `{meta['command']}`.")
    lines.append(f"Python {meta['python']}, numpy {meta['numpy']}, scipy {meta['scipy']}, platform {meta['platform']}.")
    lines.append(f"Latin hypercube: {meta['n_lhs']} points in the unit cube, seed {LHS_SEED}, scramble true, plus the Phase-1 point, {meta['n_starts']} starts in total. Grid: {meta['grid_n']}³ per window, three windows (default D, mortality, wide-γ S).")
    lines.append(f"Unique surrogate calls: {meta['n_sim_total']}. Wall time: {meta['wall_s']:.1f} s, of which the three grids took {meta['grid_wall_s']:.1f} s.")
    lines.append("Per-problem `n_simulations` counts integrations inside that problem's multistart and polish, not the shared grid. `wall_time_s` is the multistart plus polish only.")
    lines.append("Outputs: `results.csv`, `results.md`, `logs/`.")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def md5_file(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="tiny grid and few starts; not the reported run")
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    outdir = Path(args.outdir) if args.outdir else HERE / "results_rerun"  # archived run: results/
    log_dir = outdir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run.log"
    log_handle = log_path.open("w", encoding="utf-8")

    def log(message=""):
        print(message, flush=True)
        log_handle.write(str(message) + "\n")
        log_handle.flush()

    t_run = time.perf_counter()
    load_initial_state()
    grid_n = 5 if args.smoke else GRID_N
    n_lhs = 2 if args.smoke else N_LHS
    slsqp_options = dict(SLSQP_OPTIONS)
    trust_options = dict(TRUST_OPTIONS)
    if args.smoke:
        slsqp_options["maxiter"] = 25
        trust_options["maxiter"] = 25
    n_starts = n_lhs + 1

    log("Constrained optimizer experiment (frozen protocol)")
    log(f"smoke={args.smoke} grid_n={grid_n} n_lhs={n_lhs} outdir={outdir}")
    log(f"python {sys.version.split()[0]} numpy {np.__version__} scipy {__import__('scipy').__version__}")

    # --- sanity: Critic peak must match Table 2; otherwise stop ---
    t0 = time.perf_counter()
    critic_day, critic_height, critic_res = simulate_peak(_SUR, BETA0, GAMMA0, MU0)
    n_sanity = 1
    critic_day_rounded = round(critic_day, 1)
    critic_height_rounded = round(critic_height)
    parabolic = parabolic_diagnostic(_SUR)
    n_sanity += 1
    sanity = {
        "critic_day": critic_day,
        "critic_height": critic_height,
        "critic_day_rounded": critic_day_rounded,
        "critic_height_rounded": critic_height_rounded,
        "critic_r": as_float(r_of(BETA0, GAMMA0, MU0)),
        "critic_cfr": as_float(cfr_of(GAMMA0, MU0)),
        "parabolic": parabolic,
        "pass": f"{critic_day_rounded:.1f}" == "264.3" and f"{critic_height_rounded:.0f}" == "39121",
        "seconds": time.perf_counter() - t0,
    }
    (log_dir / "sanity.json").write_text(json.dumps(sanity, indent=2) + "\n", encoding="utf-8")
    log(
        f"sanity critic day {critic_day:.6f} -> {critic_day_rounded:.1f}, "
        f"height {critic_height:.4f} -> {critic_height_rounded:.0f}, pass={sanity['pass']}"
    )
    log(
        f"parabolic diagnostic day {parabolic['peak_day']:.6f} -> {parabolic['rounded_day']:.1f}, "
        f"height {parabolic['peak_height']:.4f} -> {parabolic['rounded_height']:.0f} (not used)"
    )
    if not sanity["pass"]:
        log("STOP: Phase-1 Critic peak does not match Table 2. No optimization was run.")
        (outdir / "results.md").write_text(
            "# Constrained optimizer: sanity check failed\n\n"
            "The protocol says to stop if the Phase-1 peak does not match Table 2 "
            "(day 264.3, height 39121) and not to retune the integrator.\n\n"
            f"Critic surrogate peak day {critic_day:.6f} (rounds to {critic_day_rounded:.1f}), "
            f"height {critic_height:.4f} (rounds to {critic_height_rounded:.0f}).\n\n"
            "No optimization was run.\n",
            encoding="utf-8",
        )
        log_handle.close()
        return 2

    baseline = {
        "day": critic_day,
        "height": critic_height,
        "r": sanity["critic_r"],
        "cfr": sanity["critic_cfr"],
    }
    cons0 = conservation_report(critic_res)
    log(f"baseline conservation {json.dumps(cons0)}")

    # Logic checks of the predicate. These do not depend on the optimizer.
    b_hi, g_hi, m_hi = TABLE2["higher"]["beta"], TABLE2["higher"]["gamma"], TABLE2["higher"]["mu"]
    day_hi, height_hi, _ = simulate_peak(_SUR, b_hi, g_hi, m_hi)
    n_sanity += 1
    if height_hi < 50000:
        raise SystemExit(f"Table 2 higher peak height {height_hi} is far below the published scale.")
    bounds_d = WINDOWS["D"]
    if not protocol_feasible("D", "higher", "T1", day_hi, height_hi, b_hi, g_hi, m_hi, bounds_d):
        raise SystemExit("Predicate rejected Table 2 'higher' on D/T1; the inequality direction is wrong.")
    if protocol_feasible("D", "higher", "T1", critic_day, critic_height, BETA0, GAMMA0, MU0, bounds_d):
        raise SystemExit("Predicate accepted the Phase-1 point for 'higher' T1.")
    if not protocol_feasible(
        "D", "growth", "T2", critic_day, critic_height,
        TABLE2["growth"]["beta"], TABLE2["growth"]["gamma"], TABLE2["growth"]["mu"], bounds_d,
    ):
        raise SystemExit("Predicate rejected Table 2 growth on D/T2 (algebraic r).")
    log("predicate logic checks passed")

    intent_hits, parser_hits = find_intent_logs(REPO)
    if parser_hits:
        intent_note = "parser-like files were found and were not used, because the protocol mapping is the frozen intent: " + ", ".join(parser_hits)
    elif intent_hits:
        intent_note = "files with 'intent' in the name exist but are not Intent Parser outputs: " + ", ".join(intent_hits[:12])
    else:
        intent_note = "no Intent Parser log file was found in the repository"
    log("intent: " + intent_note)

    # Reference simulations through the same surrogate.
    references = []
    ref_cache = {}
    for table_name, table in (("Table 2", TABLE2), ("Table 3", TABLE3)):
        for request, spec in table.items():
            key = (spec["beta"], spec["gamma"], spec["mu"])
            if key not in ref_cache:
                day, height, _res = simulate_peak(_SUR, *key)
                n_sanity += 1
                ref_cache[key] = (day, height)
            day, height = ref_cache[key]
            rounded_day = round(day, 1)
            rounded_height = round(height)
            match = f"{rounded_day:.1f}" == f"{spec['day']:.1f}" and f"{rounded_height:.0f}" == f"{spec['height']:.0f}"
            item = {
                "table": table_name,
                "request": request,
                "beta": spec["beta"],
                "gamma": spec["gamma"],
                "mu": spec["mu"],
                "day": day,
                "height": height,
                "rounded_day": rounded_day,
                "rounded_height": rounded_height,
                "published_day": spec["day"],
                "published_height": spec["height"],
                "published_r": spec["r"],
                "published_cfr": spec["cfr"],
                "r": as_float(r_of(*key)),
                "cfr": as_float(cfr_of(spec["gamma"], spec["mu"])),
                "J": as_float(j_of(*key)),
                "match": match,
            }
            references.append(item)
            log(
                f"ref {table_name} {request}: day {day:.4f} (pub {spec['day']}), "
                f"height {height:.2f} (pub {spec['height']}), match={match}"
            )
    (log_dir / "reference_check.json").write_text(json.dumps(references, indent=2) + "\n", encoding="utf-8")

    # Latin hypercube in the unit cube, shared by every window.
    sampler = LatinHypercube(d=3, scramble=True, seed=LHS_SEED)
    unit = sampler.random(n=n_lhs)
    if unit.shape != (n_lhs, 3):
        raise SystemExit("Latin hypercube returned an unexpected shape.")
    # The phase-1 point is packed inside each window later. The LHS samples are
    # the other starts, in this order, for every problem.
    lhs_starts = [row.astype(float) for row in unit]
    (log_dir / "lhs_unit.json").write_text(
        json.dumps({"seed": LHS_SEED, "n": n_lhs, "scramble": True, "unit": unit.tolist()}, indent=2) + "\n",
        encoding="utf-8",
    )

    ctx = mp.get_context("fork")
    pool = ctx.Pool(processes=N_WORKERS, initializer=_init_worker)
    grids = {}
    grid_wall = 0.0
    try:
        for wid, bounds in WINDOWS.items():
            axes = [np.linspace(lo, hi, grid_n) for lo, hi in bounds]
            log(f"grid {wid} {grid_n}^3 = {grid_n ** 3}")
            params, days, heights, elapsed = evaluate_grid(pool, axes, log)
            grids[wid] = {"bounds": bounds, "params": params, "days": days, "heights": heights, "elapsed": elapsed}
            grid_wall += elapsed
            np.savez_compressed(
                log_dir / f"grid_{wid}.npz",
                params=params,
                days=days,
                heights=heights,
                axes0=axes[0],
                axes1=axes[1],
                axes2=axes[2],
            )
            log(f"grid {wid} done in {elapsed:.1f}s, nan peaks {int(np.isnan(days).sum())}")

        problems = []
        for setting in ("D", "S", "G"):
            for request in REQUESTS[setting]:
                for target in ("T1", "T2"):
                    problems.append((setting, request, target))

        rows = []
        detail_path = log_dir / "starts.jsonl"
        with detail_path.open("w", encoding="utf-8") as detail:
            for setting, request, target in problems:
                wid = window_id(setting, request)
                bounds = WINDOWS[wid]
                grid = grids[wid]
                phase_u = pack(BETA0, GAMMA0, MU0, bounds)
                if np.any(phase_u < -1e-12) or np.any(phase_u > 1 + 1e-12):
                    raise SystemExit(f"Phase-1 point is outside window {wid}: {phase_u}")
                start_us = [np.clip(phase_u, 0.0, 1.0)] + lhs_starts
                payloads = [
                    (u.tolist(), bounds, setting, request, target, slsqp_options, trust_options)
                    for u in start_us
                ]
                log(f"optimize {setting} {request} {target} window {wid}")
                t0 = time.perf_counter()
                start_results = pool.map(_optimize_start, payloads, chunksize=1)
                # Polish is one extra SLSQP in the parent process, after the grid oracle is known.
                mask = protocol_mask(
                    setting, request, target,
                    grid["days"], grid["heights"],
                    grid["params"][:, 0], grid["params"][:, 1], grid["params"][:, 2],
                    bounds,
                )
                n_grid_feas = int(np.count_nonzero(mask))
                grid_best = best_mask_point(grid["params"], grid["days"], grid["heights"], mask)
                optimizer_best = None
                n_conv = 0
                n_opt_sim = 0
                for i, result in enumerate(start_results):
                    n_opt_sim += int(result.get("n_sim") or 0)
                    if result.get("feasible"):
                        n_conv += 1
                        if optimizer_best is None or sort_key(result) < sort_key(optimizer_best):
                            optimizer_best = result
                    detail.write(json.dumps({
                        "setting": setting,
                        "request": request,
                        "target": target,
                        "start": i,
                        "feasible": bool(result.get("feasible")),
                        "solver": result.get("solver"),
                        "J": result.get("J"),
                        "n_sim": result.get("n_sim"),
                        "trust_called": result.get("trust_called"),
                        "slsqp_success": result.get("slsqp_success"),
                        "slsqp_feasible": result.get("slsqp_feasible"),
                        "trust_success": result.get("trust_success"),
                        "trust_feasible": result.get("trust_feasible"),
                        "message": result.get("slsqp_message"),
                        "trust_message": result.get("trust_message"),
                        "error": result.get("error"),
                    }) + "\n")
                polish_kept = False
                reported = None
                if n_grid_feas > 0:
                    candidates = [grid_best]
                    for result in start_results:
                        if result.get("feasible"):
                            candidates.append(result)
                    reported = min(candidates, key=sort_key)
                    u_best = pack(reported["beta"], reported["gamma"], reported["mu"], bounds)
                    polished = optimize_from(
                        _SUR, u_best, bounds, setting, request, target,
                        slsqp_options, trust_options, allow_trust=False,
                    )
                    n_opt_sim += int(polished.get("n_sim") or 0)
                    if (
                        polished.get("feasible")
                        and polished.get("J") is not None
                        and polished["J"] <= reported["J"] + J_COMPARE_ATOL
                    ):
                        moved = max(
                            abs(polished["beta"] - reported["beta"]),
                            abs(polished["gamma"] - reported["gamma"]),
                            abs(polished["mu"] - reported["mu"]),
                        ) > 0.0
                        if moved:
                            origin = reported["solver"]
                            reported = polished
                            reported["solver"] = f"{origin}+SLSQP-polish"
                            polish_kept = True
                elapsed = time.perf_counter() - t0
                spec, table_name = ref_row(setting, request)
                ref_day, ref_height = ref_cache[(spec["beta"], spec["gamma"], spec["mu"])]
                ref_feas = protocol_feasible(
                    setting, request, target, ref_day, ref_height,
                    spec["beta"], spec["gamma"], spec["mu"], bounds,
                )
                ref_j = as_float(j_of(spec["beta"], spec["gamma"], spec["mu"]))
                ref_side = side_effect(request, ref_day, ref_height)
                kind, _op, level = named_level(setting, request, target)
                row = {
                    "request": request,
                    "setting": setting,
                    "target": target,
                    "phrase": PHRASES[request],
                    "window": wid,
                    "constraint": constraint_text(setting, request, target),
                    "named_quantity": {"height": "peak_height", "day": "peak_day", "r": "r", "cfr": "CFR"}[kind],
                    "n_simulations": n_opt_sim,
                    "wall_time_s": elapsed,
                    "n_starts": n_starts,
                    "n_starts_converged": n_conv,
                    "grid_n_points": grid_n ** 3,
                    "grid_n_feasible": n_grid_feas,
                    "ref_table": table_name,
                    "ref_beta": spec["beta"],
                    "ref_gamma": spec["gamma"],
                    "ref_mu": spec["mu"],
                    "ref_named_value": level if target == "T2" else {
                        "higher": spec["height"], "lower": spec["height"],
                        "earlier": spec["day"], "later": spec["day"],
                        "growth": spec["r"], "mortality": spec["cfr"],
                    }[request],
                    "ref_J": ref_j,
                    "ref_r": as_float(r_of(spec["beta"], spec["gamma"], spec["mu"])),
                    "ref_CFR": as_float(cfr_of(spec["gamma"], spec["mu"])),
                    "ref_peak_day_simulated": ref_day,
                    "ref_peak_height_simulated": ref_height,
                    "ref_peak_day_published": spec["day"],
                    "ref_peak_height_published": spec["height"],
                    "ref_peak_matches_published": "yes" if next(
                        item["match"] for item in references
                        if item["table"] == table_name and item["request"] == request
                    ) else "no",
                    "ref_feasible": "yes" if ref_feas else "no",
                    "ref_side_effect": ref_side,
                    "polish_kept": "yes" if polish_kept else "no",
                    "optimizer_n_simulations": n_opt_sim,
                }
                if optimizer_best is not None:
                    row.update({
                        "optimizer_feasible": "yes",
                        "optimizer_beta": optimizer_best["beta"],
                        "optimizer_gamma": optimizer_best["gamma"],
                        "optimizer_mu": optimizer_best["mu"],
                        "optimizer_J": optimizer_best["J"],
                        "optimizer_peak_day": optimizer_best["day"],
                        "optimizer_peak_height": optimizer_best["height"],
                        "optimizer_solver": optimizer_best["solver"],
                    })
                else:
                    row["optimizer_feasible"] = "no"
                if reported is None:
                    row.update({
                        "feasible": "no",
                        "critic_accept": "no",
                        "solver_used": "none",
                        "generator_smaller_J": "no",
                        "generator_smaller_side_effect": "no",
                        "generator_feasible_when_optimizer_not": "yes" if ref_feas else "no",
                        "ref_J_smaller_ignoring_feasibility": "n/a",
                    })
                else:
                    # Recompute J from the stored parameters. A mismatch is a bug.
                    recomputed = as_float(j_of(reported["beta"], reported["gamma"], reported["mu"]))
                    if abs(recomputed - reported["J"]) > 1e-9:
                        raise SystemExit(
                            f"J mismatch for {setting} {request} {target}: {reported['J']} vs {recomputed}"
                        )
                    reported["J"] = recomputed
                    change, unit = named_change(
                        request, reported["day"], reported["height"],
                        reported["beta"], reported["gamma"], reported["mu"],
                    )
                    this_side = side_effect(request, reported["day"], reported["height"])
                    critic_yes = critic_accepts(
                        setting, request, reported["day"], reported["height"],
                        reported["beta"], reported["gamma"], reported["mu"], baseline,
                    )
                    both = ref_feas
                    row.update(point_record(
                        reported["beta"], reported["gamma"], reported["mu"],
                        reported["day"], reported["height"],
                    ))
                    row.update({
                        "named_change": change,
                        "named_change_unit": unit,
                        "named_value": value_of(
                            kind, reported["day"], reported["height"],
                            reported["beta"], reported["gamma"], reported["mu"],
                        ),
                        "unnamed_peak_day_change_days": reported["day"] - DAY0,
                        "unnamed_peak_height_change_pct": (reported["height"] - HEIGHT0) / HEIGHT0 * 100.0,
                        "side_effect": this_side,
                        "feasible": "yes",
                        "critic_accept": "yes" if critic_yes else "no",
                        "solver_used": reported["solver"],
                        "generator_smaller_J": "yes" if both and ref_j + J_COMPARE_ATOL < reported["J"] else "no",
                        "generator_smaller_side_effect": "yes" if both and ref_side + 1e-9 < this_side else "no",
                        "generator_feasible_when_optimizer_not": "no",
                        "ref_J_smaller_ignoring_feasibility": "yes" if ref_j + J_COMPARE_ATOL < reported["J"] else "no",
                    })
                rows.append(row)
                log(
                    f"  -> feasible={row['feasible']} J={row.get('J')} "
                    f"grid_feas={n_grid_feas} conv={n_conv} sims={n_opt_sim} t={elapsed:.1f}s"
                )
    finally:
        pool.close()
        pool.join()

    # Conservation on every reported solution, plus the J identity already checked.
    checks = []
    for row in rows:
        if row["feasible"] != "yes":
            continue
        _day, _height, res = simulate_peak(_SUR, row["beta"], row["gamma"], row["mu"])
        n_sanity += 1
        report = conservation_report(res)
        report.update({
            "setting": row["setting"],
            "request": row["request"],
            "target": row["target"],
            "peak_day_resim": _day,
            "peak_height_resim": _height,
            "peak_day_reported": row["peak_day"],
            "peak_height_reported": row["peak_height"],
            "peak_day_abs_diff": abs(_day - row["peak_day"]),
            "peak_height_abs_diff": abs(_height - row["peak_height"]),
            "J_identity_abs_diff": abs(as_float(j_of(row["beta"], row["gamma"], row["mu"])) - row["J"]),
        })
        checks.append(report)
        if report["peak_day_abs_diff"] > 1e-6 or report["peak_height_abs_diff"] > 1e-4:
            log(f"WARNING resimulation peak differs for {row['setting']} {row['request']} {row['target']}")
    (log_dir / "conservation.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    bad_cons = [c for c in checks if not c["nonnegative"] or not c["conserved"]]
    bad_j = [c for c in checks if c["J_identity_abs_diff"] > 1e-9]
    check_paragraph = (
        f"Reported solutions were integrated again. "
        f"{len(checks)} feasible rows were checked. "
        f"Compartment negativity or a conservation error above 1e-4 person: {len(bad_cons)}. "
        f"J differing from the formula by more than 1e-9: {len(bad_j)}. "
        f"Baseline conservation error {cons0['max_abs_conservation_error']:.3e}, "
        f"minimum compartments S {cons0['min_S']:.3f}, I {cons0['min_I']:.3f}, "
        f"R {cons0['min_R']:.3f}, D {cons0['min_D']:.3f}."
    )
    log(check_paragraph)

    write_csv(outdir / "results.csv", rows)
    n_sim_total = (
        n_sanity
        + sum(grid_n ** 3 for _ in WINDOWS)
        + sum(int(row["n_simulations"]) for row in rows)
    )
    # n_sanity includes the reference and conservation calls. Grid calls are
    # unique. Optimizer calls are summed from the rows. The logic-check higher
    # simulation is inside n_sanity.
    wall = time.perf_counter() - t_run
    meta = {
        "sanity": sanity,
        "references": references,
        "reference_mismatches": [item for item in references if not item["match"]],
        "intent_note": intent_note,
        "data_relpath": str(DATA_CSV.relative_to(REPO)),
        "check_paragraph": check_paragraph,
        "command": "python3 appendix_I_constrained_optimizer/run_experiment.py",
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "platform": platform.platform(),
        "n_lhs": n_lhs,
        "n_starts": n_starts,
        "grid_n": grid_n,
        "n_sim_total": n_sim_total,
        "wall_s": wall,
        "grid_wall_s": grid_wall,
        "smoke": args.smoke,
    }
    write_report(outdir / "results.md", rows, meta)
    provenance = {
        "smoke": args.smoke,
        "protocol_md5": md5_file(HERE / "PROTOCOL.md"),
        "script_md5": md5_file(Path(__file__).resolve()),
        "knobs": {
            "BETA0": BETA0, "GAMMA0": GAMMA0, "MU0": MU0,
            "DAY0": DAY0, "HEIGHT0": HEIGHT0, "R_GROWTH0": R_GROWTH0, "CFR0": CFR0,
            "T_MAX": T_MAX, "N_OUT": N_OUT, "GRID_N": grid_n, "N_LHS": n_lhs,
            "LHS_SEED": LHS_SEED, "SLSQP_OPTIONS": slsqp_options, "TRUST_OPTIONS": trust_options,
            "ALGEBRAIC_ATOL": ALGEBRAIC_ATOL, "J_COMPARE_ATOL": J_COMPARE_ATOL,
            "BOUND_VIOLATION_ATOL": BOUND_VIOLATION_ATOL,
            "windows": {k: v for k, v in WINDOWS.items()},
        },
        "initial_state": {"S": S0, "I": I0, "R": R0_COMP, "D": D0, "N": POPULATION},
        "n_sim_total": n_sim_total,
        "n_sanity_and_checks": n_sanity,
        "grid_wall_s": grid_wall,
        "wall_s": wall,
        "results_md5": md5_file(outdir / "results.csv"),
        "intent_hits": intent_hits,
        "parser_hits": parser_hits,
    }
    (log_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    log(f"done in {wall:.1f}s, unique sims about {n_sim_total}")
    log_handle.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
