#!/usr/bin/env python3
"""R3.7 high-budget rerun.

Same model, data, window, objective, bounds, starts, and solver as the original
R3.7. Only maxiter and maxfun are larger. Extra labelled runs are not part of
the 16-start comparison. Does not write to out_r37 or the repository.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
import os
# Output folder: results_rerun/ next to this script (set R37_OUT to override); the archived run is in results/.
OUT = Path(os.environ.get("R37_OUT", ROOT / "results_rerun"))
# starts.csv of the earlier low-budget run (maxiter=200, maxfun=500), used only for the continuation starts.
PRIOR = ROOT / "prior_low_budget"
CFG = json.loads((ROOT / "config.json").read_text())


def load_observations():
    frame = pd.read_csv(ROOT / CFG["series"]["file"])
    cols = {name: frame[name].to_numpy(float) for name in "SIRD"}
    t = frame["t"].to_numpy(float)
    if len(t) != CFG["series"]["n_days"]:
        raise SystemExit(f"expected {CFG['series']['n_days']} rows, found {len(t)}")
    if abs(t[0] - CFG["series"]["t_min"]) > 0 or abs(t[-1] - CFG["series"]["t_max"]) > 0:
        raise SystemExit("time column is not 0..368")
    total = cols["S"] + cols["I"] + cols["R"] + cols["D"]
    if np.max(np.abs(total - CFG["series"]["N"])) > 0:
        raise SystemExit("S+I+R+D is not N on every row")
    y0 = np.array([cols["S"][0], cols["I"][0], cols["R"][0], cols["D"][0]], dtype=float)
    return t, cols, y0


def beta_of(betas, breaks, time):
    edges = np.concatenate([[-np.inf], np.asarray(breaks, dtype=float), [np.inf]])
    # day 0 belongs to the first segment; a later knot belongs to the segment that starts there
    idx = int(np.searchsorted(edges[1:-1], time, side="right"))
    return float(betas[idx])


def lengths_from_z(z, n_seg, t_max, min_len):
    excess = t_max - n_seg * min_len
    if excess <= 0:
        raise SystemExit("minimum segment length does not fit in the window")
    z = np.asarray(z, dtype=float)
    z = z - np.max(z)
    weights = np.exp(z)
    weights = weights / np.sum(weights)
    return min_len + excess * weights


def breaks_from_z(z, n_seg, t_max, min_len):
    if n_seg == 1:
        return np.array([], dtype=float)
    lengths = lengths_from_z(z, n_seg, t_max, min_len)
    return np.cumsum(lengths)[:-1]


def integrate(betas, breaks, gamma, mu, y0, t_obs, population, rtol, atol):
    betas = np.asarray(betas, dtype=float)
    breaks = np.asarray(breaks, dtype=float)
    edges = np.concatenate([[t_obs[0]], breaks, [t_obs[-1]]])
    state = np.array(y0, dtype=float)
    columns = [state.copy()]
    times = [float(t_obs[0])]
    for beta, left, right in zip(betas, edges[:-1], edges[1:]):
        grid = t_obs[(t_obs > left) & (t_obs < right)]
        targets = np.unique(np.concatenate([np.asarray(grid, dtype=float), [float(right)]]))
        targets = targets[(targets > left) & (targets <= right + 1e-9)]
        if len(targets) == 0:
            continue
        targets = np.minimum(targets, float(right))
        targets = np.unique(targets)
        if targets[-1] < right:
            targets = np.append(targets, float(right))

        def field(_time, state, beta=beta):
            susceptible, infected, _recovered, _dead = state
            force = beta * susceptible * infected / population
            return [
                -force,
                force - (gamma + mu) * infected,
                gamma * infected,
                mu * infected,
            ]

        try:
            sol = solve_ivp(
                field,
                (float(left), float(right)),
                state,
                t_eval=targets,
                method="RK45",
                rtol=rtol,
                atol=atol,
            )
        except ValueError:
            return None
        if (not sol.success) or sol.y.shape[1] != len(targets):
            return None
        for column, stamp in zip(sol.y.T, sol.t):
            if any(abs(stamp - day) <= 1e-8 for day in t_obs):
                columns.append(column.copy())
                times.append(float(stamp))
        state = sol.y[:, -1].copy()
    if len(columns) != len(t_obs):
        return None
    order = np.argsort(times)
    traj = np.vstack(columns)[order].T
    if np.max(np.abs(np.asarray(times)[order] - t_obs)) > 1e-6:
        return None
    return traj


def unpack(x, n_seg):
    betas = x[:n_seg]
    gamma = float(x[n_seg])
    mu = float(x[n_seg + 1])
    z = x[n_seg + 2:]
    return betas, gamma, mu, z


def objective(x, n_seg, y0, t_obs, infected, population, settings):
    betas, gamma, mu, z = unpack(x, n_seg)
    breaks = breaks_from_z(z, n_seg, settings["t_max"], settings["min_len"])
    traj = integrate(
        betas, breaks, gamma, mu, y0, t_obs, population,
        settings["rtol"], settings["atol"],
    )
    if traj is None or not np.isfinite(traj).all():
        return 1e30
    return float(np.sum((traj[1] - infected) ** 2))


def bounds_for(n_seg, cfg):
    lo_b, hi_b = cfg["bounds"]["beta"]
    lo_g, hi_g = cfg["bounds"]["gamma"]
    lo_m, hi_m = cfg["bounds"]["mu"]
    pairs = [(lo_b, hi_b)] * n_seg + [(lo_g, hi_g), (lo_m, hi_m)]
    if n_seg > 1:
        pairs += [(-6.0, 6.0)] * n_seg
    return pairs


def draw_start(n_seg, seed, cfg):
    opt = cfg["optimizer"]
    if seed == 0:
        fixed = opt["start0"]
        betas = np.full(n_seg, fixed["beta_all_segments"])
        gamma = fixed["gamma"]
        mu = fixed["mu"]
        z = np.zeros(n_seg if n_seg > 1 else 0)
        return np.concatenate([betas, [gamma, mu], z])
    rng = np.random.default_rng(20261108 + seed)
    lo_b, hi_b = cfg["bounds"]["beta"]
    lo_g, hi_g = cfg["bounds"]["gamma"]
    lo_m, hi_m = cfg["bounds"]["mu"]
    base = float(np.exp(rng.uniform(np.log(lo_b), np.log(hi_b))))
    multipliers = np.exp(rng.normal(0.0, 0.35, size=n_seg))
    betas = np.clip(base * multipliers, lo_b, hi_b)
    gamma = float(np.exp(rng.uniform(np.log(lo_g), np.log(hi_g))))
    mu = float(np.exp(rng.uniform(np.log(lo_m), np.log(hi_m))))
    z = rng.uniform(-2.0, 2.0, size=n_seg if n_seg > 1 else 0)
    return np.concatenate([betas, [gamma, mu], z])


def settings_of(cfg):
    return {
        "t_max": cfg["series"]["t_max"],
        "min_len": cfg["breakpoints"]["min_segment_days"],
        "rtol": cfg["solver"]["rtol"],
        "atol": cfg["solver"]["atol"],
    }


def projected_gradient_norm(x, jac, box):
    if jac is None:
        return None
    gradient = np.asarray(jac, dtype=float).ravel()
    if gradient.size != len(x):
        return None
    projected = []
    for value, component, (low, high) in zip(x, gradient, box):
        if value <= low + 1e-8:
            component = min(float(component), 0.0)
        elif value >= high - 1e-8:
            component = max(float(component), 0.0)
        projected.append(float(component))
    return float(np.linalg.norm(projected))


def bound_report(x, box):
    hits = []
    for index, (value, (low, high)) in enumerate(zip(x, box)):
        if value <= low + 1e-8:
            hits.append(f"{index}:lower")
        elif value >= high - 1e-8:
            hits.append(f"{index}:upper")
    return len(hits), ";".join(hits)


def row_from_result(name, run_set, seed, n_seg, result, box, t_max, min_len):
    betas, gamma, mu, z = unpack(np.asarray(result.x, dtype=float), n_seg)
    breaks = breaks_from_z(z, n_seg, t_max, min_len)
    jac = getattr(result, "jac", None)
    jac_list = None if jac is None else [float(v) for v in np.asarray(jac, dtype=float).ravel()]
    n_hits, hit_list = bound_report(result.x, box)
    row = {
        "model": name,
        "run_set": run_set,
        "seed": seed,
        "success": bool(result.success),
        "status": int(getattr(result, "status", -1)),
        "nit": int(result.nit),
        "nfev": int(result.nfev),
        "rss": float(result.fun),
        "gamma": float(gamma),
        "mu": float(mu),
        "message": str(result.message),
        "grad_norm": None if jac_list is None else float(np.linalg.norm(jac_list)),
        "proj_grad_norm": projected_gradient_norm(result.x, jac, box),
        "n_at_bound": n_hits,
        "bound_hits": hit_list,
        "jac": "" if jac_list is None else json.dumps(jac_list),
    }
    for index, beta in enumerate(betas, start=1):
        row[f"beta{index}"] = float(beta)
    for index, knot in enumerate(breaks, start=1):
        row[f"break{index}"] = float(knot)
    return row


def run_lbfgs(name, run_set, seed, n_seg, x0, y0, t_obs, infected, population, cfg):
    settings = settings_of(cfg)
    box = bounds_for(n_seg, cfg)
    result = minimize(
        objective,
        np.asarray(x0, dtype=float),
        args=(n_seg, y0, t_obs, infected, population, settings),
        method=cfg["optimizer"]["method"],
        bounds=box,
        options={
            "maxiter": cfg["optimizer"]["maxiter"],
            "maxfun": cfg["optimizer"]["maxfun"],
            "ftol": cfg["optimizer"]["ftol"],
            "gtol": cfg["optimizer"]["gtol"],
        },
    )
    row = row_from_result(name, run_set, seed, n_seg, result, box, settings["t_max"], settings["min_len"])
    print(
        f"{run_set} {name} seed {seed}: rss={row['rss']:.6e} success={row['success']} "
        f"status={row['status']} nit={row['nit']} nfev={row['nfev']} {row['message']}",
        flush=True,
    )
    return row


def fit_model(name, n_seg, y0, t_obs, infected, population, cfg):
    rows = []
    t0 = time.perf_counter()
    for seed in cfg["optimizer"]["seeds"]:
        x0 = draw_start(n_seg, seed, cfg)
        rows.append(run_lbfgs(name, "fresh16", int(seed), n_seg, x0, y0, t_obs, infected, population, cfg))
        pd.DataFrame(rows).to_csv(OUT / f"starts_{name}_partial.csv", index=False)
    elapsed = time.perf_counter() - t0
    finite = [row for row in rows if np.isfinite(row["rss"]) and row["rss"] < 1e29]
    best = min(finite, key=lambda row: (row["rss"], row["seed"]))
    return rows, best, elapsed


def z_from_lengths(lengths, t_max, min_len):
    lengths = np.asarray(lengths, dtype=float)
    excess = t_max - len(lengths) * min_len
    shares = (lengths - min_len) / excess
    shares = np.clip(shares, 1e-12, None)
    shares = shares / np.sum(shares)
    return np.log(shares)


def vector_from_pieces(betas, breaks, gamma, mu, t_max, min_len):
    betas = np.asarray(betas, dtype=float)
    breaks = np.asarray(breaks, dtype=float)
    if len(betas) == 1:
        return np.array([betas[0], gamma, mu], dtype=float)
    edges = np.concatenate([[0.0], breaks, [t_max]])
    lengths = np.diff(edges)
    z = z_from_lengths(lengths, t_max, min_len)
    return np.concatenate([betas, [gamma, mu], z])


def trajectory_of(best, n_seg, y0, t_obs, population, cfg, rtol=None, atol=None):
    betas = [best[f"beta{i}"] for i in range(1, n_seg + 1)]
    breaks = [best[f"break{i}"] for i in range(1, n_seg) if f"break{i}" in best]
    return integrate(
        betas,
        breaks,
        best["gamma"],
        best["mu"],
        y0,
        t_obs,
        population,
        cfg["solver"]["rtol"] if rtol is None else rtol,
        cfg["solver"]["atol"] if atol is None else atol,
    )


def growth(series, start, stop):
    window = np.asarray(series[start:stop], dtype=float)
    return float(np.mean(np.diff(np.log(window))))


def instantaneous(traj, betas, breaks, gamma, mu, population, days):
    values = []
    for day in days:
        susceptible = traj[0, day]
        beta = beta_of(betas, breaks, day)
        values.append(beta * susceptible / population - gamma - mu)
    return float(np.mean(values))


def metrics_block(name, traj, obs, betas, breaks, gamma, mu, population, rss):
    infected = traj[1]
    peak = int(np.argmax(infected))
    out = {
        "model": name,
        "rss_I": rss,
        "rmse_I": float(np.sqrt(np.mean((infected - obs["I"]) ** 2))),
        "mae_I": float(np.mean(np.abs(infected - obs["I"]))),
        "rmse_S": float(np.sqrt(np.mean((traj[0] - obs["S"]) ** 2))),
        "rmse_R": float(np.sqrt(np.mean((traj[2] - obs["R"]) ** 2))),
        "rmse_D": float(np.sqrt(np.mean((traj[3] - obs["D"]) ** 2))),
        "peak_day_I": peak,
        "peak_height_I": float(infected[peak]),
        "g_early": growth(infected, 0, 111),
        "g_accel": growth(infected, 110, 151),
        "delta_I_110_150": float(infected[150] - infected[110]),
        "g_inst_early": instantaneous(traj, betas, breaks, gamma, mu, population, range(0, 111)),
        "g_inst_accel": instantaneous(traj, betas, breaks, gamma, mu, population, range(110, 151)),
    }
    out["accel"] = out["g_accel"] - out["g_early"]
    out["accel_inst"] = out["g_inst_accel"] - out["g_inst_early"]
    return out


def validate(name, n_seg, best, y0, t_obs, population, cfg, constant_best):
    traj = trajectory_of(best, n_seg, y0, t_obs, population, cfg)
    tight = trajectory_of(best, n_seg, y0, t_obs, population, cfg, rtol=1e-8, atol=1e-10)
    betas = [best[f"beta{i}"] for i in range(1, n_seg + 1)]
    breaks = [best[f"break{i}"] for i in range(1, n_seg) if f"break{i}" in best]
    report = {"model": name, "finite": bool(traj is not None and np.isfinite(traj).all())}
    if traj is None:
        report["pass"] = False
        return report, None
    total = traj.sum(axis=0)
    report["min_compartment"] = float(np.min(traj))
    report["max_abs_N_error"] = float(np.max(np.abs(total - population)))
    report["positive"] = bool(np.min(traj) > -1e-6)
    report["conserved"] = report["max_abs_N_error"] < 1e-4
    if tight is None:
        report["tight_rel_rmse_I"] = None
        report["solver_robust"] = False
    else:
        scale = float(np.sqrt(np.mean(traj[1] ** 2)))
        report["tight_rel_rmse_I"] = float(np.sqrt(np.mean((tight[1] - traj[1]) ** 2)) / scale)
        report["solver_robust"] = report["tight_rel_rmse_I"] < 1e-4
    # Rebuild each integer day from the previous stored state, splitting at any breakpoint
    # inside that day, and compare with the stored trajectory.
    max_step = 0.0
    step_ok = True
    for day in range(1, len(t_obs)):
        left = float(t_obs[day - 1])
        right = float(t_obs[day])
        knots = [float(k) for k in breaks if left < float(k) < right]
        points = [left] + knots + [right]
        state = traj[:, day - 1].copy()
        for a, b in zip(points[:-1], points[1:]):
            beta = beta_of(betas, breaks, 0.5 * (a + b))
            sol = solve_ivp(
                lambda _t, state, beta=beta: [
                    -beta * state[0] * state[1] / population,
                    beta * state[0] * state[1] / population - (best["gamma"] + best["mu"]) * state[1],
                    best["gamma"] * state[1],
                    best["mu"] * state[1],
                ],
                (a, b),
                state,
                t_eval=[b],
                method="RK45",
                rtol=cfg["solver"]["rtol"],
                atol=cfg["solver"]["atol"],
            )
            if not sol.success:
                step_ok = False
                break
            state = sol.y[:, -1].copy()
        if not step_ok:
            break
        max_step = max(max_step, float(np.max(np.abs(state - traj[:, day]))))
    report["continuity_max_abs"] = None if not step_ok else max_step
    # nested constant limit inside this simulator
    const_betas = [constant_best["beta1"]] * n_seg
    const_breaks = breaks_from_z(np.zeros(n_seg if n_seg > 1 else 0), n_seg, cfg["series"]["t_max"], cfg["breakpoints"]["min_segment_days"])
    nested = integrate(
        const_betas, const_breaks, constant_best["gamma"], constant_best["mu"],
        y0, t_obs, population, cfg["solver"]["rtol"], cfg["solver"]["atol"],
    )
    plain = integrate(
        [constant_best["beta1"]], [], constant_best["gamma"], constant_best["mu"],
        y0, t_obs, population, cfg["solver"]["rtol"], cfg["solver"]["atol"],
    )
    if nested is None or plain is None:
        report["nested_rel_rmse_I"] = None
        report["nested_ok"] = False
    else:
        scale = float(np.sqrt(np.mean(plain[1] ** 2)))
        report["nested_rel_rmse_I"] = float(np.sqrt(np.mean((nested[1] - plain[1]) ** 2)) / scale)
        report["nested_ok"] = report["nested_rel_rmse_I"] < 1e-6
    report["pass"] = bool(
        report["finite"] and report["positive"] and report["conserved"]
        and report["solver_robust"] and report["nested_ok"]
        and report["continuity_max_abs"] is not None and report["continuity_max_abs"] < 1.0
    )
    return report, traj


def make_figure(t, obs, curves, betas, path):
    colors = {
        "observed": "#000000",
        "published_120": "#009E73",
        "constant": "#0072B2",
        "bp2": "#D55E00",
        "bp3": "#CC79A7",
    }
    labels = {
        "published_120": "Published Phase-1 constant (120-day fit)",
        "constant": "Matched refit, constant β",
        "bp2": "Piecewise β, 2 breakpoints (3 segments)",
        "bp3": "Piecewise β, 3 breakpoints (4 segments)",
    }
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 11.2), gridspec_kw={"height_ratios": [1.15, 1.0, 0.85]})
    ax = axes[0]
    ax.plot(t, obs, color=colors["observed"], linewidth=1.3, label="Observed active cases")
    for key in ("published_120", "constant", "bp2", "bp3"):
        ax.plot(t, curves[key][1], color=colors[key], linewidth=1.6, label=labels[key])
    ax.set_ylabel("Active cases, I")
    ax.set_xlabel("Day (0 = 5 July 2020)")
    ax.set_title("Full estimation window, days 0–368")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    ax = axes[1]
    mask = (t >= 90) & (t <= 170)
    ax.plot(t[mask], obs[mask], color=colors["observed"], linewidth=1.4, label="Observed")
    for key in ("constant", "bp2", "bp3"):
        ax.plot(t[mask], curves[key][1][mask], color=colors[key], linewidth=1.7, label=labels[key])
    ax.axvspan(110, 150, color="#999999", alpha=0.15, label="Days 110–150")
    ax.set_ylabel("Active cases, I")
    ax.set_xlabel("Day")
    ax.set_title("Days 90–170, with the acceleration interval marked")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax = axes[2]
    for key in ("published_120", "constant", "bp2", "bp3"):
        ax.step(t, betas[key], where="post", color=colors[key], linewidth=1.8, label=labels[key])
    ax.set_ylabel("β(t), per day")
    ax.set_xlabel("Day (0 = 5 July 2020)")
    ax.set_title("Transmission: constant refit, two piecewise fits, and the published 120-day value")
    ax.set_xlim(0, 368)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def write_results(path, summary, checks, runtime, obs_metrics):
    def pct(x):
        return 100.0 * x

    lines = []
    lines.append("# R3.7. Кусочно-постоянная β(t) и ускорение на днях 110–150")
    lines.append("")
    lines.append("Ретроспективный прямой fit SIRD на наблюдениях первой волны Санкт-Петербурга (5 июля 2020 — 8 июля 2021, 369 суток, день 0 = 5 июля 2020). Это не прогноз за пределы окна и не правка рукописи.")
    lines.append("")
    lines.append("## Что зафиксировано до оптимизации")
    lines.append("")
    lines.append("Общее окно: дни 0–368. Цель: сумма квадратов отклонений I на каждой суточной точке (Table A13, fit to I(t), n = 369). γ и μ постоянны. Сравниваются постоянная β, ровно 2 точки излома (3 участка) и ровно 3 точки излома (4 участка). Число изломов задано протоколом; даты оцениваются из данных при минимальной длине участка 14 суток. Ограничения β, γ, μ не менялись. Повторный прогон тех же 16 стартов: L-BFGS-B, maxiter = 5000, maxfun = 20000 (раньше 200 и 500). Флаг success — код остановки, не глобальный оптимум. Кривая Phase 1 (β = 0.1219, γ = 0.0990, μ = 0.0099, окно 120 суток) показана отдельно и не является парным refit.")
    lines.append("")
    lines.append("## Наблюдаемое ускорение")
    lines.append("")
    lines.append(
        f"Средний логарифмический прирост I: {pct(obs_metrics['g_early']):.2f}% в сутки на шагах t = 0…109 "
        f"и {pct(obs_metrics['g_accel']):.2f}% в сутки на шагах t = 110…149. "
        f"Разница accel = {pct(obs_metrics['accel']):.2f} процентного пункта. "
        f"I(150) − I(110) = {obs_metrics['delta_I_110_150']:.0f}."
    )
    lines.append("")
    lines.append("## Оценки")
    lines.append("")
    lines.append("| Модель | RSS(I) | g ранний, %/сут | g 110–150, %/сут | accel, п.п. | ΔI 110–150 | пик I (день) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in summary:
        lines.append(
            f"| {row['label']} | {row['rss_I']:.4e} | {pct(row['g_early']):.2f} | {pct(row['g_accel']):.2f} | "
            f"{pct(row['accel']):.2f} | {row['delta_I_110_150']:.0f} | {row['peak_height_I']:.0f} ({row['peak_day_I']}) |"
        )
    lines.append("")
    lines.append("Даты изломов и β по участкам приведены в `summary.json`. У постоянной модели лучшие старты дают одну и ту же траекторию I и одно и то же β, но разный раскол γ и μ при почти неизменной сумме γ+μ: отдельные γ и μ этим рядом не идентифицируются.")
    lines.append("")
    lines.append("## Ответ на вопрос рецензента")
    lines.append("")
    lines.append(summary[-1]["answer"])
    lines.append("")
    lines.append("## Проверки и ограничения")
    lines.append("")
    for item in checks:
        lines.append(
            f"- {item['model']}: pass={item['pass']}, min compartment={item.get('min_compartment')}, "
            f"max |sum−N|={item.get('max_abs_N_error')}, tight relRMSE(I)={item.get('tight_rel_rmse_I')}, "
            f"nested relRMSE(I)={item.get('nested_rel_rmse_I')}."
        )
    lines.append(f"- Время оптимизации, с: {runtime}.")
    lines.append("- Интервалы и p-value не считались. Отрицательный или частичный результат не доказывает, что компартментальная схема неверна: ускорение может быть и изменением выявляемости, а β(t) здесь описательная ступенчатая функция с заранее заданным числом ступенек.")
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def assert_budget():
    opt = CFG["optimizer"]
    if int(opt["maxiter"]) != 5000 or int(opt["maxfun"]) != 20000:
        raise SystemExit("high-budget limits were changed after PROTOCOL_HIGHBUDGET.md was frozen")
    if opt["method"] != "L-BFGS-B" or list(opt["seeds"]) != list(range(16)):
        raise SystemExit("optimizer method or the 16 starts no longer match the frozen protocol")
    if float(opt["ftol"]) != 1e-12 or float(opt["gtol"]) != 1e-8:
        raise SystemExit("ftol or gtol changed")


def continuation_start(prior_row, n_seg, cfg):
    betas = [prior_row[f"beta{i}"] for i in range(1, n_seg + 1)]
    breaks = [prior_row[f"break{i}"] for i in range(1, n_seg) if f"break{i}" in prior_row and pd.notna(prior_row[f"break{i}"])]
    return vector_from_pieces(
        betas, breaks, prior_row["gamma"], prior_row["mu"],
        cfg["series"]["t_max"], cfg["breakpoints"]["min_segment_days"],
    )


def embed_bp2_in_bp3(bp2_row, cfg):
    """Split the longest 2-breakpoint segment in half and duplicate its beta."""
    t_max = cfg["series"]["t_max"]
    min_len = cfg["breakpoints"]["min_segment_days"]
    breaks = [bp2_row["break1"], bp2_row["break2"]]
    betas = [bp2_row["beta1"], bp2_row["beta2"], bp2_row["beta3"]]
    edges = np.array([0.0, breaks[0], breaks[1], t_max])
    lengths = np.diff(edges)
    longest = int(np.argmax(lengths))
    if lengths[longest] < 2 * min_len:
        raise SystemExit("longest segment cannot be split while keeping the 14-day minimum")
    half = lengths[longest] / 2.0
    new_lengths = np.concatenate([lengths[:longest], [half, half], lengths[longest + 1:]])
    new_betas = betas[:longest] + [betas[longest], betas[longest]] + betas[longest + 1:]
    new_breaks = np.cumsum(new_lengths)[:-1]
    return vector_from_pieces(new_betas, new_breaks, bp2_row["gamma"], bp2_row["mu"], t_max, min_len), new_breaks.tolist(), new_betas


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert_budget()
    stamp = {
        "locked_before_optimization": True,
        "maxiter": CFG["optimizer"]["maxiter"],
        "maxfun": CFG["optimizer"]["maxfun"],
        "ftol": CFG["optimizer"]["ftol"],
        "gtol": CFG["optimizer"]["gtol"],
        "method": CFG["optimizer"]["method"],
        "seeds": CFG["optimizer"]["seeds"],
    }
    (OUT / "budget_locked.json").write_text(json.dumps(stamp, indent=2))
    print("BUDGET LOCKED", json.dumps(stamp), flush=True)
    t_obs, obs, y0 = load_observations()
    population = float(CFG["series"]["N"])
    infected = obs["I"]
    models = [("constant", 1), ("bp2", 3), ("bp3", 4)]
    all_rows = []
    best = {}
    elapsed = {}
    for name, n_seg in models:
        rows, winner, seconds = fit_model(name, n_seg, y0, t_obs, infected, population, CFG)
        all_rows.extend(rows)
        best[name] = winner
        elapsed[name] = seconds
    prior_starts = pd.read_csv(PRIOR / "starts.csv")
    prior_best = {}
    for name, _n_seg in models:
        group = prior_starts[prior_starts["model"] == name]
        prior_best[name] = group.loc[group["rss"].idxmin()].to_dict()
    extras = []
    for name, n_seg in models:
        x0 = continuation_start(prior_best[name], n_seg, CFG)
        extras.append(run_lbfgs(
            name, "continuation_previous_best", "previous_best", n_seg, x0,
            y0, t_obs, infected, population, CFG,
        ))
    embed_x, embed_breaks, embed_betas = embed_bp2_in_bp3(best["bp2"], CFG)
    embed_rss = objective(embed_x, 4, y0, t_obs, infected, population, settings_of(CFG))
    extras.append({
        "model": "bp3",
        "run_set": "nested_bp2_embed_eval",
        "seed": "embed_eval",
        "success": True,
        "status": -2,
        "nit": 0,
        "nfev": 1,
        "rss": float(embed_rss),
        "gamma": best["bp2"]["gamma"],
        "mu": best["bp2"]["mu"],
        "message": "evaluation only; 2-breakpoint solution embedded in 4 segments, not optimized",
        "grad_norm": None,
        "proj_grad_norm": None,
        "n_at_bound": None,
        "bound_hits": "",
        "jac": "",
        "beta1": embed_betas[0],
        "beta2": embed_betas[1],
        "beta3": embed_betas[2],
        "beta4": embed_betas[3],
        "break1": embed_breaks[0],
        "break2": embed_breaks[1],
        "break3": embed_breaks[2],
    })
    print(f"nested embed eval rss={embed_rss:.6e} bp2 rss={best['bp2']['rss']:.6e}", flush=True)
    extras.append(run_lbfgs(
        "bp3", "nested_bp2_embed_optimized", "embed_opt", 4, embed_x,
        y0, t_obs, infected, population, CFG,
    ))
    all_rows.extend(extras)
    frame = pd.DataFrame(all_rows)
    frame.to_csv(OUT / "starts.csv", index=False)

    curves = {}
    beta_series = {}
    metric_rows = []
    checks = []
    obs_metrics = {
        "g_early": growth(infected, 0, 111),
        "g_accel": growth(infected, 110, 151),
        "delta_I_110_150": float(infected[150] - infected[110]),
    }
    obs_metrics["accel"] = obs_metrics["g_accel"] - obs_metrics["g_early"]
    labels = {
        "constant": "Постоянная β, refit на днях 0–368",
        "bp2": "2 точки излома (3 участка)",
        "bp3": "3 точки излома (4 участка)",
        "published_120": "Phase 1, 120 суток (контекст, не refit)",
    }
    published = CFG["contextual_curve"]
    published_traj = integrate(
        [published["beta"]], [], published["gamma"], published["mu"],
        y0, t_obs, population, CFG["solver"]["rtol"], CFG["solver"]["atol"],
    )
    curves["published_120"] = published_traj
    beta_series["published_120"] = np.full(len(t_obs), published["beta"])
    pub_metrics = metrics_block(
        "published_120", published_traj, obs, [published["beta"]], [],
        published["gamma"], published["mu"], population,
        float(np.sum((published_traj[1] - infected) ** 2)),
    )
    pub_metrics["label"] = labels["published_120"]

    for name, n_seg in models:
        check, traj = validate(name, n_seg, best[name], y0, t_obs, population, CFG, best["constant"])
        checks.append(check)
        curves[name] = traj
        betas = [best[name][f"beta{i}"] for i in range(1, n_seg + 1)]
        breaks = [best[name][f"break{i}"] for i in range(1, n_seg) if f"break{i}" in best[name]]
        beta_series[name] = np.array([beta_of(betas, breaks, day) for day in t_obs])
        block = metrics_block(
            name, traj, obs, betas, breaks, best[name]["gamma"], best[name]["mu"],
            population, best[name]["rss"],
        )
        block["label"] = labels[name]
        block["gamma"] = best[name]["gamma"]
        block["mu"] = best[name]["mu"]
        block["betas"] = betas
        block["breaks"] = breaks
        block["seed"] = best[name]["seed"]
        block["success"] = best[name]["success"]
        block["status"] = best[name]["status"]
        block["nit"] = best[name]["nit"]
        block["nfev"] = best[name]["nfev"]
        block["message"] = best[name]["message"]
        block["proj_grad_norm"] = best[name]["proj_grad_norm"]
        block["n_at_bound"] = best[name]["n_at_bound"]
        block["run_set"] = "fresh16"
        metric_rows.append(block)

    # gamma/mu spread among the 16 fresh constant starts within 1e-4 relative RSS of the best
    const_rows = [
        row for row in all_rows
        if row["model"] == "constant" and row["run_set"] == "fresh16" and np.isfinite(row["rss"]) and row["rss"] < 1e29
    ]
    best_rss = min(row["rss"] for row in const_rows)
    near = [row for row in const_rows if row["rss"] <= best_rss * (1 + 1e-4)]
    ident = {
        "constant_near_best_n": len(near),
        "gamma_min": float(min(row["gamma"] for row in near)),
        "gamma_max": float(max(row["gamma"] for row in near)),
        "mu_min": float(min(row["mu"] for row in near)),
        "mu_max": float(max(row["mu"] for row in near)),
        "beta_min": float(min(row["beta1"] for row in near)),
        "beta_max": float(max(row["beta1"] for row in near)),
        "removal_min": float(min(row["gamma"] + row["mu"] for row in near)),
        "removal_max": float(max(row["gamma"] + row["mu"] for row in near)),
    }

    answer = build_answer(obs_metrics, metric_rows)
    for row in metric_rows:
        row["answer"] = answer
    pub_metrics["answer"] = answer
    ordered = [pub_metrics] + metric_rows
    pd.DataFrame(ordered).drop(columns=["answer", "betas", "breaks"], errors="ignore").to_csv(
        OUT / "summary.csv", index=False
    )
    payload = {
        "obs": obs_metrics,
        "models": metric_rows,
        "published_120": {k: v for k, v in pub_metrics.items() if k != "answer"},
        "identifiability_constant": ident,
        "checks": checks,
        "runtime_seconds": elapsed,
        "answer": answer,
        "extras": extras,
        "prior_rss": {name: float(prior_best[name]["rss"]) for name in prior_best},
        "config": CFG,
        "budget": stamp,
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2))
    (OUT / "checks.json").write_text(json.dumps(checks, indent=2))

    table = pd.DataFrame({
        "t": t_obs,
        "I_obs": obs["I"],
        "S_obs": obs["S"],
        "R_obs": obs["R"],
        "D_obs": obs["D"],
    })
    for name in ("published_120", "constant", "bp2", "bp3"):
        table[f"S_{name}"] = curves[name][0]
        table[f"I_{name}"] = curves[name][1]
        table[f"R_{name}"] = curves[name][2]
        table[f"D_{name}"] = curves[name][3]
        table[f"beta_{name}"] = beta_series[name]
    table.to_csv(OUT / "trajectories.csv", index=False)
    make_figure(t_obs, infected, curves, beta_series, OUT / "fig_I_beta_r37.png")
    write_results(OUT / "results.md", ordered, checks, elapsed, obs_metrics)
    print("DONE", json.dumps({"runtime": elapsed, "accel_obs": obs_metrics["accel"]}, indent=2), flush=True)


def build_answer(obs, rows):
    by = {row["model"]: row for row in rows}
    const = by["constant"]
    bp2 = by["bp2"]
    bp3 = by["bp3"]

    def pct(x):
        return 100.0 * x

    return (
        "На том же ряде и в том же окне постоянная β, оцененная заново, даёт прирост I "
        f"{pct(const['g_early']):.2f}% в сутки на первых 110 сутках и {pct(const['g_accel']):.2f}% на сутках 110–150 "
        f"(accel {pct(const['accel']):.2f} п.п.) при наблюдаемых {pct(obs['g_early']):.2f}% и {pct(obs['g_accel']):.2f}% "
        f"(accel {pct(obs['accel']):.2f} п.п.). "
        f"Модель с 2 точками излома даёт {pct(bp2['g_early']):.2f}% и {pct(bp2['g_accel']):.2f}% "
        f"(accel {pct(bp2['accel']):.2f} п.п.; RSS {bp2['rss_I']:.3e} против {const['rss_I']:.3e} у постоянной β). "
        f"Модель с 3 точками излома даёт {pct(bp3['g_early']):.2f}% и {pct(bp3['g_accel']):.2f}% "
        f"(accel {pct(bp3['accel']):.2f} п.п.; RSS {bp3['rss_I']:.3e}). "
        "Даты изломов оценены из данных при фиксированном числе участков и минимальной длине 14 суток; "
        "это не клинически заданные даты. Ступенчатая β(t) уменьшает ошибку по I и меняет прирост на интервале 110–150 "
        "в сторону наблюдаемого ускорения в той мере, в какой это видно из таблицы; она не идентифицирует причину "
        "(передача или выявляемость) и не является проверкой прогноза за пределами окна. "
        "Рукопись этим расчётом ещё не изменена."
    )


if __name__ == "__main__":
    main()
