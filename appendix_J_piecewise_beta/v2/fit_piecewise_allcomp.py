#!/usr/bin/env python3
"""Manuscript Appendix G: piecewise-constant beta refitted to S, I, R and D.

This is the script behind the four-series paragraph of Appendix G
(results in results_allcomp/). Same model, data, bounds, solver and
L-BFGS-B settings as the I-only fits of Table A14 (run_r37.py); the
objectives are defined in allcomp_core.py (SIRD4: one variance for S, I, R, D;
IRDsep: separate variances for I, R, D).

Per model (constant beta, 2 and 3 breakpoints) and objective: 64 seeded
starts (breakpoint positions from a Latin hypercube over the admissible
simplex of segment lengths; half of the starts with beta, gamma, mu
log-uniform over the full bounds, half in a region set by the observed R and
D: gamma in [0.01, 0.2], mu in [3e-4, 0.02], beta per segment =
(gamma+mu) x ratio, ratio log-uniform in [0.5, 2.5]); plus labelled starts:
the best I-only point (results/summary.json), the best point of an earlier
16-start search (data/allcomp_prev16_best_starts.csv), and the best simpler
model of this run embedded. The 5 best end points per model are polished:
L-BFGS-B on log(objective) (maxfun 20000, ftol 1e-13), then bounded
Nelder-Mead (8000 evaluations). The archived I-only fits are re-integrated
(not refitted) and added to summary.csv as "Ionly_archived" for comparison.

Usage, from the repository root (about 35 min with 7 workers):
  python appendix_J_piecewise_beta/fit_piecewise_allcomp.py
  python appendix_J_piecewise_beta/make_figure_allcomp.py appendix_J_piecewise_beta/results_allcomp_rerun
Output goes to appendix_J_piecewise_beta/results_allcomp_rerun/ (or --out);
the archived run in results_allcomp/ is not overwritten.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import allcomp_core as RA  # noqa: E402

N_STARTS = 64
SEED_BASE = 20261117


def wide_starts(n_seg, cfg):
    lo_b, hi_b = cfg["bounds"]["beta"]
    lo_g, hi_g = cfg["bounds"]["gamma"]
    lo_m, hi_m = cfg["bounds"]["mu"]
    if n_seg > 1:
        lhs = qmc.LatinHypercube(d=n_seg, seed=SEED_BASE + n_seg).random(N_STARTS)
        lhs = np.clip(lhs, 1e-9, 1 - 1e-9)
    starts = []
    for s in range(N_STARTS):
        rng = np.random.default_rng(SEED_BASE + 1000 * n_seg + s)
        if s % 2 == 0:
            g = float(np.exp(rng.uniform(np.log(lo_g), np.log(hi_g))))
            m = float(np.exp(rng.uniform(np.log(lo_m), np.log(hi_m))))
            b = np.exp(rng.uniform(np.log(lo_b), np.log(hi_b), size=n_seg))
        else:
            g = float(np.exp(rng.uniform(np.log(0.01), np.log(0.2))))
            m = float(np.exp(rng.uniform(np.log(3e-4), np.log(0.02))))
            b = (g + m) * np.exp(rng.uniform(np.log(0.5), np.log(2.5), size=n_seg))
        b = np.clip(b, lo_b, hi_b)
        if n_seg > 1:
            shares = -np.log(lhs[s])
            z = np.log(shares / shares.sum())
            z = np.clip(z - z.max(), -6.0, 6.0)
        else:
            z = np.array([])
        starts.append(np.concatenate([b, [g, m], z]).tolist())
    return starts


def logobj(x, *a):
    v = RA.obj_fn(x, *a)
    if a[-1] == "SIRD4":
        return float(np.log(v)) if v < 1e29 else 1e3
    return v / 1000.0 if v < 1e29 else 1e3


def polish(args):
    (jdir, objective, name, n_seg, label, x0) = args
    if RA.J is None:
        RA.load_j(Path(jdir))
    J = RA.J
    cfg = J.CFG
    t_obs, obs, y0 = J.load_observations()
    obs_mat = np.vstack([obs[c] for c in "SIRD"])
    settings = J.settings_of(cfg)
    box = J.bounds_for(n_seg, cfg)
    a = (n_seg, y0, t_obs, obs_mat, float(cfg["series"]["N"]), settings, objective)
    rows = []
    t0 = time.perf_counter()
    r1 = minimize(logobj, np.asarray(x0, float), args=a, method="L-BFGS-B", bounds=box,
                  options={"maxiter": 5000, "maxfun": 20000, "ftol": 1e-13, "gtol": 1e-8})
    r2 = minimize(RA.obj_fn, r1.x, args=a, method="Nelder-Mead", bounds=box,
                  options={"maxiter": 8000, "maxfev": 8000, "xatol": 1e-7,
                           "fatol": 1e-8 * abs(RA.obj_fn(r1.x, *a)), "adaptive": True})
    for meth, r in (("polish_lbfgs_log", r1), ("polish_lbfgs_log+NM", r2)):
        fake = type("R", (), {})()
        fake.x, fake.success, fake.status = r.x, r.success, getattr(r, "status", -1)
        fake.nit, fake.nfev, fake.message = r.nit, r.nfev, r.message
        fake.fun, fake.jac = RA.obj_fn(r.x, *a), None
        row = J.row_from_result(name, meth, label, n_seg, fake, box,
                                settings["t_max"], settings["min_len"])
        row["objective"] = objective
        row["obj_value"] = row.pop("rss")
        row["seconds"] = time.perf_counter() - t0
        row["x"] = json.dumps([float(v) for v in r.x])
        row.pop("jac", None)
        rows.append(row)
    print(f"[{objective}] polish {name} from {label}: lbfgs_log={rows[0]['obj_value']:.7e} "
          f"NM={rows[1]['obj_value']:.7e} {rows[1]['seconds']:.0f}s", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jdir", default=str(RA.DEFAULT_J),
                    help="folder with run_r37.py, config.json, data/ and results/ (default: this folder)")
    ap.add_argument("--prev", default=str(HERE / "data" / "allcomp_prev16_best_starts.csv"),
                    help="best end points of the earlier 16-start search, used as labelled starts")
    ap.add_argument("--out", default=str(HERE / "results_allcomp_rerun"))
    ap.add_argument("--workers", type=int, default=min(7, os.cpu_count() or 1))
    ap.add_argument("--objectives", default="SIRD4,IRDsep")
    args = ap.parse_args()
    jdir = Path(args.jdir).resolve()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    RA.load_j(jdir)
    J = RA.J
    cfg = J.CFG
    t_obs, obs, y0 = J.load_observations()
    population = float(cfg["series"]["N"])
    archived = json.loads((jdir / "results" / "summary.json").read_text())
    ionly = {m["model"]: m for m in archived["models"]}
    prev = pd.read_csv(Path(args.prev))
    all_rows, summary, trajs = [], [], {}
    t_start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=RA.load_j, initargs=(jdir,)) as pool:
        for objective in args.objectives.split(","):
            best = {}
            for name, n_seg in RA.MODELS:
                jobs = [(str(jdir), objective, name, n_seg, "wide64", s, x0)
                        for s, x0 in enumerate(wide_starts(n_seg, cfg))]
                m = ionly[name]
                jobs.append((str(jdir), objective, name, n_seg, "from_Ionly_best", "J",
                             RA.vec(m["betas"], m["breaks"], m["gamma"], m["mu"], cfg)))
                pv = prev[(prev.objective == objective) & (prev.model == name)]
                pb = pv.loc[pv.obj_value.idxmin()]
                jobs.append((str(jdir), objective, name, n_seg, "from_prev_allcomp_best", "prev",
                             json.loads(pb["x"])))
                for smaller in [mm for mm, ns in RA.MODELS if ns < n_seg]:
                    b, br = RA.pieces(best[smaller], dict(RA.MODELS)[smaller])
                    jobs.append((str(jdir), objective, name, n_seg, f"embed_{smaller}", "this_run",
                                 RA.embed(b, br, best[smaller]["gamma"], best[smaller]["mu"], n_seg, cfg)))
                rows = list(pool.map(RA.job, jobs))
                finite = sorted([r for r in rows if r["obj_value"] < 1e29], key=lambda r: r["obj_value"])
                pjobs = [(str(jdir), objective, name, n_seg, f"{r['run_set']}:{r['seed']}",
                          json.loads(r["x"])) for r in finite[:5]]
                for pr in pool.map(polish, pjobs):
                    rows.extend(pr)
                all_rows.extend(rows)
                pd.DataFrame(all_rows).to_csv(out / "starts.csv", index=False)
                finite = [r for r in rows if r["obj_value"] < 1e29]
                best[name] = min(finite, key=lambda r: r["obj_value"])
                bv = best[name]["obj_value"]
                met, traj = RA.metrics(name, best[name], n_seg, objective, t_obs, obs, y0, population, cfg)
                met["obj_value"] = bv
                met["prev_best_obj"] = float(pb["obj_value"])
                lo = [r for r in rows if r["run_set"] == "wide64" and r["obj_value"] < 1e29]
                met["best_wide64_obj"] = min(r["obj_value"] for r in lo)
                tol = 1e-3 * abs(bv) if objective == "SIRD4" else 1e-3 * abs(bv)
                met["n_wide64_within_0.1pct"] = int(sum(r["obj_value"] - bv <= tol for r in lo))
                met["n_wide64_within_1pct"] = int(sum(r["obj_value"] - bv <= 10 * tol for r in lo))
                met["n_wide64_finite"] = len(lo)
                summary.append(met)
                trajs[(objective, name)] = traj
    # archived I-only fits (Table A14), re-integrated, same metrics
    for name, n_seg in RA.MODELS:
        m = ionly[name]
        row = {f"beta{i+1}": b for i, b in enumerate(m["betas"])}
        row.update({f"break{i+1}": b for i, b in enumerate(m["breaks"])})
        row.update({"gamma": m["gamma"], "mu": m["mu"], "run_set": m["run_set"], "seed": m["seed"],
                    "status": m["status"], "message": m["message"], "nit": m["nit"], "nfev": m["nfev"],
                    "n_at_bound": m["n_at_bound"], "bound_hits": ""})
        met, traj = RA.metrics(name, row, n_seg, "Ionly_archived", t_obs, obs, y0, population, cfg)
        met["obj_value"] = met["rss_I"]
        summary.append(met)
        trajs[("Ionly_archived", name)] = traj
    df = pd.DataFrame(summary)
    df.to_csv(out / "summary.csv", index=False)
    tr = {"t": t_obs}
    for c_i, c in enumerate("SIRD"):
        tr[f"{c}_obs"] = obs[c]
    for (objective, name), traj in trajs.items():
        for c_i, c in enumerate("SIRD"):
            tr[f"{c}_{objective}_{name}"] = traj[c_i]
    pd.DataFrame(tr).to_csv(out / "trajectories.csv", index=False)
    obs_info = {
        "left_S_frac_368_registered": float(1.0 - obs["S"][-1] / population),
        "R_368": float(obs["R"][-1]), "D_368": float(obs["D"][-1]),
        "peak_I": float(obs["I"].max()), "peak_day_I": int(np.argmax(obs["I"])),
        "g_early": J.growth(obs["I"], 0, 111), "g_accel": J.growth(obs["I"], 110, 151),
        "delta_I_110_150": float(obs["I"][150] - obs["I"][110]),
    }
    meta = {"observed": obs_info, "elapsed_seconds": time.perf_counter() - t_start,
            "n_starts": N_STARTS, "seed_base": SEED_BASE, "n_polished": 5,
            "workers": args.workers, "python": sys.version.split()[0],
            "numpy": np.__version__, "scipy": __import__("scipy").__version__,
            "data_sha256": cfg["series"]["sha256"]}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
