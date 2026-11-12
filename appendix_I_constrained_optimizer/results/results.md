# Constrained optimizer versus the Parameter Generator

The protocol in `PROTOCOL.md` was frozen before this run. Windows, Critic thresholds, the objective, the Latin-hypercube seed, the number of starts, the 41³ grids, and the solver options were not changed after the results were known. This comparison is not a forecast test: it asks whether a constrained minimizer of the relative parameter change can meet the same Critic inequalities as the published Generator and starter points.

## Phase-1 sanity check

The Critic surrogate `SIRDSurrogate.simulate` (RK45, rtol 1e-6, atol 1e-8, horizon 400 days, 1000 output times, peak = argmax of I) at β = 0.1219, γ = 0.099, μ = 0.0099 gives peak day 264.2642642643 and peak height 39121.313522. Rounded to the published precision these are day 264.3 and height 39121. The published Table 2 anchors are day 264.3 and height 39121. The check passes.

A diagnostic that the protocol also names, a time step of 0.1 day with three-point parabolic interpolation of the same ODE, gives day 264.095655 and height 39121.3617, which round to 264.1 and 39121. The day does not match 264.3. That interpolant was not used for the objective, the constraints, or the grid, and it was not retuned. The 24 problems use the Critic argmax, which is the definition that matches Table 2.

Initial state from `Phase 1 (model calibration)/real_datasets/covid-19_Kouprianov.csv`, first row: S = 5995513, I = 4197, R = 248, D = 42, N = 6000000. Intent Parser logs for the Table 2 runs were not found (no Intent Parser log file was found in the repository). No LLM was called. The six requests use the protocol mapping: Peak should be higher → higher, Peak should be lower → lower, Peak should be earlier → earlier, Peak should be later → later, faster initial growth → growth, higher mortality → mortality.

## Reference peaks

Each published parameter vector was integrated with the same surrogate. A mismatch is reported and the T2 target is left at the printed number.

| Table | Request | Published day, height | Simulated day, height | Rounded match |
| --- | --- | --- | --- | --- |
| Table 2 | higher | 197.8, 100455 | 197.7978, 100454.50 | yes |
| Table 2 | lower | 305.9, 18273 | 305.9059, 18273.30 | yes |
| Table 2 | earlier | 197.8, 100455 | 197.7978, 100454.50 | yes |
| Table 2 | later | 312.7, 12555 | 312.7127, 12554.75 | yes |
| Table 2 | growth | 195.4, 104848 | 195.3954, 104847.68 | yes |
| Table 2 | mortality | 268.3, 35894 | 268.2683, 35894.26 | yes |
| Table 3 | higher | 271.5, 42257 | 271.4715, 42257.20 | yes |
| Table 3 | lower | 257.1, 36356 | 257.0571, 36356.28 | yes |
| Table 3 | earlier | 244.2, 39121 | 243.8438, 39292.55 | no |
| Table 3 | later | 287.9, 39121 | 287.8879, 38920.12 | no |

Mismatches at the published rounding:
- Table 3 earlier: simulated day 243.8438 (rounds to 243.8) and height 39292.55 (rounds to 39293); published day 244.2, height 39121.
- Table 3 later: simulated day 287.8879 (rounds to 287.9) and height 38920.12 (rounds to 38920); published day 287.9, height 39121.

## All 24 problems

J = ((β−β0)/β0)² + ((γ−γ0)/γ0)² + ((μ−μ0)/μ0)² with the Phase-1 point (0.1219, 0.0990, 0.0099). A problem is feasible only when at least one point of its 41³ grid meets the protocol inequalities inside the window. The reported point is then the feasible point with the smallest J among the successful optimizer starts and the feasible grid points, polished with SLSQP when that polish stays feasible and does not increase J. If the grid has no feasible point, the row is infeasible; an optimizer point may still be listed in the CSV and is not the reported solution. Algebraic inequalities on r and CFR use an absolute tolerance of 1e-12. Peak day and peak height do not. `critic_accept` applies `DeterministicCriticAgent`'s checks to the simulated Phase-1 baseline (day 264.2643, height 39121.31, r = 0.013, CFR = 0.0909091), which is not identical to the printed anchors 264.3, 39121, 0.0130 and 0.0909.

| Setting | Request | Target | Feasible | Critic | J | Named change | Unnamed / peak axes | Reference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D | higher | T1 | yes | no | 2.049739e-06 | +3 % height | day -2.04 d, height +3.00% | Table 2 J 6.729652e-03, feasible yes |
| D | higher | T2 | yes | yes | 5.914011e-03 | +156.8 % height | day -65.70 d, height +156.78% | Table 2 J 6.729652e-03, feasible no |
| D | lower | T1 | yes | yes | 2.116110e-06 | -3 % height | day +1.57 d, height -3.00% | Table 2 J 1.682413e-03, feasible yes |
| D | lower | T2 | yes | yes | 1.356050e-03 | -53.29 % height | day +40.00 d, height -53.29% | Table 2 J 1.682413e-03, feasible no |
| D | earlier | T1 | yes | yes | 1.532385e-06 | -1.237 days | day -1.24 d, height +1.69% | Table 2 J 6.729652e-03, feasible yes |
| D | earlier | T2 | yes | yes | 6.102143e-03 | -66.5 days | day -66.50 d, height +159.12% | Table 2 J 6.729652e-03, feasible yes |
| D | later | T1 | yes | yes | 2.845885e-06 | +1.165 days | day +1.17 d, height -2.34% | Table 2 J 3.203987e-03, feasible yes |
| D | later | T2 | yes | yes | 2.807242e-03 | +48.41 days | day +48.41 d, height -63.89% | Table 2 J 3.203987e-03, feasible yes |
| D | growth | T1 | yes | no | 6.143315e-06 | +3 % r | day -3.24 d, height +5.21% | Table 2 J 6.755159e-03, feasible yes |
| D | growth | T2 | yes | yes | 6.713539e-03 | +80.77 % r | day -68.50 d, height +168.21% | Table 2 J 6.755159e-03, feasible yes |
| D | mortality | T1 | yes | no | 7.965816e-04 | +3 % CFR | day -1.64 d, height +3.20% | Table 2 J 3.673095e-03, feasible yes |
| D | mortality | T2 | yes | yes | 3.090922e-03 | +5.501 % CFR | day -0.04 d, height -0.68% | Table 2 J 3.673095e-03, feasible no |
| S | higher | T1 | yes | no | 2.049739e-06 | +3 % height | day -2.04 d, height +3.00% | Table 3 J 4.233173e-03, feasible yes |
| S | higher | T2 | yes | yes | 1.428675e-05 | +8.016 % height | day -4.84 d, height +8.02% | Table 3 J 4.233173e-03, feasible yes |
| S | lower | T1 | yes | yes | 2.116110e-06 | -3 % height | day +1.57 d, height -3.00% | Table 3 J 4.233173e-03, feasible yes |
| S | lower | T2 | yes | yes | 1.199893e-05 | -7.068 % height | day +3.97 d, height -7.07% | Table 3 J 4.233173e-03, feasible no |
| S | earlier | T1 | yes | yes | 4.993320e-07 | -1.237 days | day -1.24 d, height +1.35% | Table 3 J 1.481148e-02, feasible yes |
| S | earlier | T2 | no | no | infeasible |  |  | Table 3 J 1.481148e-02, feasible yes |
| S | later | T1 | yes | yes | 1.922613e-06 | +1.165 days | day +1.17 d, height -2.01% | Table 3 J 1.481148e-02, feasible yes |
| S | later | T2 | no | no | infeasible |  |  | Table 3 J 1.481148e-02, feasible no |
| G | growth | T1 | no | no | infeasible |  |  | Table 2 J 6.755159e-03, feasible no |
| G | growth | T2 | no | no | infeasible |  |  | Table 2 J 6.755159e-03, feasible no |
| G | mortality | T1 | yes | no | 7.965985e-04 | +3 % CFR | day -1.64 d, height +3.00% | Table 2 J 3.673095e-03, feasible no |
| G | mortality | T2 | yes | yes | 3.090922e-03 | +5.501 % CFR | day -0.04 d, height -0.68% | Table 2 J 3.673095e-03, feasible no |

20 of the 24 problems are feasible on the grid oracle. 4 are infeasible.

Per-problem detail (solver, grid count, and the comparison flags) is in `results.csv`.

- D higher T1: β 0.122029, γ 0.098905, μ 0.009899, R0 1.1215, r 0.0132, CFR 0.0910, peak day 262.2623, height 40294.63, J 2.049739e-06, max relative change 0.001056, solver SLSQP, grid feasible 33657, converged starts 21/21, critic no.
- D higher T2: β 0.131234, γ 0.098500, μ 0.009850, R0 1.2112, r 0.0229, CFR 0.0909, peak day 198.5986, height 100455.00, J 5.914011e-03, max relative change 0.07657, solver SLSQP, grid feasible 1000, converged starts 21/21, critic yes.
- D lower T1: β 0.121769, γ 0.099097, μ 0.009901, R0 1.1172, r 0.0128, CFR 0.0908, peak day 265.8659, height 37947.37, J 2.116110e-06, max relative change 0.001075, solver SLSQP, grid feasible 33653, converged starts 21/21, critic yes.
- D lower T2: β 0.117470, γ 0.099500, μ 0.009931, R0 1.0735, r 0.0080, CFR 0.0908, peak day 304.3043, height 18273.00, J 1.356050e-03, max relative change 0.03634, solver SLSQP, grid feasible 17648, converged starts 21/21, critic yes.
- D earlier T1: β 0.121900, γ 0.098885, μ 0.009896, R0 1.1206, r 0.0131, CFR 0.0910, peak day 263.0631, height 39783.84, J 1.532385e-06, max relative change 0.001166, solver SLSQP, grid feasible 35512, converged starts 12/21, critic yes.
- D earlier T2: β 0.131400, γ 0.098500, μ 0.009882, R0 1.2124, r 0.0230, CFR 0.0912, peak day 197.7978, height 101368.85, J 6.102143e-03, max relative change 0.07793, solver grid, grid feasible 975, converged starts 1/21, critic yes.
- D later T1: β 0.121900, γ 0.099167, μ 0.009900, R0 1.1177, r 0.0128, CFR 0.0908, peak day 265.4655, height 38203.87, J 2.845885e-06, max relative change 0.001686, solver SLSQP, grid feasible 32616, converged starts 15/21, critic yes.
- D later T2: β 0.115443, γ 0.098885, μ 0.009896, R0 1.0612, r 0.0067, CFR 0.0910, peak day 312.7127, height 14127.35, J 2.807242e-03, max relative change 0.05297, solver trust-constr, grid feasible 2368, converged starts 1/21, critic yes.
- D growth T1: β 0.122134, γ 0.098846, μ 0.009898, R0 1.1231, r 0.0134, CFR 0.0910, peak day 261.0611, height 41160.32, J 6.143315e-06, max relative change 0.00192, solver SLSQP, grid feasible 33165, converged starts 21/21, critic no.
- D growth T2: β 0.131850, γ 0.098500, μ 0.009850, R0 1.2169, r 0.0235, CFR 0.0909, peak day 195.7958, height 104926.92, J 6.713539e-03, max relative change 0.08162, solver SLSQP, grid feasible 33, converged starts 21/21, critic yes.
- D mortality T1: β 0.121900, γ 0.098500, μ 0.010175, R0 1.1217, r 0.0132, CFR 0.0936, peak day 262.6627, height 40373.93, J 7.965816e-04, max relative change 0.02777, solver SLSQP, grid feasible 23493, converged starts 21/21, critic no.
- D mortality T2: β 0.121900, γ 0.098500, μ 0.010448, R0 1.1189, r 0.0130, CFR 0.0959, peak day 264.2643, height 38856.48, J 3.090922e-03, max relative change 0.05537, solver SLSQP, grid feasible 14227, converged starts 21/21, critic yes.
- S higher T1: β 0.122029, γ 0.098905, μ 0.009899, R0 1.1215, r 0.0132, CFR 0.0910, peak day 262.2623, height 40294.63, J 2.049739e-06, max relative change 0.001056, solver SLSQP, grid feasible 2859, converged starts 20/21, critic no.
- S higher T2: β 0.122239, γ 0.098748, μ 0.009897, R0 1.1251, r 0.0136, CFR 0.0911, peak day 259.4595, height 42257.00, J 1.428675e-05, max relative change 0.002784, solver SLSQP, grid feasible 2356, converged starts 20/21, critic yes.
- S lower T1: β 0.121769, γ 0.099097, μ 0.009901, R0 1.1172, r 0.0128, CFR 0.0908, peak day 265.8659, height 37947.37, J 2.116110e-06, max relative change 0.001075, solver SLSQP, grid feasible 4146, converged starts 21/21, critic yes.
- S lower T2: β 0.121588, γ 0.099230, μ 0.009902, R0 1.1141, r 0.0125, CFR 0.0907, peak day 268.2683, height 36356.00, J 1.199893e-05, max relative change 0.002563, solver SLSQP, grid feasible 3671, converged starts 18/21, critic yes.
- S earlier T1: β 0.121978, γ 0.098973, μ 0.009901, R0 1.1204, r 0.0131, CFR 0.0909, peak day 263.0631, height 39649.19, J 4.993320e-07, max relative change 0.0006404, solver SLSQP, grid feasible 704, converged starts 17/21, critic yes.
- S earlier T2: infeasible. Grid feasible points: 0. Starts that reached a feasible point: 0 of 21.
- S later T1: β 0.121824, γ 0.099065, μ 0.009910, R0 1.1179, r 0.0128, CFR 0.0909, peak day 265.4655, height 38333.14, J 1.922613e-06, max relative change 0.001048, solver SLSQP, grid feasible 632, converged starts 9/21, critic yes.
- S later T2: infeasible. Grid feasible points: 0. Starts that reached a feasible point: 0 of 21.
- G growth T1: infeasible. Grid feasible points: 0. Starts that reached a feasible point: 0 of 21.
- G growth T2: infeasible. Grid feasible points: 0. Starts that reached a feasible point: 0 of 21.

The empty grids have a direct reading from the stored 41³ simulations. For S earlier T2, every grid point that keeps peak height within 3% has peak day at least 244.24, and the printed target is day ≤ 244.2. The Table 3 earlier parameters are not a grid node. Their simulated day is 243.84 and their height is inside the 3% band, so that single point is feasible while the grid oracle is empty, and none of the 21 starts found it. For S later T2, no point in the wide-γ window keeps peak height within 3% and has peak day ≥ 287.9. The latest height-held grid day is 287.49. The Table 3 later simulation falls at day 287.89, which is still short of the printed threshold 287.9, so the starter is not feasible for T2 either. For G growth, the points in the default window that hold both peak axes have r between 0.01273 and 0.01326. The T1 threshold is r ≥ 0.01339, so a 3% faster growth rate and a held peak do not occur together inside that window. The Table 2 growth parameters move the peak by about 69 days and are not feasible for G.
- G mortality T1: β 0.121884, γ 0.098500, μ 0.010175, R0 1.1215, r 0.0132, CFR 0.0936, peak day 262.6627, height 40294.63, J 7.965985e-04, max relative change 0.02777, solver SLSQP, grid feasible 544, converged starts 21/21, critic no.
- G mortality T2: β 0.121900, γ 0.098500, μ 0.010448, R0 1.1189, r 0.0130, CFR 0.0959, peak day 264.2643, height 38856.48, J 3.090922e-03, max relative change 0.05537, solver SLSQP, grid feasible 329, converged starts 21/21, critic yes.

## Comparison with the Generator and the starter

Settings D and G are compared with the Table 2 Generator parameters. Setting S is compared with the Table 3 starter. Reference J uses the same formula at those published parameters. A reference point is feasible only when the same protocol inequalities accept it. Side effects use the simulated peak, not the printed peak: absolute day shift for higher/lower, absolute height percent for earlier/later, and the larger of |Δday|/8 and |Δheight|/3% for growth and mortality.

On J, among problems where both the reported point and the reference point are feasible, the reference has the smaller J in 0 problems.
On the side-effect scalar, the same both-feasible rule gives the reference the smaller side effect in 4 problems: D earlier T2, D growth T2, S earlier T1, S later T1.
On feasibility, the reference point meets the protocol inequalities in 1 problem whose grid oracle is empty: S earlier T2.
Ignoring feasibility, the reference parameter vector has a smaller J than the reported point in 0 feasible optimizer rows. That count includes reference vectors that miss the printed constraint, so it is not by itself evidence that the Generator solved the problem with a smaller change.
The optimizer reports a feasible point in 6 problems where the published reference parameters do not meet the same inequalities: D higher T2, D lower T2, D mortality T2, S lower T2, G mortality T1, G mortality T2.
Protocol-feasible reported points that the code Critic rejects: 5: D higher T1, D growth T1, D mortality T1, S higher T1, G mortality T1.

## Checks

Reported solutions were integrated again. 20 feasible rows were checked. Compartment negativity or a conservation error above 1e-4 person: 0. J differing from the formula by more than 1e-9: 0. Baseline conservation error 2.794e-09, minimum compartments S 4915595.871, I 4197.000, R 248.000, D 42.000.

## Reproducibility

Command: `python3 appendix_I_constrained_optimizer/run_experiment.py`.
Python 3.12.3, numpy 2.5.4, scipy 1.18.1, platform Linux-6.12.113+-x86_64-with-glibc2.39.
Latin hypercube: 20 points in the unit cube, seed 0, scramble true, plus the Phase-1 point, 21 starts in total. Grid: 41³ per window, three windows (default D, mortality, wide-γ S).
Unique surrogate calls: 682334. Wall time: 263.5 s, of which the three grids took 64.4 s.
Per-problem `n_simulations` counts integrations inside that problem's multistart and polish, not the shared grid. `wall_time_s` is the multistart plus polish only.
Outputs: `results.csv`, `results.md`, `logs/`.

