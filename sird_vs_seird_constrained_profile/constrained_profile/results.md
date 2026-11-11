
# Constrained SEIR-D E(0) profile, latent period held to 2–14 days

Direct least-squares on the St. Petersburg series, days 0–368, N = 6,000,000. σ ∈ [1/14, 1/2]. Fixed E(0)/I(0) grid only. ΔAIC = n ln(RSS_SEIR-D / RSS_SIRD) + 2 against the SIRD fit from this same run and the same error model. Protocol: `PROTOCOL.md`.

## Copy-ready headline

Best point on the constrained profile for each error model (lowest ΔAIC).

### I only (fit I(t) only)

| field | value |
|---|---|
| error model | I only |
| best E(0)/I(0) | 0 |
| E(0) | 0 |
| ΔAIC vs SIRD | -352.872745 |
| AIC SEIR-D | 8771.998367 |
| AIC SIRD | 9124.871112 |
| RSS SEIR-D | 2.064363922e+10 |
| RSS SIRD | 5.400744998e+10 |
| latent period 1/σ (days) | 14.00000000 |
| latent bound | upper_14d |
| ever infected | 62.0787% (0.62078731 of N) |
| dip_ratio | 0.28385756 |
| dip_day | 11 |
| peak I (day) | 89273.399501 (day 198) |
| β | 0.4594096319 |
| σ | 0.07142857143 |
| γ | 0.2710892215 |
| μ | 0.02172214711 |
| γ+μ | 0.2928113686 |
| R0 | 1.56896105 |
| at_bound | sigma |
| starts within 1 AIC | 14 / 26 |

### Four compartments (S, I, R, D, one variance; SEIR-D S residual uses S+E)

| field | value |
|---|---|
| error model | S, I, R, D, one variance |
| best E(0)/I(0) | 2 |
| E(0) | 8394 |
| ΔAIC vs SIRD | +55.733022 |
| AIC SEIR-D | 41653.371463 |
| AIC SIRD | 41597.638441 |
| RSS SEIR-D | 1.793197251e+12 |
| RSS SIRD | 1.729090788e+12 |
| latent period 1/σ (days) | 2.00000000 |
| latent bound | lower_2d |
| ever infected | 8.8658% (0.08865848 of N) |
| dip_ratio | 1.00000000 |
| dip_day | 0 |
| peak I (day) | 39962.458787 (day 368) |
| β | 0.0569641853 |
| σ | 0.5 |
| γ | 0.04888848602 |
| μ | 0.002152702315 |
| γ+μ | 0.05104118834 |
| R0 | 1.11604348 |
| at_bound | sigma |
| starts within 1 AIC | 26 / 26 |

## Comparison with the released fits

On active cases alone, the lowest ΔAIC on this constrained grid is -352.873 at E(0)/I(0) = 0, with latent period 14.00 days, on the 14-day upper edge, ever-infected share 62.08% of N, and dip_ratio 0.2839 on day 11. The fixed-ratio point E(0)/I(0) = 0.5 on the same grid has ΔAIC -289.265, latent period 14.00 days, on the 14-day upper edge, ever-infected 60.19%, and dip_ratio 0.3859. The released constrained row at that same ratio (`ode_lsq_reference.csv`, incubation 2–14 d, fixed E(0)) is ΔAIC -289.265, latent period 14.00 days, ever-infected 60.19%, dip_ratio 0.3859. The unconstrained I-only profile in that file is lowest at E(0)/I(0) = 0 (ΔAIC -413.404, latent period about 28.8 days, ever-infected 82.9%, dip_ratio 0.174); that latent period lies outside the 2–14 day box, so the constrained profile cannot recover it. Every ratio on this I-only grid sits on the 14-day edge. ΔAIC rises with E(0)/I(0) and changes sign between 2 (-118.7) and 4 (+43.1). The ΔAIC of about -35 with a 6.1-day latent period is the self-consistent E(0) rule already stored in the same file (E(0)/I(0) ≈ 1.03), which this grid does not recompute.

On all four compartments with one variance, the lowest ΔAIC on this constrained grid is +55.733 at E(0)/I(0) = 2, with latent period 2.00 days, on the 2-day lower edge, ever-infected share 8.87% of N, and dip_ratio 1.0000 on day 0. The fixed-ratio point E(0)/I(0) = 0.5 has ΔAIC +69.818, latent period 2.00 days, on the 2-day lower edge, ever-infected 8.71%, and dip_ratio 0.8699. The released constrained row at that ratio is ΔAIC +69.819, latent period 2.00 days, ever-infected 8.71%, dip_ratio 0.8701. The unconstrained four-compartment profile is lowest at E(0)/I(0) = 0 (ΔAIC +2.297, latent period about 0.01 days, σ on its unconstrained upper edge). The E(0)/I(0) = 4 point on this grid has latent period 10.38 days, on neither edge, and ΔAIC +83.194. It matches the released free-σ row at that ratio (ΔAIC +83.194, latent period 10.38 days), whose optimum already lay inside [2, 14]. At E(0)/I(0) = 24, μ = 1e-07 is on its lower edge as well as σ on the 14-day edge. Positive ΔAIC at every ratio means the SIRD baseline remains preferred under the four-compartment residual once 1/σ is held in [2, 14] days.

## I-only profile (10 fixed ratios)

γ and μ are not separately identified on this criterion; γ+μ is. The split below is the one returned by the winning start.

| E(0)/I(0) | ΔAIC | 1/σ (days) | latent bound | ever infected % | dip_ratio | dip_day | peak I | peak day | β | σ | γ | μ |
|---:|---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | -352.873 | 14.0000 | upper_14d | 62.08 | 0.2839 | 11 | 89273.4 | 198 | 0.45941 | 0.0714286 | 0.271089 | 0.0217221 |
| 0.25 | -321.372 | 14.0000 | upper_14d | 61.09 | 0.3343 | 11 | 87918.8 | 198 | 0.441461 | 0.0714286 | 0.284353 | 2.28408e-06 |
| 0.5 | -289.265 | 14.0000 | upper_14d | 60.19 | 0.3859 | 11 | 86694.0 | 197 | 0.425887 | 0.0714286 | 0.256641 | 0.0202734 |
| 1 | -227.142 | 14.0000 | upper_14d | 58.60 | 0.4913 | 10 | 84536.3 | 197 | 0.399869 | 0.0714286 | 0.240276 | 0.0239847 |
| 2 | -118.658 | 14.0000 | upper_14d | 55.95 | 0.7065 | 8 | 80996.2 | 196 | 0.360778 | 0.0714286 | 0.244597 | 0.000108092 |
| 4 | +43.108 | 14.0000 | upper_14d | 51.84 | 1.0000 | 0 | 75658.4 | 194 | 0.308632 | 0.0714286 | 0.212066 | 0.00546699 |
| 8 | +250.219 | 14.0000 | upper_14d | 45.68 | 1.0000 | 0 | 68226.6 | 189 | 0.245768 | 0.0714286 | 0.182957 | 6.08171e-06 |
| 12 | +389.212 | 14.0000 | upper_14d | 40.82 | 1.0000 | 0 | 63043.0 | 180 | 0.205404 | 0.0714286 | 0.15913 | 0.000484775 |
| 16 | +497.156 | 14.0000 | upper_14d | 36.89 | 1.0000 | 0 | 59527.0 | 166 | 0.17708 | 0.0714286 | 0.142592 | 0.000156664 |
| 24 | +666.016 | 14.0000 | upper_14d | 32.18 | 1.0000 | 0 | 57182.6 | 126 | 0.149069 | 0.0714286 | 0.127035 | 2.9094e-05 |

## Four-compartment profile (10 fixed ratios)

Observed S is compared with S+E. n = 1476.

| E(0)/I(0) | ΔAIC | 1/σ (days) | latent bound | ever infected % | dip_ratio | dip_day | peak I | peak day | β | σ | γ | μ |
|---:|---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | +66.249 | 2.0000 | lower_2d | 8.54 | 0.5651 | 59 | 11719.3 | 286 | 0.167129 | 0.5 | 0.146655 | 0.0102453 |
| 0.25 | +71.506 | 2.0000 | lower_2d | 8.63 | 0.7175 | 59 | 15047.1 | 307 | 0.133165 | 0.5 | 0.116477 | 0.00772405 |
| 0.5 | +69.818 | 2.0000 | lower_2d | 8.71 | 0.8699 | 59 | 18580.6 | 326 | 0.110723 | 0.5 | 0.0965341 | 0.00606058 |
| 1 | +60.037 | 2.0000 | lower_2d | 8.81 | 1.0000 | 0 | 26031.0 | 361 | 0.0832235 | 0.5 | 0.0721107 | 0.00403586 |
| 2 | +55.733 | 2.0000 | lower_2d | 8.87 | 1.0000 | 0 | 39962.5 | 368 | 0.0569642 | 0.5 | 0.0488885 | 0.0021527 |
| 4 | +83.194 | 10.3833 | neither | 9.20 | 1.0000 | 0 | 46033.2 | 368 | 0.0523824 | 0.0963088 | 0.0436943 | 0.00168955 |
| 8 | +242.598 | 14.0000 | upper_14d | 8.98 | 1.0000 | 0 | 62524.7 | 368 | 0.0331159 | 0.0714286 | 0.0276877 | 0.000625039 |
| 12 | +609.815 | 14.0000 | upper_14d | 8.53 | 1.0000 | 0 | 70211.9 | 368 | 0.0235857 | 0.0714286 | 0.0204309 | 0.000225822 |
| 16 | +991.115 | 14.0000 | upper_14d | 8.13 | 1.0000 | 0 | 72213.5 | 368 | 0.0184016 | 0.0714286 | 0.0167189 | 5.99916e-05 |
| 24 | +1639.616 | 14.0000 | upper_14d | 7.44 | 1.0000 | 0 | 86779.2 | 60 | 0.0125534 | 0.0714286 | 0.0128591 | 1e-07 |

## Sanity check against ode_lsq_reference.csv

The optimizer was not started from the stored parameter vectors. Pass means the refit is in the same regime as the stored constrained row at E(0)/I(0) = 0.5.

| error model | ΔAIC this run | ΔAIC stored | latent days | bound | ever infected | dip_ratio | RSS rel | pass |
|---|---:|---:|---:|:---|---:|---:|---:|:---|
| I only | -289.265 | -289.265 | 14.0000 | upper_14d | 60.19% | 0.3859 | 6.575e-08 | yes |
| Four compartments | +69.818 | +69.819 | 2.0000 | lower_2d | 8.71% | 0.8699 | 7.681e-07 | yes |

## SIRD baselines from this run

| error model | RSS | AIC | β | γ | μ | R0 | ever infected % | peak I (day) | dip_ratio (day) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| I only | 5.400744998e+10 | 9124.8711 | 0.141584 | 0.115941 | 0.0039567 | 1.1809 | 28.49 | 77790.2 (196) | 1.0000 (0) |
| S, I, R, D, one variance | 1.729090788e+12 | 41597.6384 | 0.125264 | 0.110184 | 0.00734989 | 1.0658 | 8.50 | 15589.7 (288) | 0.7491 (59) |

Machine-readable copy: `profile.csv` and `summary.json`. Log: `run.log`.

