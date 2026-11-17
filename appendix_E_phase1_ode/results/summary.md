# Phase-1 PINN with and without the ODE term

Observed: peak on day 199 (104 932 active cases). Errors are for the network output for I on days 185-214. Peak day: argmax of the network output over days 0-368; window peak: argmax over days 185-214 (the definition used in Section 4.1.1).

| arch | lambda_ode | n | RMSE | RMSE range | MAE | peak day | peak day range | peak height | window peak day | beta | gamma | mu | pairs better than lambda_ode=0: RMSE / MAE / peak day / peak height |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 128-256-128 | 0 | 10 | 5613 ± 1252 | 4297-8008 | 4692 ± 707 | 204.8 ± 3.6 | 201-210 | 102472 ± 3302 | 204.8 ± 3.6 | 0.1000 ± 0.0000 | 0.1000 ± 0.0000 | 0.00100 ± 0.00000 |  |
| 128-256-128 | 0.01 | 10 | 5576 ± 1213 | 4334-7897 | 4680 ± 662 | 204.9 ± 3.3 | 201-210 | 102476 ± 3132 | 204.9 ± 3.3 | 0.0383 ± 0.0037 | 0.0319 ± 0.0018 | 0.00179 ± 0.00041 | 7 / 7 / 2 (ties 6) / 5 of 10 |
| 128-256-128 | 0.1 | 10 | 4955 ± 710 | 4210-6252 | 4289 ± 442 | 203.2 ± 2.1 | 201-207 | 101469 ± 2413 | 203.2 ± 2.1 | 0.0401 ± 0.0027 | 0.0330 ± 0.0012 | 0.00197 ± 0.00034 | 9 / 9 / 8 (ties 1) / 2 of 10 |
| 128-256-128 | 1 | 10 | 4149 ± 882 | 2897-5666 | 3652 ± 764 | 198.1 ± 0.7 | 197-199 | 98743 ± 1486 | 198.1 ± 0.7 | 0.0459 ± 0.0010 | 0.0372 ± 0.0005 | 0.00228 ± 0.00033 | 7 / 8 / 10 (ties 0) / 0 of 10 |
| 128-256-128 | 10 | 10 | 7811 ± 891 | 6542-8980 | 7613 ± 890 | 193.7 ± 0.7 | 193-195 | 95652 ± 1124 | 193.7 ± 0.7 | 0.0496 ± 0.0003 | 0.0418 ± 0.0004 | 0.00246 ± 0.00027 | 1 / 0 / 4 (ties 1) / 0 of 10 |

With lambda_ode = 0 the parameters receive no gradient and stay at their initial values (0.1, 0.1, 0.001); they are listed only for completeness.
