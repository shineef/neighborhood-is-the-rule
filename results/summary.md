# Results

All numbers reported in the paper, and the 25 public games one by one. Unless stated otherwise the score is the share of test steps whose next frame is predicted exactly (all 4096 cells).

**Protocol.** In each game a random agent plays 900 moves (uniform over the available actions; clicks land on uniform cells). The first 300 moves train the predictors, the last 600 test them. Networks are pretrained on 3000 random moves of each of the other 24 games. *Real events* are steps that change more than three cells.

## Table 1: after 300 moves of a game (mean over 25 games)

| Predictor | Exact frame | Exact on real events | Precision | Recall |
|---|---|---|---|---|
| Nothing changes | 0.184 | 0.000 | | |
| Network from scratch (10 seeds) | 0.364 ± 0.006 | 0.010 ± 0.005 | | |
| Network pretrained on 24 games (10 seeds) | 0.448 ± 0.014 | 0.111 ± 0.014 | | |
| Network, Reptile initialization (10 seeds) | 0.450 ± 0.013 | 0.113 ± 0.013 | | |
| Table, 3×3 | 0.352 | 0.016 | 0.810 | 0.467 |
| Table, 7×7 | 0.422 | 0.086 | 0.825 | 0.614 |
| Table, 13×13 | 0.551 | 0.229 | 0.854 | 0.625 |
| Table, 25×25 | 0.504 | 0.177 | 0.896 | 0.489 |
| Table, 41×41 | 0.443 | 0.122 | 0.907 | 0.365 |
| Table, widest neighborhood seen (7, 13, 25, 41) | 0.578 | 0.270 | 0.886 | 0.708 |
| Network alone, in the 3-seed comparison below | 0.447 ± 0.017 | 0.108 ± 0.019 | 0.839 | 0.654 |
| Table 13×13, network where unseen (3 seeds) | 0.576 ± 0.002 | 0.279 ± 0.003 | 0.859 | 0.742 |
| Widest seen, network where unseen (3 seeds) | 0.586 ± 0.004 | 0.295 ± 0.004 | 0.878 | 0.758 |

Precision: of the cells a predictor changes, the share that really change. Recall: of the cells that really change, the share the predictor changes. Choosing the best table width separately for each game gives 0.580.

Games won (exact frame, win / lose / tie): 13×13 table against the pretrained network 17 / 7 / 1; table-first hybrid (13×13) against the 13×13 table 17 / 7 / 1 and against the network 19 / 2 / 4; widest-seen table against the 13×13 table 16 / 7 / 2.

## Share of changing cells whose neighborhood the table has seen (300 moves)

| Table | 3×3 | 7×7 | 13×13 | 25×25 | 41×41 |
|---|---|---|---|---|---|
| Split of Table 1 | 0.980 | 0.895 | 0.766 | 0.583 | 0.417 |
| Split of Fig. 2 (see below) | 0.970 | 0.873 | 0.713 | 0.537 | 0.400 |

## Fig. 2: tables filled from a growing number of moves

A separate random run per game (3600 moves, last 600 for testing). Each cell: exact frame / share of changing cells whose neighborhood was seen.

| Moves | 3×3 | 7×7 | 13×13 | 25×25 | 41×41 |
|---|---|---|---|---|---|
| 50 | 0.331 / 0.926 | 0.404 / 0.733 | 0.435 / 0.503 | 0.375 / 0.313 | 0.314 / 0.209 |
| 100 | 0.344 / 0.951 | 0.427 / 0.805 | 0.481 / 0.599 | 0.421 / 0.404 | 0.363 / 0.291 |
| 300 | 0.352 / 0.970 | 0.445 / 0.873 | 0.539 / 0.713 | 0.500 / 0.537 | 0.434 / 0.400 |
| 1000 | 0.356 / 0.978 | 0.466 / 0.915 | 0.588 / 0.801 | 0.556 / 0.640 | 0.501 / 0.496 |

## Networks, seed by seed (mean over 25 games)

| Seed | Scratch | Pretrained | Reptile | Reptile − pretrained |
|---|---|---|---|---|
| 0 | 0.3562 | 0.4506 | 0.4624 | +0.0117 |
| 1 | 0.3709 | 0.4496 | 0.4398 | -0.0098 |
| 2 | 0.3546 | 0.4436 | 0.4576 | +0.0140 |
| 3 | 0.3609 | 0.4444 | 0.4426 | -0.0018 |
| 4 | 0.3633 | 0.4549 | 0.4428 | -0.0122 |
| 5 | 0.3667 | 0.4138 | 0.4534 | +0.0397 |
| 6 | 0.3669 | 0.4632 | 0.4699 | +0.0068 |
| 7 | 0.3667 | 0.4548 | 0.4410 | -0.0137 |
| 8 | 0.3696 | 0.4422 | 0.4623 | +0.0201 |
| 9 | 0.3692 | 0.4640 | 0.4278 | -0.0362 |
| mean | 0.3645 | 0.4481 | 0.4500 | +0.0019 |

Reptile − pretrained over the ten seeds: mean +0.0019, SD 0.0213, t = 0.28, positive in 5 of 10. Pretrained − scratch: mean +0.0836, SD 0.0149, t = 17.77, positive in 10 of 10.

## Networks with wider views (3 seeds, same 300 moves)

| View | Scratch: exact / events | Pretrained: exact / events |
|---|---|---|
| 9×9 | 0.362 ± 0.008 / 0.007 | 0.454 ± 0.004 / 0.119 |
| 13×13 | 0.352 ± 0.005 / 0.003 | 0.448 ± 0.010 / 0.109 |
| 25×25 | 0.297 ± 0.019 / 0.001 | 0.409 ± 0.012 / 0.065 |

## Fig. 3b: predictors fed their own predictions

Exact frame after k imagined steps, on the same 100 windows of real play per game (networks: 3 seeds).

| Predictor | k = 1 | 2 | 3 | 5 | 10 |
|---|---|---|---|---|---|
| Table, widest seen | 0.564 | 0.450 | 0.387 | 0.322 | 0.232 |
| Widest seen, network where unseen | 0.541 | 0.437 | 0.384 | 0.321 | 0.240 |
| Table, 13×13 | 0.539 | 0.397 | 0.315 | 0.245 | 0.165 |
| Table 13×13, network where unseen | 0.520 | 0.390 | 0.308 | 0.241 | 0.175 |
| Table, 25×25 | 0.489 | 0.352 | 0.277 | 0.205 | 0.128 |
| Table, 41×41 | 0.427 | 0.299 | 0.240 | 0.168 | 0.100 |
| Table, 7×7 | 0.414 | 0.294 | 0.234 | 0.198 | 0.150 |
| Network, pretrained (9×9) | 0.405 | 0.273 | 0.217 | 0.174 | 0.144 |
| Network, from scratch (9×9) | 0.315 | 0.226 | 0.187 | 0.157 | 0.137 |
| Nothing changes | 0.188 | 0.095 | 0.072 | 0.058 | 0.043 |

One-step accuracy to the power k, averaged over runs, for the pretrained network: 0.272 (k = 2), 0.216 (k = 3), 0.121 (k = 10).

## Sec. 5: the compounding law, game by game

On the windows of Fig. 3b. *Law* is one-step accuracy to the power k, computed for each game and seed and then averaged. Each cell: measured / law.

| Predictor | 1 step | 2 steps | 3 steps | 5 steps | 10 steps |
|---|---|---|---|---|---|
| Table, widest seen | 0.564 | 0.450 / 0.410 | 0.387 / 0.332 | 0.322 / 0.253 | 0.232 / 0.175 |
| Widest seen, network where unseen | 0.541 | 0.437 / 0.394 | 0.384 / 0.322 | 0.321 / 0.248 | 0.240 / 0.175 |
| Table, 13×13 | 0.539 | 0.397 / 0.382 | 0.315 / 0.301 | 0.245 / 0.220 | 0.165 / 0.142 |
| Table 13×13, network where unseen | 0.520 | 0.390 / 0.362 | 0.308 / 0.282 | 0.241 / 0.202 | 0.175 / 0.134 |
| Table, 25×25 | 0.489 | 0.352 / 0.328 | 0.277 / 0.248 | 0.205 / 0.168 | 0.128 / 0.096 |
| Table, 41×41 | 0.427 | 0.299 / 0.275 | 0.240 / 0.208 | 0.168 / 0.145 | 0.100 / 0.090 |
| Table, 7×7 | 0.414 | 0.294 / 0.276 | 0.234 / 0.219 | 0.198 / 0.169 | 0.150 / 0.125 |
| Network, pretrained (9×9), trained in `exp_hybrid.py` | 0.401 | 0.269 / 0.270 | 0.216 / 0.215 | 0.177 / 0.166 | 0.148 / 0.125 |

## Sec. 5: scheduled sampling

`code/exp_sched.py`: 8 games (ar25, cd82, m0r0, re86, sc25, sp80, tr87, tu93) and 3 seeds, view 25×25, 4000 moves per game, 25 epochs, rate 0.15, 60 windows per game. Plain training and scheduled sampling start from the same weights and see the same batches. Exact frame after k imagined steps, mean of 24 paired runs; better / worse / equal counts runs. The comparison was run twice.

| Steps | Plain | Scheduled | Difference | Better / worse / equal | Plain (second run) | Scheduled | Difference | Better / worse / equal |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.444 | 0.422 | −0.022 | 8 / 11 / 5 | 0.416 | 0.405 | −0.011 | 10 / 10 / 4 |
| 2 | 0.229 | 0.212 | −0.017 | 8 / 8 / 8 | 0.205 | 0.206 | +0.001 | 11 / 10 / 3 |
| 3 | 0.108 | 0.110 | +0.002 | 11 / 9 / 4 | 0.094 | 0.092 | −0.002 | 9 / 8 / 7 |
| 5 | 0.058 | 0.052 | −0.006 | 9 / 6 / 9 | 0.044 | 0.040 | −0.003 | 5 / 8 / 11 |
| 10 | 0.029 | 0.036 | +0.007 | 2 / 3 / 19 | 0.028 | 0.024 | −0.003 | 4 / 4 / 16 |
| 20 | 0.027 | 0.030 | +0.003 | 3 / 3 / 18 | 0.026 | 0.028 | +0.002 | 2 / 0 / 22 |

One-step accuracy on the held-out fifth of the moves: 0.561 plain against 0.525 scheduled in the first run, 0.529 against 0.508 in the second.

## Networks without the second click plane

`RO_CLICK_T=0` leaves the transposed click plane empty. Three seeds, 25 games; run twice.

| | Both planes | First plane only | First plane only (second run) |
|---|---|---|---|
| Pretrained network, exact frame | 0.454 | 0.450 | 0.447 |
| Pretrained network, exact on real events | 0.119 | 0.111 | 0.105 |
| Pretrained network, exact after 1, 2, 3, 5, 10 imagined steps | 0.405, 0.273, 0.217, 0.174, 0.144 | 0.399, 0.271, 0.216, 0.177, 0.146 | 0.401, 0.266, 0.210, 0.174, 0.146 |
| Network from scratch, exact frame | 0.362 | 0.359 | 0.361 |
| Network from scratch, exact on real events | 0.0068 | 0.0051 | 0.0065 |
| Network from scratch, exact after 1, 2, 3, 5, 10 imagined steps | 0.315, 0.226, 0.187, 0.157, 0.137 | 0.314, 0.225, 0.187, 0.158, 0.139 | 0.316, 0.226, 0.187, 0.158, 0.139 |

## The 25 games one by one

Real events: number of the 600 test steps that change more than three cells. Networks: mean of ten seeds. "Seen at 7×7": share of changing cells whose 7×7 neighborhood was in the table.

| Game | Real events | Nothing changes | Network, scratch | Network, pretrained | Network, Reptile | Table 7×7 | Table 13×13 | Table 41×41 | Seen at 7×7 |
|---|---|---|---|---|---|---|---|---|---|
| ar25 | 375 | 0.204 | 0.386 | 0.506 | 0.536 | 0.716 | 0.706 | 0.275 | 0.968 |
| bp35 | 356 | 0.000 | 0.277 | 0.292 | 0.300 | 0.337 | 0.509 | 0.385 | 0.982 |
| cd82 | 216 | 0.226 | 0.165 | 0.197 | 0.203 | 0.186 | 0.228 | 0.274 | 0.971 |
| cn04 | 414 | 0.170 | 0.053 | 0.132 | 0.119 | 0.200 | 0.287 | 0.166 | 0.941 |
| dc22 | 348 | 0.216 | 0.028 | 0.455 | 0.476 | 0.444 | 0.423 | 0.383 | 0.906 |
| ft09 | 47 | 0.921 | 0.893 | 0.851 | 0.899 | 0.268 | 0.870 | 0.921 | 0.314 |
| g50t | 343 | 0.210 | 0.076 | 0.090 | 0.084 | 0.135 | 0.337 | 0.359 | 0.959 |
| ka59 | 404 | 0.113 | 0.116 | 0.403 | 0.428 | 0.410 | 0.417 | 0.329 | 0.692 |
| lf52 | 2 | 0.000 | 0.937 | 0.980 | 0.986 | 0.954 | 0.920 | 0.635 | 0.932 |
| lp85 | 19 | 0.968 | 0.968 | 0.968 | 0.968 | 0.968 | 0.968 | 0.968 | 0.998 |
| ls20 | 450 | 0.007 | 0.000 | 0.006 | 0.003 | 0.008 | 0.840 | 0.278 | 0.982 |
| m0r0 | 320 | 0.276 | 0.171 | 0.172 | 0.173 | 0.175 | 0.383 | 0.266 | 0.987 |
| r11l | 400 | 0.000 | 0.000 | 0.000 | 0.000 | 0.007 | 0.031 | 0.102 | 0.743 |
| re86 | 464 | 0.000 | 0.011 | 0.119 | 0.113 | 0.125 | 0.078 | 0.014 | 0.900 |
| s5i5 | 21 | 0.000 | 0.690 | 0.717 | 0.667 | 0.727 | 0.715 | 0.767 | 0.988 |
| sb26 | 20 | 0.481 | 0.943 | 0.968 | 0.977 | 0.970 | 0.970 | 0.966 | 0.736 |
| sc25 | 451 | 0.232 | 0.208 | 0.565 | 0.526 | 0.479 | 0.656 | 0.468 | 0.987 |
| sk48 | 397 | 0.290 | 0.238 | 0.237 | 0.233 | 0.240 | 0.519 | 0.449 | 0.943 |
| sp80 | 378 | 0.000 | 0.163 | 0.256 | 0.226 | 0.177 | 0.414 | 0.153 | 0.975 |
| su15 | 21 | 0.163 | 0.750 | 0.731 | 0.751 | 0.732 | 0.709 | 0.441 | 0.801 |
| tn36 | 22 | 0.000 | 0.948 | 0.961 | 0.962 | 0.956 | 0.935 | 0.783 | 0.902 |
| tr87 | 594 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.051 | 0.008 | 0.935 |
| tu93 | 261 | 0.000 | 0.250 | 0.346 | 0.359 | 0.413 | 0.657 | 0.743 | 0.944 |
| vc33 | 5 | 0.000 | 0.699 | 0.724 | 0.719 | 0.787 | 0.782 | 0.808 | 0.999 |
| wa30 | 459 | 0.136 | 0.142 | 0.528 | 0.539 | 0.131 | 0.362 | 0.136 | 0.891 |

The eight quiet games (fewer than 100 real events): ft09, lf52, lp85, s5i5, sb26, su15, tn36, vc33. On these the 13×13 table and the pretrained network both average 0.86. On the other 17 the table averages 0.41 and the network 0.25.

## Games named in the paper

3-seed means from the comparison runs.

| Game | Network 9×9 | Network 13×13 | Network 25×25 | Table 13×13 | Table, widest seen | Table 13×13 + network | Widest seen + network |
|---|---|---|---|---|---|---|---|
| ls20 | 0.000 | 0.266 | 0.000 | 0.840 | 0.791 | 0.801 | 0.795 |
| wa30 | 0.563 | 0.558 | 0.550 | 0.362 | 0.411 | 0.557 | 0.424 |

Networks are pretrained on the other 24 games and finetuned on the 300 moves. In ls20 the 13×13 network scores 0.266, 0.197 and 0.335 in its three seeds.
