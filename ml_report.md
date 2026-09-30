# Machine-learning layer: out-of-sample evaluation

Gradient-boosted models trained on every pre-match feature (form, xG, shots, corners, venue form, rest, table, Elo, bookmaker probabilities, Asian-handicap line) plus the Dixon-Coles model's output. Each test season is never seen in training.

## Test season 2526 (trained on 2223, 2324, 2425; 23310 training matches)

### Result (1X2)

| model | games | winner correct | log loss | Brier |
|---|---|---|---|---|
| Dixon-Coles (previous) | 7634 | 49.2% | 1.0169 | 0.6094 |
| Bookmakers (no-vig) | 7634 | 50.6% | 1.003 | 0.6001 |
| **ML layer** | 7634 | 50.5% | 1.0027 | 0.6001 |

### Over/under 2.5 and BTTS

| market | model | games | call correct | log loss |
|---|---|---|---|---|
| Over/under 2.5 | dixon coles | 7646 | 54.3% | 0.6884 |
| Over/under 2.5 | bookmakers | 7634 | 55.8% | 0.6805 |
| Over/under 2.5 | ml | 7646 | 55.6% | 0.6808 |
| BTTS | dixon coles | 7646 | 54.3% | 0.6886 |
| BTTS | ml | 7646 | 55.2% | 0.6857 |

### Winner hit rate when only confident games are called

| ML favourite at least | games called | share of games | winner correct | double chance correct |
|---|---|---|---|---|
| 0% | 7646 | 100.0% | 50.5% | 78.1% |
| 45% | 4479 | 58.6% | 56.9% | 82.0% |
| 50% | 3069 | 40.1% | 62.2% | 85.0% |
| 55% | 2195 | 28.7% | 66.4% | 87.7% |
| 60% | 1376 | 18.0% | 71.4% | 89.7% |
| 65% | 832 | 10.9% | 76.3% | 92.8% |
| 70% | 555 | 7.3% | 78.6% | 94.4% |
| 75% | 308 | 4.0% | 81.8% | 95.8% |

## Test season 2627 (trained on 2223, 2324, 2425, 2526; 30956 training matches)

### Result (1X2)

| model | games | winner correct | log loss | Brier |
|---|---|---|---|---|
| Dixon-Coles (previous) | 1339 | 47.9% | 1.0276 | 0.6169 |
| Bookmakers (no-vig) | 1339 | 48.4% | 1.0115 | 0.6058 |
| **ML layer** | 1339 | 48.7% | 1.0112 | 0.6055 |

### Over/under 2.5 and BTTS

| market | model | games | call correct | log loss |
|---|---|---|---|---|
| Over/under 2.5 | dixon coles | 1339 | 59.1% | 0.6703 |
| Over/under 2.5 | bookmakers | 1338 | 59.1% | 0.667 |
| Over/under 2.5 | ml | 1339 | 59.2% | 0.6665 |
| BTTS | dixon coles | 1339 | 55.9% | 0.6821 |
| BTTS | ml | 1339 | 55.8% | 0.6805 |

### Winner hit rate when only confident games are called

| ML favourite at least | games called | share of games | winner correct | double chance correct |
|---|---|---|---|---|
| 0% | 1339 | 100.0% | 48.7% | 75.3% |
| 45% | 754 | 56.3% | 55.6% | 78.6% |
| 50% | 512 | 38.2% | 62.7% | 83.4% |
| 55% | 366 | 27.3% | 67.8% | 86.3% |
| 60% | 235 | 17.5% | 72.8% | 88.9% |
| 65% | 146 | 10.9% | 78.1% | 93.2% |
| 70% | 99 | 7.4% | 81.8% | 97.0% |
| 75% | 68 | 5.1% | 80.9% | 98.5% |

