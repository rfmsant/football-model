# Backtest results

_Generated 2026-11-01T13:38:16Z. Parameters tuned on season 2425, evaluated out-of-sample on season 2526 with weekly walk-forward refits (no look-ahead)._

## Tuned parameters

| parameter | value |
|---|---|
| half_life_days | 120 |
| xg_weight | 0.9 |
| elo_weight | 0.3 |
| prior_matches | 4.0 |
| sharpness | 1.1 |
| rho | -0.08 |
| min_ev | 0.04 |
| market_weight | 0.8 |
| min_odds | 1.3 |
| max_odds | 3.5 |
| bet_markets | ['1X2', 'O/U', 'BTTS'] |

## 1X2 accuracy (lower is better)

| probabilities | matches | Brier | log loss |
|---|---|---|---|
| Model | 7646 | 0.6093 | 1.0167 |
| Market (no-vig average odds) | 7634 | 0.6001 | 1.003 |
| Model/market blend (w=0.8) | 7634 | 0.6009 | 1.0043 |

## Calibration (all 1X2 outcomes)

| predicted bucket | n | mean predicted | actual |
|---|---|---|---|
| 0%-10% | 317 | 7.5% | 5.4% |
| 10%-20% | 2110 | 16.2% | 15.0% |
| 20%-30% | 9677 | 25.8% | 26.2% |
| 30%-40% | 4796 | 34.3% | 34.5% |
| 40%-50% | 3121 | 44.7% | 43.2% |
| 50%-60% | 1782 | 54.3% | 53.9% |
| 60%-70% | 776 | 64.5% | 67.5% |
| 70%-80% | 297 | 74.1% | 79.1% |
| 80%-90% | 62 | 83.4% | 85.5% |

## Betting simulation (1 unit flat stake on the best-EV bet per match)

| metric | value |
|---|---|
| Bets | 279 |
| Hit rate | 30.8% |
| Average odds | 2.83 |
| ROI at best available price | -17.9% |
| ROI at average price | -22.1% |
| Profit (units, best price) | -49.9 |
| Mean closing line value | -0.6% |
| Bets beating the closing line | 45.2% |

| market | bets | ROI |
|---|---|---|
| 1X2 | 137 | -34.8% |
| O/U 2.5 | 142 | -1.5% |

## By league

| league | matches | log loss | bets | ROI (best price) | CLV |
|---|---|---|---|---|---|
| B1 Belgian Pro League | 311 | 1.0283 | 11 | -9.0% | -1.3% |
| D1 Bundesliga | 306 | 0.9678 | 10 | -35.0% | 3.0% |
| D2 2. Bundesliga | 306 | 1.0408 | 14 | -77.1% | -0.9% |
| E0 Premier League | 380 | 1.0273 | 18 | 15.7% | -1.6% |
| E1 Championship | 552 | 1.0565 | 16 | -26.6% | 2.5% |
| E2 League One | 552 | 1.035 | 15 | -2.9% | -2.9% |
| E3 League Two | 552 | 1.0377 | 18 | 11.7% | -1.7% |
| EC National League | 552 | 0.9921 | 23 | -14.3% | -3.0% |
| F1 Ligue 1 | 306 | 0.9866 | 12 | -42.4% | -6.6% |
| F2 Ligue 2 | 305 | 1.0791 | 13 | -38.5% | -1.8% |
| G1 Super League Greece | 236 | 0.9617 | 16 | -9.4% | 2.1% |
| I1 Serie A | 380 | 0.999 | 17 | 37.5% | -2.2% |
| I2 Serie B | 380 | 1.0323 | 12 | -54.8% | 3.1% |
| N1 Eredivisie | 306 | 0.9831 | 9 | 10.1% | 4.5% |
| P1 Primeira Liga | 306 | 0.9231 | 9 | 32.2% | -0.9% |
| SC0 Scottish Premiership | 228 | 0.989 | 8 | 7.9% | 5.2% |
| SC1 Scottish Championship | 180 | 1.0732 | 4 | -100.0% | -1.7% |
| SC2 Scottish League One | 180 | 1.0399 | 3 | -100.0% | -0.2% |
| SC3 Scottish League Two | 180 | 1.0854 | 7 | 5.7% | -5.6% |
| SP1 La Liga | 380 | 0.9802 | 18 | -38.0% | -0.3% |
| SP2 Segunda Division | 462 | 1.0488 | 14 | -39.6% | -1.9% |
| T1 Super Lig | 306 | 0.9887 | 12 | -43.2% | 1.8% |

## Weight tuning grid (tuning season, 1X2 log loss)

| xG weight | Elo weight | half-life (days) | log loss |
|---|---|---|---|
| 0.9 | 0.3 | 120 | 1.0126 |
| 0.9 | 0.3 | 120 | 1.0127 |
| 0.9 | 0.3 | 180 | 1.0128 |
| 0.9 | 0.3 | 120 | 1.0128 |
| 0.9 | 0.3 | 180 | 1.0129 |
| 0.9 | 0.3 | 180 | 1.0129 |
| 0.9 | 0.3 | 180 | 1.013 |
| 0.9 | 0.3 | 120 | 1.013 |
| 0.9 | 0.3 | 270 | 1.0131 |
| 0.9 | 0.3 | 120 | 1.0131 |
| 0.9 | 0.3 | 120 | 1.0131 |
| 0.9 | 0.3 | 180 | 1.0132 |
| 0.9 | 0.15 | 180 | 1.0132 |
| 0.9 | 0.3 | 120 | 1.0132 |
| 0.9 | 0.3 | 180.0 | 1.0134 |
| 0.9 | 0.45 | 180.0 | 1.0135 |
| 0.9 | 0.3 | 270 | 1.0135 |
| 0.7 | 0.3 | 180 | 1.0136 |
| 0.7 | 0.3 | 180.0 | 1.0137 |
| 0.9 | 0.3 | 120 | 1.0137 |
| 0.9 | 0.3 | 180 | 1.0138 |
| 0.9 | 0.45 | 180 | 1.0138 |
| 0.5 | 0.3 | 180.0 | 1.0145 |
| 0.9 | 0.15 | 180.0 | 1.0146 |
| 0.9 | 0.3 | 120 | 1.0146 |
| 0.9 | 0.6 | 180.0 | 1.0149 |
| 0.5 | 0.3 | 180 | 1.0152 |
| 0.9 | 0.3 | 180 | 1.0156 |
| 0.3 | 0.3 | 180.0 | 1.0159 |
| 0.9 | 0.6 | 180 | 1.0161 |
| 0.3 | 0.3 | 180 | 1.0178 |

## How to read this

- **Brier / log loss**: probability accuracy. Beating the bookmaker's no-vig probabilities is very hard; getting close means the model is well calibrated.
- **ROI at best price** assumes you always get the highest odds among the bookmakers football-data.co.uk tracks; **ROI at average price** is the more realistic figure.
- **Closing line value (CLV)**: how the odds taken compare with the no-vig closing odds. Consistently positive CLV is the best evidence of a real edge; ROI over one season is noisy.
- Injury, rest and motivation adjustments (extras.py) are not part of the backtest because historical injury data isn't available on free tiers.

This is a statistical model, not financial advice. Bet responsibly.
