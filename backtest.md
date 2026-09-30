# Backtest results

_Generated 2026-09-30T15:26:46Z. Parameters tuned on season 2425, evaluated out-of-sample on season 2526 with weekly walk-forward refits (no look-ahead)._

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
| max_odds | 2.5 |

## 1X2 accuracy (lower is better)

| probabilities | matches | Brier | log loss |
|---|---|---|---|
| Model | 7646 | 0.6094 | 1.0168 |
| Market (no-vig average odds) | 7634 | 0.6001 | 1.003 |
| Model/market blend (w=0.8) | 7634 | 0.6009 | 1.0043 |

## Calibration (all 1X2 outcomes)

| predicted bucket | n | mean predicted | actual |
|---|---|---|---|
| 0%-10% | 318 | 7.5% | 5.3% |
| 10%-20% | 2096 | 16.2% | 15.1% |
| 20%-30% | 9763 | 25.9% | 26.2% |
| 30%-40% | 4715 | 34.3% | 34.8% |
| 40%-50% | 3135 | 44.8% | 42.9% |
| 50%-60% | 1779 | 54.4% | 54.2% |
| 60%-70% | 777 | 64.5% | 67.4% |
| 70%-80% | 293 | 74.1% | 79.2% |
| 80%-90% | 62 | 83.4% | 85.5% |

## Betting simulation (1 unit flat stake on the best-EV bet per match)

| metric | value |
|---|---|
| Bets | 143 |
| Hit rate | 41.3% |
| Average odds | 2.1 |
| ROI at best available price | -9.8% |
| ROI at average price | -14.1% |
| Profit (units, best price) | -14.0 |
| Mean closing line value | -1.2% |
| Bets beating the closing line | 35.0% |

| market | bets | ROI |
|---|---|---|
| 1X2 | 15 | -7.0% |
| AH | 75 | -12.9% |
| O/U 2.5 | 53 | -6.2% |

## By league

| league | matches | log loss | bets | ROI (best price) | CLV |
|---|---|---|---|---|---|
| B1 Belgian Pro League | 311 | 1.0283 | 6 | 18.8% | 3.1% |
| D1 Bundesliga | 306 | 0.9678 | 5 | -60.0% | -5.2% |
| D2 2. Bundesliga | 306 | 1.0405 | 2 | 19.0% | -8.9% |
| E0 Premier League | 380 | 1.0273 | 8 | 30.2% | -4.5% |
| E1 Championship | 552 | 1.0565 | 12 | -0.8% | 0.3% |
| E2 League One | 552 | 1.0344 | 8 | -18.8% | -1.1% |
| E3 League Two | 552 | 1.0378 | 9 | 54.4% | -1.4% |
| EC National League | 552 | 0.9921 | 20 | -16.8% | -2.8% |
| F1 Ligue 1 | 306 | 0.9866 | 5 | -23.8% | -4.0% |
| F2 Ligue 2 | 305 | 1.079 | 8 | 65.4% | -3.7% |
| G1 Super League Greece | 236 | 0.9676 | 11 | 7.6% | -0.6% |
| I1 Serie A | 380 | 0.999 | 7 | -31.6% | 1.1% |
| I2 Serie B | 380 | 1.032 | 8 | -41.5% | 3.7% |
| N1 Eredivisie | 306 | 0.9816 | 1 | -100.0% | 3.7% |
| P1 Primeira Liga | 306 | 0.9233 | 4 | -1.0% | -1.0% |
| SC0 Scottish Premiership | 228 | 0.9904 | 1 | -100.0% | -0.3% |
| SC1 Scottish Championship | 180 | 1.0726 | 3 | 18.0% | -0.7% |
| SC2 Scottish League One | 180 | 1.0384 | 1 | -100.0% | -1.8% |
| SC3 Scottish League Two | 180 | 1.0853 | 9 | -2.7% | -2.5% |
| SP1 La Liga | 380 | 0.9803 | 5 | -50.0% | -0.3% |
| SP2 Segunda Division | 462 | 1.0486 | 4 | -100.0% | 0.4% |
| T1 Super Lig | 306 | 0.9886 | 6 | -83.3% | -6.9% |

## Weight tuning grid (tuning season, 1X2 log loss)

| xG weight | Elo weight | half-life (days) | log loss |
|---|---|---|---|
| 0.9 | 0.3 | 120 | 1.0128 |
| 0.9 | 0.3 | 120 | 1.0128 |
| 0.9 | 0.3 | 180 | 1.0129 |
| 0.9 | 0.3 | 120 | 1.0129 |
| 0.9 | 0.3 | 180 | 1.013 |
| 0.9 | 0.3 | 180 | 1.013 |
| 0.9 | 0.3 | 180 | 1.0131 |
| 0.9 | 0.3 | 270 | 1.0131 |
| 0.9 | 0.3 | 120 | 1.0131 |
| 0.9 | 0.3 | 120 | 1.0131 |
| 0.9 | 0.3 | 120 | 1.0132 |
| 0.9 | 0.3 | 180 | 1.0133 |
| 0.9 | 0.15 | 180 | 1.0133 |
| 0.9 | 0.3 | 120 | 1.0133 |
| 0.9 | 0.3 | 180.0 | 1.0135 |
| 0.9 | 0.45 | 180.0 | 1.0136 |
| 0.9 | 0.3 | 270 | 1.0136 |
| 0.7 | 0.3 | 180 | 1.0136 |
| 0.7 | 0.3 | 180.0 | 1.0138 |
| 0.9 | 0.3 | 120 | 1.0138 |
| 0.9 | 0.3 | 180 | 1.0138 |
| 0.9 | 0.45 | 180 | 1.0139 |
| 0.5 | 0.3 | 180.0 | 1.0146 |
| 0.9 | 0.15 | 180.0 | 1.0147 |
| 0.9 | 0.3 | 120 | 1.0147 |
| 0.9 | 0.6 | 180.0 | 1.015 |
| 0.5 | 0.3 | 180 | 1.0153 |
| 0.9 | 0.3 | 180 | 1.0157 |
| 0.3 | 0.3 | 180.0 | 1.016 |
| 0.9 | 0.6 | 180 | 1.0163 |
| 0.3 | 0.3 | 180 | 1.0179 |

## How to read this

- **Brier / log loss**: probability accuracy. Beating the bookmaker's no-vig probabilities is very hard; getting close means the model is well calibrated.
- **ROI at best price** assumes you always get the highest odds among the bookmakers football-data.co.uk tracks; **ROI at average price** is the more realistic figure.
- **Closing line value (CLV)**: how the odds taken compare with the no-vig closing odds. Consistently positive CLV is the best evidence of a real edge; ROI over one season is noisy.
- Injury, rest and motivation adjustments (extras.py) are not part of the backtest because historical injury data isn't available on free tiers.

This is a statistical model, not financial advice. Bet responsibly.
