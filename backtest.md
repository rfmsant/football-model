# Backtest results

_Generated 2026-09-30T16:52:38Z. Parameters tuned on season 2425, evaluated out-of-sample on season 2526 with weekly walk-forward refits (no look-ahead)._

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
| Bets | 271 |
| Hit rate | 31.4% |
| Average odds | 2.86 |
| ROI at best available price | -15.0% |
| ROI at average price | -19.4% |
| Profit (units, best price) | -40.7 |
| Mean closing line value | -0.6% |
| Bets beating the closing line | 44.6% |

| market | bets | ROI |
|---|---|---|
| 1X2 | 138 | -32.9% |
| O/U 2.5 | 133 | 3.5% |

## By league

| league | matches | log loss | bets | ROI (best price) | CLV |
|---|---|---|---|---|---|
| B1 Belgian Pro League | 311 | 1.0283 | 10 | -28.7% | -0.3% |
| D1 Bundesliga | 306 | 0.9678 | 10 | -35.0% | 3.0% |
| D2 2. Bundesliga | 306 | 1.0405 | 14 | -60.1% | -2.1% |
| E0 Premier League | 380 | 1.0273 | 18 | 15.7% | -1.6% |
| E1 Championship | 552 | 1.0565 | 16 | -26.6% | 2.5% |
| E2 League One | 552 | 1.0344 | 15 | -2.9% | -2.9% |
| E3 League Two | 552 | 1.0378 | 18 | 11.7% | -1.9% |
| EC National League | 552 | 0.9921 | 23 | -14.3% | -3.0% |
| F1 Ligue 1 | 306 | 0.9866 | 12 | -42.4% | -6.6% |
| F2 Ligue 2 | 305 | 1.079 | 11 | -27.3% | -1.8% |
| G1 Super League Greece | 236 | 0.9676 | 10 | 0.5% | 3.8% |
| I1 Serie A | 380 | 0.999 | 17 | 37.5% | -2.2% |
| I2 Serie B | 380 | 1.032 | 13 | -33.0% | 1.9% |
| N1 Eredivisie | 306 | 0.9816 | 14 | 41.5% | 4.4% |
| P1 Primeira Liga | 306 | 0.9233 | 7 | 40.7% | 2.7% |
| SC0 Scottish Premiership | 228 | 0.9904 | 8 | 6.6% | 1.8% |
| SC1 Scottish Championship | 180 | 1.0726 | 4 | -100.0% | -1.7% |
| SC2 Scottish League One | 180 | 1.0384 | 2 | -100.0% | -0.9% |
| SC3 Scottish League Two | 180 | 1.0853 | 5 | -3.0% | -4.9% |
| SP1 La Liga | 380 | 0.9803 | 18 | -38.0% | -0.3% |
| SP2 Segunda Division | 462 | 1.0486 | 15 | -43.7% | -0.8% |
| T1 Super Lig | 306 | 0.9886 | 11 | -59.5% | 1.3% |

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
