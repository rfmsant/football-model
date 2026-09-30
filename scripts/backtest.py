"""Walk-forward backtest.

* Tuning season (two seasons ago): grid-search model weights by 1X2 log loss.
* Test season (last season): out-of-sample report - calibration, Brier, log loss, ROI and CLV.

Ratings are refitted every week using only matches played before that week, so there is no
look-ahead. Writes data/params.json, data/backtest.json and backtest.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import itertools
import math

import numpy as np
import pandas as pd

from common import DATA, DEFAULT_PARAMS, LEAGUES, ROOT, get_logger, read_json, season_code, write_json
from fetch import fetch_history, merge_understat
from model import Model, best_bet, compute_elo, dixon_coles_matrix, market_odds_from_row, no_vig

log = get_logger("backtest")


def settle(market: str, sel: str, hg: int, ag: int, odds: float) -> float:
    """Profit for a 1-unit stake."""
    if market == "1X2":
        res = "H" if hg > ag else "D" if hg == ag else "A"
        return odds - 1 if sel == res else -1.0
    if market.startswith("O/U"):
        k = float(market.split()[1])
        over = hg + ag > k
        return odds - 1 if (over == (sel == "Over")) else -1.0
    if market == "BTTS":
        both = hg > 0 and ag > 0
        return odds - 1 if (both == (sel == "Yes")) else -1.0
    if market == "Double chance":
        res = "H" if hg > ag else "D" if hg == ag else "A"
        covered = {"1X": "HD", "X2": "DA", "12": "HA"}[sel]
        return odds - 1 if res in covered else -1.0
    if market == "Draw no bet":
        if hg == ag:
            return 0.0
        return odds - 1 if (hg > ag) == (sel == "H") else -1.0
    if market.startswith("AH"):
        line = float(market.split()[1])
        if sel == "Away":
            hg, ag, line = ag, hg, -line
        halves = [line - 0.25, line + 0.25] if abs(line * 4) % 2 == 1 else [line]
        pl = 0.0
        for hl in halves:
            d = hg - ag + hl
            pl += ((odds - 1) if d > 0 else 0.0 if d == 0 else -1.0) / len(halves)
        return pl
    return 0.0


def closing_odds(row, market: str, sel: str) -> float | None:
    def g(c):
        v = row.get(c)
        return float(v) if v is not None and not pd.isna(v) and float(v) > 1 else None
    if market == "1X2":
        cols = {"H": "AvgCH", "D": "AvgCD", "A": "AvgCA"}
        odds = [g(cols[s]) for s in "HDA"]
        if not all(odds):
            return None
        fair = no_vig(odds)
        return 1 / fair["HDA".index(sel)]
    if market == "O/U 2.5":
        o, u = g("AvgC>2.5"), g("AvgC<2.5")
        if not (o and u):
            return None
        fair = no_vig([o, u])
        return 1 / fair[0 if sel == "Over" else 1]
    if market.startswith("AH"):
        line = float(market.split()[1])
        ch = row.get("AHCh")
        if ch is None or pd.isna(ch) or abs(float(ch) - line) > 1e-9:
            return None
        h, a = g("AvgCAHH"), g("AvgCAHA")
        if not (h and a):
            return None
        fair = no_vig([h, a])
        return 1 / fair[0 if sel == "Home" else 1]
    return None


def simulate(model: Model, season: str, params: dict, collect_bets: bool = True) -> pd.DataFrame:
    h = model.hist
    test = h[h["Season"] == season]
    recs = []
    for (lg, week), grp in test.groupby(["Div", test["Date"].dt.to_period("W-MON")]):
        ref = pd.Timestamp(week.start_time)
        for _, row in grp.iterrows():
            try:
                if collect_bets:
                    p = model.predict(lg, row.HomeTeam, row.AwayTeam, ref, season, odds=market_odds_from_row(row),
                                      ah_line=row.get("AHh"), elo_h=row.EloH, elo_a=row.EloA)
                else:  # fast path for tuning: scoreline matrix only
                    lh, la, info = model.lambdas(lg, row.HomeTeam, row.AwayTeam, ref, season, row.EloH, row.EloA)
                    mat = dixon_coles_matrix(lh, la, params["rho"])
                    p = {"markets": {"1X2": {"H": np.tril(mat, -1).sum(), "D": np.trace(mat), "A": np.triu(mat, 1).sum()},
                                     "O/U 2.5": {"Over": 1 - mat[np.add.outer(range(7), range(7)) <= 2].sum()},
                                     "BTTS": {"Yes": 1 - mat[0, :].sum() - mat[:, 0].sum() + mat[0, 0]}},
                         "info": info, "lambda_home": lh, "lambda_away": la}
            except Exception as e:  # noqa: BLE001
                log.debug("skip %s: %s", row.HomeTeam, e)
                continue
            x = p["markets"]["1X2"]
            res = "H" if row.FTHG > row.FTAG else "D" if row.FTHG == row.FTAG else "A"
            rec = {"Div": lg, "Date": row.Date, "home": row.HomeTeam, "away": row.AwayTeam,
                   "FTHG": row.FTHG, "FTAG": row.FTAG, "res": res,
                   "pH": x["H"], "pD": x["D"], "pA": x["A"],
                   "pO25": p["markets"]["O/U 2.5"]["Over"], "xg_share": p["info"].get("xg_share", 0),
                   "pBTTS": (p["markets"].get("BTTS") or {}).get("Yes"),
                   "lh": p.get("lambda_home"), "la": p.get("lambda_away")}
            fair = no_vig([row.get("AvgH"), row.get("AvgD"), row.get("AvgA")])
            if fair:
                rec.update(mH=fair[0], mD=fair[1], mA=fair[2])
            if collect_bets:
                b = best_bet(p["evals"], params)
                if b:
                    rec.update(bet_market=b["market"], bet_sel=b["selection"], bet_odds=b["odds"],
                               bet_avg_odds=b["avg_odds"], bet_ev=b["ev"], bet_edge=b["edge"],
                               bet_p=b["p_bet"],
                               bet_pl=settle(b["market"], b["selection"], row.FTHG, row.FTAG, b["odds"]),
                               bet_pl_avg=settle(b["market"], b["selection"], row.FTHG, row.FTAG, b["avg_odds"]))
                    co = closing_odds(row, b["market"], b["selection"])
                    if co:
                        rec["bet_clv"] = b["odds"] / co - 1
            recs.append(rec)
    return pd.DataFrame(recs)


def metrics(df: pd.DataFrame, prefix: str = "p") -> dict:
    P = df[[f"{prefix}H", f"{prefix}D", f"{prefix}A"]].to_numpy(float)
    Y = np.stack([(df["res"] == s).to_numpy(float) for s in "HDA"], axis=1)
    ok = ~np.isnan(P).any(axis=1)
    P, Y = P[ok], Y[ok]
    brier = float(np.mean(np.sum((P - Y) ** 2, axis=1)))
    ll = float(-np.mean(np.log(np.clip(np.sum(P * Y, axis=1), 1e-12, 1))))
    return {"n": int(ok.sum()), "brier": round(brier, 4), "log_loss": round(ll, 4)}


def calibration(df: pd.DataFrame) -> list[dict]:
    p = np.concatenate([df["pH"], df["pD"], df["pA"]])
    y = np.concatenate([(df["res"] == s).to_numpy(float) for s in "HDA"])
    bins = np.linspace(0, 1, 11)
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() >= 20:
            out.append({"bin": f"{lo:.0%}-{hi:.0%}", "n": int(m.sum()),
                        "predicted": round(float(p[m].mean()), 4), "actual": round(float(y[m].mean()), 4)})
    return out


def bet_stats(df: pd.DataFrame) -> dict:
    if "bet_pl" not in df:
        return {"bets": 0}
    b = df.dropna(subset=["bet_pl"])
    if b.empty:
        return {"bets": 0}
    out = {"bets": int(len(b)), "roi_best_price": round(float(b["bet_pl"].mean()), 4),
           "roi_avg_price": round(float(b["bet_pl_avg"].mean()), 4),
           "profit_units": round(float(b["bet_pl"].sum()), 1),
           "hit_rate": round(float((b["bet_pl"] > 0).mean()), 4),
           "avg_odds": round(float(b["bet_odds"].mean()), 2)}
    if "bet_clv" in b and b["bet_clv"].notna().any():
        c = b["bet_clv"].dropna()
        out.update(clv_mean=round(float(c.mean()), 4), clv_beat_close=round(float((c > 0).mean()), 4), clv_n=int(len(c)))
    by_mkt = b.assign(mk=b["bet_market"].str.replace(r"AH .*", "AH", regex=True)).groupby("mk")
    out["by_market"] = {k: {"bets": int(len(g)), "roi": round(float(g["bet_pl"].mean()), 4)} for k, g in by_mkt}
    return out


def load_backtest_history(today: dt.date) -> pd.DataFrame:
    seasons = [season_code(today, k) for k in (-3, -2, -1)]
    h = fetch_history(seasons)
    h = merge_understat(h)
    h, _ = compute_elo(h)
    return h


def run(today: dt.date | None = None, quick: bool = False, skip_grid: bool = False) -> dict:
    today = today or dt.date.today()
    tune_season, test_season = season_code(today, -2), season_code(today, -1)
    hist = load_backtest_history(today)
    log.info("backtest history: %d matches", len(hist))
    if hist.empty:
        return {}

    # ---- 1. tune model weights on the tuning season (1X2 log loss), coordinate descent
    grid = {"xg_weight": [0.3, 0.5, 0.7, 0.9], "elo_weight": [0.15, 0.3, 0.45, 0.6],
            "half_life_days": [120, 180, 270], "prior_matches": [2.0, 4.0, 6.0],
            "sharpness": [1.0, 1.1, 1.2, 1.3], "rho": [-0.04, -0.08, -0.12]}
    params = dict(DEFAULT_PARAMS)
    results, seen = [], {}

    def score(prm):
        key = tuple(prm[k] for k in grid)
        if key not in seen:
            m = metrics(simulate(Model(hist, prm), tune_season, prm, collect_bets=False))
            seen[key] = m["log_loss"]
            results.append({**{k: prm[k] for k in grid}, **m})
            log.info("tune %s -> logloss %.4f", {k: prm[k] for k in grid}, m["log_loss"])
        return seen[key]

    if skip_grid:  # reuse the model weights from the last tuning, re-tune only the betting thresholds
        prev = read_json(DATA / "backtest.json", {}) or {}
        params.update({k: v for k, v in (prev.get("params") or {}).items() if k in grid})
        results = prev.get("grid", [])
    for _ in range(0 if skip_grid else 1 if quick else 2):
        for name, values in grid.items():
            if quick:
                break
            params[name] = min(values, key=lambda v: score(dict(params, **{name: v})))
    if not skip_grid:
        score(params)

    # ---- 2. market weight: blend that minimises log loss on the tuning season
    tune_sim = simulate(Model(hist, params), tune_season, params, collect_bets=False)
    tune_sim = tune_sim.dropna(subset=["mH"])
    mw_scores = {}
    for mw in [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8]:  # capped so the model always contributes
        b = tune_sim.copy()
        for s in "HDA":
            b[f"b{s}"] = (1 - mw) * b[f"p{s}"] + mw * b[f"m{s}"]
        mw_scores[mw] = metrics(b, "b")["log_loss"]
    params["market_weight"] = min(mw_scores, key=mw_scores.get)
    # ---- 3. betting thresholds by ROI at average prices on the tuning season (min 150 bets)
    ev_scores = {}
    for mev, mx in itertools.product([0.02, 0.04, 0.06], [2.5, 3.5, 6.0]):
        prm = dict(params, min_ev=mev, max_odds=mx)
        ev_scores[(mev, mx)] = bet_stats(simulate(Model(hist, prm), tune_season, prm))
        log.info("thresholds min_ev=%.2f max_odds=%.1f -> %s", mev, mx,
                 {k: ev_scores[(mev, mx)].get(k) for k in ("bets", "roi_avg_price", "clv_mean")})
        if quick:
            break
    params["min_ev"], params["max_odds"] = max(
        ev_scores, key=lambda k: ev_scores[k].get("roi_avg_price", -1) if ev_scores[k].get("bets", 0) >= 150 else -1)

    # ---- 4. out-of-sample test season
    test = simulate(Model(hist, params), test_season, params)
    model_m = metrics(test)
    market_m = metrics(test.dropna(subset=["mH"]), "m")
    blend = test.dropna(subset=["mH"]).copy()
    for s in "HDA":
        blend[f"b{s}"] = (1 - params["market_weight"]) * blend[f"p{s}"] + params["market_weight"] * blend[f"m{s}"]
    blend_m = metrics(blend, "b")
    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tune_season": tune_season, "test_season": test_season,
        "params": params, "grid": results, "market_weight_scores": mw_scores,
        "threshold_scores": {f"min_ev={k[0]} max_odds={k[1]}": v for k, v in ev_scores.items()},
        "test": {"model": model_m, "market": market_m, "blend": blend_m,
                 "calibration": calibration(test), "bets": bet_stats(test),
                 "by_league": {lg: {**metrics(g), **{k: v for k, v in bet_stats(g).items() if k != "by_market"}}
                               for lg, g in test.groupby("Div")}},
    }
    write_json(DATA / "params.json", {"params": params, "generated_at": report["generated_at"],
                                      "source": f"tuned on {tune_season}, tested on {test_season}"})
    write_json(DATA / "backtest.json", report)
    write_markdown(report)
    return report


def pct(x):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.1f}%"


def write_markdown(r: dict) -> None:
    t = r["test"]
    p = r["params"]
    b = t["bets"]
    L = [f"# Backtest results\n",
         f"_Generated {r['generated_at']}. Parameters tuned on season {r['tune_season']}, "
         f"evaluated out-of-sample on season {r['test_season']} with weekly walk-forward refits "
         f"(no look-ahead)._\n",
         "## Tuned parameters\n", "| parameter | value |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in p.items()]
    L += ["\n## 1X2 accuracy (lower is better)\n", "| probabilities | matches | Brier | log loss |", "|---|---|---|---|",
          f"| Model | {t['model']['n']} | {t['model']['brier']} | {t['model']['log_loss']} |",
          f"| Market (no-vig average odds) | {t['market']['n']} | {t['market']['brier']} | {t['market']['log_loss']} |",
          f"| Model/market blend (w={p['market_weight']}) | {t['blend']['n']} | {t['blend']['brier']} | {t['blend']['log_loss']} |",
          "\n## Calibration (all 1X2 outcomes)\n", "| predicted bucket | n | mean predicted | actual |", "|---|---|---|---|"]
    L += [f"| {c['bin']} | {c['n']} | {pct(c['predicted'])} | {pct(c['actual'])} |" for c in t["calibration"]]
    L += ["\n## Betting simulation (1 unit flat stake on the best-EV bet per match)\n"]
    if b.get("bets"):
        L += ["| metric | value |", "|---|---|",
              f"| Bets | {b['bets']} |", f"| Hit rate | {pct(b['hit_rate'])} |", f"| Average odds | {b['avg_odds']} |",
              f"| ROI at best available price | {pct(b['roi_best_price'])} |",
              f"| ROI at average price | {pct(b['roi_avg_price'])} |",
              f"| Profit (units, best price) | {b['profit_units']} |",
              f"| Mean closing line value | {pct(b.get('clv_mean'))} |",
              f"| Bets beating the closing line | {pct(b.get('clv_beat_close'))} |",
              "\n| market | bets | ROI |", "|---|---|---|"]
        L += [f"| {k} | {v['bets']} | {pct(v['roi'])} |" for k, v in b["by_market"].items()]
    else:
        L += ["No bets met the thresholds."]
    L += ["\n## By league\n", "| league | matches | log loss | bets | ROI (best price) | CLV |", "|---|---|---|---|---|---|"]
    for lg, s in sorted(t["by_league"].items()):
        L.append(f"| {lg} {LEAGUES[lg][0]} | {s['n']} | {s['log_loss']} | {s.get('bets', 0)} | "
                 f"{pct(s.get('roi_best_price'))} | {pct(s.get('clv_mean'))} |")
    L += ["\n## Weight tuning grid (tuning season, 1X2 log loss)\n", "| xG weight | Elo weight | half-life (days) | log loss |",
          "|---|---|---|---|"]
    L += [f"| {g['xg_weight']} | {g['elo_weight']} | {g['half_life_days']} | {g['log_loss']} |"
          for g in sorted(r["grid"], key=lambda g: g["log_loss"])]
    L += ["\n## How to read this\n",
          "- **Brier / log loss**: probability accuracy. Beating the bookmaker's no-vig probabilities is very hard; "
          "getting close means the model is well calibrated.",
          "- **ROI at best price** assumes you always get the highest odds among the bookmakers football-data.co.uk "
          "tracks; **ROI at average price** is the more realistic figure.",
          "- **Closing line value (CLV)**: how the odds taken compare with the no-vig closing odds. Consistently "
          "positive CLV is the best evidence of a real edge; ROI over one season is noisy.",
          "- Injury, rest and motivation adjustments (extras.py) are not part of the backtest because historical "
          "injury data isn't available on free tiers.",
          "\nThis is a statistical model, not financial advice. Bet responsibly."]
    (ROOT / "backtest.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the grid search")
    ap.add_argument("--skip-grid", action="store_true", help="reuse model weights from data/backtest.json")
    a = ap.parse_args()
    rep = run(quick=a.quick, skip_grid=a.skip_grid)
    print({k: rep["test"][k] for k in ("model", "market", "blend", "bets")} if rep else "no data")
