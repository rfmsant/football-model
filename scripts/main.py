"""Run the whole pipeline and write data/predictions.json.

    python scripts/main.py                      # normal weekly run
    python scripts/main.py --backtest-if-stale  # also re-tune if params.json is > 30 days old
    python scripts/main.py --demo 2026-09-19    # replay a past week from history (local testing)
    python scripts/main.py --no-fetch           # reuse data/ from the last fetch
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import sys
import traceback

import numpy as np
import pandas as pd

import extras
import fetch
import notify
import oddsapi
import summary
import track
from common import DATA, LEAGUES, get_logger, load_params, read_json, season_code, write_json
from model import Model, base_confidence, best_bet, market_odds_from_row

log = get_logger("main")
N_FLAG = 25


def _clean(o):
    """Make numpy/NaN values JSON-safe."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if math.isnan(float(o)) else round(float(o), 4)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def load_inputs(args, today: dt.date):
    if args.demo:
        demo = pd.Timestamp(args.demo)
        hist = pd.read_csv(DATA / "history.csv", parse_dates=["Date"], low_memory=False)
        hist["Season"] = hist["Season"].astype(str).str.zfill(4)
        fx = hist[(hist["Date"] >= demo) & (hist["Date"] < demo + pd.Timedelta(days=7))].copy()
        fx["BookSpread"] = np.nan
        return fx.reset_index(drop=True), hist[hist["Date"] < demo].copy(), pd.DataFrame()
    if args.no_fetch:
        fx = pd.read_csv(DATA / "fixtures.csv", parse_dates=["Date"]) if (DATA / "fixtures.csv").exists() else pd.DataFrame()
        hist = pd.read_csv(DATA / "history.csv", parse_dates=["Date"], low_memory=False)
        hist["Season"] = hist["Season"].astype(str).str.zfill(4)
        ce = pd.read_csv(DATA / "clubelo.csv") if (DATA / "clubelo.csv").exists() else pd.DataFrame()
        return fx, hist, ce
    d = fetch.run(today)
    return d["fixtures"], d["history"], d["clubelo"]


def params_stale(days: int = 30) -> bool:
    meta = read_json(DATA / "params.json")
    if not meta or "generated_at" not in meta:
        return True
    ts = pd.Timestamp(meta["generated_at"]).tz_localize(None)
    return (pd.Timestamp.utcnow().tz_localize(None) - ts).days > days


def predict_game(model: Model, row, params, ref, season, adj=(1.0, 1.0), extra_odds=None) -> dict:
    odds = market_odds_from_row(row)
    if extra_odds:
        odds = extras.merge_odds(odds, extra_odds)
    allowed = tuple(params.get("bet_markets", ["1X2", "O/U", "BTTS"]))
    odds = {m: v for m, v in odds.items() if m.startswith(allowed)}   # only markets the user can bet
    ah_lines = sorted({float(m.split()[1]) for m in odds if m.startswith("AH")})
    return model.predict(row["Div"], row["HomeTeam"], row["AwayTeam"], ref, season, odds=odds,
                         ah_line=ah_lines, adj=adj) | {"odds": odds}


def game_record(gid, row, p, params, flagged=False, ext=None) -> dict:
    mk, info = p["markets"], p["info"]
    evals = p["evals"]
    bb = best_bet(evals, params)
    x = mk["1X2"]
    one = {e["selection"]: e for e in evals if e["market"] == "1X2"}
    edge_1x2 = max((abs(e.get("edge", 0)) for e in one.values()), default=None) if one else None
    has_odds = any("ev" in e for e in evals)
    conf = base_confidence(info, has_odds, bb["edge"] if bb else edge_1x2)
    ext = ext or {}
    injury_data = bool(ext.get("injury_data"))
    conf = conf + 5 if injury_data else conf - 8
    conf = int(max(5, min(95, conf)))
    t = row.get("Time")
    rec = {
        "id": gid, "league": row["Div"], "league_name": LEAGUES[row["Div"]][0], "country": LEAGUES[row["Div"]][1],
        "date": pd.Timestamp(row["Date"]).strftime("%Y-%m-%d"), "time": None if pd.isna(t) else str(t),
        "home": row["HomeTeam"], "away": row["AwayTeam"],
        "probs": x, "xg": {"home": p["lambda_home"], "away": p["lambda_away"]},
        "likely_score": mk["likely_score"], "top_scores": mk["top_scores"],
        "markets": {k: {s: v for s, v in sel.items() if not s.startswith("_")}
                    for k, sel in mk.items() if isinstance(sel, dict)},
        "evals": evals, "best_bet": bb,
        "max_ev": max((e["ev"] for e in evals if "ev" in e), default=None),
        "edge": bb["edge"] if bb else None,
        "confidence": conf, "flagged": flagged, "has_xg": info.get("xg_share", 0) >= 0.5,
        "xg_share": info.get("xg_share", 0), "elo_diff": info.get("elo_diff"), "elo_source": info.get("elo_source"),
        "injury_data": injury_data, "book_spread": row.get("BookSpread"),
        "factors": ext.get("factors", []),
        "context": {"home": ext.get("home_context", {}), "away": ext.get("away_context", {})},
        "table": {"home": ext.get("home_table"), "away": ext.get("away_table")},
        "absences": {"home": ext.get("home_absences", []), "away": ext.get("away_absences", [])},
        "adjustment": ext.get("lambda_mult"),
    }
    if bb:
        rec["best_bet"] = dict(bb, label=summary.bet_label(bb, rec["home"], rec["away"]))
    # the model's lean: best-EV option among priced markets, shown even when it isn't a value bet
    priced = [e for e in evals if "ev" in e]
    lean = max(priced, key=lambda e: e["ev"]) if priced else None
    rec["lean"] = dict(lean, label=summary.bet_label(lean, rec["home"], rec["away"]), value=bool(bb)) if lean else None
    return rec


def select_flagged(records: list[dict], n: int = N_FLAG, today: dt.date | None = None) -> list[str]:
    """Deep-check the next 48 hours first (daily analysis), then the rest by EV and disagreement."""
    soon_cut = ((today or dt.date.today()) + dt.timedelta(days=1)).isoformat()
    soon = sorted((r for r in records if r["date"] <= soon_cut), key=lambda r: -(r["max_ev"] or -1))
    flagged = [r["id"] for r in soon[:n]]
    by_ev = sorted((r for r in records if r["max_ev"] is not None and r["id"] not in flagged), key=lambda r: -r["max_ev"])
    flagged += [r["id"] for r in by_ev[: max(0, int(n * 0.7) - len(flagged))]]

    def disagreement(r):
        e = max((abs(x.get("edge", 0)) for x in r["evals"] if x["market"] == "1X2"), default=0)
        spread = r.get("book_spread")
        return e + (2 * spread if spread is not None and not pd.isna(spread) else 0)
    for r in sorted(records, key=lambda r: -disagreement(r)):
        if len(flagged) >= n:
            break
        if r["id"] not in flagged:
            flagged.append(r["id"])
    return flagged


def run(args) -> dict:
    today = dt.date.fromisoformat(args.demo) if args.demo else dt.date.today()
    season = season_code(today)
    if args.backtest or (args.backtest_if_stale and params_stale()):
        try:
            import backtest
            backtest.run(today)
        except Exception:  # noqa: BLE001
            log.error("backtest failed, keeping existing params:\n%s", traceback.format_exc())
    params = load_params()
    fixtures, hist, clubelo = load_inputs(args, today)
    if hist is None or hist.empty:
        log.error("no history available; aborting without overwriting predictions")
        return {}
    odds_client = oddsapi.OddsClient(key="" if args.demo else None, today=today)
    n_fd = len(fixtures)
    try:
        fixtures = oddsapi.discover_fixtures(odds_client, fixtures, hist, season, today)
    except Exception:  # noqa: BLE001
        log.warning("fixture discovery failed:\n%s", traceback.format_exc(limit=2))
    status = {"fixtures": len(fixtures), "fixtures_football_data": n_fd, "fixtures_odds_api": len(fixtures) - n_fd,
              "history": len(hist), "clubelo": bool(len(clubelo))}
    model = Model(hist, params, clubelo)
    ref = pd.Timestamp(today)

    # ---- stage 1: scan every game
    rows, records = {}, []
    for i, row in fixtures.iterrows():
        row = row.to_dict()
        gid = f"{row['Div']}-{pd.Timestamp(row['Date']):%Y%m%d}-{i}"
        try:
            p = predict_game(model, row, params, ref, season)
            records.append(game_record(gid, row, p, params))
            rows[gid] = row
        except Exception:  # noqa: BLE001
            log.warning("prediction failed for %s v %s:\n%s", row.get("HomeTeam"), row.get("AwayTeam"),
                        traceback.format_exc(limit=2))
    flagged = set(select_flagged(records, today=today))
    log.info("stage 1: %d games scanned, %d flagged", len(records), len(flagged))

    # ---- stage 2: extras for flagged games
    games = [{"id": r["id"], "league": r["league"], "home": r["home"], "away": r["away"], "date": r["date"],
              "priority": r["max_ev"] if r["max_ev"] is not None else -1}
             for r in records if r["id"] in flagged]
    try:
        ex = extras.run(games, hist, season, odds_client) if games else {"games": {}}
    except Exception:  # noqa: BLE001
        log.error("extras failed:\n%s", traceback.format_exc())
        ex = {"games": {}}
    status.update(api_football=ex.get("api_football_error") or "ok", api_football_requests=ex.get("api_football_used", 0),
                  odds_api=odds_client.status())
    for k, r in enumerate(records):
        if r["id"] not in flagged:
            continue
        e = ex["games"].get(r["id"], {})
        try:
            p = predict_game(model, rows[r["id"]], params, ref, season,
                             adj=tuple(e.get("lambda_mult", (1.0, 1.0))), extra_odds=e.get("extra_odds"))
            records[k] = game_record(r["id"], rows[r["id"]], p, params, flagged=True, ext=e)
        except Exception:  # noqa: BLE001
            log.warning("stage 2 failed for %s:\n%s", r["id"], traceback.format_exc(limit=2))
            records[k]["flagged"] = True

    for r in records:
        try:
            r["summary"] = summary.generate(r)
        except Exception:  # noqa: BLE001
            r["summary"] = ""
        r.pop("evals_full", None)

    records.sort(key=lambda r: (r["date"], r["time"] or "", r["league"]))
    days = {}
    for r in records:
        days.setdefault(r["date"], []).append(r)
    daily = []
    for d, gs in sorted(days.items()):
        try:
            daily.append(summary.daily_overview(d, gs, today))
        except Exception:  # noqa: BLE001
            log.warning("daily overview failed for %s:\n%s", d, traceback.format_exc(limit=2))
    top = [r for r in records if r["best_bet"] and r["flagged"]]
    top.sort(key=lambda r: -(r["best_bet"]["ev"] * (0.5 + r["confidence"] / 200)))
    bt = read_json(DATA / "backtest.json", {}) or {}
    out = {
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "demo": bool(args.demo), "reference_date": today.isoformat(), "season": season,
        "n_games": len(records), "n_leagues": len({r["league"] for r in records}),
        "n_flagged": len(flagged), "params": params, "status": status,
        "backtest": {k: bt.get("test", {}).get(k) for k in ("model", "market", "bets")} if bt else None,
        "leagues": {k: v[0] for k, v in LEAGUES.items()},
        "top_bets": [{k: r[k] for k in ("id", "league", "league_name", "date", "time", "home", "away",
                                        "best_bet", "confidence", "probs")} for r in top[:5]],
        "daily": daily,
        "games": records,
    }
    out = _clean(out)
    target = DATA / ("predictions_demo.json" if args.demo else "predictions.json")
    write_json(target, out)
    log.info("wrote %s: %d games, %d value bets", target.name, len(records), sum(1 for r in records if r["best_bet"]))
    if not args.demo:
        try:
            track.update(out, hist, today)
        except Exception:  # noqa: BLE001
            log.error("track record update failed:\n%s", traceback.format_exc())
    if not args.demo and not args.no_notify:
        notify.run(out, os.environ.get("SITE_URL"))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", help="replay the week starting YYYY-MM-DD using history as fixtures")
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--no-notify", action="store_true")
    ap.add_argument("--backtest", action="store_true", help="always run the backtest first")
    ap.add_argument("--backtest-if-stale", action="store_true")
    try:
        run(ap.parse_args())
    except Exception:  # noqa: BLE001
        log.error("pipeline failed:\n%s", traceback.format_exc())
        sys.exit(1)
