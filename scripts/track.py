"""Track record: log every prediction, settle it against the real result, and summarise.

data/track_record.json holds one entry per game:
  * the latest prediction made before kick-off (source "live"), or a walk-forward replay made with
    data available before that day (source "replay", used to backfill the season);
  * once played: the score and whether each call / bet won.

    python scripts/track.py --backfill 2026-08-01   # replay the season up to yesterday
    python scripts/track.py                         # settle finished games and print the summary
"""
from __future__ import annotations

import argparse
import datetime as dt
import json

import pandas as pd

from backtest import settle
from common import DATA, get_logger, load_params, read_json, season_code, write_json

log = get_logger("track")
LEDGER = DATA / "track_record.json"
BET_TYPE = {"1X2": "Winner (1X2)", "O/U": "Goals (over/under)", "BTTS": "Both teams to score"}


def bet_type(market: str) -> str:
    return next((v for k, v in BET_TYPE.items() if market.startswith(k)), market)


def game_key(league: str, date: str, home: str, away: str) -> str:
    return f"{league}|{date}|{home}|{away}"


def load() -> dict:
    return read_json(LEDGER, None) or {"games": {}}


def save(ledger: dict) -> None:
    ledger["updated_at"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    LEDGER.write_text(json.dumps(ledger, separators=(",", ":"), ensure_ascii=False, default=str), encoding="utf-8")


def _bet(b: dict | None) -> dict | None:
    if not b:
        return None
    return {k: b.get(k) for k in ("market", "selection", "label", "odds", "model_p", "market_p", "ev", "edge")}


def snapshot(g: dict, source: str, made_at: str) -> dict:
    """Compact copy of a game record from predictions.json."""
    mk = g.get("markets", {})
    return {
        "league": g["league"], "league_name": g["league_name"], "date": g["date"], "time": g.get("time"),
        "home": g["home"], "away": g["away"], "source": source, "made_at": made_at,
        "probs": g["probs"], "likely_score": g["likely_score"],
        "p_over25": (mk.get("O/U 2.5") or {}).get("Over"), "p_btts": (mk.get("BTTS") or {}).get("Yes"),
        "confidence": g.get("confidence"),
        "lean": _bet(g.get("lean")), "value": _bet(g.get("best_bet")),
        "pick_tier": (g.get("pick") or {}).get("tier"),
        "dc_pick": ((g.get("pick") or {}).get("double_chance") or {}).get("selections"),
        "dc_label": ((g.get("pick") or {}).get("double_chance") or {}).get("label"),
        "prob_source": g.get("prob_source", "dixon-coles"),
        "result": None, "outcome": None,
    }


def record_predictions(ledger: dict, pred: dict, today: dt.date) -> int:
    """Store today's and future games (latest pre-match prediction wins). Returns entries written."""
    n = 0
    for g in pred.get("games", []):
        if g["date"] < today.isoformat():
            continue
        k = game_key(g["league"], g["date"], g["home"], g["away"])
        old = ledger["games"].get(k)
        if old and old.get("result"):
            continue
        ledger["games"][k] = snapshot(g, "live", pred.get("generated_at"))
        n += 1
    return n


def evaluate(e: dict, hg: int, ag: int) -> dict:
    res = "H" if hg > ag else "D" if hg == ag else "A"
    p = e["probs"]
    pick = max(("H", "D", "A"), key=lambda s: p[s])
    out = {"winner_pick": pick, "winner_ok": pick == res, "score_ok": e["likely_score"] == f"{hg}-{ag}"}
    out["tier"] = e.get("pick_tier")
    if e.get("dc_pick"):
        out["dc_ok"] = res in e["dc_pick"]
    if e.get("p_over25") is not None:
        out["ou_pick"] = "Over" if e["p_over25"] >= 0.5 else "Under"
        out["ou_ok"] = (hg + ag > 2.5) == (out["ou_pick"] == "Over")
    if e.get("p_btts") is not None:
        out["btts_pick"] = "Yes" if e["p_btts"] >= 0.5 else "No"
        out["btts_ok"] = (hg > 0 and ag > 0) == (out["btts_pick"] == "Yes")
    for kind in ("lean", "value"):
        b = e.get(kind)
        if b and b.get("odds"):
            pl = settle(b["market"], b["selection"], hg, ag, float(b["odds"]))
            out[f"{kind}_pl"] = round(pl, 3)
            out[f"{kind}_ok"] = pl > 0
            out[f"{kind}_type"] = bet_type(b["market"])
    return out


def settle_results(ledger: dict, hist: pd.DataFrame) -> int:
    """Attach final scores from football-data history (same team names; +/-3 days for rescheduling)."""
    if hist is None or hist.empty:
        return 0
    idx = {}
    for r in hist.itertuples(index=False):
        idx.setdefault((r.Div, r.HomeTeam, r.AwayTeam), []).append((pd.Timestamp(r.Date), int(r.FTHG), int(r.FTAG)))
    n = 0
    for e in ledger["games"].values():
        if e.get("result"):
            continue
        d = pd.Timestamp(e["date"])
        cands = [c for c in idx.get((e["league"], e["home"], e["away"]), []) if abs((c[0] - d).days) <= 3]
        if not cands:
            continue
        when, hg, ag = min(cands, key=lambda c: abs((c[0] - d).days))
        e["result"] = {"hg": hg, "ag": ag, "played": when.strftime("%Y-%m-%d")}
        e["outcome"] = evaluate(e, hg, ag)
        n += 1
    return n


def backfill(ledger: dict, hist: pd.DataFrame, start: dt.date, end: dt.date) -> int:
    """Walk-forward replay: for every match day in [start, end), predict with data before that day."""
    import features
    import main  # lazy: main imports this module
    import ml
    from model import Model
    params = load_params()
    season = season_code(start)
    # ML ensemble trained only on seasons before this one, so the replay stays out-of-sample
    bundle = ml.train_final(before_season=season, path=ml.DATA / "ml_model_replay.pkl")
    full = ml.all_history(end)                      # same 6-season span (and Elo) the ML was trained on
    hist = full[full["Date"] >= pd.Timestamp(start) - pd.Timedelta(days=400)].reset_index(drop=True)
    feats = features.build(hist)                    # pre-match features only use earlier dates
    hist = hist.drop(columns=["EloH", "EloA"])      # Model computes its own Elo from what it is given
    full = full.drop(columns=["EloH", "EloA"])
    days = sorted(d for d in hist["Date"].dt.date.unique() if start <= d < end)
    n = 0
    for day in days:
        before = full[full["Date"] < pd.Timestamp(day)]
        todays = hist[hist["Date"] == pd.Timestamp(day)]
        model = Model(before, params)
        made = f"replay (data to {day - dt.timedelta(days=1)})"
        recs, odds, idx = [], {}, {}
        for i, row in todays.iterrows():
            row = row.to_dict()
            k = game_key(row["Div"], day.isoformat(), row["HomeTeam"], row["AwayTeam"])
            if k in ledger["games"] and ledger["games"][k].get("source") == "live":
                continue  # never overwrite a genuine live prediction
            try:
                p = main.predict_game(model, row, params, pd.Timestamp(day), season)
                recs.append(main._clean(main.game_record(k, row, p, params)))
                odds[k], idx[k] = p["odds"], i
            except Exception as ex:  # noqa: BLE001
                log.debug("replay failed for %s: %s", k, ex)
        if recs:
            X = feats.loc[[idx[r["id"]] for r in recs]]
            X.index = [r["id"] for r in recs]
            ml.apply(bundle, recs, X)
            for r in recs:
                if r.get("prob_source") == "ml":
                    main.rescore(r, odds[r["id"]], params)
                ledger["games"][r["id"]] = snapshot(main._clean(r), "replay", made)
                n += 1
        log.info("replayed %s: %d games", day, len(recs))
    return n


# ---------------------------------------------------------------- summary (also computed in the browser)
def summarise(ledger: dict, source: str | None = None) -> dict:
    rows = [e for e in ledger["games"].values() if e.get("outcome") and (source is None or e["source"] == source)]

    def rate(key):
        v = [e["outcome"][key] for e in rows if key in e["outcome"]]
        return {"n": len(v), "hit": sum(v), "pct": round(sum(v) / len(v), 4) if v else None}

    def bets(kind):
        out = {}
        for e in rows:
            o = e["outcome"]
            if f"{kind}_pl" not in o:
                continue
            for t in ("All", o[f"{kind}_type"]):
                s = out.setdefault(t, {"n": 0, "won": 0, "profit": 0.0})
                s["n"] += 1
                s["won"] += int(o[f"{kind}_ok"])
                s["profit"] += o[f"{kind}_pl"]
        for s in out.values():
            s["hit_pct"] = round(s["won"] / s["n"], 4)
            s["roi"] = round(s["profit"] / s["n"], 4)
            s["profit"] = round(s["profit"], 2)
        return out
    return {"games": len(rows), "winner": rate("winner_ok"), "exact_score": rate("score_ok"),
            "over_under_2_5": rate("ou_ok"), "btts": rate("btts_ok"),
            "value_bets": bets("value"), "leans": bets("lean")}


def update(pred: dict, hist: pd.DataFrame, today: dt.date) -> dict:
    ledger = load()
    w = record_predictions(ledger, pred, today)
    s = settle_results(ledger, hist)
    save(ledger)
    log.info("track record: %d predictions stored, %d games settled, %d total", w, s, len(ledger["games"]))
    return ledger


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", help="replay from YYYY-MM-DD up to yesterday")
    a = ap.parse_args()
    hist = pd.read_csv(DATA / "history.csv", parse_dates=["Date"], low_memory=False)
    hist["Season"] = hist["Season"].astype(str).str.zfill(4)
    ledger = load()
    if a.backfill:
        backfill(ledger, hist, dt.date.fromisoformat(a.backfill), dt.date.today())
    settle_results(ledger, hist)
    save(ledger)
    print(json.dumps(summarise(ledger), indent=1))
