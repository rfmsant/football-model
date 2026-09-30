"""Publish and settle the daily deep-dive reports (see DEEP_DIVE.md).

    python scripts/deep.py publish data/deep/2026-10-01.json   # validate + update latest.json / index.json
    python scripts/deep.py settle                               # settle finished bets -> data/deep_record.json
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

from backtest import settle
from common import DATA, get_logger, read_json, write_json

log = get_logger("deep")
DEEP = DATA / "deep"
DEEP.mkdir(exist_ok=True)
MARKETS = {"1X2": {"H", "D", "A"}, "Double chance": {"1X", "X2", "12"}, "Draw no bet": {"H", "A"},
           "O/U 1.5": {"Over", "Under"}, "O/U 2.5": {"Over", "Under"}, "O/U 3.5": {"Over", "Under"}, "BTTS": {"Yes", "No"}}
VERDICTS = {"BET", "BET SMALL", "WAIT FOR LINE-UPS", "PASS"}
REQUIRED = ("rank", "match", "league", "kickoff", "market", "selection", "label", "model_p", "final_p", "odds_seen",
            "min_odds", "meets_bar", "adjustments", "analysis", "verdict")


def validate(rep: dict) -> list[str]:
    errs = []
    for k in ("date", "bets"):
        if k not in rep:
            errs.append(f"missing top-level '{k}'")
    for i, b in enumerate(rep.get("bets", []), 1):
        tag = f"bet {i} ({b.get('label', '?')})"
        for k in REQUIRED:
            if k not in b:
                errs.append(f"{tag}: missing '{k}'")
        m, s = b.get("market"), b.get("selection")
        if m not in MARKETS or s not in MARKETS.get(m, set()):
            errs.append(f"{tag}: unsupported market/selection {m}/{s}")
        if b.get("verdict") not in VERDICTS:
            errs.append(f"{tag}: verdict must be one of {sorted(VERDICTS)}")
        try:
            fp, mp, o = float(b["final_p"]), float(b["model_p"]), float(b["odds_seen"])
            total = round(sum(float(a["delta"]) for a in b.get("adjustments", [])), 4)
            if abs(round(mp + total, 4) - round(fp, 4)) > 0.011:
                errs.append(f"{tag}: final_p {fp} != model_p {mp} + adjustments {total}")
            want = fp >= 0.70 and o >= 1.40
            if bool(b.get("meets_bar")) != want:
                errs.append(f"{tag}: meets_bar should be {want} (final_p {fp}, odds {o})")
        except (KeyError, TypeError, ValueError) as e:
            errs.append(f"{tag}: bad numbers ({e})")
        for a in b.get("adjustments", []):
            if not {"factor", "delta", "reason"} <= set(a):
                errs.append(f"{tag}: adjustment needs factor, delta, reason")
    return errs


def publish(path: str) -> None:
    p = Path(path)
    rep = json.loads(p.read_text(encoding="utf-8"))
    errs = validate(rep)
    if errs:
        print("NOT published, fix these:\n  - " + "\n  - ".join(errs))
        sys.exit(1)
    rep.setdefault("generated_at", dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    rep["n_meets_bar"] = sum(1 for b in rep["bets"] if b["meets_bar"])
    write_json(p, rep)
    write_json(DEEP / "latest.json", rep)
    idx = read_json(DEEP / "index.json", []) or []
    idx = [x for x in idx if x["date"] != rep["date"]] + [{"date": rep["date"], "file": p.name, "bets": len(rep["bets"]),
                                                            "meets_bar": rep["n_meets_bar"]}]
    write_json(DEEP / "index.json", sorted(idx, key=lambda x: x["date"], reverse=True))
    print(f"published {p.name}: {len(rep['bets'])} bets, {rep['n_meets_bar']} meet the bar")


def settle_all(hist: pd.DataFrame | None = None) -> dict:
    """Settle every bet in every deep-dive report against football-data results."""
    if hist is None:
        hist = pd.read_csv(DATA / "history.csv", parse_dates=["Date"], low_memory=False)
    from common import match_name
    by_div = {d: g for d, g in hist.groupby("Div")}
    rows = []
    for f in sorted(DEEP.glob("20*.json")):
        rep = json.loads(f.read_text(encoding="utf-8"))
        for b in rep.get("bets", []):
            row = {"report_date": rep["date"], **{k: b.get(k) for k in ("rank", "match", "league", "league_name", "kickoff", "market",
                                                                        "selection", "label", "model_p", "final_p", "odds_seen",
                                                                        "min_odds", "meets_bar", "confidence", "verdict")}}
            g = by_div.get(b.get("league"))
            if g is not None and " v " in (b.get("match") or ""):
                home, away = b["match"].split(" v ", 1)
                teams = set(g.HomeTeam) | set(g.AwayTeam)
                h, a = match_name(home, teams), match_name(away, teams)
                ko = pd.Timestamp(str(b.get("kickoff", ""))[:10]) if b.get("kickoff") else None
                m = g[(g.HomeTeam == h) & (g.AwayTeam == a)]
                if ko is not None and len(m):
                    m = m[(m.Date - ko).abs() <= pd.Timedelta(days=3)]
                if len(m):
                    r = m.iloc[0]
                    pl = settle(b["market"], b["selection"], int(r.FTHG), int(r.FTAG), float(b["odds_seen"]))
                    row.update(score=f"{int(r.FTHG)}-{int(r.FTAG)}", profit=round(pl, 3),
                               result="won" if pl > 0 else "void" if pl == 0 else "lost")
            rows.append(row)
    settled = [r for r in rows if "result" in r]

    def summ(rs):
        dec = [r for r in rs if r["result"] != "void"]
        return {"bets": len(rs), "won": sum(r["result"] == "won" for r in rs), "void": sum(r["result"] == "void" for r in rs),
                "hit_pct": round(sum(r["result"] == "won" for r in dec) / len(dec), 4) if dec else None,
                "profit": round(sum(r["profit"] for r in rs), 2), "roi": round(sum(r["profit"] for r in rs) / len(rs), 4) if rs else None,
                "avg_odds": round(sum(float(r["odds_seen"]) for r in rs) / len(rs), 2) if rs else None,
                "avg_final_p": round(sum(float(r["final_p"]) for r in rs) / len(rs), 4) if rs else None}
    out = {"updated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "all": summ(settled), "meets_bar": summ([r for r in settled if r["meets_bar"]]),
           "best_available": summ([r for r in settled if not r["meets_bar"]]),
           "by_market": {m: summ([r for r in settled if r["market"] == m]) for m in sorted({r["market"] for r in settled})},
           "by_day": {d: summ([r for r in settled if r["report_date"] == d]) for d in sorted({r["report_date"] for r in settled}, reverse=True)},
           "bets": rows}
    write_json(DATA / "deep_record.json", out)
    log.info("deep record: %d bets, %d settled", len(rows), len(settled))
    return out


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "publish":
        publish(sys.argv[2])
    elif len(sys.argv) >= 2 and sys.argv[1] == "settle":
        print(json.dumps({k: v for k, v in settle_all().items() if k != "bets"}, indent=1))
    else:
        print(__doc__)
