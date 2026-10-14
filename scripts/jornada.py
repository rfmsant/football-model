"""Weekly Liga Portugal round report (jornada): every game of the next round with the model's chances, the current
prices and the best bets, posted to Discord a few days before the round. The user picks and places the bets.

A "best bet" needs positive value at the current price (model chance x price > 1 + MIN_VALUE). Most games will
have none: bookmakers price Liga Portugal well, and the report says so instead of forcing a pick. Each best bet
gets a minimum price (below it the bet is no longer worth taking at Betclic).

    python scripts/jornada.py build            # writes data/jornada/<first date>.json + latest.json
    python scripts/jornada.py send             # posts latest.json to DISCORD_WEBHOOK
    python scripts/jornada.py run              # build, and post once per round when it is <= POST_DAYS away
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

import requests

import card
import shortlist
from common import DATA, get_logger, read_json, write_json

log = get_logger("jornada")
OUT = DATA / "jornada"
LEAGUE = "P1"
ROUND_SPAN_DAYS = 4        # a round runs Friday to Monday
POST_DAYS = 3              # post when the round's first game is this close
MIN_VALUE = 0.02           # model chance x price must beat 1 by at least 2%
MIN_P, MIN_ODDS, MAX_ODDS = 0.25, 1.30, 6.00
MAX_BEST = 3
PT_DAYS = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]


def _price(g: dict, market: str, sel: str) -> float | None:
    e = next((e for e in g.get("evals", []) if e["market"] == market and e["selection"] == sel and e.get("avg_odds")), None)
    return e["avg_odds"] if e else None


def next_round(games: list[dict], now: dt.datetime | None = None) -> list[dict]:
    """Liga Portugal games of the next round that haven't kicked off: the earliest upcoming date + ROUND_SPAN_DAYS."""
    up = sorted((g for g in games if g.get("league") == LEAGUE and not card.started(g["date"], g.get("time"), now)),
                key=lambda g: (g["date"], g.get("time") or ""))
    if not up:
        return []
    end = (dt.date.fromisoformat(up[0]["date"]) + dt.timedelta(days=ROUND_SPAN_DAYS)).isoformat()
    return [g for g in up if g["date"] <= end]


def _deep(date: str) -> tuple[list[dict], list[dict]]:
    """(researched, passed) deep-dive bets for that date."""
    deep = read_json(DATA / "deep" / "latest.json", {}) or {}
    bets = [b for b in deep.get("bets", []) if str(b.get("kickoff", ""))[:10] == date]
    return [b for b in bets if b.get("verdict") != "PASS"], [b for b in bets if b.get("verdict") == "PASS"]


def game_bets(g: dict) -> dict:
    """{"best": [...], "closest": {...} | None} for one game."""
    try:
        cands = shortlist.candidates_for(g)
    except Exception as e:  # noqa: BLE001
        log.warning("candidates failed for %s: %s", g.get("id"), e)
        cands = []
    researched, passed = _deep(g["date"])
    rows = []
    for c in cands:
        if c["market"] not in card.BETCLIC_MARKETS:
            continue
        if any(card._same_bet(c, b["match"], b["market"], b["selection"]) for b in passed):
            continue
        b = next((b for b in researched if card._same_bet(c, b["match"], b["market"], b["selection"])), None)
        if b:
            c.update(p_model=b["final_p"], researched=True)
            if b.get("odds_seen"):
                c["odds"] = b["odds_seen"]
        elif c.get("odds_source") == "estimated":
            c["odds"] = round(c["odds"] * card.ESTIMATED_PRICE_HAIRCUT, 2)
        w, push = card.leg_probs(c)
        c["value"] = round(w * c["odds"] + push - 1, 4)
        c["min_odds"] = card.min_price(c)
        if c["p_model"] >= MIN_P and MIN_ODDS <= c["odds"] <= MAX_ODDS:
            rows.append(c)
    keep = ("market", "selection", "label", "p_model", "odds", "odds_source", "value", "min_odds", "researched")
    slim = lambda c: {k: c.get(k) for k in keep}  # noqa: E731
    best = sorted((c for c in rows if c["value"] >= MIN_VALUE and c["odds"] >= c["min_odds"]), key=lambda c: -c["value"])
    best = _one_per_outcome(best)[:MAX_BEST]
    closest = max((c for c in rows if c not in best), key=lambda c: c["value"], default=None)
    return {"best": [slim(c) for c in best], "closest": slim(closest) if closest and not best else None}


def _one_per_outcome(cs: list[dict]) -> list[dict]:
    """Drop near-duplicates: one bet per market family (e.g. not 'Porto win' and 'Porto DNB' and 'Porto or draw')."""
    fam = lambda c: "goals" if c["market"].startswith("O/U") else "btts" if c["market"] == "BTTS" else "result"  # noqa: E731
    out, seen = [], set()
    for c in cs:
        if fam(c) not in seen:
            seen.add(fam(c))
            out.append(c)
    return out


def build(now: dt.datetime | None = None) -> dict:
    pred = read_json(DATA / "predictions.json", {}) or {}
    games = next_round(pred.get("games", []), now)
    rep = {"league": "Liga Portugal Betclic", "generated_at": card._now(now).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "round_start": games[0]["date"] if games else None, "round_end": games[-1]["date"] if games else None,
           "games": []}
    for g in games:
        p = g["probs"]
        o25 = (g["markets"].get("O/U 2.5") or {}).get("Over")
        btts = (g["markets"].get("BTTS") or {}).get("Yes")
        rep["games"].append({
            "match": f"{g['home']} v {g['away']}", "date": g["date"], "time": g.get("time"),
            "chances": {"H": round(p["H"], 3), "D": round(p["D"], 3), "A": round(p["A"], 3),
                        "over25": round(o25, 3) if o25 is not None else None, "btts": round(btts, 3) if btts is not None else None},
            "prices": {"H": _price(g, "1X2", "H"), "D": _price(g, "1X2", "D"), "A": _price(g, "1X2", "A"),
                       "over25": _price(g, "O/U 2.5", "Over"), "under25": _price(g, "O/U 2.5", "Under")},
            **game_bets(g)})
    if games:
        old = read_json(OUT / f"{rep['round_start']}.json", {}) or {}
        if old.get("posted_at"):
            rep["posted_at"] = old["posted_at"]
        OUT.mkdir(exist_ok=True)
        write_json(OUT / f"{rep['round_start']}.json", rep)
    write_json(OUT / "latest.json", rep)
    log.info("round %s..%s: %d games, %d best bets", rep["round_start"], rep["round_end"], len(rep["games"]),
             sum(len(x["best"]) for x in rep["games"]))
    return rep


def _odd(x) -> str:
    return f"{x:.2f}" if x else "–"


def _game_text(x: dict) -> str:
    d = dt.date.fromisoformat(x["date"])
    c, pr = x["chances"], x["prices"]
    head = f"`{PT_DAYS[d.weekday()]} {d.day:02d} · {(x.get('time') or '--:--')[:5]}` **{card._short_match(x['match'])}**"
    line = f"1 {_odd(pr['H'])} ({c['H']:.0%}) · X {_odd(pr['D'])} ({c['D']:.0%}) · 2 {_odd(pr['A'])} ({c['A']:.0%})"
    if pr.get("over25") and c.get("over25") is not None:
        line += f" · O2.5 {_odd(pr['over25'])} ({c['over25']:.0%})"
    out = [head, line]
    for b in x["best"]:
        est = " (estimated, check Betclic)" if b.get("odds_source") == "estimated" else ""
        res = " 🔎" if b.get("researched") else ""
        out.append(f"✅ **{card._short_bet(b)}** {b['odds']:.2f}{est} · {b['p_model']:.0%}{res} · value {b['value']:+.0%} · **min {b['min_odds']:.2f}**")
    if not x["best"]:
        cl = x.get("closest")
        out.append(f"— no value bet" + (f" (closest: {card._short_bet(cl)} {cl['p_model']:.0%}, worth it from {cl['min_odds']:.2f})" if cl else ""))
    return "\n".join(out)


def discord_payload(rep: dict) -> dict:
    if not rep.get("games"):
        return {"username": "football-model", "embeds": [{"title": "🇵🇹 Liga Portugal · next round", "url": card.SITE,
                "description": "No Liga Portugal games with prices yet.", "color": 0x95A5A6}]}
    a, b = (dt.date.fromisoformat(rep[k]) for k in ("round_start", "round_end"))
    n_best = sum(len(x["best"]) for x in rep["games"])
    blocks = [_game_text(x) for x in rep["games"]]
    embeds, cur = [], []
    for blk in blocks:                                 # Discord: 4096 characters per embed description
        if sum(len(s) + 2 for s in cur) + len(blk) > 3800:
            embeds.append(cur)
            cur = []
        cur.append(blk)
    embeds.append(cur)
    out = []
    for k, part in enumerate(embeds):
        e = {"description": "\n\n".join(part), "color": 0x006600}
        if k == 0:
            e.update(title=f"🇵🇹 Liga Portugal · {a.strftime('%d %b')}–{b.strftime('%d %b')} · {n_best} value bet{'s' * (n_best != 1)}",
                     url=card.SITE)
        if k == len(embeds) - 1:
            e["footer"] = {"text": "Chance = model · price = bookmaker average · bet only at Betclic ≥ min · fixed small stake · 18+"}
        out.append(e)
    return {"username": "football-model", "embeds": out}


def send(rep: dict) -> bool:
    hook = os.environ.get("DISCORD_WEBHOOK")
    if not hook:
        print("DISCORD_WEBHOOK not set; round report not sent")
        return True
    r = requests.post(hook, json=discord_payload(rep), timeout=20)
    ok = r.status_code < 300
    print("sent to Discord" if ok else f"Discord error {r.status_code}: {r.text[:200]}")
    return ok


def due(rep: dict, now: dt.datetime | None = None) -> bool:
    """Post once per round, from POST_DAYS before its first game."""
    if not rep.get("games") or rep.get("posted_at"):
        return False
    first = rep["games"][0]
    ko = card.kickoff_utc(first["date"], first.get("time")) or dt.datetime.combine(
        dt.date.fromisoformat(first["date"]), dt.time(12), tzinfo=dt.timezone.utc)
    return ko - card._now(now) <= dt.timedelta(days=POST_DAYS)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["build", "send", "run", "show"])
    a = ap.parse_args()
    if a.action == "build":
        print(json.dumps(discord_payload(build()), indent=1, ensure_ascii=False))
    elif a.action == "send":
        sys.exit(0 if send(read_json(OUT / "latest.json", {}) or {}) else 1)
    elif a.action == "run":
        rep = build()
        if due(rep) and not os.environ.get("DISCORD_WEBHOOK"):
            print("round report due, but DISCORD_WEBHOOK not set; not marked as posted")
        elif due(rep):
            if not send(rep):
                sys.exit(1)
            rep["posted_at"] = card._now().strftime("%Y-%m-%dT%H:%M:%SZ")
            write_json(OUT / f"{rep['round_start']}.json", rep)
            write_json(OUT / "latest.json", rep)
        else:
            print("round report not due (already posted, or round more than", POST_DAYS, "days away)")
    else:
        print(json.dumps(discord_payload(read_json(OUT / "latest.json", {}) or {}), indent=1, ensure_ascii=False))
