"""Weekly Liga Portugal deep dive (see PORTUGAL_WEEKLY.md): validate, publish and post to Discord.

    python scripts/portugal.py publish data/portugal/2026-10-18.json   # validate + write latest.json
    python scripts/portugal.py send                                     # post latest.json to DISCORD_WEBHOOK
    python scripts/portugal.py show                                     # print the Discord payload
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import requests

import card
from common import DATA, get_logger, read_json, write_json
from deep import MARKETS

log = get_logger("portugal")
OUT = DATA / "portugal"
OUTCOMES = ("H", "D", "A", "over15", "over25", "over35", "btts")
FACTORS = ("injuries", "fatigue", "lineups", "form", "home_away", "manager_news", "motivation", "referee", "weather",
           "tactics", "market", "h2h")
PT_DAYS = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]


def _check_bet(b: dict, tag: str) -> list[str]:
    errs = []
    for k in ("market", "selection", "label", "p", "min_odds"):
        if k not in b:
            errs.append(f"{tag}: missing '{k}'")
    m, s = b.get("market"), b.get("selection")
    if m not in MARKETS or s not in MARKETS.get(m, set()):
        errs.append(f"{tag}: unsupported market/selection {m}/{s}")
    p = b.get("p")
    if not isinstance(p, (int, float)) or not 0 < p < 1:
        errs.append(f"{tag}: p must be between 0 and 1")
    elif isinstance(b.get("min_odds"), (int, float)) and abs(b["min_odds"] - round(1 / p + 0.02, 2)) > 0.011:
        errs.append(f"{tag}: min_odds should be {round(1 / p + 0.02, 2)}")
    o = b.get("odds_seen")
    if o is not None and (not isinstance(o, (int, float)) or o < 1.01):
        errs.append(f"{tag}: odds_seen must be a price or null")
    return errs


def validate(rep: dict) -> list[str]:
    errs = [f"missing top-level '{k}'" for k in ("date", "week_start", "week_end", "summary", "games") if k not in rep]
    for i, g in enumerate(rep.get("games", []), 1):
        tag = f"game {i} ({g.get('match', '?')})"
        for k in ("match", "kickoff", "outcomes", "best_bet", "why", "confidence", "factors", "risks", "sources"):
            if k not in g:
                errs.append(f"{tag}: missing '{k}'")
        try:
            dt.datetime.strptime(str(g.get("kickoff")), "%Y-%m-%d %H:%M")
        except ValueError:
            errs.append(f"{tag}: kickoff must be 'YYYY-MM-DD HH:MM' (Lisbon time)")
        oc = g.get("outcomes") or {}
        missing = [k for k in OUTCOMES if k not in oc]
        if missing:
            errs.append(f"{tag}: outcomes missing {missing}")
        elif abs(oc["H"] + oc["D"] + oc["A"] - 1) > 0.02:
            errs.append(f"{tag}: H+D+A = {oc['H'] + oc['D'] + oc['A']:.3f}, should be 1")
        missing = [k for k in FACTORS if k not in (g.get("factors") or {})]
        if missing:
            errs.append(f"{tag}: factors missing {missing}")
        if g.get("confidence") not in ("high", "medium", "low"):
            errs.append(f"{tag}: confidence must be high, medium or low")
        if isinstance(g.get("best_bet"), dict):
            errs += _check_bet(g["best_bet"], f"{tag} best_bet")
        for j, a in enumerate(g.get("alternatives", []), 1):
            errs += _check_bet(a, f"{tag} alternative {j}")
    return errs


def publish(path: Path) -> dict:
    rep = read_json(path, {}) or {}
    errs = validate(rep)
    if errs:
        print("validation errors:\n- " + "\n- ".join(errs))
        sys.exit(1)
    rep["games"].sort(key=lambda g: g["kickoff"])
    write_json(path, rep)
    write_json(OUT / "latest.json", rep)
    idx = []
    for f in sorted(OUT.glob("20*.json"), reverse=True):     # newest first, for the site's week selector
        r = read_json(f, {}) or {}
        idx.append({"file": f.name, "date": r.get("date"), "week_start": r.get("week_start"), "week_end": r.get("week_end"),
                    "games": len(r.get("games", []))})
    write_json(OUT / "index.json", idx)
    print(f"published {path.name}: {len(rep['games'])} games, {rep['week_start']}..{rep['week_end']}")
    return rep


def _price(b: dict) -> str:
    return f"{b['odds_seen']:.2f}" if b.get("odds_seen") else "no price yet"


def _game_text(g: dict) -> str:
    ko = dt.datetime.strptime(g["kickoff"], "%Y-%m-%d %H:%M")
    b, oc = g["best_bet"], g["outcomes"]
    out = [f"`{PT_DAYS[ko.weekday()]} {ko.day:02d} · {ko:%H:%M}` **{card._short_match(g['match'])}** ({g['confidence']})",
           f"✅ **{b['label']}** · {b['p']:.0%} · {_price(b)} · **min {b['min_odds']:.2f}**",
           f"1 {oc['H']:.0%} · X {oc['D']:.0%} · 2 {oc['A']:.0%} · O2.5 {oc['over25']:.0%} · BTTS {oc['btts']:.0%}"]
    for a in g.get("alternatives", [])[:2]:
        out.append(f"↳ {a['label']} · {a['p']:.0%} · {_price(a)} · min {a['min_odds']:.2f}")
    if g.get("risks"):
        out.append(f"⚠️ {g['risks'][0]}")
    return "\n".join(out)


def discord_payload(rep: dict) -> dict:
    if not rep.get("games"):
        return {"username": "football-model", "embeds": [{"title": "🇵🇹 Liga Portugal · week ahead", "url": card.SITE,
                "description": rep.get("summary") or "No Liga Portugal games next week.", "color": 0x95A5A6}]}
    a, b = (dt.date.fromisoformat(rep[k]) for k in ("week_start", "week_end"))
    blocks = [rep["summary"]] + [_game_text(g) for g in rep["games"]]
    embeds, cur = [], []
    for blk in blocks:                                 # Discord: 4096 characters per embed description
        if cur and sum(len(s) + 2 for s in cur) + len(blk) > 3800:
            embeds.append(cur)
            cur = []
        cur.append(blk)
    embeds.append(cur)
    out = []
    for k, part in enumerate(embeds):
        e = {"description": "\n\n".join(part), "color": 0x006600}
        if k == 0:
            e.update(title=f"🇵🇹 Liga Portugal · best bet per game · {a.strftime('%d %b')}–{b.strftime('%d %b')}", url=card.SITE)
        if k == len(embeds) - 1:
            e["footer"] = {"text": "Chance = researched estimate · bet only at Betclic ≥ min · fixed small stake · 18+"}
        out.append(e)
    return {"username": "football-model", "embeds": out}


def send(rep: dict) -> bool:
    hook = os.environ.get("DISCORD_WEBHOOK")
    if not hook:
        print("DISCORD_WEBHOOK not set; weekly report not sent")
        return True
    r = requests.post(hook, json=discord_payload(rep), timeout=20)
    ok = r.status_code < 300
    print("sent to Discord" if ok else f"Discord error {r.status_code}: {r.text[:200]}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["publish", "send", "show"])
    ap.add_argument("file", nargs="?")
    a = ap.parse_args()
    if a.action == "publish":
        if not a.file:
            sys.exit("usage: portugal.py publish data/portugal/<date>.json")
        publish(Path(a.file).resolve())
    elif a.action == "send":
        sys.exit(0 if send(read_json(OUT / "latest.json", {}) or {}) else 1)
    else:
        print(json.dumps(discord_payload(read_json(OUT / "latest.json", {}) or {}), indent=1, ensure_ascii=False))
