"""Daily betting card: combined bets (doubles/trebles/4-folds) at 2.00+ for the user's bookmaker (Betclic),
sent to Discord.

Leg pool for the card day: club games (data/predictions.json), international games (data/intl.json) and the
deep-dive research (data/deep/latest.json overrides the model's probability with the researched one).
Combos use one leg per game, combined odds between TARGET_MIN and TARGET_MAX, and are ranked by the chance that
every leg wins. Combos built from poor-value legs are excluded. Each leg gets a minimum Betclic price
(below it the leg is bad value).

    python scripts/card.py build [--date YYYY-MM-DD]   # writes data/cards/<date>.json
    python scripts/card.py send  [--date YYYY-MM-DD]   # posts that card to the DISCORD_WEBHOOK
    python scripts/card.py settle                      # settles past cards -> data/card_record.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import os
import sys

import pandas as pd
import requests

import shortlist
from backtest import settle
from common import DATA, get_logger, match_name, read_json, write_json

log = get_logger("card")
CARDS = DATA / "cards"
CARDS.mkdir(exist_ok=True)
TARGET_MIN, TARGET_MAX = 2.00, 3.20
MIN_LEG_PROB = 0.62
MIN_LEG_EV = -0.04          # legs worse than this are never used
MAX_LEGS = 4
BETCLIC_MARKETS = {"1X2", "Double chance", "Draw no bet", "O/U 1.5", "O/U 2.5", "O/U 3.5", "BTTS"}
SITE = "https://rfmsant.github.io/football-model/"


def leg_probs(c: dict) -> tuple[float, float]:
    """(P leg wins, P leg refunded)."""
    if c["market"] == "Draw no bet":
        push = c.get("push_prob", 0.25)
        return c["p_model"] * (1 - push), push
    return c["p_model"], 0.0


def min_price(c: dict) -> float:
    """Lowest price at which the leg is still within ~2% of fair value."""
    w, push = leg_probs(c)
    return round(max(1.01, (0.98 - push) / max(w, 1e-6)), 2)


def pool(date: str) -> list[dict]:
    legs = []
    pred = read_json(DATA / "predictions.json", {}) or {}
    for g in pred.get("games", []):
        if g["date"] != date or shortlist.is_hard(g):
            continue
        try:
            legs += [dict(c, source="club model") for c in shortlist.candidates_for(g)]
        except Exception as e:  # noqa: BLE001
            log.debug("skip %s: %s", g.get("id"), e)
    intl = read_json(DATA / "intl.json", {}) or {}
    for g in intl.get("games", []):
        if g["date"] == date and not g.get("hard"):
            legs += [dict(c, source="international model") for c in g["candidates"]]
    # deep-dive research overrides the model where it covered the same bet
    deep = read_json(DATA / "deep" / "latest.json", {}) or {}
    researched = {}
    for b in deep.get("bets", []):
        if str(b.get("kickoff", ""))[:10] == date and b.get("verdict") != "PASS":
            researched[(b["match"], b["market"], b["selection"])] = b
    for c in legs:
        b = next((v for (m, mk, s), v in researched.items() if mk == c["market"] and s == c["selection"]
                  and match_name(c["match"].split(" v ")[0], [m.split(" v ")[0]])), None)
        if b:
            c.update(p_model=b["final_p"], researched=True, research_note="; ".join(a["reason"] for a in b.get("adjustments", []))[:200])
            if b.get("odds_seen"):
                c["odds"] = b["odds_seen"]
    out = []
    for c in legs:
        if c["market"] not in BETCLIC_MARKETS:
            continue
        w, push = leg_probs(c)
        c["ev"] = round(w * c["odds"] + push - 1, 4)
        if c["p_model"] >= MIN_LEG_PROB and c["ev"] >= MIN_LEG_EV and c["odds"] >= 1.08:
            c["min_odds"] = min_price(c)
            if c["odds"] >= c["min_odds"]:          # never list a leg whose expected price is below its own minimum
                out.append(c)
    return out


def combos(legs: list[dict]) -> list[dict]:
    res = []
    legs = sorted(legs, key=lambda c: -c["ev"])[:40]
    for k in range(2, MAX_LEGS + 1):
        for cs in itertools.combinations(legs, k):
            if len({c["match"] for c in cs}) < k:
                continue
            odds = 1.0
            p_all = 1.0
            p_safe = 1.0
            ev = 1.0
            for c in cs:
                w, push = leg_probs(c)
                odds *= c["odds"]
                p_all *= w
                p_safe *= w + push
                ev *= w * c["odds"] + push
            if TARGET_MIN <= odds <= TARGET_MAX:
                res.append({"legs": list(cs), "odds": round(odds, 2), "p_win": round(p_all, 4), "p_not_lose": round(p_safe, 4),
                            "ev": round(ev - 1, 4), "n": k})
    return res


def choose(cands: list[dict]) -> dict:
    """Main = highest chance of winning among combos at no worse than -3% value; alternatives = best value and
    a different safest one, without reusing the main combo's games."""
    ok = [c for c in cands if c["ev"] >= -0.03]
    if not ok:
        ok = sorted(cands, key=lambda c: -c["ev"])[:20]
    if not ok:
        return {}
    main = max(ok, key=lambda c: (c["p_win"], c["ev"]))   # the goal is the full 2.00+ payout
    used = {l["match"] for l in main["legs"]}
    rest = [c for c in ok if not ({l["match"] for l in c["legs"]} & used)]
    alts = []
    if rest:
        alts.append(max(rest, key=lambda c: c["ev"]))
        used |= {l["match"] for l in alts[-1]["legs"]}
        rest = [c for c in rest if not ({l["match"] for l in c["legs"]} & used)]
        if rest:
            alts.append(max(rest, key=lambda c: (c["p_win"], c["ev"])))
    return {"main": main, "alternatives": alts}


def build(date: str) -> dict:
    legs = pool(date)
    pick = choose(combos(legs))
    slim = lambda c: {k: c.get(k) for k in ("match", "league_name", "date", "time", "market", "selection", "label", "p_model",  # noqa: E731
                                            "odds", "odds_source", "min_odds", "ev", "source", "researched", "research_note")}
    card = {"date": date, "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "bookmaker": "Betclic", "target_odds": TARGET_MIN, "legs_considered": len(legs),
            "main": None, "alternatives": []}
    if pick:
        card["main"] = {**{k: pick["main"][k] for k in ("odds", "p_win", "p_not_lose", "ev", "n")}, "legs": [slim(l) for l in pick["main"]["legs"]]}
        card["alternatives"] = [{**{k: a[k] for k in ("odds", "p_win", "p_not_lose", "ev", "n")}, "legs": [slim(l) for l in a["legs"]]}
                                for a in pick["alternatives"]]
    write_json(CARDS / f"{date}.json", card)
    write_json(CARDS / "latest.json", card)
    log.info("card %s: %d legs considered, main %s", date, len(legs), card["main"] and card["main"]["odds"])
    return card


def _combo_text(c: dict) -> str:
    lines = []
    for l in c["legs"]:
        tag = " (researched)" if l.get("researched") else ""
        tag += " · price estimated, check Betclic" if l.get("odds_source") == "estimated" else ""
        lines.append(f"• **{l['label']}**: {l['match']} ({l.get('time') or ''})\n"
                     f"  model {l['p_model']:.0%}{tag} · bet only at Betclic **≥ {l['min_odds']:.2f}** (expected ~{l['odds']:.2f})")
    safe = c.get("p_not_lose", c["p_win"])
    chance = f"chance it wins ≈ **{c['p_win']:.0%}**" + (f" (≈ {safe:.0%} not losing; a draw refunds the draw-no-bet leg)" if safe - c["p_win"] > 0.005 else "")
    lines.append(f"**Combined ≈ {c['odds']:.2f}** · {chance} · value {c['ev']:+.1%}")
    return "\n".join(lines)


def _short_bet(l: dict) -> str:
    """Compact bet name: 'Under 3.5', 'Ukraine or draw', 'Kazakhstan DNB', 'France win', 'BTTS yes'."""
    lab = l["label"].replace(" goals", "").replace(" draw no bet", " DNB").replace(" to win", " win")
    return {"Both teams to score": "BTTS yes", "Both teams NOT to score": "BTTS no"}.get(lab, lab)


def _short_match(m: str) -> str:
    return m.replace("Northern Ireland", "N. Ireland").replace("Republic of Ireland", "Ireland") \
            .replace("Bosnia and Herzegovina", "Bosnia")


def _combo_lines(c: dict) -> str:
    legs = sorted(c["legs"], key=lambda l: (l.get("date") or "", l.get("time") or ""))
    return "\n".join(f"`{(l.get('time') or '--:--')[:5]}` {_short_match(l['match'])} → **{_short_bet(l)}** · min {l['min_odds']:.2f}"
                     for l in legs)


def discord_payload(card: dict) -> dict:
    """Short, phone-friendly card: one line per bet. Details live on the site."""
    d = dt.date.fromisoformat(card["date"])
    title = f"🎯 {d.strftime('%a %d %b')} · Betclic card"
    if not card.get("main"):
        return {"username": "football-model", "embeds": [{"title": title, "url": SITE, "color": 0x95A5A6,
                "description": "No fair-value combo at 2.00+ today. **No bet.**"}]}
    m = card["main"]
    parts = [f"⭐ **MAIN · {m['odds']:.2f} · {m['p_win']:.0%} chance**", _combo_lines(m)]
    if card.get("alternatives"):
        a = card["alternatives"][0]
        parts += ["", f"**Alt · {a['odds']:.2f} · {a['p_win']:.0%} chance**", _combo_lines(a)]
    return {"username": "football-model", "embeds": [{"title": title, "url": SITE, "description": "\n".join(parts),
            "color": 0x2ECC71, "footer": {"text": "Skip any leg priced below its min · 1 unit · 18+, not financial advice"}}]}


def send(date: str) -> bool:
    hook = os.environ.get("DISCORD_WEBHOOK")
    card = read_json(CARDS / f"{date}.json")
    if not card:
        print(f"no card for {date}")
        return False
    if not hook:
        print("DISCORD_WEBHOOK not set; card not sent (add the repository secret to enable Discord)")
        return True   # not an error: Discord is optional
    r = requests.post(hook, json=discord_payload(card), timeout=20)
    ok = r.status_code < 300
    print("sent to Discord" if ok else f"Discord error {r.status_code}: {r.text[:200]}")
    return ok


def settle_cards(hist: pd.DataFrame | None = None) -> dict:
    """Settle club legs against football-data results and international legs against the international results file."""
    if hist is None:
        hist = pd.read_csv(DATA / "history.csv", parse_dates=["Date"], low_memory=False)
    results = {}
    for r in hist.itertuples(index=False):
        results[(r.HomeTeam, r.AwayTeam, str(r.Date.date()))] = (int(r.FTHG), int(r.FTAG))
    try:
        ir = pd.read_csv(DATA / "raw" / "intl" / "results.csv", parse_dates=["date"])
        for r in ir.dropna(subset=["home_score"]).itertuples(index=False):
            results[(r.home_team, r.away_team, str(r.date.date()))] = (int(r.home_score), int(r.away_score))
    except Exception:  # noqa: BLE001
        pass
    rows = []
    for f in sorted(CARDS.glob("20*.json")):
        card = json.loads(f.read_text(encoding="utf-8"))
        for kind, combo in [("main", card.get("main"))] + [("alternative", a) for a in card.get("alternatives", [])]:
            if not combo:
                continue
            payout, status = 1.0, "won"
            for l in combo["legs"]:
                h, a = l["match"].split(" v ", 1)
                res = next((v for (rh, ra, d), v in results.items() if d == l["date"] and rh == h and ra == a), None)
                if res is None:
                    status = "pending"
                    break
                pl = settle(l["market"], l["selection"], res[0], res[1], float(l["odds"]))
                if pl < 0:
                    status, payout = "lost", 0.0
                    break
                payout *= 1 + pl
            rows.append({"date": card["date"], "kind": kind, "odds": combo["odds"], "status": status,
                         "profit": round(payout - 1, 3) if status != "pending" else None})
    done = [r for r in rows if r["status"] != "pending"]
    main = [r for r in done if r["kind"] == "main"]
    out = {"cards": rows,
           "main": {"played": len(main), "won": sum(r["status"] == "won" for r in main),
                    "profit_units": round(sum(r["profit"] for r in main), 2)},
           "all": {"played": len(done), "won": sum(r["status"] == "won" for r in done),
                   "profit_units": round(sum(r["profit"] for r in done), 2)}}
    write_json(DATA / "card_record.json", out)
    return out


def default_date() -> str:
    """Today if there are still enough games today, otherwise tomorrow."""
    today = dt.date.today()
    return today.isoformat() if len(pool(today.isoformat())) >= 4 and dt.datetime.now().hour < 12 else (today + dt.timedelta(days=1)).isoformat()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["build", "send", "settle", "show"])
    ap.add_argument("--date")
    a = ap.parse_args()
    date = a.date or default_date()
    if a.action == "build":
        c = build(date)
        print(json.dumps(discord_payload(c), indent=1, ensure_ascii=False))
    elif a.action == "send":
        sys.exit(0 if send(date) else 1)
    elif a.action == "settle":
        print(json.dumps({k: v for k, v in settle_cards().items() if k != "cards"}, indent=1))
    else:
        print(json.dumps(discord_payload(read_json(CARDS / f"{date}.json", {})), indent=1, ensure_ascii=False))
