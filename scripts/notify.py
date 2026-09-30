"""Post the top bets to Discord via the webhook in DISCORD_WEBHOOK."""
from __future__ import annotations

import os

import requests

from common import DATA, get_logger, read_json
from summary import bet_label

log = get_logger("notify")


def build_payload(pred: dict, site_url: str | None = None, n: int = 5) -> dict:
    top = pred.get("top_bets", [])[:n]
    fields = []
    for i, t in enumerate(top, 1):
        b = t["best_bet"]
        fields.append({
            "name": f"{i}. {t['home']} v {t['away']} ({t['league_name']}, {t['date']} {t.get('time') or ''})".strip(),
            "value": (f"**{bet_label(b, t['home'], t['away'])}** @ {b['odds']:.2f}\n"
                      f"Model {b['model_p']:.0%} vs market {b['market_p']:.0%} | edge {b['edge']:+.1%} | "
                      f"EV {b['ev']:+.1%} | confidence {t['confidence']}/100"),
            "inline": False})
    desc = (f"Top {len(top)} value bets from {pred.get('n_games', 0)} fixtures across "
            f"{pred.get('n_leagues', 0)} leagues." if top else "No bets cleared the value threshold this run.")
    embed = {"title": "Football model: weekly value bets", "description": desc, "fields": fields,
             "color": 0x2ECC71, "footer": {"text": "Statistical model, not financial advice. Bet responsibly."}}
    if site_url:
        embed["url"] = site_url
    return {"username": "football-model", "embeds": [embed]}


def run(pred: dict | None = None, site_url: str | None = None) -> bool:
    hook = os.environ.get("DISCORD_WEBHOOK")
    if not hook:
        log.info("DISCORD_WEBHOOK not set; skipping Discord post")
        return False
    pred = pred or read_json(DATA / "predictions.json", {})
    try:
        r = requests.post(hook, json=build_payload(pred, site_url), timeout=20)
        if r.status_code >= 300:
            log.warning("Discord responded %s: %s", r.status_code, r.text[:200])
            return False
        log.info("posted %d bets to Discord", len(pred.get("top_bets", [])[:5]))
        return True
    except requests.RequestException as e:
        log.warning("Discord post failed: %s", e)
        return False


if __name__ == "__main__":
    run(site_url=os.environ.get("SITE_URL"))
