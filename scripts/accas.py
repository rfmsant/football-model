"""Accumulator builder: combine single bets from DIFFERENT games into doubles/trebles.

Combined odds multiply, and so do the probabilities (legs from different games are treated as independent;
same-game legs are never combined because they are correlated). Draw-no-bet legs refund on a draw: a refunded
leg counts as odds 1.00. Rules: combined probability of not losing >= MIN_PROB and combined odds >= MIN_ODDS.
Ranked by expected value, so a combination of negative-value legs never looks better than it is.
"""
from __future__ import annotations

import itertools

MIN_PROB = 0.70
MIN_ODDS = 1.40


def leg_stats(c: dict) -> tuple[float, float, float]:
    """(probability the leg wins, probability it is refunded, probability it loses)."""
    if c["market"] == "Draw no bet":
        push = c.get("push_prob", 0.0)
        win = c["p_model"] * (1 - push)
        return win, push, 1 - win - push
    return c["p_model"], 0.0, 1 - c["p_model"]


def combine(legs: list[dict]) -> dict:
    odds = 1.0
    ev_mult = 1.0
    p_no_loss = 1.0
    for c in legs:
        w, push, lose = leg_stats(c)
        odds *= c["odds"]
        ev_mult *= w * c["odds"] + push           # expected return per unit for this leg
        p_no_loss *= 1 - lose
    return {"legs": [f"{c['match']}: {c['label']} @{c['odds']}" for c in legs], "n": len(legs),
            "odds": round(odds, 2), "p_not_lose": round(p_no_loss, 4), "ev": round(ev_mult - 1, 4),
            "estimated_prices": any(c.get("odds_source") == "estimated" for c in legs)}


def build(candidates: list[dict], max_legs: int = 3, min_leg_prob: float = 0.75, min_leg_ev: float = -0.03,
          top: int = 10) -> list[dict]:
    pool = [c for c in candidates if c["p_model"] >= min_leg_prob and c["ev"] >= min_leg_ev and c["odds"] > 1.0]
    out = []
    for k in range(2, max_legs + 1):
        for legs in itertools.combinations(pool, k):
            if len({c["match"] for c in legs}) < k:     # one leg per game
                continue
            a = combine(list(legs))
            if a["p_not_lose"] >= MIN_PROB and a["odds"] >= MIN_ODDS:
                out.append(a)
    out.sort(key=lambda a: (-a["ev"], -a["p_not_lose"]))
    return out[:top]
