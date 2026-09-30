"""Daily shortlist for the deep dive.

Rules (from the user):
  * look at tomorrow's games first; widen to later days until there are enough candidates;
  * drop games that are hard to call (finely balanced);
  * candidate bets must have probability >= MIN_PROB and odds >= MIN_ODDS (no 1.05 favourites);
  * keep ~TARGET candidates, ranked by expected value; each gets a full dossier (dossier.py).

Markets (European bookmakers): 1X2, double chance, draw no bet, over/under 1.5/2.5/3.5, BTTS.
Where the free data has no real price (double chance, DNB, O/U 1.5 & 3.5, BTTS) the bookmaker price is
ESTIMATED: a score model is fitted to the bookmakers' own 1X2 and O/U 2.5 prices and a normal margin is
added. Estimated prices are labelled; the deep dive gives a "bet only at odds >=" threshold instead.
"""
from __future__ import annotations

import datetime as dt
import math
import traceback

import numpy as np
import pandas as pd

import dossier
from common import DATA, get_logger, write_json
from model import dixon_coles_matrix

log = get_logger("shortlist")
FINAL_MIN_PROB = 0.70   # the bar a final bet must clear after the deep dive
MIN_PROB = 0.60         # shortlist: the model alone rarely shows 70% at 1.40+ (it is anchored to the market),
MIN_ODDS = 1.40         # so candidates are the bets research could plausibly lift over the bar
TARGET = 20
MAX_PER_GAME = 2
MAX_DAYS_AHEAD = 10
RHO = -0.08


def _mk(mat: np.ndarray) -> dict:
    i, j = np.indices(mat.shape)
    t = i + j
    pH, pD, pA = float(mat[i > j].sum()), float(mat[i == j].sum()), float(mat[i < j].sum())
    return {"H": pH, "D": pD, "A": pA, "O15": float(mat[t > 1.5].sum()), "O25": float(mat[t > 2.5].sum()),
            "O35": float(mat[t > 3.5].sum()), "BTTS": float(mat[(i > 0) & (j > 0)].sum())}


def fit_market_lambdas(pH: float, pA: float, pO25: float | None) -> tuple[float, float]:
    """Home/away goal expectations that reproduce the bookmakers' no-vig 1X2 (and O/U 2.5) prices."""
    best, bl = 9e9, (1.4, 1.1)
    for tot in np.arange(1.6, 4.6, 0.1):
        for sup in np.arange(-2.5, 2.55, 0.1):
            lh, la = (tot + sup) / 2, (tot - sup) / 2
            if lh < 0.15 or la < 0.15:
                continue
            m = _mk(dixon_coles_matrix(lh, la, RHO))
            err = (m["H"] - pH) ** 2 + (m["A"] - pA) ** 2 + ((m["O25"] - pO25) ** 2 if pO25 else 0)
            if err < best:
                best, bl = err, (lh, la)
    lh0, la0 = bl
    for lh in np.arange(lh0 - 0.06, lh0 + 0.061, 0.02):           # refine
        for la in np.arange(la0 - 0.06, la0 + 0.061, 0.02):
            if lh < 0.1 or la < 0.1:
                continue
            m = _mk(dixon_coles_matrix(lh, la, RHO))
            err = (m["H"] - pH) ** 2 + (m["A"] - pA) ** 2 + ((m["O25"] - pO25) ** 2 if pO25 else 0)
            if err < best:
                best, bl = err, (lh, la)
    return bl


def model_goal_probs(g: dict) -> dict:
    """O/U 1.5/3.5 and BTTS consistent with the ML over-2.5 probability: scale the model's expected goals
    until the Dixon-Coles over-2.5 matches the ML value."""
    lh, la = g["xg"]["home"], g["xg"]["away"]
    target = (g["markets"].get("O/U 2.5") or {}).get("Over")
    k = 1.0
    if target:
        lo, hi = 0.4, 2.5
        for _ in range(30):
            k = (lo + hi) / 2
            if _mk(dixon_coles_matrix(lh * k, la * k, RHO))["O25"] < target:
                lo = k
            else:
                hi = k
    m = _mk(dixon_coles_matrix(lh * k, la * k, RHO))
    btts = (g["markets"].get("BTTS") or {}).get("Yes")
    return {"O15": m["O15"], "O25": target or m["O25"], "O35": m["O35"], "BTTS": btts if btts is not None else m["BTTS"]}


def _real(g: dict, market: str, sel: str):
    e = next((e for e in g.get("evals", []) if e["market"] == market and e["selection"] == sel and e.get("avg_odds")), None)
    return (e["avg_odds"], e["odds"]) if e else (None, None)


def candidates_for(g: dict) -> list[dict]:
    p = g["probs"]
    H, D, A = p["H"], p["D"], p["A"]
    oh, oa, od = (_real(g, "1X2", s)[0] for s in "HAD")
    if not (oh and od and oa):
        return []
    imp = np.array([1 / oh, 1 / od, 1 / oa])
    overround = float(imp.sum())
    mH, mD, mA = imp / overround
    ov = _real(g, "O/U 2.5", "Over")[0]
    un = _real(g, "O/U 2.5", "Under")[0]
    mO25 = (1 / ov) / (1 / ov + 1 / un) if ov and un else None
    mlh, mla = fit_market_lambdas(mH, mA, mO25)
    mm = _mk(dixon_coles_matrix(mlh, mla, RHO))
    margin = max(overround, 1.05)
    est = lambda q: round(1 / (q * margin), 2) if q > 0 else None  # noqa: E731
    gp = model_goal_probs(g)
    home, away = g["home"], g["away"]
    rows = []

    def add(market, sel, label, prob, mkt_p, real=None, push=0.0):
        odds = real or est(mkt_p)
        if not odds:
            return
        ev = prob * (odds - 1) - (1 - prob - push)
        rows.append({"market": market, "selection": sel, "label": label, "p_model": round(prob, 4),
                     "p_market": round(mkt_p, 4), "odds": round(odds, 2), "odds_source": "real" if real else "estimated",
                     "ev": round(ev, 4), "edge": round(prob - mkt_p, 4), "fair_odds": round(1 / prob, 2) if prob else None})
    add("1X2", "H", f"{home} to win", H, mH, oh)
    add("1X2", "A", f"{away} to win", A, mA, oa)
    add("Double chance", "1X", f"{home} or draw", H + D, mH + mD)
    add("Double chance", "X2", f"{away} or draw", A + D, mA + mD)
    add("Double chance", "12", f"{home} or {away} (no draw)", H + A, mH + mA)
    # draw no bet: stake returned on a draw; probability shown is P(win | no draw)
    for s, pw, pl, mw, ml, team in (("H", H, A, mH, mA, home), ("A", A, H, mA, mH, away)):
        pc, mc = pw / (pw + pl), mw / (mw + ml)
        odds = round(1 / (mc * margin), 2)
        ev = pw * (odds - 1) - pl
        rows.append({"market": "Draw no bet", "selection": s, "label": f"{team} draw no bet", "p_model": round(pc, 4),
                     "p_market": round(mc, 4), "odds": odds, "odds_source": "estimated", "ev": round(ev, 4),
                     "edge": round(pc - mc, 4), "fair_odds": round(1 / pc, 2), "push_prob": round(D, 4)})
    for line, key in ((1.5, "O15"), (2.5, "O25"), (3.5, "O35")):
        ro, ru = _real(g, f"O/U {line}", "Over")[0], _real(g, f"O/U {line}", "Under")[0]
        mo = (1 / ro) / (1 / ro + 1 / ru) if ro and ru else mm[key]
        add(f"O/U {line}", "Over", f"Over {line} goals", gp[key], mo, ro)
        add(f"O/U {line}", "Under", f"Under {line} goals", 1 - gp[key], 1 - mo, ru)
    add("BTTS", "Yes", "Both teams to score", gp["BTTS"], mm["BTTS"])
    add("BTTS", "No", "Both teams NOT to score", 1 - gp["BTTS"], 1 - mm["BTTS"])
    for r in rows:
        r.update(game_id=g["id"], match=f"{home} v {away}", league=g["league"], league_name=g["league_name"],
                 date=g["date"], time=g.get("time"), pick_tier=(g.get("pick") or {}).get("tier"))
    return rows


def is_hard(g: dict) -> bool:
    """Finely balanced games: no side is a real favourite."""
    p = g["probs"]
    return abs(p["H"] - p["A"]) < 0.15 and max(p["H"], p["A"]) < 0.45


def select(records: list[dict], today: dt.date) -> tuple[list[dict], list[str], list[dict]]:
    chosen, dates_used, rejected = [], [], []
    for ahead in range(1, MAX_DAYS_AHEAD + 1):
        day = (today + dt.timedelta(days=ahead)).isoformat()
        games = [g for g in records if g["date"] == day]
        if not games:
            continue
        dates_used.append(day)
        pool = []
        for g in games:
            if is_hard(g):
                rejected.append({"match": f"{g['home']} v {g['away']}", "date": day, "reason": "hard to call (balanced game)"})
                continue
            try:
                cands = [c for c in candidates_for(g) if c["p_model"] >= MIN_PROB and c["odds"] >= MIN_ODDS]
            except Exception:  # noqa: BLE001
                log.warning("candidates failed for %s:\n%s", g["id"], traceback.format_exc(limit=2))
                continue
            for c in cands:
                c["gap_to_bar"] = round(max(0.0, FINAL_MIN_PROB - c["p_model"]), 4)
            cands.sort(key=lambda c: -c["ev"])
            pool += cands[:MAX_PER_GAME]
        pool.sort(key=lambda c: -c["ev"])
        chosen += pool
        if len(chosen) >= TARGET:
            break
    chosen.sort(key=lambda c: (c["date"], -c["ev"]))
    return chosen[:TARGET], dates_used, rejected


def run(records: list[dict], rows: dict, hist_elo: pd.DataFrame, season: str, today: dt.date, stamp: str) -> dict:
    try:
        dossier.record_odds(records, stamp)
    except Exception:  # noqa: BLE001
        log.warning("odds history failed:\n%s", traceback.format_exc(limit=2))
    cands, dates, rejected = select(records, today)
    by_id = {g["id"]: g for g in records}
    elo_hist = dossier.elo_history(hist_elo)
    hist = hist_elo
    dossiers = {}
    for c in cands:
        gid = c["game_id"]
        if gid in dossiers:
            continue
        try:
            dossiers[gid] = dossier.build(by_id[gid], hist, rows.get(gid, {}), season, elo_hist)
        except Exception:  # noqa: BLE001
            log.warning("dossier failed for %s:\n%s", gid, traceback.format_exc(limit=3))
    for k, c in enumerate(cands, 1):
        c["rank"] = k
    out = {"generated_at": stamp, "for_dates": dates, "rules": {"final_min_prob": FINAL_MIN_PROB, "candidate_min_prob": MIN_PROB, "min_odds": MIN_ODDS, "target": TARGET,
                                                                "max_per_game": MAX_PER_GAME},
           "n_candidates": len(cands), "candidates": cands, "rejected_hard_games": rejected, "dossiers": dossiers}
    write_json(DATA / "shortlist.json", _clean(out))
    log.info("shortlist: %d candidates from %s, %d dossiers", len(cands), dates, len(dossiers))
    return out


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if math.isnan(float(o)) else round(float(o), 4)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (pd.Timestamp, dt.date)):
        return str(o)
    return o
