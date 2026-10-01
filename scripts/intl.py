"""International (national-team) model for Nations League / qualifiers / friendlies.

Elo for every national team from every international result since 1872 (martj42/international_results),
a Poisson goal model fitted on Elo difference + home advantage, Dixon-Coles scoreline matrix, then the same
market logic as the club model (no-vig odds, 50/50 blend with the bookmakers, candidate bets and rules).

    python scripts/intl.py --date 2026-10-02            # bets for that day's Nations League games
    python scripts/intl.py --date 2026-10-02 --validate # also print a walk-forward accuracy check
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import os

import numpy as np
import pandas as pd

from common import DATA, RAW, get_logger, http_get, match_name
from model import dixon_coles_matrix

log = get_logger("intl")
RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
CACHE = RAW / "intl"
CACHE.mkdir(parents=True, exist_ok=True)
RHO = -0.05
MARKET_BLEND = 0.5


def k_factor(tournament: str) -> float:
    t = tournament.lower()
    if t == "fifa world cup":
        return 60
    if t in ("uefa euro", "copa américa", "african cup of nations", "afc asian cup", "gold cup", "confederations cup"):
        return 50
    if "qualification" in t or "nations league" in t:
        return 40
    if t == "friendly":
        return 20
    return 30


def load_results() -> pd.DataFrame:
    f = CACHE / "results.csv"
    r = http_get(RESULTS_URL, timeout=60)
    if r is not None:
        f.write_bytes(r.content)
    df = pd.read_csv(f, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")
    return df.sort_values("date").reset_index(drop=True)


def compute_elo(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    elo: dict = {}
    pre_h, pre_a = np.empty(len(df)), np.empty(len(df))
    for i, r in enumerate(df.itertuples(index=False)):
        eh, ea = elo.get(r.home_team, 1500.0), elo.get(r.away_team, 1500.0)
        pre_h[i], pre_a[i] = eh, ea
        adv = 0 if r.neutral else 100
        we = 1 / (1 + 10 ** (-(eh + adv - ea) / 400))
        gd = r.home_score - r.away_score
        w = 1.0 if gd > 0 else 0.5 if gd == 0 else 0.0
        g = 1 if abs(gd) <= 1 else 1.5 if abs(gd) == 2 else (11 + abs(gd)) / 8
        d = k_factor(r.tournament) * g * (w - we)
        elo[r.home_team], elo[r.away_team] = eh + d, ea - d
    df = df.copy()
    df["elo_h"], df["elo_a"] = pre_h, pre_a
    return df, elo


def fit_goal_model(df: pd.DataFrame, since="2012-01-01") -> np.ndarray:
    """Poisson regression: log(goals) = a + b*(elo diff/400) + c*home + d*friendly, fitted on both sides of each match."""
    m = df[df.date >= since]
    d = ((m.elo_h - m.elo_a) / 400).to_numpy()
    home = (~m.neutral).astype(float).to_numpy()
    fr = (m.tournament == "Friendly").astype(float).to_numpy()
    X = np.vstack([np.column_stack([np.ones(len(m)), d, home, fr]), np.column_stack([np.ones(len(m)), -d, -home * 0, fr])])
    # away side gets no home term (the -home*0 keeps the column for a clean design matrix)
    y = np.concatenate([m.home_score.to_numpy(float), m.away_score.to_numpy(float)])
    beta = np.array([0.2, 0.8, 0.2, 0.0])
    for _ in range(25):  # Newton-Raphson
        mu = np.exp(X @ beta)
        grad = X.T @ (y - mu)
        hess = X.T @ (X * mu[:, None])
        beta = beta + np.linalg.solve(hess, grad)
    return beta


def lambdas(beta, elo_h, elo_a, neutral=False, friendly=False):
    d = (elo_h - elo_a) / 400
    lh = math.exp(beta[0] + beta[1] * d + (0 if neutral else beta[2]) + beta[3] * friendly)
    la = math.exp(beta[0] - beta[1] * d + beta[3] * friendly)
    return lh, la


def validate(df: pd.DataFrame, beta, since="2024-01-01") -> dict:
    t = df[(df.date >= since) & (df.tournament != "Friendly")]
    ll, hit, n = 0.0, 0, 0
    for r in t.itertuples(index=False):
        lh, la = lambdas(beta, r.elo_h, r.elo_a, r.neutral)
        mat = dixon_coles_matrix(lh, la, RHO)
        i, j = np.indices(mat.shape)
        p = {"H": mat[i > j].sum(), "D": mat[i == j].sum(), "A": mat[i < j].sum()}
        res = "H" if r.home_score > r.away_score else "D" if r.home_score == r.away_score else "A"
        ll -= math.log(max(p[res], 1e-9))
        hit += max(p, key=p.get) == res
        n += 1
    return {"matches": n, "winner_correct": round(hit / n, 4), "log_loss": round(ll / n, 4)}


INTL_SPORTS = {"soccer_uefa_nations_league": "Nations League",
               "soccer_uefa_euro_qualification": "Euro qualifiers",
               "soccer_fifa_world_cup_qualifiers_europe": "World Cup qualifiers",
               "soccer_fifa_world_cup": "World Cup", "soccer_uefa_european_championship": "Euro"}
_ODDS_CACHE: dict = {}


def odds_for(date: str) -> list[dict]:
    """Odds for every international competition with games on `date`. Event lists are free; odds (2 credits per
    competition) are only fetched when that competition has games in the window, and cached per run."""
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        return []
    out = []
    for sport, name in INTL_SPORTS.items():
        if sport not in _ODDS_CACHE:
            ev = http_get(f"https://api.the-odds-api.com/v4/sports/{sport}/events", params={"apiKey": key}, retries=1)
            evs = ev.json() if ev is not None and ev.status_code == 200 and isinstance(ev.json(), list) else []
            if not any(e["commence_time"][:10] == date for e in evs):
                continue
            r = http_get(f"https://api.the-odds-api.com/v4/sports/{sport}/odds",
                         params={"apiKey": key, "regions": "eu", "markets": "h2h,totals", "oddsFormat": "decimal"})
            _ODDS_CACHE[sport] = r.json() if r is not None and r.status_code == 200 else []
            log.info("Odds API %s: %d events, credits left %s", sport, len(_ODDS_CACHE[sport]),
                     r.headers.get("x-requests-remaining") if r is not None else "?")
        out += [dict(e, _competition=name) for e in _ODDS_CACHE.get(sport, []) if e["commence_time"][:10] == date]
    return out


def run(date: str, do_validate: bool = False) -> list[dict]:
    import shortlist
    from oddsapi import parse_event
    df, elo = compute_elo(load_results())
    beta = fit_goal_model(df)
    log.info("goal model: %s; data up to %s", np.round(beta, 3), df.date.max().date())
    if do_validate:
        print("walk-forward check (competitive internationals since 2024):", validate(df, beta))
    teams = set(elo)
    out = []
    for ev in odds_for(date):
        h, a = match_name(ev["home_team"], teams), match_name(ev["away_team"], teams)
        if not h or not a:
            log.warning("no Elo for %s v %s", ev["home_team"], ev["away_team"])
            continue
        lh, la = lambdas(beta, elo[h], elo[a])
        mat = dixon_coles_matrix(lh, la, RHO)
        i, j = np.indices(mat.shape)
        pm = {"H": float(mat[i > j].sum()), "D": float(mat[i == j].sum()), "A": float(mat[i < j].sum())}
        o25 = float(mat[(i + j) > 2.5].sum())
        btts = float(mat[(i > 0) & (j > 0)].sum())
        odds = parse_event(ev)
        x = odds.get("1X2", {})
        if not all(s in x for s in "HDA"):
            continue
        imp = np.array([1 / x[s]["avg"] for s in "HDA"])
        mk = imp / imp.sum()
        probs = {s: (1 - MARKET_BLEND) * pm[s] + MARKET_BLEND * float(mk[k]) for k, s in enumerate("HDA")}
        ou = odds.get("O/U 2.5", {})
        if "Over" in ou and "Under" in ou:
            mo = (1 / ou["Over"]["avg"]) / (1 / ou["Over"]["avg"] + 1 / ou["Under"]["avg"])
            o25 = (1 - MARKET_BLEND) * o25 + MARKET_BLEND * mo
        evals = [{"market": "1X2", "selection": s, "avg_odds": x[s]["avg"], "odds": x[s]["max"]} for s in "HDA"]
        if "Over" in ou and "Under" in ou:
            evals += [{"market": "O/U 2.5", "selection": s, "avg_odds": ou[s]["avg"], "odds": ou[s]["max"]} for s in ("Over", "Under")]
        g = {"id": f"INT-{date}-{h}-{a}", "league": "INT", "league_name": ev.get("_competition", "International"), "date": date,
             "time": pd.Timestamp(ev["commence_time"]).tz_convert("Europe/Lisbon").strftime("%H:%M"), "home": h, "away": a, "probs": probs,
             "xg": {"home": lh, "away": la}, "markets": {"O/U 2.5": {"Over": o25}, "BTTS": {"Yes": btts}},
             "evals": evals, "elo": (round(elo[h]), round(elo[a])), "model_only": pm}
        cands = shortlist.candidates_for(g)
        out.append({"game": g, "hard": shortlist.is_hard(g), "candidates": cands})
    return out


def export(dates: list[str]) -> dict:
    """Write data/intl.json: international games + candidate bets for the given dates (used by card.py)."""
    import json
    games = []
    for d in dates:
        for r in run(d):
            g = r["game"]
            games.append({**{k: g[k] for k in ("id", "league", "league_name", "date", "time", "home", "away", "probs", "xg", "elo")},
                          "hard": r["hard"], "candidates": r["candidates"]})
    out = {"generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "dates": dates, "games": games}
    (DATA / "intl.json").write_text(json.dumps(out, default=float, indent=1), encoding="utf-8")
    log.info("intl.json: %d games for %s", len(games), dates)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=(dt.date.today() + dt.timedelta(days=1)).isoformat())
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    res = run(a.date, a.validate)
    for r in res:
        g = r["game"]
        p = g["probs"]
        print(f"\n{g['time']} {g['home']} v {g['away']}  Elo {g['elo'][0]}-{g['elo'][1]}  "
              f"H {p['H']:.0%} D {p['D']:.0%} A {p['A']:.0%}  xG {g['xg']['home']:.2f}-{g['xg']['away']:.2f}"
              f"  over2.5 {g['markets']['O/U 2.5']['Over']:.0%}{'  [hard to call]' if r['hard'] else ''}")
        for c in sorted(r["candidates"], key=lambda c: -c["p_model"]):
            if c["p_model"] >= 0.6:
                bar = "MEETS BAR" if c["p_model"] >= 0.70 and c["odds"] >= 1.40 else ""
                print(f"   {c['label']:34s} p={c['p_model']:.0%} mkt={c['p_market']:.0%} @{c['odds']} ({c['odds_source'][:4]}) EV {c['ev']:+.1%} {bar}")
