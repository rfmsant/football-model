"""International (national-team) model for Nations League / qualifiers / friendlies.

Elo for every national team from every international result since 1872 (martj42/international_results, topped up
with recent results from API-Football), a Poisson goal model fitted on Elo difference + home advantage, Dixon-Coles
scoreline matrix, per-market recalibration of the goal markets, then a blend with the bookmakers' no-vig
probabilities for every market (MARKET_BLEND) and the same candidate-bet logic as the club model.

    python scripts/intl.py --date 2026-10-02            # bets for that day's Nations League games
    python scripts/intl.py --date 2026-10-02 --validate # also print walk-forward accuracy + calibration checks
    python scripts/intl.py --fetch-results [--max-calls 40]   # top up recent results (needs APIFOOTBALL_KEY)
    python scripts/intl.py --tune-blend                 # re-tune MARKET_BLEND from logged predictions
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
# Weight of the bookmakers' no-vig probabilities in the final numbers, for every market. It was 0.5 (a guess) and
# only applied to 1X2 and over/under 2.5. There is no history of international odds to tune it on yet, so it follows
# the tuned club value (market_weight 0.8 in data/params.json): the Elo model is cruder than the club model, so it
# shouldn't get more say. `python scripts/intl.py --tune-blend` re-tunes it from data/intl_predictions.csv once
# enough logged games have finished.
MARKET_BLEND = 0.8
PRED_LOG = DATA / "intl_predictions.csv"
MIN_TUNE_GAMES = 150


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


# ---------------------------------------------------------------- results
# The upstream file is updated by hand and can lag by weeks (on 6 Oct 2026 it stopped at 26 Aug). Without recent
# results the Elo ratings go stale and the cards can't be settled, so finished games are topped up from API-Football
# into data/intl_results_extra.csv (tracked in git; rows can also be added by hand, same columns as upstream).
EXTRA_RESULTS = DATA / "intl_results_extra.csv"
FETCHED = DATA / "intl_results_fetched.json"
COLUMNS = ["date", "home_team", "away_team", "home_score", "away_score", "tournament", "city", "country", "neutral"]
AF_LEAGUES = {1: "FIFA World Cup", 4: "UEFA Euro", 5: "UEFA Nations League", 6: "African Cup of Nations",
              7: "AFC Asian Cup", 9: "Copa América", 10: "Friendly", 22: "Gold Cup",
              29: "FIFA World Cup qualification", 30: "FIFA World Cup qualification", 31: "FIFA World Cup qualification",
              32: "FIFA World Cup qualification", 33: "FIFA World Cup qualification", 34: "FIFA World Cup qualification",
              536: "CONCACAF Nations League", 960: "UEFA Euro qualification"}
FINALS = {1, 4, 6, 7, 9, 22}                 # tournament finals: treat as neutral venue
EXILED_HOME = {"Ukraine", "Israel", "Russia"}  # home games currently played abroad
AF_ALIASES = {"Ireland": "Republic of Ireland", "Czechia": "Czech Republic", "Türkiye": "Turkey", "USA": "United States",
              "Korea Republic": "South Korea", "Korea DPR": "North Korea", "Cape Verde Islands": "Cape Verde",
              "Congo DR": "DR Congo", "Bosnia & Herzegovina": "Bosnia and Herzegovina", "Macedonia": "North Macedonia",
              "FYR Macedonia": "North Macedonia", "Chinese Taipei": "Taiwan", "Curacao": "Curaçao",
              "Côte d'Ivoire": "Ivory Coast", "Swaziland": "Eswatini", "Brunei Darussalam": "Brunei",
              "Kyrgyz Republic": "Kyrgyzstan", "St. Kitts and Nevis": "Saint Kitts and Nevis", "St. Lucia": "Saint Lucia",
              "St. Vincent / Grenadines": "Saint Vincent and the Grenadines"}


def _read_results(path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")
    return df


def merge_results(base: pd.DataFrame, extra: pd.DataFrame | None) -> pd.DataFrame:
    """Upstream results plus our top-up rows. A top-up row is dropped when upstream already has the same pairing
    within a day (upstream catches up eventually, sometimes dated a day apart because of time zones)."""
    if extra is None or extra.empty:
        return base.sort_values("date").reset_index(drop=True)
    keep = []
    for r in extra.itertuples(index=False):
        same = base[(base.home_team == r.home_team) & (base.away_team == r.away_team)]
        keep.append(not ((same.date - r.date).abs() <= pd.Timedelta(days=1)).any())
    df = pd.concat([base, extra[keep]], ignore_index=True)
    return df.sort_values("date", kind="stable").reset_index(drop=True)


def load_results() -> pd.DataFrame:
    f = CACHE / "results.csv"
    r = http_get(RESULTS_URL, timeout=60)
    if r is not None:
        f.write_bytes(r.content)
    base = _read_results(f)
    extra = _read_results(EXTRA_RESULTS) if EXTRA_RESULTS.exists() else None
    df = merge_results(base, extra)
    log.info("international results: upstream to %s, %d top-up rows used, latest %s",
             base.date.max().date(), len(df) - len(base), df.date.max().date())
    return df


def _team(name: str, teams: set) -> str | None:
    name = AF_ALIASES.get(name, name)
    return name if name in teams else match_name(name, teams, cutoff=0.9)


def parse_fixtures(resp: list, teams: set) -> list[dict]:
    """API-Football /fixtures response -> upstream-style result rows (finished national-team games only)."""
    rows = []
    for f in resp or []:
        lg = (f.get("league") or {}).get("id")
        if lg not in AF_LEAGUES:
            continue
        st = ((f.get("fixture") or {}).get("status") or {}).get("short")
        gh, ga = (f.get("goals") or {}).get("home"), (f.get("goals") or {}).get("away")
        if st not in ("FT", "AET", "PEN") or gh is None or ga is None:
            continue
        h, a = _team(f["teams"]["home"]["name"], teams), _team(f["teams"]["away"]["name"], teams)
        if not h or not a:
            log.warning("international top-up: unknown team in %s v %s", f["teams"]["home"]["name"], f["teams"]["away"]["name"])
            continue
        venue = f["fixture"].get("venue") or {}
        rows.append({"date": f["fixture"]["date"][:10], "home_team": h, "away_team": a, "home_score": int(gh),
                     "away_score": int(ga), "tournament": AF_LEAGUES[lg], "city": venue.get("city") or "",
                     "country": "", "neutral": lg in FINALS or h in EXILED_HOME})
    return rows


def append_results(rows: list[dict]) -> int:
    """Add rows to the top-up file, skipping duplicates. Returns the number added."""
    if not rows:
        return 0
    old = pd.read_csv(EXTRA_RESULTS, dtype=str) if EXTRA_RESULTS.exists() else pd.DataFrame(columns=COLUMNS)
    have = set(zip(old.date, old.home_team, old.away_team))
    new = [r for r in rows if (r["date"], r["home_team"], r["away_team"]) not in have]
    if new:
        out = pd.concat([old, pd.DataFrame(new, columns=COLUMNS).astype(str)], ignore_index=True)
        out["neutral"] = out["neutral"].str.upper()
        out.sort_values(["date", "home_team"]).to_csv(EXTRA_RESULTS, index=False, encoding="utf-8")
    return len(new)


def update_recent(api=None, today: dt.date | None = None, max_calls: int = 10) -> int:
    """Top up finished internationals after the upstream file's last date, newest days first, at most `max_calls`
    API calls per run (one per day). A day counts as done once it has been fetched 2+ days after it."""
    import json
    if api is None:
        from extras import ApiFootball
        api = ApiFootball(os.environ.get("APIFOOTBALL_KEY") or None, budget=max_calls)
    if api.disabled:
        log.info("international top-up skipped: %s", api.error)
        return 0
    today = today or dt.date.today()
    f = CACHE / "results.csv"
    if not f.exists():
        load_results()
    base = _read_results(f)
    teams = set(base.home_team) | set(base.away_team)
    start = base.date.max().date() + dt.timedelta(days=1)
    done = set(json.loads(FETCHED.read_text()) if FETCHED.exists() else [])
    added, calls = 0, 0
    for k in range(1, (today - start).days + 1):
        d = today - dt.timedelta(days=k)
        if d.isoformat() in done:
            continue
        if calls >= max_calls:
            break
        resp = api.get("fixtures", date=d.isoformat())
        calls += 1
        if resp is None:
            break
        added += append_results(parse_fixtures(resp, teams))
        if k >= 2:
            done.add(d.isoformat())
    FETCHED.write_text(json.dumps(sorted(done), indent=0))
    log.info("international top-up: %d API calls, %d results added", calls, added)
    return added


# ---------------------------------------------------------------- Elo + goal model
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


# ---------------------------------------------------------------- calibration of the goal markets
# Walk-forward check (goal model + calibration fitted before 2024, tested on 2,011 competitive games 2024-26): the raw
# Elo/Poisson goal markets are off: over 1.5 too high (73% predicted v 68.5% actual at the 70-75% level), the
# strongest under-3.5 calls too high (78% v 67%), BTTS-no too low (72% v 91%). Per-market Platt scaling fixes most
# of it (log loss on the test games: over 1.5 0.5845 -> 0.5800, over 2.5 0.6844 -> 0.6822, over 3.5
# 0.6181 -> 0.6163); `--validate` prints the before/after numbers.
CAL_MARKETS = ("O15", "O25", "O35", "BTTS")
APPLY_CAL = ("O15", "O25", "O35")   # BTTS: calibration made test log loss slightly worse (0.6787 -> 0.6800), so off
CAL_SINCE = "2010-01-01"


def _goal_probs(mat) -> dict:
    i, j = np.indices(mat.shape)
    t = i + j
    return {"O15": float(mat[t > 1.5].sum()), "O25": float(mat[t > 2.5].sum()), "O35": float(mat[t > 3.5].sum()),
            "BTTS": float(mat[(i > 0) & (j > 0)].sum())}


def _outcomes(hs: int, as_: int) -> dict:
    return {"O15": hs + as_ > 1, "O25": hs + as_ > 2, "O35": hs + as_ > 3, "BTTS": hs > 0 and as_ > 0}


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit_logistic(p, y) -> tuple[float, float]:
    """Platt scaling: P(y) = sigmoid(a + b*logit(p)), Newton-Raphson."""
    x = _logit(np.asarray(p, float))
    X = np.column_stack([np.ones_like(x), x])
    y = np.asarray(y, float)
    w = np.array([0.0, 1.0])
    for _ in range(50):
        mu = 1 / (1 + np.exp(-(X @ w)))
        H = X.T @ (X * (mu * (1 - mu))[:, None]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(H, X.T @ (y - mu))
        w = w + step
        if np.abs(step).max() < 1e-8:
            break
    return float(w[0]), float(w[1])


def apply_cal(p: float, ab: tuple[float, float] | None) -> float:
    if not ab:
        return p
    return float(1 / (1 + math.exp(-(ab[0] + ab[1] * float(_logit(p))))))


def raw_goal_table(df: pd.DataFrame, beta, start: str, end: str | None = None) -> pd.DataFrame:
    """Raw model goal-market probabilities and outcomes for competitive games in [start, end)."""
    t = df[(df.date >= start) & (df.tournament != "Friendly")]
    if end:
        t = t[t.date < end]
    rows = []
    for r in t.itertuples(index=False):
        lh, la = lambdas(beta, r.elo_h, r.elo_a, r.neutral)
        p = _goal_probs(dixon_coles_matrix(lh, la, RHO))
        y = _outcomes(int(r.home_score), int(r.away_score))
        rows.append({**{f"p_{k}": p[k] for k in CAL_MARKETS}, **{f"y_{k}": float(y[k]) for k in CAL_MARKETS}})
    return pd.DataFrame(rows)


def fit_calibration(df: pd.DataFrame, beta, start: str = CAL_SINCE, end: str | None = None) -> dict:
    tab = raw_goal_table(df, beta, start, end)
    if len(tab) < 300:
        return {}
    return {k: fit_logistic(tab[f"p_{k}"].to_numpy(), tab[f"y_{k}"].to_numpy()) for k in CAL_MARKETS}


def _ll(p, y) -> float:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration_report(df: pd.DataFrame, split: str = "2024-01-01") -> dict:
    """Fit the goal model and the calibration on games before `split`; evaluate on competitive games after it.
    For each market: log loss raw v calibrated, and [mean predicted, actual, n] for predictions in 65-80%."""
    beta = fit_goal_model(df[df.date < split])
    cal = fit_calibration(df, beta, end=split)
    test = raw_goal_table(df, beta, split)
    out = {"test_games": len(test)}
    for k in CAL_MARKETS:
        p, y = test[f"p_{k}"].to_numpy(), test[f"y_{k}"].to_numpy()
        pc = np.array([apply_cal(v, cal.get(k)) for v in p])
        res = {"log_loss_raw": round(_ll(p, y), 4), "log_loss_cal": round(_ll(pc, y), 4)}
        for name, q, yy in (("yes", p, y), ("no", 1 - p, 1 - y), ("yes_cal", pc, y), ("no_cal", 1 - pc, 1 - y)):
            band = (q >= 0.65) & (q < 0.80)
            res[f"{name}_65_80"] = [round(float(q[band].mean()), 3), round(float(yy[band].mean()), 3), int(band.sum())] if band.any() else None
        out[k] = res
    return out


# ---------------------------------------------------------------- odds + predictions
INTL_SPORTS = {"soccer_uefa_nations_league": "Nations League",
               "soccer_uefa_euro_qualification": "Euro qualifiers",
               "soccer_fifa_world_cup_qualifiers_europe": "World Cup qualifiers",
               "soccer_fifa_world_cup": "World Cup", "soccer_uefa_european_championship": "Euro"}
_ODDS_CACHE: dict = {}
_MODEL_CACHE: dict = {}


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


def fitted_model() -> tuple:
    """(results with pre-match Elo, current Elo, goal model, calibration), built once per process."""
    if not _MODEL_CACHE:
        df, elo = compute_elo(load_results())
        beta = fit_goal_model(df)
        cal = {k: v for k, v in fit_calibration(df, beta).items() if k in APPLY_CAL}
        log.info("goal model: %s; calibration %s; data up to %s", np.round(beta, 3),
                 {k: tuple(round(x, 3) for x in v) for k, v in cal.items()}, df.date.max().date())
        _MODEL_CACHE["m"] = (df, elo, beta, cal)
    return _MODEL_CACHE["m"]


def game_probs(beta, cal: dict, elo_h: float, elo_a: float, odds: dict, w: float = MARKET_BLEND) -> dict | None:
    """Model (Elo goal model, goal markets recalibrated) blended with the bookmakers for every market.
    The bookmakers' goal markets other than O/U 2.5 come from the goal expectations that reproduce their 1X2 and
    O/U 2.5 prices (the same derivation shortlist.py uses for estimated prices)."""
    import shortlist
    x = odds.get("1X2", {})
    if not all(s in x for s in "HDA"):
        return None
    lh, la = lambdas(beta, elo_h, elo_a)
    mat = dixon_coles_matrix(lh, la, RHO)
    i, j = np.indices(mat.shape)
    pm = {"H": float(mat[i > j].sum()), "D": float(mat[i == j].sum()), "A": float(mat[i < j].sum())}
    gm = {k: apply_cal(v, cal.get(k)) for k, v in _goal_probs(mat).items()}
    imp = np.array([1 / x[s]["avg"] for s in "HDA"])
    mk = dict(zip("HDA", (imp / imp.sum()).tolist()))
    ou = odds.get("O/U 2.5", {})
    mo25 = (1 / ou["Over"]["avg"]) / (1 / ou["Over"]["avg"] + 1 / ou["Under"]["avg"]) if "Over" in ou and "Under" in ou else None
    mm = shortlist._mk(dixon_coles_matrix(*shortlist.fit_market_lambdas(mk["H"], mk["A"], mo25), shortlist.RHO))
    if mo25 is not None:
        mm["O25"] = mo25
    blend = lambda m, b: (1 - w) * m + w * b  # noqa: E731
    return {"lh": lh, "la": la, "model_1x2": pm, "market_1x2": mk, "model_goals": gm, "market_goals": mm,
            "probs": {s: blend(pm[s], mk[s]) for s in "HDA"},
            "goals": {k: blend(gm[k], mm[k]) for k in CAL_MARKETS}}


def run(date: str, do_validate: bool = False) -> list[dict]:
    import shortlist
    from oddsapi import parse_event
    df, elo, beta, cal = fitted_model()
    if do_validate:
        print("walk-forward check (competitive internationals since 2024):", validate(df, beta))
        import json
        print("goal-market calibration (fit before 2024, test 2024+):")
        print(json.dumps(calibration_report(df), indent=1))
    teams = set(elo)
    out = []
    for ev in odds_for(date):
        h, a = match_name(ev["home_team"], teams), match_name(ev["away_team"], teams)
        if not h or not a:
            log.warning("no Elo for %s v %s", ev["home_team"], ev["away_team"])
            continue
        odds = parse_event(ev)
        gp = game_probs(beta, cal, elo[h], elo[a], odds)
        if gp is None:
            continue
        x, ou = odds["1X2"], odds.get("O/U 2.5", {})
        evals = [{"market": "1X2", "selection": s, "avg_odds": x[s]["avg"], "odds": x[s]["max"]} for s in "HDA"]
        if "Over" in ou and "Under" in ou:
            evals += [{"market": "O/U 2.5", "selection": s, "avg_odds": ou[s]["avg"], "odds": ou[s]["max"]} for s in ("Over", "Under")]
        gb = gp["goals"]
        g = {"id": f"INT-{date}-{h}-{a}", "league": "INT", "league_name": ev.get("_competition", "International"), "date": date,
             "time": pd.Timestamp(ev["commence_time"]).tz_convert("Europe/Lisbon").strftime("%H:%M"), "home": h, "away": a,
             "probs": gp["probs"], "xg": {"home": gp["lh"], "away": gp["la"]},
             "markets": {"O/U 2.5": {"Over": gb["O25"]}, "BTTS": {"Yes": gb["BTTS"]}},
             "goal_probs": {"O15": gb["O15"], "O35": gb["O35"]},
             "evals": evals, "elo": (round(elo[h]), round(elo[a])), "model_only": gp["model_1x2"],
             "_log": {"model_1x2": gp["model_1x2"], "market_1x2": gp["market_1x2"], "model_goals": gp["model_goals"],
                      "market_goals": gp["market_goals"]}}
        cands = shortlist.candidates_for(g)
        out.append({"game": g, "hard": shortlist.is_hard(g), "candidates": cands})
    return out


def log_predictions(games: list[dict]) -> None:
    """Append model-only and market probabilities for every priced game to data/intl_predictions.csv (latest
    snapshot per game wins), so MARKET_BLEND can be tuned once results come in."""
    rows = []
    for g in games:
        lg = g["_log"]
        rows.append({"date": g["date"], "home": g["home"], "away": g["away"],
                     **{f"model_{s}": round(lg["model_1x2"][s], 4) for s in "HDA"},
                     **{f"market_{s}": round(lg["market_1x2"][s], 4) for s in "HDA"},
                     **{f"model_{k}": round(lg["model_goals"][k], 4) for k in CAL_MARKETS},
                     **{f"market_{k}": round(lg["market_goals"][k], 4) for k in CAL_MARKETS}})
    if not rows:
        return
    new = pd.DataFrame(rows)
    old = pd.read_csv(PRED_LOG) if PRED_LOG.exists() else pd.DataFrame(columns=new.columns)
    out = pd.concat([old, new], ignore_index=True).drop_duplicates(["date", "home", "away"], keep="last")
    out.sort_values(["date", "home"]).to_csv(PRED_LOG, index=False)


def tune_blend(min_games: int = MIN_TUNE_GAMES) -> dict:
    """Market weight that minimises log loss over logged games that have finished (1X2 + the four goal markets)."""
    if not PRED_LOG.exists():
        return {"games": 0, "note": "no logged predictions yet"}
    lg = pd.read_csv(PRED_LOG, parse_dates=["date"])
    res = load_results()
    m = lg.merge(res[["date", "home_team", "away_team", "home_score", "away_score"]],
                 left_on=["date", "home", "away"], right_on=["date", "home_team", "away_team"])
    out = {"games": len(m), "current": MARKET_BLEND}
    if len(m) < min_games:
        out["note"] = f"need {min_games} finished games to tune; keeping {MARKET_BLEND}"
        return out
    hs, as_ = m.home_score.to_numpy(), m.away_score.to_numpy()
    y1 = np.select([hs > as_, hs == as_], ["H", "D"], "A")
    ys = {"O15": hs + as_ > 1, "O25": hs + as_ > 2, "O35": hs + as_ > 3, "BTTS": (hs > 0) & (as_ > 0)}
    grid = {}
    for w in np.round(np.arange(0, 1.01, 0.05), 2):
        p1 = np.array([(1 - w) * m[f"model_{s}"].to_numpy()[k] + w * m[f"market_{s}"].to_numpy()[k] for k, s in enumerate(y1)])
        ll = -np.mean(np.log(np.clip(p1, 1e-6, 1)))
        ll += sum(_ll((1 - w) * m[f"model_{k}"] + w * m[f"market_{k}"], ys[k]) for k in CAL_MARKETS)
        grid[float(w)] = round(float(ll), 4)
    out["grid"] = grid
    out["best"] = min(grid, key=grid.get)
    return out


def export(dates: list[str]) -> dict:
    """Write data/intl.json: international games + candidate bets for the given dates (used by card.py)."""
    import json
    try:
        update_recent()
    except Exception as e:  # noqa: BLE001
        log.warning("international results top-up failed: %s", e)
    games, logged = [], []
    for d in dates:
        for r in run(d):
            g = r["game"]
            logged.append(g)
            games.append({**{k: g[k] for k in ("id", "league", "league_name", "date", "time", "home", "away", "probs", "xg", "elo")},
                          "hard": r["hard"], "candidates": r["candidates"]})
    try:
        log_predictions(logged)
    except Exception as e:  # noqa: BLE001
        log.warning("prediction log failed: %s", e)
    out = {"generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "dates": dates,
           "market_blend": MARKET_BLEND, "games": games}
    (DATA / "intl.json").write_text(json.dumps(out, default=float, indent=1), encoding="utf-8")
    log.info("intl.json: %d games for %s", len(games), dates)
    return out


if __name__ == "__main__":
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=(dt.date.today() + dt.timedelta(days=1)).isoformat())
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--fetch-results", action="store_true")
    ap.add_argument("--max-calls", type=int, default=10)
    ap.add_argument("--tune-blend", action="store_true")
    a = ap.parse_args()
    if a.fetch_results:
        print("results added:", update_recent(max_calls=a.max_calls))
        raise SystemExit(0)
    if a.tune_blend:
        print(json.dumps(tune_blend(), indent=1))
        raise SystemExit(0)
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
