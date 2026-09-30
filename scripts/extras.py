"""Stage 2 adjustments for flagged games only.

* Injuries/suspensions (API-Football free tier, hard request budget) weighted by the player's
  minutes share and goal contribution.
* Rest days, fixture congestion, manager change and motivation (league-table stakes).
* Extra bookmaker odds from The Odds API.

Everything degrades gracefully: without API keys only the free, history-based factors are applied.
"""
from __future__ import annotations

import datetime as dt
import json
import os

import numpy as np
import pandas as pd

from common import LEAGUES, RAW, get_logger, http_get, match_name, season_code, season_start_year
from oddsapi import OddsClient, find_event, parse_event

log = get_logger("extras")

AF_BASE = "https://v3.football.api-sports.io"
MAX_ATT_LOSS, MAX_DEF_LOSS = 0.20, 0.20


# ================================================================ free, history-based context
def team_context(hist: pd.DataFrame, league: str, team: str, date: pd.Timestamp, season: str) -> dict:
    h = hist[(hist["Date"] < date) & ((hist["HomeTeam"] == team) | (hist["AwayTeam"] == team))]
    ctx = {"rest_days": None, "matches_21d": 0, "form": "", "form_pts": None}
    if h.empty:
        return ctx
    last = h["Date"].max()
    ctx["rest_days"] = int((date - last).days)
    ctx["matches_21d"] = int((h["Date"] >= date - pd.Timedelta(days=21)).sum())
    recent = h.sort_values("Date").tail(5)
    form = []
    for _, r in recent.iterrows():
        gf, ga = (r.FTHG, r.FTAG) if r.HomeTeam == team else (r.FTAG, r.FTHG)
        form.append("W" if gf > ga else "D" if gf == ga else "L")
    ctx["form"] = "".join(form)
    ctx["form_pts"] = sum(3 if f == "W" else 1 if f == "D" else 0 for f in form)
    return ctx


def league_table(hist: pd.DataFrame, league: str, season: str, before: pd.Timestamp) -> pd.DataFrame:
    m = hist[(hist["Div"] == league) & (hist["Season"] == season) & (hist["Date"] < before)]
    rows = {}
    for r in m.itertuples(index=False):
        for team, gf, ga in ((r.HomeTeam, r.FTHG, r.FTAG), (r.AwayTeam, r.FTAG, r.FTHG)):
            t = rows.setdefault(team, {"team": team, "P": 0, "Pts": 0, "GD": 0})
            t["P"] += 1
            t["GD"] += gf - ga
            t["Pts"] += 3 if gf > ga else 1 if gf == ga else 0
    if not rows:
        return pd.DataFrame(columns=["team", "P", "Pts", "GD", "Pos"])
    tab = pd.DataFrame(rows.values()).sort_values(["Pts", "GD"], ascending=False).reset_index(drop=True)
    tab["Pos"] = tab.index + 1
    return tab


def motivation(tab: pd.DataFrame, league: str, team: str) -> tuple[float, str | None]:
    """Returns (attack multiplier, description)."""
    if tab.empty or team not in set(tab["team"]):
        return 1.0, None
    n = len(tab)
    total_games = 2 * (n - 1) if league not in ("SC0", "SC1", "SC2", "SC3") else 36 if n <= 10 else 38
    row = tab[tab["team"] == team].iloc[0]
    played_frac = row.P / max(total_games, 1)
    remaining = max(total_games - row.P, 0)
    europe, releg = LEAGUES[league][6]
    pos = int(row.Pos)
    if played_frac < 0.6:
        return 1.0, f"{pos}{_ord(pos)} in the table"
    pts = tab["Pts"].to_numpy()
    safe_line = pts[n - releg - 1] if n > releg else 0
    top_line = pts[min(europe, n) - 1]
    max_gain = remaining * 3
    if pos > n - releg or row.Pts - safe_line <= 4:
        return 1.03, f"{pos}{_ord(pos)}, fighting relegation"
    if pos == 1 or (pts[0] - row.Pts) <= 4:
        return 1.03, f"{pos}{_ord(pos)}, in the title race"
    if pos <= europe or top_line - row.Pts <= 3:
        return 1.02, f"{pos}{_ord(pos)}, chasing a top-{europe} finish"
    if row.Pts - safe_line > max_gain * 0.6 and top_line - row.Pts > max_gain * 0.6:
        return 0.97, f"{pos}{_ord(pos)}, little left to play for"
    return 1.0, f"{pos}{_ord(pos)} in the table"


def _ord(n: int) -> str:
    return "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


# ================================================================ API-Football
class ApiFootball:
    def __init__(self, key: str | None, budget: int = 90):
        self.key = key
        self.budget = budget
        self.used = 0
        self.disabled = not key
        self.error = None if key else "APIFOOTBALL_KEY not set"

    def get(self, path: str, **params):
        if self.disabled or self.used >= self.budget:
            return None
        self.used += 1
        r = http_get(f"{AF_BASE}/{path}", params=params, headers={"x-apisports-key": self.key}, retries=2)
        if r is None:
            return None
        try:
            data = r.json()
        except ValueError:
            return None
        errs = data.get("errors")
        if errs:
            msg = json.dumps(errs)
            log.warning("API-Football %s: %s", path, msg)
            if any(w in msg.lower() for w in ("token", "suspended", "limit", "application key")):
                self.disabled, self.error = True, msg
            if "plan" in msg.lower():  # e.g. free plan without access to the current season
                self.disabled, self.error = True, msg
            return None
        remaining = r.headers.get("x-ratelimit-requests-remaining")
        if remaining is not None and remaining.isdigit() and int(remaining) <= 2:
            log.warning("API-Football daily quota almost exhausted")
            self.disabled = True
        return data.get("response")


def _understat_players(league: str, season: str) -> list[dict]:
    key = LEAGUES[league][3]
    if not key:
        return []
    f = RAW / f"understat_players_{key}_{season_start_year(season)}.json"
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []


def player_impact(minutes_share: float, contrib_share: float, position: str, questionable: bool) -> tuple[float, float]:
    """(attack loss, defence loss) as fractions. A replacement recovers roughly half the output."""
    w = 0.5 if questionable else 1.0
    pos = (position or "").upper()[:1]
    att = 0.5 * contrib_share * min(1.0, minutes_share / 0.6) + 0.02 * minutes_share
    dfn = {"G": 0.06, "D": 0.035, "M": 0.02}.get(pos, 0.005) * minutes_share
    return w * att, w * dfn


def injury_adjustments(api: ApiFootball, games: list[dict], hist: pd.DataFrame, season: str) -> dict:
    """Returns {game_id: {"home": {...}, "away": {...}}} with absences and multipliers."""
    out = {}
    if api.disabled:
        return out
    year = season_start_year(season)
    by_league: dict = {}
    for g in games:
        by_league.setdefault(g["league"], []).append(g)
    for lg, gs in by_league.items():
        dates = sorted(g["date"] for g in gs)
        fixtures = api.get("fixtures", league=LEAGUES[lg][4], season=year, **{"from": dates[0], "to": dates[-1]})
        if not fixtures:
            continue
        af_teams = {}
        for f in fixtures:
            for side in ("home", "away"):
                af_teams[f["teams"][side]["name"]] = f["teams"][side]["id"]
        us_players = _understat_players(lg, season)
        for g in gs:
            h, a = match_name(g["home"], af_teams), match_name(g["away"], af_teams)
            fx = next((f for f in fixtures if f["teams"]["home"]["name"] == h and f["teams"]["away"]["name"] == a), None)
            if not fx:
                log.info("API-Football: no fixture match for %s v %s", g["home"], g["away"])
                continue
            inj = api.get("injuries", fixture=fx["fixture"]["id"])
            if inj is None:
                continue
            res = {"fixture_id": fx["fixture"]["id"], "home": _side_impact(api, inj, fx["teams"]["home"], g["home"], lg, season, hist, us_players),
                   "away": _side_impact(api, inj, fx["teams"]["away"], g["away"], lg, season, hist, us_players),
                   "home_team_id": fx["teams"]["home"]["id"], "away_team_id": fx["teams"]["away"]["id"]}
            out[g["id"]] = res
    return out


def _side_impact(api, injuries, af_team, fd_team, league, season, hist, us_players) -> dict:
    cur = hist[(hist["Season"] == season) & ((hist["HomeTeam"] == fd_team) | (hist["AwayTeam"] == fd_team))]
    team_matches = max(len(cur), 1)
    team_goals = max(float(np.where(cur["HomeTeam"] == fd_team, cur["FTHG"], cur["FTAG"]).sum()), 1.0)
    us_team = [p for p in us_players if match_name(fd_team, {p["team_title"].split(",")[-1]}) is not None]
    us_names = {p["player_name"]: p for p in us_team}
    absences, att_loss, def_loss = [], 0.0, 0.0
    for item in injuries:
        if item["team"]["id"] != af_team["id"]:
            continue
        pl = item["player"]
        questionable = "questionable" in (pl.get("type") or "").lower()
        minutes = goals = assists = None
        position = ""
        m = match_name(pl["name"], us_names, cutoff=0.8) if us_names else None
        if m:
            p = us_names[m]
            minutes, goals, assists = float(p["time"]), float(p["goals"]), float(p["assists"])
            position = p.get("position", "")
        else:
            st = api.get("players", id=pl["id"], season=season_start_year(season))
            if st:
                stats = [s for s in st[0].get("statistics", []) if s["league"]["id"] == LEAGUES[league][4]] \
                    or st[0].get("statistics", [])[:1]
                if stats:
                    s = stats[0]
                    minutes = float(s["games"].get("minutes") or 0)
                    goals = float(s["goals"].get("total") or 0)
                    assists = float(s["goals"].get("assists") or 0)
                    position = s["games"].get("position") or ""
        if minutes is None:  # unknown player -> small generic impact
            a, d = (0.01, 0.005) if not questionable else (0.005, 0.0025)
            ms = cs = None
        else:
            ms = min(1.0, minutes / (team_matches * 90))
            cs = min(0.6, (goals + 0.7 * assists) / team_goals)
            a, d = player_impact(ms, cs, position, questionable)
        att_loss += a
        def_loss += d
        absences.append({"name": pl["name"], "reason": pl.get("reason") or pl.get("type"),
                         "status": "doubtful" if questionable else "out",
                         "minutes_share": None if ms is None else round(ms, 2),
                         "goal_share": None if cs is None else round(cs, 2),
                         "impact": round(a + d, 3)})
    absences.sort(key=lambda x: -x["impact"])
    return {"absences": absences, "att_mult": round(1 - min(att_loss, MAX_ATT_LOSS), 3),
            "def_mult": round(1 + min(def_loss, MAX_DEF_LOSS), 3)}


def manager_change(api: ApiFootball, team_id: int, date: pd.Timestamp) -> bool | None:
    coaches = api.get("coachs", team=team_id)
    if not coaches:
        return None
    for c in coaches:
        for job in c.get("career", []):
            if job.get("team", {}).get("id") == team_id and not job.get("end") and job.get("start"):
                start = pd.Timestamp(job["start"])
                return bool((date - start).days <= 60)
    return None


# ================================================================ The Odds API
def fetch_extra_odds(client: OddsClient, games: list[dict]) -> dict:
    """{game_id: {market: {selection: {"avg": x, "max": y}}}} for flagged games.
    Leagues are fetched in order of their most valuable flagged game until the credit budget is spent;
    leagues already fetched for fixture discovery cost nothing extra."""
    if not client.enabled:
        return {}
    out = {}
    prio: dict = {}
    for g in games:
        prio[g["league"]] = max(prio.get(g["league"], -9), g.get("priority") or 0)
    for lg in sorted(prio, key=lambda lg: (lg not in client.cache, -prio[lg])):
        events = client.league(lg)
        if not events:
            continue
        for g in (g for g in games if g["league"] == lg):
            ev = find_event(events, g["home"], g["away"])
            if ev:
                out[g["id"]] = parse_event(ev)
    return out


def merge_odds(base: dict, extra: dict) -> dict:
    out = {m: {s: dict(v) for s, v in sel.items()} for m, sel in base.items()}
    for m, sel in extra.items():
        if m not in out:
            if m == "1X2" and len(sel) < 3 or len(sel) < 2:
                continue
            out[m] = {s: {"avg": v["avg"], "max": v["max"]} for s, v in sel.items()}
        else:
            for s, v in sel.items():
                if s in out[m]:
                    out[m][s]["max"] = max(out[m][s]["max"], v["max"])
    return out


# ================================================================ orchestration
def run(games: list[dict], hist: pd.DataFrame, season: str | None = None, odds_client: OddsClient | None = None) -> dict:
    """games: [{id, league, home, away, date (YYYY-MM-DD), priority}]. Returns per-game adjustments."""
    season = season or season_code()
    odds_client = odds_client or OddsClient()
    api = ApiFootball(os.environ.get("APIFOOTBALL_KEY") or None,
                      budget=int(os.environ.get("APIFOOTBALL_BUDGET", "90")))
    try:
        injuries = injury_adjustments(api, games, hist, season)
    except Exception as e:  # noqa: BLE001
        log.warning("injury stage failed: %s", e)
        injuries = {}
    try:
        odds = fetch_extra_odds(odds_client, games)
    except Exception as e:  # noqa: BLE001
        log.warning("odds stage failed: %s", e)
        odds = {}

    out = {}
    tables = {}
    for g in games:
        date = pd.Timestamp(g["date"])
        lg = g["league"]
        if lg not in tables:
            tables[lg] = league_table(hist, lg, season, date)
        res = {"factors": [], "injury_data": g["id"] in injuries, "extra_odds": odds.get(g["id"], {})}
        mult = {"home": [1.0, 1.0], "away": [1.0, 1.0]}  # [attack, defence(conceded)]
        for side in ("home", "away"):
            team = g[side]
            ctx = team_context(hist, lg, team, date, season)
            res[f"{side}_context"] = ctx
            rd = ctx["rest_days"]
            if rd is not None and rd < 4:
                mult[side][0] *= 0.97
                mult[side][1] *= 1.03
                res["factors"].append(f"{team} on short rest ({rd} days)")
            if ctx["matches_21d"] >= 5:
                mult[side][0] *= 0.98
                mult[side][1] *= 1.02
                res["factors"].append(f"{team} congested schedule ({ctx['matches_21d']} games in 21 days)")
            mot, desc = motivation(tables[lg], lg, team)
            mult[side][0] *= mot
            res[f"{side}_table"] = desc
            if mot != 1.0 and desc:
                res["factors"].append(f"{team}: {desc}")
            inj = injuries.get(g["id"], {}).get(side)
            if inj:
                mult[side][0] *= inj["att_mult"]
                mult[side][1] *= inj["def_mult"]
                res[f"{side}_absences"] = inj["absences"]
                key_out = [a["name"] for a in inj["absences"] if a["impact"] >= 0.03]
                if key_out:
                    res["factors"].append(f"{team} missing key players: {', '.join(key_out[:3])}")
            tid = injuries.get(g["id"], {}).get(f"{side}_team_id")
            if tid and not api.disabled:
                mc = manager_change(api, tid, date)
                if mc:
                    mult[side][0] *= 1.03
                    res["factors"].append(f"{team} new manager bounce")
        # home lambda: home attack x away defence; away lambda: away attack x home defence
        res["lambda_mult"] = (round(mult["home"][0] * mult["away"][1], 4), round(mult["away"][0] * mult["home"][1], 4))
        out[g["id"]] = res
    log.info("extras: %d games, injuries for %d, extra odds for %d, API-Football requests used %d (%s)",
             len(games), len(injuries), len(odds), api.used, api.error or "ok")
    return {"games": out, "api_football_error": api.error, "api_football_used": api.used}
