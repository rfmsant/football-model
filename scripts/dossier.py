"""Deep per-game dossier: every metric the free data offers, for the shortlisted games.

Sections: model, market (incl. line movement), table & context, detailed form, scoring patterns,
venue records, head-to-head, referee, weather/travel, Elo trend, and - for the top 5 leagues -
Understat team style (situations, game state, timing, formation) and individual player form
(season vs last 5 matches: minutes, goals, xG, assists, xA, shots, key passes, xGChain).
"""
from __future__ import annotations

import datetime as dt
import json
import math
import time
from urllib.parse import quote

import numpy as np
import pandas as pd

from common import DATA, LEAGUES, RAW, get_logger, http_get, match_name, norm_name, season_start_year

log = get_logger("dossier")
CACHE = RAW / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
STADIUMS = DATA / "stadiums.json"
COUNTRY_CODE = {"ENG": "GB", "SCO": "GB", "GER": "DE", "ITA": "IT", "ESP": "ES", "FRA": "FR", "NED": "NL",
                "BEL": "BE", "POR": "PT", "TUR": "TR", "GRE": "GR"}


def _num(x):
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None


def _r(x, n=2):
    v = _num(x)
    return None if v is None else round(v, n)


# ---------------------------------------------------------------- cached JSON fetch
def cached_json(url: str, key: str, max_age_h: float = 12, headers=None):
    f = CACHE / f"{key}.json"
    if f.exists() and (time.time() - f.stat().st_mtime) < max_age_h * 3600:
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except ValueError:
            pass
    time.sleep(0.3)  # be polite to free services
    r = http_get(url, headers=headers, retries=2, timeout=25)
    if r is None or r.status_code != 200:
        return None
    try:
        data = r.json()
    except ValueError:
        return None
    f.write_text(json.dumps(data), encoding="utf-8")
    return data


US_HEADERS = {"X-Requested-With": "XMLHttpRequest"}


def understat_team(league: str, team: str, season: str) -> dict | None:
    key = LEAGUES[league][3]
    if not key:
        return None
    year = season_start_year(season)
    lg = cached_json(f"https://understat.com/getLeagueData/{key}/{year}", f"us_league_{key}_{year}", 12, US_HEADERS)
    if not lg:
        return None
    titles = [t["title"] for t in lg.get("teams", {}).values()]
    title = match_name(team, titles)
    if not title:
        return None
    data = cached_json(f"https://understat.com/getTeamData/{quote(title.replace(' ', '_'))}/{year}",
                       f"us_team_{norm_name(title).replace(' ', '_')}_{year}", 12, US_HEADERS)
    if data:
        data["_title"] = title
        tinfo = next((t for t in lg["teams"].values() if t["title"] == title), None)
        data["_history"] = tinfo.get("history", []) if tinfo else []
    return data


def understat_player_last(pid: str, n: int = 5) -> list[dict]:
    d = cached_json(f"https://understat.com/getPlayerData/{pid}", f"us_player_{pid}", 12, US_HEADERS)
    if not d:
        return []
    ms = sorted(d.get("matches", []), key=lambda m: m["date"], reverse=True)
    return ms[:n]


# ---------------------------------------------------------------- team sections
def _team_matches(hist: pd.DataFrame, team: str, before: pd.Timestamp) -> pd.DataFrame:
    m = hist[((hist.HomeTeam == team) | (hist.AwayTeam == team)) & (hist.Date < before)].sort_values("Date")
    return m


def _persp(r, team) -> dict:
    home = r.HomeTeam == team
    g = lambda h, a: (getattr(r, h), getattr(r, a)) if home else (getattr(r, a), getattr(r, h))  # noqa: E731
    gf, ga = g("FTHG", "FTAG")
    hgf, hga = g("HTHG", "HTAG")
    xgf, xga = g("HxG", "AxG")
    sf, sa = g("HS", "AS")
    sotf, sota = g("HST", "AST")
    cf, ca = g("HC", "AC")
    yf, _ = g("HY", "AY")
    rf, _ = g("HR", "AR")
    res = "W" if gf > ga else "D" if gf == ga else "L"
    return {"date": str(r.Date.date()), "venue": "H" if home else "A", "opponent": r.AwayTeam if home else r.HomeTeam,
            "competition": r.Div, "score": f"{int(gf)}-{int(ga)}", "ht": None if _num(hgf) is None else f"{int(hgf)}-{int(hga)}",
            "result": res, "gf": int(gf), "ga": int(ga), "ht_gf": _num(hgf), "ht_ga": _num(hga),
            "xg_for": _r(xgf), "xg_against": _r(xga), "shots": _num(sf), "shots_against": _num(sa),
            "sot": _num(sotf), "sot_against": _num(sota), "corners": _num(cf), "corners_against": _num(ca),
            "yellows": _num(yf), "reds": _num(rf)}


def _agg(rows: list[dict]) -> dict:
    if not rows:
        return {}
    n = len(rows)

    def mean(k):
        v = [r[k] for r in rows if r.get(k) is not None]
        return round(float(np.mean(v)), 2) if v else None

    def share(f):
        return round(sum(1 for r in rows if f(r)) / n, 2)
    return {"games": n, "ppg": round(sum(3 if r["result"] == "W" else 1 if r["result"] == "D" else 0 for r in rows) / n, 2),
            "record": f"{sum(r['result'] == 'W' for r in rows)}W-{sum(r['result'] == 'D' for r in rows)}D-{sum(r['result'] == 'L' for r in rows)}L",
            "goals_for": mean("gf"), "goals_against": mean("ga"), "xg_for": mean("xg_for"), "xg_against": mean("xg_against"),
            "shots": mean("shots"), "shots_against": mean("shots_against"), "sot": mean("sot"), "sot_against": mean("sot_against"),
            "corners": mean("corners"), "corners_against": mean("corners_against"), "yellows": mean("yellows"),
            "clean_sheet_pct": share(lambda r: r["ga"] == 0), "failed_to_score_pct": share(lambda r: r["gf"] == 0),
            "btts_pct": share(lambda r: r["gf"] > 0 and r["ga"] > 0),
            "over15_pct": share(lambda r: r["gf"] + r["ga"] > 1.5), "over25_pct": share(lambda r: r["gf"] + r["ga"] > 2.5),
            "over35_pct": share(lambda r: r["gf"] + r["ga"] > 3.5),
            "first_half_goals_for": mean("ht_gf"), "first_half_goals_against": mean("ht_ga")}


def _game_state(rows: list[dict]) -> dict:
    """Half-time leads converted, comebacks from half-time deficits."""
    led = [r for r in rows if r["ht_gf"] is not None and r["ht_gf"] > r["ht_ga"]]
    trailed = [r for r in rows if r["ht_gf"] is not None and r["ht_gf"] < r["ht_ga"]]
    return {"ht_leads": len(led), "ht_leads_won": sum(r["result"] == "W" for r in led),
            "ht_deficits": len(trailed), "points_from_ht_deficits": sum(3 if r["result"] == "W" else 1 if r["result"] == "D" else 0 for r in trailed),
            "second_half_goals_for": round(float(np.mean([r["gf"] - r["ht_gf"] for r in rows if r["ht_gf"] is not None])), 2) if rows else None,
            "second_half_goals_against": round(float(np.mean([r["ga"] - r["ht_ga"] for r in rows if r["ht_ga"] is not None])), 2) if rows else None}


def team_section(hist, league, team, side, date, season, elo_hist) -> dict:
    ms = _team_matches(hist, team, date)
    rows = [_persp(r, team) for r in ms.itertuples(index=False)]
    cur = [r for r, (_, m) in zip(rows, ms.iterrows()) if m.Season == season and m.Div == league]
    venue_rows = [r for r in cur if r["venue"] == ("H" if side == "home" else "A")]
    last = rows[-6:][::-1]
    out = {"name": team, "last6": last, "last5": _agg(rows[-5:]), "last10": _agg(rows[-10:]), "season": _agg(cur),
           "season_venue": _agg(venue_rows), "game_state_season": _game_state(cur)}
    if out["season"] and out["last5"] and out["season"].get("xg_for") is not None and out["last5"].get("xg_for") is not None:
        out["trend_xgd_last5_vs_season"] = round((out["last5"]["xg_for"] - out["last5"]["xg_against"]) -
                                                 (out["season"]["xg_for"] - out["season"]["xg_against"]), 2)
    if rows:
        out["rest_days"] = (date - pd.Timestamp(rows[-1]["date"])).days
        out["games_last_14d"] = sum(1 for r in rows if (date - pd.Timestamp(r["date"])).days <= 14)
        out["games_last_30d"] = sum(1 for r in rows if (date - pd.Timestamp(r["date"])).days <= 30)
    e = elo_hist.get(team)
    if e:
        now = e[-1][1]
        past = next((v for d, v in reversed(e) if (date - d).days >= 30), e[0][1])
        out["elo"] = round(now)
        out["elo_change_30d"] = round(now - past)
    return out


def table_section(hist, league, season, date, home, away) -> dict:
    m = hist[(hist.Div == league) & (hist.Season == season) & (hist.Date < date)]
    t = {}
    for r in m.itertuples(index=False):
        for team, gf, ga in ((r.HomeTeam, r.FTHG, r.FTAG), (r.AwayTeam, r.FTAG, r.FTHG)):
            x = t.setdefault(team, [0, 0, 0, 0])
            x[0] += 1
            x[1] += 3 if gf > ga else 1 if gf == ga else 0
            x[2] += gf - ga
            x[3] += gf
    ranked = sorted(t.items(), key=lambda kv: (-kv[1][1], -kv[1][2], -kv[1][3]))
    n = len(ranked)
    europe, releg = LEAGUES[league][6]
    pts = [v[1] for _, v in ranked]

    def row(team):
        for i, (tm, v) in enumerate(ranked):
            if tm == team:
                return {"pos": i + 1, "of": n, "played": v[0], "points": v[1], "gd": v[2],
                        "ppg": round(v[1] / v[0], 2) if v[0] else None,
                        "gap_to_top": pts[0] - v[1], "gap_to_europe": (pts[europe - 1] - v[1]) if n >= europe else None,
                        "gap_above_relegation": (v[1] - pts[n - releg]) if n > releg else None}
        return None
    return {"home": row(home), "away": row(away), "teams": n}


def h2h_section(hist, home, away, date) -> dict:
    m = hist[(((hist.HomeTeam == home) & (hist.AwayTeam == away)) | ((hist.HomeTeam == away) & (hist.AwayTeam == home)))
             & (hist.Date < date)].sort_values("Date", ascending=False).head(8)
    games = [{"date": str(r.Date.date()), "home": r.HomeTeam, "away": r.AwayTeam, "score": f"{int(r.FTHG)}-{int(r.FTAG)}",
              "competition": r.Div} for r in m.itertuples(index=False)]
    hw = sum(1 for g, r in zip(games, m.itertuples(index=False)) if (r.HomeTeam == home and r.FTHG > r.FTAG) or (r.AwayTeam == home and r.FTAG > r.FTHG))
    aw = sum(1 for g, r in zip(games, m.itertuples(index=False)) if (r.HomeTeam == away and r.FTHG > r.FTAG) or (r.AwayTeam == away and r.FTAG > r.FTHG))
    goals = [r.FTHG + r.FTAG for r in m.itertuples(index=False)]
    return {"games": games, "home_wins": hw, "away_wins": aw, "draws": len(games) - hw - aw,
            "avg_goals": round(float(np.mean(goals)), 2) if goals else None,
            "over25_pct": round(float(np.mean([g > 2.5 for g in goals])), 2) if goals else None}


def referee_section(hist, league, referee) -> dict | None:
    if not referee or (isinstance(referee, float) and math.isnan(referee)):
        return None
    lg = hist[hist.Div == league]
    m = hist[hist.Referee == referee]
    if m.empty:
        return {"name": referee, "games": 0}

    def stats(x):
        cards = (x.HY + x.AY + 2 * (x.HR + x.AR))
        return {"games": int(len(x)), "cards_per_game": _r(cards.mean()), "reds_per_game": _r((x.HR + x.AR).mean(), 3),
                "fouls_per_game": _r((x.HF + x.AF).mean()), "home_win_pct": _r((x.FTHG > x.FTAG).mean()),
                "draw_pct": _r((x.FTHG == x.FTAG).mean()), "goals_per_game": _r((x.FTHG + x.FTAG).mean()),
                "over25_pct": _r(((x.FTHG + x.FTAG) > 2.5).mean()), "home_cards": _r((x.HY + 2 * x.HR).mean()),
                "away_cards": _r((x.AY + 2 * x.AR).mean())}
    return {"name": referee, **stats(m), "league_average": stats(lg)}


# ---------------------------------------------------------------- weather & travel
def _load_stadiums():
    try:
        return json.loads(STADIUMS.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def geocode(team: str, league: str) -> dict | None:
    st = _load_stadiums()
    key = f"{league}|{team}"
    if key in st:
        return st[key]
    cc = COUNTRY_CODE.get(LEAGUES[league][1])
    tried = []
    for q in [norm_name(team), team.split()[0], norm_name(team).split()[0]]:
        if not q or q in tried:
            continue
        tried.append(q)
        d = cached_json(f"https://geocoding-api.open-meteo.com/v1/search?name={quote(q)}&count=5",
                        f"geo_{q.replace(' ', '_')}", 24 * 90)
        res = [x for x in (d or {}).get("results", []) if not cc or x.get("country_code") == cc]
        if res:
            best = max(res, key=lambda x: x.get("population") or 0)
            loc = {"query": q, "place": best["name"], "lat": best["latitude"], "lon": best["longitude"], "approx": True}
            st[key] = loc
            STADIUMS.write_text(json.dumps(st, indent=1, ensure_ascii=False), encoding="utf-8")
            return loc
    return None


def weather_section(loc: dict | None, date: str, time_str: str | None) -> dict | None:
    if not loc:
        return None
    d = cached_json(f"https://api.open-meteo.com/v1/forecast?latitude={loc['lat']}&longitude={loc['lon']}"
                    f"&hourly=temperature_2m,precipitation,precipitation_probability,wind_speed_10m,wind_gusts_10m"
                    f"&timezone=auto&start_date={date}&end_date={date}", f"wx_{loc['lat']}_{loc['lon']}_{date}", 6)
    if not d or "hourly" not in d:
        return None
    hour = int((time_str or "15:00")[:2])
    h = d["hourly"]
    idx = min(range(len(h["time"])), key=lambda i: abs(int(h["time"][i][11:13]) - hour))
    w = {k: h[k][idx] for k in h if k != "time"}
    w["time"] = h["time"][idx]
    w["place"] = loc.get("place")
    notes = []
    if (w.get("precipitation") or 0) >= 2:
        notes.append("heavy rain")
    if (w.get("wind_speed_10m") or 0) >= 30 or (w.get("wind_gusts_10m") or 0) >= 55:
        notes.append("strong wind")
    if (w.get("temperature_2m") or 15) >= 30:
        notes.append("very hot")
    if (w.get("temperature_2m") or 15) <= 0:
        notes.append("freezing")
    w["impact"] = ", ".join(notes) or "none expected"
    return w


def haversine_km(a, b) -> float | None:
    if not a or not b:
        return None
    la1, lo1, la2, lo2 = map(math.radians, (a["lat"], a["lon"], b["lat"], b["lon"]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return round(2 * 6371 * math.asin(math.sqrt(h)))


# ---------------------------------------------------------------- Understat style & players
def style_section(us: dict | None) -> dict | None:
    if not us:
        return None
    st = us.get("statistics", {})
    out = {}
    sit = st.get("situation", {})
    tot_g = sum(v.get("goals", 0) for v in sit.values()) or 1
    tot_ga = sum(v.get("against", {}).get("goals", 0) for v in sit.values()) or 1
    out["goals_by_situation"] = {k: {"goals": v.get("goals"), "xG": _r(v.get("xG")), "share": round(v.get("goals", 0) / tot_g, 2),
                                     "conceded": v.get("against", {}).get("goals"), "xGA": _r(v.get("against", {}).get("xG")),
                                     "conceded_share": round(v.get("against", {}).get("goals", 0) / tot_ga, 2)}
                                 for k, v in sit.items()}
    out["game_state"] = {k: {"minutes": v.get("time"), "xG": _r(v.get("xG")), "xGA": _r(v.get("against", {}).get("xG"))}
                         for k, v in st.get("gameState", {}).items()}
    out["timing"] = {k: {"goals": v.get("goals"), "conceded": v.get("against", {}).get("goals")} for k, v in st.get("timing", {}).items()}
    fm = st.get("formation", {})
    if fm:
        main = max(fm.items(), key=lambda kv: kv[1].get("time", 0))
        out["main_formation"] = main[0]
        out["formations_used"] = {k: v.get("time") for k, v in fm.items()}
    hist = us.get("_history", [])
    if hist:
        n = len(hist)
        out["xpts"] = round(sum(float(h["xpts"]) for h in hist), 1)
        out["pts"] = sum(int(h["pts"]) for h in hist)
        out["luck_pts_minus_xpts"] = round(out["pts"] - out["xpts"], 1)
        out["ppda"] = round(sum(h["ppda"]["att"] for h in hist) / max(sum(h["ppda"]["def"] for h in hist), 1), 2)
        out["ppda_allowed"] = round(sum(h["ppda_allowed"]["att"] for h in hist) / max(sum(h["ppda_allowed"]["def"] for h in hist), 1), 2)
        out["deep_completions_pg"] = round(sum(h["deep"] for h in hist) / n, 1)
        out["deep_allowed_pg"] = round(sum(h["deep_allowed"] for h in hist) / n, 1)
    return out


def players_section(us: dict | None, top_n: int = 10) -> list[dict] | None:
    """Key players: season numbers, share of team output, and last-5-match form vs season."""
    if not us:
        return None
    ps = us.get("players", [])
    if not ps:
        return None
    team_min = max(float(p["time"]) for p in ps) or 1
    team_xg = sum(float(p["xG"]) for p in ps) or 1
    team_xa = sum(float(p["xA"]) for p in ps) or 1
    ps = sorted(ps, key=lambda p: -(float(p["xGChain"]) if p.get("xGChain") else float(p["time"])))[:top_n]
    out = []
    for p in ps:
        mins = float(p["time"])
        per90 = lambda v: round(float(v) / mins * 90, 2) if mins else None  # noqa: E731
        row = {"name": p["player_name"], "position": p.get("position"), "games": int(p["games"]), "minutes": int(mins),
               "minutes_share": round(mins / team_min, 2) if team_min else None,
               "goals": int(p["goals"]), "xG": _r(p["xG"]), "npxG": _r(p.get("npxG")), "assists": int(p["assists"]), "xA": _r(p["xA"]),
               "shots": int(p["shots"]), "key_passes": int(p["key_passes"]), "xGChain": _r(p.get("xGChain")), "xGBuildup": _r(p.get("xGBuildup")),
               "yellow": int(p.get("yellow_cards", 0)), "red": int(p.get("red_cards", 0)),
               "share_of_team_xG": round(float(p["xG"]) / team_xg, 2), "share_of_team_xA": round(float(p["xA"]) / team_xa, 2),
               "season_xg_xa_per90": per90(float(p["xG"]) + float(p["xA"])),
               "finishing_goals_minus_xG": round(int(p["goals"]) - float(p["xG"]), 2)}
        last = understat_player_last(p["id"], 5)
        if last:
            lm = sum(float(m["time"]) for m in last)
            row["last5"] = [{"date": m["date"], "match": f"{m['h_team']} {m['h_goals']}-{m['a_goals']} {m['a_team']}",
                             "minutes": int(m["time"]), "goals": int(m["goals"]), "assists": int(m["assists"]),
                             "xG": _r(m["xG"]), "xA": _r(m["xA"]), "shots": int(m["shots"]), "key_passes": int(m["key_passes"])}
                            for m in last]
            row["last5_minutes"] = int(lm)
            row["last5_xg_xa_per90"] = round((sum(float(m["xG"]) + float(m["xA"]) for m in last)) / lm * 90, 2) if lm else None
            if row["last5_xg_xa_per90"] is not None and row["season_xg_xa_per90"]:
                ratio = row["last5_xg_xa_per90"] / max(row["season_xg_xa_per90"], 0.05)
                row["form"] = "hot" if ratio >= 1.3 else "cold" if ratio <= 0.7 else "steady"
            row["days_since_last_match"] = (dt.date.today() - dt.date.fromisoformat(last[0]["date"])).days
            row["started_recently"] = sum(1 for m in last if int(m["time"]) >= 60)
        out.append(row)
    return out


# ---------------------------------------------------------------- odds history (line movement)
ODDS_HISTORY = DATA / "odds_history.json"


def record_odds(games: list[dict], stamp: str) -> None:
    """Append today's prices for every upcoming game (used for line movement)."""
    try:
        oh = json.loads(ODDS_HISTORY.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        oh = {}
    today = stamp[:10]
    for g in games:
        k = f"{g['league']}|{g['date']}|{g['home']}|{g['away']}"
        snap = {"t": stamp}
        for e in g.get("evals", []):
            if e.get("avg_odds"):
                snap[f"{e['market']}|{e['selection']}"] = e["avg_odds"]
        if len(snap) > 1:
            hist = oh.setdefault(k, [])
            if not hist or hist[-1]["t"][:10] != today:
                hist.append(snap)
            else:
                hist[-1] = snap
    cutoff = (dt.date.today() - dt.timedelta(days=14)).isoformat()
    oh = {k: v for k, v in oh.items() if k.split("|")[1] >= cutoff}
    ODDS_HISTORY.write_text(json.dumps(oh, separators=(",", ":")), encoding="utf-8")


def line_movement(g: dict) -> dict | None:
    try:
        oh = json.loads(ODDS_HISTORY.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    h = oh.get(f"{g['league']}|{g['date']}|{g['home']}|{g['away']}")
    if not h or len(h) < 2:
        return {"snapshots": len(h or []), "note": "no movement yet (first price seen)"}
    first, last = h[0], h[-1]
    mv = {}
    for k in last:
        if k != "t" and k in first:
            mv[k] = {"open": first[k], "now": last[k], "change_pct": round((last[k] / first[k] - 1) * 100, 1)}
    steam = [k for k, v in mv.items() if v["change_pct"] <= -5]
    return {"snapshots": len(h), "since": first["t"], "moves": mv, "shortening_5pct_plus": steam}


# ---------------------------------------------------------------- assemble
def build(g: dict, hist: pd.DataFrame, fixture_row: dict, season: str, elo_hist: dict) -> dict:
    date = pd.Timestamp(g["date"])
    lg = g["league"]
    home_loc, away_loc = geocode(g["home"], lg), geocode(g["away"], lg)
    us_h = understat_team(lg, g["home"], season)
    us_a = understat_team(lg, g["away"], season)
    d = {
        "id": g["id"], "league": lg, "league_name": g["league_name"], "date": g["date"], "time": g.get("time"),
        "home": g["home"], "away": g["away"],
        "model": {"probs": g["probs"], "probs_dixon_coles": g.get("probs_dc"), "expected_goals": g["xg"],
                  "likely_scores": g.get("top_scores"), "pick": g.get("pick"),
                  "markets": g.get("markets"), "confidence": g.get("confidence")},
        "market": {"prices": [{k: e.get(k) for k in ("market", "selection", "avg_odds", "odds", "market_p", "model_p", "edge", "ev")}
                              for e in g.get("evals", []) if e.get("odds")],
                   "bookmaker_spread": g.get("book_spread"), "line_movement": line_movement(g)},
        "table": table_section(hist, lg, season, date, g["home"], g["away"]),
        "home": team_section(hist, lg, g["home"], "home", date, season, elo_hist),
        "away": team_section(hist, lg, g["away"], "away", date, season, elo_hist),
        "h2h": h2h_section(hist, g["home"], g["away"], date),
        "referee": referee_section(hist, lg, fixture_row.get("Referee")),
        "weather": weather_section(home_loc, g["date"], g.get("time")),
        "travel_km_away_team": haversine_km(home_loc, away_loc),
        "style": {"home": style_section(us_h), "away": style_section(us_a)},
        "players": {"home": players_section(us_h), "away": players_section(us_a)},
        "data_coverage": {"xg": g.get("has_xg"), "understat": bool(us_h and us_a),
                          "referee": bool(fixture_row.get("Referee") and not (isinstance(fixture_row.get("Referee"), float))),
                          "weather": home_loc is not None},
        "research_checklist": [
            "Confirmed injuries & suspensions for both teams (with expected return dates) and who replaces them",
            "Predicted line-ups (and confirmed XIs if within ~1h of kick-off); rotation risk from cup/European games",
            "Key players' recent form and ratings (FotMob/Sofascore/WhoScored pages) - especially outside the top 5 leagues",
            "Manager situation, press-conference quotes, dressing-room news, new signings",
            "Motivation: derby, title/relegation/Europe stakes, dead rubber, next fixture priorities",
            "Referee appointment and tendencies if not listed (non-UK leagues)",
            "Pitch/weather confirmation; travel; any off-field issues",
            "Latest odds and whether the price has moved since the model ran",
        ],
    }
    return d


def elo_history(hist_elo: pd.DataFrame) -> dict:
    """team -> [(date, pre-match elo)] (last ~60 days is enough for trend)."""
    out: dict = {}
    recent = hist_elo[hist_elo.Date >= hist_elo.Date.max() - pd.Timedelta(days=90)]
    for r in recent.itertuples(index=False):
        out.setdefault(r.HomeTeam, []).append((r.Date, r.EloH))
        out.setdefault(r.AwayTeam, []).append((r.Date, r.EloA))
    return out
