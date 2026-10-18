"""Live data for the weekly Liga Portugal report: referee appointments and their record, kick-off changes and
postponements (Liga Portugal's own API), weather at the stadium for kick-off (Open-Meteo) and current prices
(the model's bookmaker feed, then ESPN). Writes data/portugal/live.json, which the site merges into the report.

    python scripts/portugal_live.py update      # refresh live.json for the games in data/portugal/latest.json

No API keys and no odds-API credits are used. Runs every few hours from .github/workflows/portugal-live.yml.
"""
from __future__ import annotations

import datetime as dt
import sys
import unicodedata

from common import DATA, get_logger, http_get, read_json, write_json

log = get_logger("portugal_live")
OUT = DATA / "portugal"
LP = "https://www.ligaportugal.pt/api/v1"
COMP = "ligaportugalbetclic"
UA = {"User-Agent": "Mozilla/5.0 (football-model; +https://github.com/rfmsant/football-model)", "Accept": "application/json"}
STOP = {"fc", "sc", "cd", "ac", "sl", "cp", "sp", "m", "futebol", "clube", "sad", "ud", "de"}
ALIAS = {"lisbon": "sporting", "guimaraes": "vitoria", "academico viseu": "academico"}

# home stadiums (lat, lon) for the forecast; approximate is fine for weather
STADIUMS = {
    "benfica": ("Estadio da Luz", 38.7527, -9.1847), "porto": ("Estadio do Dragao", 41.1618, -8.5839),
    "sporting": ("Estadio Jose Alvalade", 38.7611, -9.1608), "braga": ("Estadio Municipal de Braga", 41.5617, -8.4298),
    "vitoria": ("Estadio D. Afonso Henriques", 41.4459, -8.3006), "estrela": ("Estadio Jose Gomes", 38.7522, -9.2236),
    "alverca": ("Complexo Desportivo FC Alverca", 38.8956, -9.0393), "gil": ("Estadio Cidade de Barcelos", 41.5444, -8.6338),
    "estoril": ("Estadio Antonio Coimbra da Mota", 38.7135, -9.4013), "nacional": ("Estadio da Madeira", 32.6704, -16.8834),
    "rio": ("Estadio do Rio Ave", 41.3616, -8.7402), "santa": ("Estadio de Sao Miguel", 37.7489, -25.6631),
    "casa": ("Estadio Municipal de Rio Maior", 39.3397, -8.9367), "arouca": ("Estadio Municipal de Arouca", 40.9287, -8.2448),
    "moreirense": ("Estadio Comendador Joaquim de Almeida Freitas", 41.3812, -8.3473),
    "famalicao": ("Estadio Municipal 22 de Junho", 41.4061, -8.5161), "maritimo": ("Estadio do Maritimo", 32.6457, -16.9256),
    "academico": ("Estadio do Fontelo", 40.6571, -7.9023),
}


def key(name: str) -> str:
    """Club name -> short key shared by the report, Liga Portugal, the model and ESPN ('SC Braga' -> 'braga')."""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower().replace(".", " ")
    s = " ".join(t for t in s.split() if t not in STOP)
    s = ALIAS.get(s, s)
    return ALIAS.get(s.split()[0], s.split()[0]) if s else ""


def pair(match: str) -> tuple[str, str]:
    h, a = match.split(" v ", 1)
    return key(h), key(a)


def _get(url: str, **params):
    r = http_get(url, params=params or None, headers=UA, timeout=20)
    try:
        return r.json() if r is not None and r.status_code == 200 else None
    except ValueError:
        return None


# ---------------------------------------------------------------- Liga Portugal: fixtures and referees
def season_code(today: dt.date) -> str:
    y = today.year if today.month >= 7 else today.year - 1
    return f"{y}{y + 1}"


def lp_fixtures(season: str) -> dict:
    """{(home key, away key): fixture} for every round of the season that isn't fully played."""
    rounds = _get(f"{LP}/competition/season/rounds", competition=COMP, season=season) or []
    out = {}
    for r in rounds:
        if r.get("has_been_played"):
            continue
        for f in _get(f"{LP}/competition/matches", competition=COMP, season=season, round=r["round_number"]) or []:
            out[(key(f["homeTeam"]["name"]), key(f["awayTeam"]["name"]))] = f
    return out


def referee_record(ref_id: int, season: str) -> dict:
    """The referee's league games this season as main referee: cards, goals and home wins per game."""
    d = _get(f"{LP}/referee/{ref_id}/matches", season=season) or {}
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    games = [f for f in d.get("fixtures", []) if f.get("competitionName") == "Liga Portugal Betclic"
             and f.get("refereeType") in ("RefereePrincipal", "RefereeFPF") and str(f.get("fixtureDate", "9")) < now
             and not f.get("postponedFixture")]
    if not games:
        return {"games": 0}
    carded = [f for f in games if f.get("yellowCards") is not None]
    n = len(games)
    return {
        "games": n,
        "yellow_pg": round(sum(len(f["yellowCards"]) for f in carded) / len(carded), 1) if carded else None,
        "red_pg": round(sum(len(f.get("redCards") or []) for f in carded) / len(carded), 2) if carded else None,
        "goals_pg": round(sum(f["homeTeamGoals"] + f["awayTeamGoals"] for f in games) / n, 1),
        "home_win_pct": round(sum(f["homeTeamGoals"] > f["awayTeamGoals"] for f in games) / n, 2),
        "draw_pct": round(sum(f["homeTeamGoals"] == f["awayTeamGoals"] for f in games) / n, 2),
    }


def referees(fixtures: dict, season: str, wanted: set) -> dict:
    """{(home, away): referee dict} for the wanted games whose referee has been appointed."""
    out, rounds = {}, {f["round"] for k, f in fixtures.items() if k in wanted}
    for rnd in rounds:
        for n in _get(f"{LP}/competition/round/{rnd}/nominations", competition=COMP, season=season) or []:
            k = (key(n["fixture"]["homeTeam"]["name"]), key(n["fixture"]["awayTeam"]["name"]))
            if k in wanted and n.get("refereeName"):
                out[k] = {"name": n["refereeName"], "var": n.get("refereeVarName") or None,
                          **referee_record(n["refereeId"], season)}
    return out


# ---------------------------------------------------------------- weather (Open-Meteo, no key)
def weather(lat: float, lon: float, kickoff_utc: dt.datetime) -> dict | None:
    if kickoff_utc - dt.datetime.now(dt.timezone.utc) > dt.timedelta(days=15):
        return None
    d = _get("https://api.open-meteo.com/v1/forecast", latitude=lat, longitude=lon, timezone="UTC",
             hourly="temperature_2m,precipitation_probability,precipitation,wind_speed_10m,wind_gusts_10m,weather_code",
             start_date=kickoff_utc.date().isoformat(), end_date=kickoff_utc.date().isoformat())
    if not d or "hourly" not in d:
        return None
    hh = d["hourly"]
    i = min(range(len(hh["time"])), key=lambda j: abs(dt.datetime.fromisoformat(hh["time"][j]).replace(tzinfo=dt.timezone.utc) - kickoff_utc))
    w = {k: hh[k][i] for k in ("temperature_2m", "precipitation_probability", "precipitation", "wind_speed_10m", "wind_gusts_10m", "weather_code")}
    heavy = (w["precipitation"] or 0) >= 2 or (w["wind_gusts_10m"] or 0) >= 50
    w["impact"] = "heavy rain or strong wind: fewer goals, more errors" if heavy else \
        "some rain or wind" if (w["precipitation_probability"] or 0) >= 60 or (w["wind_speed_10m"] or 0) >= 30 else "no impact expected"
    return w


# ---------------------------------------------------------------- prices
def model_prices() -> dict:
    pred = read_json(DATA / "predictions.json", {}) or {}
    out = {}
    for g in pred.get("games", []):
        if g.get("league") != "P1":
            continue
        ev = {(e["market"], e["selection"]): e.get("avg_odds") or e.get("odds") for e in g.get("evals", [])}
        p = {"H": ev.get(("1X2", "H")), "D": ev.get(("1X2", "D")), "A": ev.get(("1X2", "A")),
             "over25": ev.get(("O/U 2.5", "Over")), "under25": ev.get(("O/U 2.5", "Under")), "source": "bookmaker average (model feed)"}
        if p["H"]:
            out[(key(g["home"]), key(g["away"]))] = p
    return out


def _dec(american) -> float | None:
    try:
        a = float(american)
    except (TypeError, ValueError):
        return None
    return round(1 + (a / 100 if a > 0 else 100 / -a), 2)


def espn_prices(start: dt.date, end: dt.date) -> dict:
    events = []
    for i in range((end - start).days + 1):             # ESPN rejects date ranges here; one call per day
        sb = _get("https://site.api.espn.com/apis/site/v2/sports/soccer/por.1/scoreboard",
                  dates=f"{start + dt.timedelta(days=i):%Y%m%d}") or {}
        events += sb.get("events", [])
    out = {}
    for e in events:
        comp = (e.get("competitions") or [{}])[0]
        teams = {c["homeAway"]: c["team"]["displayName"] for c in comp.get("competitors", [])}
        odds = (comp.get("odds") or [{}])[0]
        if not odds or "home" not in teams:
            continue
        ml, tot = odds.get("moneyline") or {}, odds.get("total") or {}
        cur = lambda side: ((side or {}).get("close") or (side or {}).get("open") or {}).get("odds")  # noqa: E731
        p = {"H": _dec(cur(ml.get("home")) or (odds.get("homeTeamOdds") or {}).get("moneyLine")),
             "A": _dec(cur(ml.get("away")) or (odds.get("awayTeamOdds") or {}).get("moneyLine")),
             "D": _dec(cur(ml.get("draw")) or (odds.get("drawOdds") or {}).get("moneyLine")), "over25": None, "under25": None,
             "source": f"{(odds.get('provider') or {}).get('name', 'ESPN')} via ESPN"}
        if odds.get("overUnder") == 2.5:
            p["over25"] = _dec(cur(tot.get("over")) or odds.get("overOdds"))
            p["under25"] = _dec(cur(tot.get("under")) or odds.get("underOdds"))
        if p["H"]:
            out[(key(teams["home"]), key(teams["away"]))] = p
    return out


def bet_price(b: dict, p: dict | None) -> dict | None:
    """Price for the best bet from 1X2 and over/under 2.5 prices; double chance and draw no bet are derived."""
    if not p:
        return None
    m, s = b["market"], b["selection"]
    H, D, A = p.get("H"), p.get("D"), p.get("A")
    try:
        if m == "1X2":
            x = {"H": H, "D": D, "A": A}[s]
            return {"price": x, "derived": False} if x else None
        if m == "O/U 2.5":
            x = p.get("over25" if s == "Over" else "under25")
            return {"price": x, "derived": False} if x else None
        if m == "Double chance" and H and D and A:
            a, c = {"1X": (H, D), "X2": (D, A), "12": (H, A)}[s]
            return {"price": round(1 / (1 / a + 1 / c), 2), "derived": True}
        if m == "Draw no bet" and H and D and A:
            x = H if s == "H" else A
            return {"price": round(x * (D - 1) / D, 2), "derived": True}
    except (KeyError, ZeroDivisionError):
        return None
    return None


# ---------------------------------------------------------------- update
def update(now: dt.datetime | None = None) -> dict:
    now = now or dt.datetime.now(dt.timezone.utc)
    rep = read_json(OUT / "latest.json", {}) or {}
    games = rep.get("games", [])
    season = season_code(now.date())
    wanted = {pair(g["match"]) for g in games}
    fx = lp_fixtures(season)
    refs = referees(fx, season, wanted)
    prices = model_prices()
    if rep.get("week_start"):
        ws, we = dt.date.fromisoformat(rep["week_start"]), dt.date.fromisoformat(rep["week_end"])
        for k, v in espn_prices(ws, we).items():
            prices.setdefault(k, v)
    out = {"updated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "report": rep.get("date"), "games": {}}
    for g in games:
        k = pair(g["match"])
        f = fx.get(k)
        ko = dt.datetime.fromisoformat(f["matchDate"].replace("Z", "+00:00")) if f and f.get("matchDate") else None
        st = STADIUMS.get(k[0])
        w = weather(st[1], st[2], ko) if st and ko else None
        p = prices.get(k)
        out["games"][g["match"]] = {
            "kickoff_utc": ko.strftime("%Y-%m-%dT%H:%M:%SZ") if ko else None,
            "postponed": bool(f and (f.get("postponedFixture") or f.get("canceledFixture"))),
            "round": f.get("round") if f else None,
            "stadium": {"name": st[0], "lat": st[1], "lon": st[2]} if st else None,
            "referee": refs.get(k), "weather": w, "prices": p, "best_bet_price": bet_price(g["best_bet"], p),
        }
    write_json(OUT / "live.json", out)
    found = sum(1 for v in out["games"].values() if v["kickoff_utc"])
    log.info("live: %d/%d fixtures found, %d referees, %d with weather, %d with prices", found, len(games),
             sum(1 for v in out["games"].values() if v["referee"]), sum(1 for v in out["games"].values() if v["weather"]),
             sum(1 for v in out["games"].values() if v["prices"]))
    return out


if __name__ == "__main__":
    if sys.argv[1:] != ["update"]:
        sys.exit(__doc__)
    update()
