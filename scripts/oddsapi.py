"""The Odds API client with a hard monthly credit budget.

Free plan: 500 credits/month. One league request with markets h2h,totals in one region costs
2 credits (no Asian handicap lines: European bookmakers don't offer them). Before each run we read the remaining credits (free /sports call) and allow
    (remaining - reserve) / scheduled runs left this month
so the scheduled Tuesday/Friday runs can never exhaust the month. Each league is fetched at most
once per run and the response is shared by fixture discovery and the stage-2 extra odds.
"""
from __future__ import annotations

import calendar
import json
import datetime as dt
import os

import numpy as np
import pandas as pd

from common import LEAGUES, RAW, get_logger, http_get, match_name

log = get_logger("oddsapi")

BASE = "https://api.the-odds-api.com/v4"
MARKETS = "h2h,totals"
COST_PER_LEAGUE = 2            # markets x regions
RESERVE = 15                   # credits kept back for manual runs
MAX_PER_RUN = 40
# fetched every run so the biggest leagues always have upcoming games, even between
# football-data.co.uk refreshes
CORE_LEAGUES = ["E0", "SP1", "D1", "I1", "F1"]
EXCHANGES = {"betfair_ex_uk", "betfair_ex_eu", "matchbook", "smarkets"}


def runs_left_this_month(today: dt.date, weekdays=(1, 4)) -> int:
    """Scheduled runs (Tue=1, Fri=4) from today to the end of the month, inclusive."""
    last = calendar.monthrange(today.year, today.month)[1]
    return max(1, sum(1 for d in range(today.day, last + 1)
                      if dt.date(today.year, today.month, d).weekday() in weekdays))


class OddsClient:
    def __init__(self, key: str | None = None, today: dt.date | None = None):
        self.key = key if key is not None else (os.environ.get("ODDS_API_KEY") or None)
        self.today = today or dt.date.today()
        self.cache: dict[str, list] = {}
        self.used = 0
        self.remaining = None
        self.budget = 0
        self.error = None if self.key else "ODDS_API_KEY not set"
        if self.key:
            self._init_budget()

    def _init_budget(self):
        r = http_get(f"{BASE}/sports", params={"apiKey": self.key}, retries=2)   # costs 0 credits
        if r is None or r.status_code != 200:
            self.error = "Odds API unreachable or key rejected"
            self.key = None
            return
        rem = r.headers.get("x-requests-remaining")
        self.remaining = int(float(rem)) if rem not in (None, "") else 500
        per_run = (self.remaining - RESERVE) // runs_left_this_month(self.today)
        self.budget = int(max(0, min(MAX_PER_RUN, per_run, int(os.environ.get("ODDS_API_BUDGET", MAX_PER_RUN)))))
        log.info("Odds API: %s credits left this month, budget this run %d (%d leagues)",
                 self.remaining, self.budget, self.budget // COST_PER_LEAGUE)

    @property
    def enabled(self) -> bool:
        return bool(self.key)

    def can_fetch(self) -> bool:
        return self.enabled and self.used + COST_PER_LEAGUE <= self.budget

    def league(self, lg: str) -> list | None:
        """Events for a league (cached). None if unavailable or over budget."""
        if lg in self.cache:
            return self.cache[lg]
        sport = LEAGUES.get(lg, (None,) * 6)[5]
        if not sport or not self.can_fetch():
            return None
        self.used += COST_PER_LEAGUE
        r = http_get(f"{BASE}/sports/{sport}/odds", retries=2,
                     params={"apiKey": self.key, "regions": "eu", "markets": MARKETS, "oddsFormat": "decimal"})
        if r is None or r.status_code != 200:
            self.cache[lg] = []
            return []
        rem = r.headers.get("x-requests-remaining")
        if rem:
            self.remaining = int(float(rem))
        events = r.json() if isinstance(r.json(), list) else []
        self.cache[lg] = events
        try:
            (RAW / f"oddsapi_{lg}.json").write_text(json.dumps(events), encoding="utf-8")
        except OSError:
            pass
        log.info("Odds API %s: %d events, %s credits left", lg, len(events), self.remaining)
        return events

    def status(self) -> str:
        if not self.enabled:
            return self.error or "disabled"
        return f"ok: used {self.used}/{self.budget} credits this run, {self.remaining} left this month"


# ---------------------------------------------------------------- parsing
def parse_event(ev: dict) -> dict:
    """{market: {selection: {"avg", "max", "n"}}} in the model's market naming."""
    prices: dict = {}
    for bk in ev.get("bookmakers", []):
        if bk["key"] in EXCHANGES:
            continue  # exchanges: commission not included in the price
        for mk in bk.get("markets", []):
            for o in mk.get("outcomes", []):
                if mk["key"] == "h2h":
                    m = "1X2"
                    s = "H" if o["name"] == ev["home_team"] else "A" if o["name"] == ev["away_team"] else "D"
                elif mk["key"] == "totals" and o.get("point") in (1.5, 2.5, 3.5):
                    m, s = f"O/U {o['point']}", o["name"]
                elif mk["key"] == "spreads" and o.get("point") is not None:
                    line = o["point"] if o["name"] == ev["home_team"] else -o["point"]
                    m, s = f"AH {line + 0:+g}", "Home" if o["name"] == ev["home_team"] else "Away"
                else:
                    continue
                prices.setdefault(m, {}).setdefault(s, []).append(float(o["price"]))
    return {m: {s: {"avg": float(np.mean(v)), "max": float(np.max(v)), "n": len(v)} for s, v in sel.items()}
            for m, sel in prices.items()}


def find_event(events: list, home: str, away: str) -> dict | None:
    names = {e["home_team"] for e in events} | {e["away_team"] for e in events}
    h, a = match_name(home, names), match_name(away, names)
    return next((e for e in events if e["home_team"] == h and e["away_team"] == a), None) if h and a else None


def event_to_row(lg: str, ev: dict, home: str, away: str) -> dict | None:
    """Convert an Odds API event into a football-data style fixture row."""
    odds = parse_event(ev)
    x = odds.get("1X2", {})
    if not all(s in x for s in "HDA"):
        return None
    t = pd.Timestamp(ev["commence_time"]).tz_convert("Europe/London")
    row = {"Div": lg, "Date": pd.Timestamp(t.date()), "Time": t.strftime("%H:%M"), "HomeTeam": home, "AwayTeam": away,
           "Source": "odds-api", "BookSpread": np.nan}
    for s in "HDA":
        row[f"Avg{s}"], row[f"Max{s}"] = x[s]["avg"], x[s]["max"]
    ou = odds.get("O/U 2.5", {})
    if "Over" in ou and "Under" in ou:
        row["Avg>2.5"], row["Max>2.5"] = ou["Over"]["avg"], ou["Over"]["max"]
        row["Avg<2.5"], row["Max<2.5"] = ou["Under"]["avg"], ou["Under"]["max"]
    # main Asian line = the one quoted by most bookmakers, ties broken by closeness to even money
    ah = [(m, sel) for m, sel in odds.items() if m.startswith("AH") and "Home" in sel and "Away" in sel]
    if ah:
        m, sel = max(ah, key=lambda it: (it[1]["Home"]["n"], -abs(it[1]["Home"]["avg"] - it[1]["Away"]["avg"])))
        row["AHh"] = float(m.split()[1])
        row["AvgAHH"], row["MaxAHH"] = sel["Home"]["avg"], sel["Home"]["max"]
        row["AvgAHA"], row["MaxAHA"] = sel["Away"]["avg"], sel["Away"]["max"]
    return row


def discover_fixtures(client: OddsClient, fixtures: pd.DataFrame, hist: pd.DataFrame, season: str,
                      today: dt.date, days: int = 12) -> pd.DataFrame:
    """Add core-league games (next `days` days) that football-data's fixtures.csv doesn't list yet.
    12 days rather than 7 so the site still shows the next round during international breaks."""
    if not client.enabled:
        return fixtures
    new_rows = []
    start, end = pd.Timestamp(today, tz="UTC"), pd.Timestamp(today + dt.timedelta(days=days + 1), tz="UTC")
    for lg in CORE_LEAGUES:
        events = client.league(lg)
        if not events:
            continue
        cur = hist[(hist["Div"] == lg) & (hist["Season"] == season)]
        teams = set(cur["HomeTeam"]) | set(cur["AwayTeam"])
        have = fixtures[fixtures["Div"] == lg] if len(fixtures) else fixtures
        for ev in events:
            ct = pd.Timestamp(ev["commence_time"])
            if not start <= ct < end:
                continue
            home, away = match_name(ev["home_team"], teams), match_name(ev["away_team"], teams)
            if not home or not away:
                log.info("Odds API %s: cannot match %s v %s to history names", lg, ev["home_team"], ev["away_team"])
                continue
            if len(have) and ((have["HomeTeam"] == home) & (have["AwayTeam"] == away)).any():
                continue
            row = event_to_row(lg, ev, home, away)
            if row:
                new_rows.append(row)
    if new_rows:
        log.info("Odds API discovery: added %d games missing from fixtures.csv", len(new_rows))
        fixtures = pd.concat([fixtures, pd.DataFrame(new_rows)], ignore_index=True)
    return fixtures
