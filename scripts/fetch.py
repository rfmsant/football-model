"""Stage 0: download fixtures, match history, xG and Elo into data/.

Every source is optional: a failed download is logged and the pipeline continues
with whatever data is available.
"""
from __future__ import annotations

import datetime as dt
import io
import json

import numpy as np
import pandas as pd

from common import DATA, LEAGUES, RAW, get_logger, http_get, match_name, season_code, season_start_year

log = get_logger("fetch")

FD_BASE = "https://www.football-data.co.uk"
KEEP = ["Div", "Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HS", "AS", "HST", "AST",
        "HC", "AC", "HY", "AY", "HR", "AR", "HxG", "AxG",
        "AvgH", "AvgD", "AvgA", "MaxH", "MaxD", "MaxA", "B365H", "B365D", "B365A", "PSH", "PSD", "PSA",
        "Avg>2.5", "Avg<2.5", "Max>2.5", "Max<2.5", "AHh", "AvgAHH", "AvgAHA", "MaxAHH", "MaxAHA",
        "AvgCH", "AvgCD", "AvgCA", "MaxCH", "MaxCD", "MaxCA", "PSCH", "PSCD", "PSCA",
        "AvgC>2.5", "AvgC<2.5", "AHCh", "AvgCAHH", "AvgCAHA"]
BOOK_PREFIXES = ["B365", "BFD", "BV", "BW", "PP", "SKB", "PS", "WH", "IW", "VC", "LB", "CL", "BMGM"]


def _read_csv_bytes(content: bytes) -> pd.DataFrame | None:
    for enc in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(content), encoding=enc, on_bad_lines="skip", low_memory=False)
            df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
            return df.dropna(how="all")
        except Exception:  # noqa: BLE001
            continue
    return None


def parse_dates(s: pd.Series) -> pd.Series:
    d = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    d2 = pd.to_datetime(s, format="%d/%m/%y", errors="coerce")
    return d.fillna(d2)


def bookmaker_spread(df: pd.DataFrame) -> pd.Series:
    """Std-dev of implied home-win probability across individual bookmakers (disagreement)."""
    cols = [f"{p}H" for p in BOOK_PREFIXES if f"{p}H" in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    imp = 1.0 / df[cols].apply(pd.to_numeric, errors="coerce")
    return imp.std(axis=1)


# ---------------------------------------------------------------- fixtures
def fetch_fixtures(days: int = 7, today: dt.date | None = None) -> pd.DataFrame:
    today = today or dt.date.today()
    r = http_get(f"{FD_BASE}/fixtures.csv")
    if r is None:
        log.error("fixtures.csv unavailable")
        return pd.DataFrame()
    df = _read_csv_bytes(r.content)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df[df["Div"].isin(LEAGUES)].copy()
    df["Date"] = parse_dates(df["Date"])
    start, end = pd.Timestamp(today), pd.Timestamp(today + dt.timedelta(days=days))
    df = df[(df["Date"] >= start) & (df["Date"] <= end)]
    df["BookSpread"] = bookmaker_spread(df)
    keep = [c for c in KEEP if c in df.columns] + ["BookSpread"]
    df = df[keep].reset_index(drop=True)
    df.to_csv(DATA / "fixtures.csv", index=False)
    log.info("fixtures: %d games in %d leagues", len(df), df["Div"].nunique() if len(df) else 0)
    return df


# ---------------------------------------------------------------- history
def fetch_league_season(league: str, season: str, force: bool = False) -> pd.DataFrame | None:
    cache = RAW / f"{season}_{league}.csv"
    current = season == season_code()
    if cache.exists() and not (force or current):
        return pd.read_csv(cache, low_memory=False)
    r = http_get(f"{FD_BASE}/mmz4281/{season}/{league}.csv")
    if r is None:
        if cache.exists():
            return pd.read_csv(cache, low_memory=False)
        log.warning("no history for %s %s", league, season)
        return None
    df = _read_csv_bytes(r.content)
    if df is None or df.empty or "HomeTeam" not in df.columns:
        return None
    df["Div"] = league
    df = df[[c for c in KEEP if c in df.columns]].copy()
    df["Season"] = season
    df.to_csv(cache, index=False)
    return df


def fetch_history(seasons: list[str]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        for lg in LEAGUES:
            df = fetch_league_season(lg, s)
            if df is not None and len(df):
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    h = pd.concat(frames, ignore_index=True)
    h["Date"] = parse_dates(h["Date"].astype(str))
    h = h.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    for c in h.columns:
        if c not in ("Div", "Date", "Time", "HomeTeam", "AwayTeam", "Season"):
            h[c] = pd.to_numeric(h[c], errors="coerce")
    h["Season"] = h["Season"].astype(str).str.zfill(4)
    return h.sort_values("Date").reset_index(drop=True)


# ---------------------------------------------------------------- Understat xG
def fetch_understat(league_key: str, year: int) -> list[dict]:
    cache = RAW / f"understat_{league_key}_{year}.json"
    past = year < season_start_year(season_code())
    if cache.exists() and past:
        return json.loads(cache.read_text(encoding="utf-8"))
    r = http_get(f"https://understat.com/getLeagueData/{league_key}/{year}",
                 headers={"X-Requested-With": "XMLHttpRequest",
                          "Referer": f"https://understat.com/league/{league_key}/{year}"})
    out = []
    if r is not None:
        try:
            data = r.json()
            for m in data.get("dates", []):
                if not m.get("isResult"):
                    continue
                out.append({"date": m["datetime"][:10], "home": m["h"]["title"], "away": m["a"]["title"],
                            "hxg": float(m["xG"]["h"]), "axg": float(m["xG"]["a"])})
            (RAW / f"understat_players_{league_key}_{year}.json").write_text(
                json.dumps(data.get("players", [])), encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            log.warning("understat parse failed for %s %s: %s", league_key, year, e)
    if out:
        cache.write_text(json.dumps(out), encoding="utf-8")
    elif cache.exists():
        out = json.loads(cache.read_text(encoding="utf-8"))
    return out


def merge_understat(hist: pd.DataFrame) -> pd.DataFrame:
    """Fill missing HxG/AxG for top-5 leagues from Understat (matched by date +/-1 day and team name)."""
    if "HxG" not in hist.columns:
        hist["HxG"] = np.nan
        hist["AxG"] = np.nan
    hist["xGSource"] = np.where(hist["HxG"].notna(), "fd", None)
    for lg, meta in LEAGUES.items():
        key = meta[3]
        if not key:
            continue
        for season in hist.loc[hist["Div"] == lg, "Season"].unique():
            rows = hist.index[(hist["Div"] == lg) & (hist["Season"] == season) & hist["HxG"].isna()]
            if len(rows) == 0:
                continue
            us = fetch_understat(key, season_start_year(season))
            if not us:
                continue
            us_teams = {m["home"] for m in us} | {m["away"] for m in us}
            fd_teams = set(hist.loc[rows, "HomeTeam"]) | set(hist.loc[rows, "AwayTeam"])
            tmap = {t: match_name(t, us_teams) for t in fd_teams}
            unmatched = [t for t, v in tmap.items() if v is None]
            if unmatched:
                log.info("understat %s %s unmatched teams: %s", lg, season, unmatched)
            idx = {(m["date"], m["home"], m["away"]): m for m in us}
            filled = 0
            for i in rows:
                h, a = tmap.get(hist.at[i, "HomeTeam"]), tmap.get(hist.at[i, "AwayTeam"])
                if not h or not a:
                    continue
                d = hist.at[i, "Date"]
                for off in (0, -1, 1):
                    m = idx.get(((d + pd.Timedelta(days=off)).strftime("%Y-%m-%d"), h, a))
                    if m:
                        hist.at[i, "HxG"], hist.at[i, "AxG"] = m["hxg"], m["axg"]
                        hist.at[i, "xGSource"] = "understat"
                        filled += 1
                        break
            log.info("understat %s %s: filled xG for %d/%d matches", lg, season, filled, len(rows))
    return hist


# ---------------------------------------------------------------- ClubElo
def fetch_clubelo(date: dt.date | None = None) -> pd.DataFrame:
    date = date or dt.date.today()
    for back in range(4):
        d = date - dt.timedelta(days=back)
        r = http_get(f"http://api.clubelo.com/{d.isoformat()}", retries=2, timeout=20)
        if r is not None and r.text.startswith("Rank"):
            df = pd.read_csv(io.StringIO(r.text))
            df.to_csv(DATA / "clubelo.csv", index=False)
            log.info("clubelo: %d clubs (%s)", len(df), d)
            return df
    log.warning("ClubElo unavailable; the built-in Elo will be used on its own")
    return pd.DataFrame()


def run(today: dt.date | None = None) -> dict:
    today = today or dt.date.today()
    fixtures = fetch_fixtures(today=today)
    seasons = [season_code(today, -1), season_code(today)]
    hist = fetch_history(seasons)
    if len(hist):
        hist = merge_understat(hist)
        hist.to_csv(DATA / "history.csv", index=False)
    elo = fetch_clubelo(today)
    log.info("history: %d matches", len(hist))
    return {"fixtures": fixtures, "history": hist, "clubelo": elo}


if __name__ == "__main__":
    run()
