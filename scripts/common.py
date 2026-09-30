"""Shared configuration and helpers for the football-model pipeline."""
from __future__ import annotations

import datetime as dt
import difflib
import json
import logging
import os
import re
import time
import unicodedata
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
DATA.mkdir(exist_ok=True)
RAW.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


log = get_logger("common")

# code: (name, country, tier, understat, api-football id, odds-api key, (europe spots, relegation spots))
LEAGUES = {
    "E0": ("Premier League", "ENG", 1, "EPL", 39, "soccer_epl", (5, 3)),
    "E1": ("Championship", "ENG", 2, None, 40, "soccer_efl_champ", (6, 3)),
    "E2": ("League One", "ENG", 3, None, 41, "soccer_england_league1", (6, 4)),
    "E3": ("League Two", "ENG", 4, None, 42, "soccer_england_league2", (7, 2)),
    "EC": ("National League", "ENG", 5, None, 43, None, (7, 4)),
    "SC0": ("Scottish Premiership", "SCO", 1, None, 179, "soccer_spl", (4, 1)),
    "SC1": ("Scottish Championship", "SCO", 2, None, 180, None, (4, 1)),
    "SC2": ("Scottish League One", "SCO", 3, None, 183, None, (4, 1)),
    "SC3": ("Scottish League Two", "SCO", 4, None, 184, None, (4, 1)),
    "D1": ("Bundesliga", "GER", 1, "Bundesliga", 78, "soccer_germany_bundesliga", (6, 3)),
    "D2": ("2. Bundesliga", "GER", 2, None, 79, "soccer_germany_bundesliga2", (3, 3)),
    "I1": ("Serie A", "ITA", 1, "Serie_A", 135, "soccer_italy_serie_a", (6, 3)),
    "I2": ("Serie B", "ITA", 2, None, 136, "soccer_italy_serie_b", (8, 4)),
    "SP1": ("La Liga", "ESP", 1, "La_liga", 140, "soccer_spain_la_liga", (6, 3)),
    "SP2": ("Segunda Division", "ESP", 2, None, 141, "soccer_spain_segunda_division", (6, 4)),
    "F1": ("Ligue 1", "FRA", 1, "Ligue_1", 61, "soccer_france_ligue_one", (5, 3)),
    "F2": ("Ligue 2", "FRA", 2, None, 62, "soccer_france_ligue_two", (5, 3)),
    "N1": ("Eredivisie", "NED", 1, None, 88, "soccer_netherlands_eredivisie", (5, 3)),
    "B1": ("Belgian Pro League", "BEL", 1, None, 144, "soccer_belgium_first_div", (4, 2)),
    "P1": ("Primeira Liga", "POR", 1, None, 94, "soccer_portugal_primeira_liga", (5, 3)),
    "T1": ("Super Lig", "TUR", 1, None, 203, "soccer_turkey_super_league", (4, 4)),
    "G1": ("Super League Greece", "GRE", 1, None, 197, "soccer_greece_super_league", (4, 2)),
}

# Starting Elo per tier for the built-in Elo (used when a team first appears).
TIER_ELO = {1: 1600, 2: 1450, 3: 1350, 4: 1260, 5: 1180}

# Default model parameters (overwritten by data/params.json produced by backtest.py).
DEFAULT_PARAMS = {
    "half_life_days": 180.0,   # time-decay of match weights
    "xg_weight": 0.6,          # share of xG (or shots-on-target proxy) vs actual goals
    "elo_weight": 0.3,         # blend of Elo-implied goals with rating-implied goals (log space)
    "prior_matches": 6.0,      # shrinkage toward league average, in pseudo-matches
    "sharpness": 1.0,          # >1 stretches the home/away goal ratio (calibration)
    "rho": -0.08,              # Dixon-Coles low-score correlation
    "min_ev": 0.03,            # minimum EV for a bet to be recommended
    "market_weight": 0.5,      # blend of model and no-vig market probability used for EV
    "min_odds": 1.3,
    "max_odds": 6.0,
    # markets a bet may be recommended in (European bookmakers: no Asian handicap)
    "bet_markets": ["1X2", "O/U", "BTTS"],
}


def load_params() -> dict:
    p = dict(DEFAULT_PARAMS)
    f = DATA / "params.json"
    if f.exists():
        try:
            p.update(json.loads(f.read_text()).get("params", {}))
        except Exception as e:  # noqa: BLE001
            log.warning("could not read params.json: %s", e)
    return p


def season_code(date: dt.date | None = None, offset: int = 0) -> str:
    """football-data season code, e.g. '2627'. offset=-1 gives the previous season."""
    d = date or dt.date.today()
    start = d.year if d.month >= 7 else d.year - 1
    start += offset
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


def season_start_year(code: str) -> int:
    return 2000 + int(code[:2])


_session = requests.Session()
_session.headers["User-Agent"] = "football-model/1.0 (+https://github.com)"


def http_get(url: str, *, params=None, headers=None, timeout=30, retries=3, backoff=2.0):
    """GET with retries. Returns a Response or None; never raises."""
    for attempt in range(retries):
        try:
            r = _session.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code in (401, 403, 404, 429):
                log.warning("GET %s -> %s", url, r.status_code)
                return r if r.status_code == 429 else None
            log.warning("GET %s -> %s (attempt %d)", url, r.status_code, attempt + 1)
        except requests.RequestException as e:
            log.warning("GET %s failed: %s (attempt %d)", url, e, attempt + 1)
        time.sleep(backoff * (attempt + 1))
    return None


# ---------------------------------------------------------------- name matching
_STOP = {"fc", "cf", "afc", "sc", "ac", "as", "ssc", "sv", "fk", "sk", "cd", "ud", "rc", "rcd", "sd",
         "club", "de", "calcio", "1", "vfb", "vfl", "tsg", "bsc", "fsv", "the", "and", "ca", "olympique", "stade"}

ALIASES = {
    "man united": "manchester united", "man utd": "manchester united", "man city": "manchester city",
    "nott'm forest": "nottingham forest", "forest": "nottingham forest", "wolves": "wolverhampton",
    "spurs": "tottenham", "sheffield weds": "sheffield wednesday", "sheffield utd": "sheffield united",
    "qpr": "queens park rangers", "west brom": "west bromwich", "mk dons": "milton keynes dons",
    "rb leipzig": "rasenballsport leipzig", "leipzig": "rasenballsport leipzig",
    "fc koln": "fc cologne", "koln": "fc cologne", "cologne": "fc cologne", "ath bilbao": "athletic club bilbao",
    "atl madrid": "atletico madrid",
    "sociedad": "real sociedad", "betis": "real betis", "espanol": "espanyol", "vallecano": "rayo vallecano",
    "celta": "celta vigo", "la coruna": "deportivo la coruna", "sp gijon": "sporting gijon",
    "m'gladbach": "borussia monchengladbach", "gladbach": "borussia monchengladbach",
    "dortmund": "borussia dortmund", "ein frankfurt": "eintracht frankfurt",
    "leverkusen": "bayer leverkusen", "bayern munich": "bayern", "hertha": "hertha berlin",
    "inter": "internazionale", "milan": "ac milan", "paris sg": "paris saint germain", "psg": "paris saint germain",
    "st etienne": "saint etienne", "sp lisbon": "sporting cp", "sporting": "sporting cp",
    "psv eindhoven": "psv", "for sittard": "fortuna sittard", "nijmegen": "nec", "st truiden": "sint truiden",
    "club brugge": "brugge", "st gilloise": "union saint gilloise", "olympiakos": "olympiacos",
}


def norm_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower().strip()
    s = s.replace("&", "and").replace(".", "").replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    s = ALIASES.get(s, s)
    toks = [t for t in re.split(r"[^a-z0-9']+", s) if t and t not in _STOP]
    return " ".join(toks) or s


def match_name(name: str, candidates, cutoff: float = 0.72) -> str | None:
    """Fuzzy-match `name` against candidates (same country/league). Returns best candidate or None."""
    cands = list(candidates)
    if not cands:
        return None
    if name in cands:
        return name
    target = norm_name(name)
    normed = {c: norm_name(c) for c in cands}
    for c, n in normed.items():
        if n == target:
            return c
    best, best_score = None, 0.0
    for c, n in normed.items():
        score = difflib.SequenceMatcher(None, target, n).ratio()
        # token containment bonus (e.g. "bayern" vs "bayern munchen")
        tt, nt = set(target.split()), set(n.split())
        if tt and nt and (tt <= nt or nt <= tt):
            score = max(score, 0.85)
        if score > best_score:
            best, best_score = c, score
    return best if best_score >= cutoff else None


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, default=str, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default
