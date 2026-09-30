"""Rating model: time-weighted attack/defence ratings blended with Elo, Dixon-Coles scorelines,
market derivation, margin removal, edge and EV.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from common import LEAGUES, TIER_ELO, get_logger, match_name

log = get_logger("model")

MAX_GOALS = 6
ELO_K = 20.0
ELO_HFA = 65.0
COUNTRY_ELO = {"ENG": "ENG", "SCO": "SCO", "GER": "GER", "ITA": "ITA", "ESP": "ESP", "FRA": "FRA",
               "NED": "NED", "BEL": "BEL", "POR": "POR", "TUR": "TUR", "GRE": "GRE"}


# ================================================================ built-in Elo
def compute_elo(hist: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Sequential Elo over all leagues of each country (teams keep ratings across divisions).
    Adds EloH/EloA (pre-match) columns; returns (hist, {(country, team): rating})."""
    ratings: dict = {}
    eh, ea = np.empty(len(hist)), np.empty(len(hist))
    for k, row in enumerate(hist.itertuples(index=False)):
        meta = LEAGUES.get(row.Div)
        country, tier = (meta[1], meta[2]) if meta else ("?", 3)
        kh, ka = (country, row.HomeTeam), (country, row.AwayTeam)
        rh = ratings.setdefault(kh, TIER_ELO.get(tier, 1300))
        ra = ratings.setdefault(ka, TIER_ELO.get(tier, 1300))
        eh[k], ea[k] = rh, ra
        exp_h = 1.0 / (1.0 + 10 ** (-(rh + ELO_HFA - ra) / 400.0))
        gd = row.FTHG - row.FTAG
        res = 1.0 if gd > 0 else 0.5 if gd == 0 else 0.0
        mult = 1.0 if abs(gd) <= 1 else (1.5 if abs(gd) == 2 else (11 + abs(gd)) / 8.0)
        delta = ELO_K * mult * (res - exp_h)
        ratings[kh] = rh + delta
        ratings[ka] = ra - delta
    hist = hist.copy()
    hist["EloH"], hist["EloA"] = eh, ea
    return hist, ratings


# ================================================================ performance metric
def sot_conversion(h: pd.DataFrame) -> pd.Series:
    """Goals per shot on target for each row's league, taken from the PREVIOUS season (no look-ahead).
    Falls back to 0.32 (the long-run European average) when there is no previous season."""
    if "HST" not in h:
        return pd.Series(0.32, index=h.index)
    tot = h.groupby(["Div", "Season"]).agg(g=("FTHG", "sum"), g2=("FTAG", "sum"), s=("HST", "sum"), s2=("AST", "sum"))
    rate = ((tot.g + tot.g2) / (tot.s + tot.s2)).where((tot.s + tot.s2) > 200).clip(0.2, 0.5)
    prev = {(div, f"{(int(season[:2]) + 1) % 100:02d}{(int(season[2:]) + 1) % 100:02d}"): v
            for (div, season), v in rate.items() if pd.notna(v)}
    return pd.Series([prev.get((d, s), 0.32) for d, s in zip(h["Div"], h["Season"].astype(str))], index=h.index)


def add_performance(hist: pd.DataFrame, xg_weight: float) -> pd.DataFrame:
    """Blend actual goals with xG (or a shots-on-target proxy) into a 'performance goals' metric."""
    h = hist.copy()
    conv = sot_conversion(h)
    for side, g, st, xg in (("H", "FTHG", "HST", "HxG"), ("A", "FTAG", "AST", "AxG")):
        proxy = h[st] * conv if st in h else pd.Series(np.nan, index=h.index)
        underlying = h[xg].where(h[xg].notna(), proxy) if xg in h else proxy
        h[f"Perf{side}"] = np.where(underlying.notna(),
                                    (1 - xg_weight) * h[g] + xg_weight * underlying, h[g])
    h["HasXG"] = h["HxG"].notna() if "HxG" in h else False
    h["HasSOT"] = h["HST"].notna() if "HST" in h else False
    return h


# ================================================================ ratings
class LeagueRatings:
    def __init__(self, teams, att, dfn, mu_h, mu_a, n_matches, xg_share, promoted):
        self.idx = {t: i for i, t in enumerate(teams)}
        self.att, self.dfn = att, dfn
        self.mu_h, self.mu_a = mu_h, mu_a
        self.n_matches, self.xg_share, self.promoted = n_matches, xg_share, promoted

    def get(self, team):
        i = self.idx.get(team)
        if i is None:  # unseen team: treat as a promoted side
            return 0.85, 1.15, 0, 0.0
        return self.att[i], self.dfn[i], self.n_matches[i], self.xg_share[i]


def fit_league(m: pd.DataFrame, ref_date: pd.Timestamp, params: dict, current_season: str) -> LeagueRatings | None:
    m = m[(m["Date"] < ref_date) & (m["Date"] >= ref_date - pd.Timedelta(days=730))]
    if len(m) < 20:
        return None
    teams = sorted(set(m["HomeTeam"]) | set(m["AwayTeam"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi = m["HomeTeam"].map(idx).to_numpy()
    ai = m["AwayTeam"].map(idx).to_numpy()
    age = (ref_date - m["Date"]).dt.days.to_numpy()
    w = 0.5 ** (age / params["half_life_days"])
    gh, ga = m["PerfH"].to_numpy(float), m["PerfA"].to_numpy(float)

    # teams new to this league this season get a weaker prior
    prev = m[m["Season"] != current_season]
    prev_teams = set(prev["HomeTeam"]) | set(prev["AwayTeam"])
    promoted = np.array([len(prev) > 0 and t not in prev_teams for t in teams])
    att_p = np.where(promoted, 0.85, 1.0)
    def_p = np.where(promoted, 1.15, 1.0)
    P = params["prior_matches"]

    att, dfn = np.ones(n), np.ones(n)
    mu_h = np.average(gh, weights=w)
    mu_a = np.average(ga, weights=w)
    for _ in range(40):
        exp_for = np.bincount(hi, w * mu_h * dfn[ai], n) + np.bincount(ai, w * mu_a * dfn[hi], n)
        gf = np.bincount(hi, w * gh, n) + np.bincount(ai, w * ga, n)
        att = (gf + P * att_p) / (exp_for + P)
        exp_ag = np.bincount(hi, w * mu_a * att[ai], n) + np.bincount(ai, w * mu_h * att[hi], n)
        gag = np.bincount(hi, w * ga, n) + np.bincount(ai, w * gh, n)
        dfn = (gag + P * def_p) / (exp_ag + P)
        att /= np.exp(np.mean(np.log(att)))
        dfn /= np.exp(np.mean(np.log(dfn)))
        mu_h = np.sum(w * gh) / np.sum(w * att[hi] * dfn[ai])
        mu_a = np.sum(w * ga) / np.sum(w * att[ai] * dfn[hi])
    cur = m[m["Season"] == current_season]
    n_matches = np.bincount(cur["HomeTeam"].map(idx), minlength=n) + np.bincount(cur["AwayTeam"].map(idx), minlength=n)
    xg_rows = m["HasXG"].to_numpy(bool)
    xg_share = (np.bincount(hi, w * xg_rows, n) + np.bincount(ai, w * xg_rows, n)) / \
        np.maximum(np.bincount(hi, w, n) + np.bincount(ai, w, n), 1e-9)
    return LeagueRatings(teams, att, dfn, mu_h, mu_a, n_matches, xg_share, promoted)


def fit_elo_slope(hist: pd.DataFrame, ref_date: pd.Timestamp) -> float:
    """Goals of supremacy per Elo point, fitted on recent matches before ref_date."""
    m = hist[(hist["Date"] < ref_date) & (hist["Date"] >= ref_date - pd.Timedelta(days=730))]
    if len(m) < 200:
        return 0.0045
    x = (m["EloH"] + ELO_HFA - m["EloA"]).to_numpy()
    y = (m["FTHG"] - m["FTAG"]).to_numpy()
    x = x - x.mean()
    return float(np.clip(np.dot(x, y - y.mean()) / np.dot(x, x), 0.002, 0.008))


# ================================================================ Dixon-Coles
def dixon_coles_matrix(lh: float, la: float, rho: float) -> np.ndarray:
    g = np.arange(MAX_GOALS + 1)
    ph = np.exp(-lh) * lh ** g / np.array([math.factorial(k) for k in g])
    pa = np.exp(-la) * la ** g / np.array([math.factorial(k) for k in g])
    mat = np.outer(ph, pa)
    mat[0, 0] *= max(1 - lh * la * rho, 0)
    mat[0, 1] *= max(1 + lh * rho, 0)
    mat[1, 0] *= max(1 + la * rho, 0)
    mat[1, 1] *= max(1 - rho, 0)
    return mat / mat.sum()


def ah_outcomes(mat: np.ndarray, line: float) -> tuple[float, float, float]:
    """Asian handicap for the HOME side at `line` (e.g. -0.75). Returns (win, push, loss) weights,
    already averaged over the two halves of quarter lines."""
    halves = [line - 0.25, line + 0.25] if abs(line * 4) % 2 == 1 else [line]
    diff = np.subtract.outer(np.arange(MAX_GOALS + 1), np.arange(MAX_GOALS + 1))
    win = push = loss = 0.0
    for hl in halves:
        d = diff + hl
        win += mat[d > 1e-9].sum() / len(halves)
        push += mat[np.abs(d) <= 1e-9].sum() / len(halves)
        loss += mat[d < -1e-9].sum() / len(halves)
    return win, push, loss


def derive_markets(mat: np.ndarray, ah_line: float | None = None) -> dict:
    i, j = np.indices(mat.shape)
    tot = i + j
    out = {
        "1X2": {"H": float(mat[i > j].sum()), "D": float(mat[i == j].sum()), "A": float(mat[i < j].sum())},
        "BTTS": {"Yes": float(mat[(i > 0) & (j > 0)].sum())},
    }
    out["BTTS"]["No"] = 1 - out["BTTS"]["Yes"]
    for k in (1.5, 2.5, 3.5):
        over = float(mat[tot > k].sum())
        out[f"O/U {k}"] = {"Over": over, "Under": 1 - over}
    lines = ah_line if isinstance(ah_line, (list, tuple, set)) else [ah_line]
    for line in lines:
        if line is None or pd.isna(line):
            continue
        line = float(line)
        w, p, lo = ah_outcomes(mat, line)
        # probabilities exclude pushes; the away side wins exactly when the home side loses
        out[f"AH {line:+g}"] = {"Home": w / max(w + lo, 1e-9), "Away": lo / max(w + lo, 1e-9),
                                "_raw": (w, p, lo)}
    best = np.unravel_index(np.argmax(mat), mat.shape)
    out["likely_score"] = f"{best[0]}-{best[1]}"
    top = np.argsort(mat, axis=None)[::-1][:5]
    out["top_scores"] = [{"score": f"{a}-{b}", "p": float(mat[a, b])}
                         for a, b in (np.unravel_index(t, mat.shape) for t in top)]
    return out


# ================================================================ odds handling
def no_vig(odds: list[float]) -> list[float] | None:
    try:
        imp = [1.0 / float(o) for o in odds]
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if any(math.isnan(x) or x <= 0 for x in imp):
        return None
    s = sum(imp)
    return [x / s for x in imp]


def _num(row, col):
    v = row.get(col) if hasattr(row, "get") else None
    try:
        v = float(v)
        return v if v > 1.0 and not math.isnan(v) else None
    except (TypeError, ValueError):
        return None


MAX_OVER_AVG = 1.07  # best prices more than 7% above the average are treated as stale/erroneous


def cap_price(avg: float, mx: float | None) -> float:
    return min(mx or avg, avg * MAX_OVER_AVG)


def market_odds_from_row(row) -> dict:
    """{market: {selection: {'avg': x, 'max': y}}} from a football-data style row."""
    out = {}
    avg = [_num(row, f"Avg{s}") or _num(row, f"B365{s}") for s in "HDA"]
    mx = [_num(row, f"Max{s}") or a for s, a in zip("HDA", avg)]
    if all(avg):
        out["1X2"] = {s: {"avg": a, "max": m} for s, a, m in zip("HDA", avg, mx)}
    o, u = _num(row, "Avg>2.5"), _num(row, "Avg<2.5")
    if o and u:
        out["O/U 2.5"] = {"Over": {"avg": o, "max": _num(row, "Max>2.5") or o},
                          "Under": {"avg": u, "max": _num(row, "Max<2.5") or u}}
    line = row.get("AHh") if hasattr(row, "get") else None
    ah, aa = _num(row, "AvgAHH"), _num(row, "AvgAHA")
    if line is not None and not pd.isna(line) and ah and aa:
        out[f"AH {float(line):+g}"] = {"Home": {"avg": ah, "max": _num(row, "MaxAHH") or ah},
                                       "Away": {"avg": aa, "max": _num(row, "MaxAHA") or aa}}
    return {m: sel for m, sel in out.items() if _sane(m, sel)}


def _sane(market: str, sel: dict) -> bool:
    """Reject corrupt rows: implausible overround, or AH main-line prices far from evens."""
    over = sum(1.0 / v["avg"] for v in sel.values())
    if not 0.99 <= over <= 1.15:
        return False
    if market.startswith("AH") and any(not 1.5 <= v["avg"] <= 2.8 for v in sel.values()):
        return False
    return True


def evaluate_markets(model_mk: dict, odds: dict, market_weight: float) -> list[dict]:
    """Edge/EV for every market that has odds. model probs are blended with the no-vig market
    probability (market_weight) before EV, which guards against model overconfidence."""
    rows = []
    for mk, sels in model_mk.items():
        if not isinstance(sels, dict):
            continue
        names = [s for s in sels if not s.startswith("_")]
        mo = odds.get(mk)
        fair = no_vig([mo[s]["avg"] for s in names]) if mo and all(s in mo for s in names) else None
        for k, s in enumerate(names):
            p_model = float(sels[s])
            r = {"market": mk, "selection": s, "model_p": round(p_model, 4),
                 "fair_odds": round(1 / p_model, 2) if p_model > 0 else None}
            if fair:
                price = cap_price(mo[s]["avg"], mo[s]["max"])
                p_mkt = fair[k]
                p_bet = (1 - market_weight) * p_model + market_weight * p_mkt
                if mk.startswith("AH") and "_raw" in sels:
                    w, push, lo = sels["_raw"]
                    if s == "Away":
                        w, lo = lo, w
                    scale = p_bet / max(p_model, 1e-9)
                    w_b = min(w * scale, 1 - push)
                    ev = float(w_b * (price - 1) - (1 - push - w_b))
                else:
                    ev = p_bet * price - 1
                r.update({"market_p": round(p_mkt, 4), "odds": round(price, 2), "avg_odds": round(mo[s]["avg"], 2),
                          "edge": round(p_model - p_mkt, 4), "p_bet": round(p_bet, 4), "ev": round(ev, 4)})
            rows.append(r)
    return rows


# ================================================================ the model
class Model:
    def __init__(self, hist: pd.DataFrame, params: dict, clubelo: pd.DataFrame | None = None):
        self.params = params
        h = add_performance(hist, params["xg_weight"])
        if "EloH" not in h:
            h, self.elo = compute_elo(h)
        else:
            self.elo = {}
        self.hist = h
        cols = ["Date", "Season", "HomeTeam", "AwayTeam", "PerfH", "PerfA", "HasXG"]
        self.by_league = {lg: g[cols].reset_index(drop=True) for lg, g in h.groupby("Div")}
        self.clubelo = clubelo if clubelo is not None and len(clubelo) else None
        self._cache: dict = {}
        self._slope: dict = {}

    def ratings(self, league: str, ref_date: pd.Timestamp, season: str):
        key = (league, ref_date, season)
        if key not in self._cache:
            m = self.by_league.get(league)
            self._cache[key] = None if m is None else fit_league(m, ref_date, self.params, season)
        return self._cache[key]

    def elo_slope(self, ref_date):
        if ref_date not in self._slope:
            self._slope[ref_date] = fit_elo_slope(self.hist, ref_date)
        return self._slope[ref_date]

    def clubelo_diff(self, league, home, away):
        if self.clubelo is None:
            return None
        country = LEAGUES[league][1]
        ce = self.clubelo[self.clubelo["Country"] == country]
        names = list(ce["Club"])
        h, a = match_name(home, names), match_name(away, names)
        if not h or not a:
            return None
        get = dict(zip(ce["Club"], ce["Elo"]))
        return float(get[h] - get[a])

    def lambdas(self, league, home, away, ref_date, season, elo_h=None, elo_a=None):
        """Expected goals for home/away + diagnostics."""
        p = self.params
        R = self.ratings(league, ref_date, season)
        info = {"elo_source": "built-in"}
        if R is None:
            lh_r, la_r = 1.45, 1.15
            info.update(n_home=0, n_away=0, xg_share=0.0)
        else:
            ah, dh, nh, xh = R.get(home)
            aa, da, na, xa = R.get(away)
            lh_r, la_r = R.mu_h * ah * da, R.mu_a * aa * dh
            info.update(att_home=ah, def_home=dh, att_away=aa, def_away=da, n_home=int(nh), n_away=int(na),
                        xg_share=float((xh + xa) / 2))
        country = LEAGUES[league][1]
        tier = LEAGUES[league][2]
        if elo_h is None:
            elo_h = self.elo.get((country, home), TIER_ELO.get(tier, 1300))
            elo_a = self.elo.get((country, away), TIER_ELO.get(tier, 1300))
        d_elo = elo_h - elo_a
        ce = self.clubelo_diff(league, home, away)
        if ce is not None:
            d_elo = 0.5 * d_elo + 0.5 * ce
            info["elo_source"] = "ClubElo+built-in"
        info["elo_diff"] = round(d_elo, 1)
        total = lh_r + la_r
        sup = self.elo_slope(ref_date) * (d_elo + ELO_HFA)
        lh_e, la_e = max(0.15, (total + sup) / 2), max(0.15, (total - sup) / 2)
        w = p["elo_weight"]
        lh = math.exp((1 - w) * math.log(max(lh_r, 0.1)) + w * math.log(lh_e))
        la = math.exp((1 - w) * math.log(max(la_r, 0.1)) + w * math.log(la_e))
        # sharpness: stretch the home/away goal ratio (keeps total goals) to fix shrinkage bias
        s = p.get("sharpness", 1.0)
        if s != 1.0:
            tot, r = lh + la, s * math.log(lh / la)
            lh, la = tot / (1 + math.exp(-r)), tot / (1 + math.exp(r))
        return lh, la, info

    def predict(self, league, home, away, ref_date, season, odds=None, ah_line=None,
                elo_h=None, elo_a=None, adj=(1.0, 1.0)):
        lh, la, info = self.lambdas(league, home, away, ref_date, season, elo_h, elo_a)
        lh, la = lh * adj[0], la * adj[1]
        mat = dixon_coles_matrix(lh, la, self.params["rho"])
        mk = derive_markets(mat, ah_line)
        evals = evaluate_markets(mk, odds or {}, self.params.get("market_weight", 0.5))
        return {"lambda_home": lh, "lambda_away": la, "markets": mk, "evals": evals, "info": info, "matrix": mat}


def best_bet(evals: list[dict], params: dict) -> dict | None:
    allowed = tuple(params.get("bet_markets", ["1X2", "O/U", "BTTS"]))
    cands = [e for e in evals if "ev" in e and e["ev"] >= params["min_ev"] and e["market"].startswith(allowed)
             and params.get("min_odds", 1.3) <= e["odds"] <= params.get("max_odds", 6.0)]
    return max(cands, key=lambda e: e["ev"]) if cands else None


def base_confidence(info: dict, has_odds: bool, edge: float | None) -> int:
    c = 75.0
    c -= 20 * (1 - min(1.0, info.get("xg_share", 0.0)))          # no xG -> up to -20
    nmin = min(info.get("n_home", 0), info.get("n_away", 0))
    if nmin < 8:
        c -= (8 - nmin) * 2.0                                       # early season / new team
    if info.get("elo_source") == "built-in":
        c -= 3
    if not has_odds:
        c -= 10
    if edge is not None and abs(edge) > 0.10:
        c -= min(15, (abs(edge) - 0.10) * 100)                      # huge disagreement with market = suspect
    return int(round(max(5, min(95, c))))
