"""Pre-match feature engineering for the machine-learning layer.

Every feature of a match is computed only from matches played on earlier dates (no look-ahead):
form (points, goals, xG, shots on target, corners over the last 5/10 games), venue-specific form,
rest and congestion, league-table position and points per game, Elo, bookmaker no-vig
probabilities (1X2, over/under 2.5, Asian-handicap line) and the Dixon-Coles model's own output.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from common import LEAGUES

WINDOW = 10


def _nv(*odds):
    try:
        imp = [1.0 / float(o) for o in odds]
    except (TypeError, ValueError, ZeroDivisionError):
        return [np.nan] * len(odds), np.nan
    if any(not np.isfinite(x) or x <= 0 for x in imp):
        return [np.nan] * len(odds), np.nan
    s = sum(imp)
    return [x / s for x in imp], s


def market_features(df: pd.DataFrame) -> pd.DataFrame:
    """No-vig bookmaker probabilities from opening average odds (falls back to Bet365)."""
    g = lambda c: df[c] if c in df else pd.Series(np.nan, index=df.index)  # noqa: E731
    H, D, A = (g(f"Avg{s}").fillna(g(f"B365{s}")) for s in "HDA")
    out = pd.DataFrame(index=df.index)
    probs = [_nv(h, d, a) for h, d, a in zip(H, D, A)]
    out["mH"] = [p[0][0] for p in probs]
    out["mD"] = [p[0][1] for p in probs]
    out["mA"] = [p[0][2] for p in probs]
    out["m_overround"] = [p[1] for p in probs]
    ou = [_nv(o, u)[0][0] for o, u in zip(g("Avg>2.5"), g("Avg<2.5"))]
    out["mO25"] = ou
    out["ah_line"] = g("AHh")
    ah = [_nv(h, a)[0][0] for h, a in zip(g("AvgAHH"), g("AvgAHA"))]
    out["ah_home_p"] = ah
    # sanity: drop implausible rows (corrupt odds) rather than feed them to the model
    bad = (out["m_overround"] < 0.99) | (out["m_overround"] > 1.2)
    out.loc[bad, ["mH", "mD", "mA", "m_overround"]] = np.nan
    return out


def _xg(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    from model import sot_conversion
    conv = sot_conversion(df)   # previous season's rate: no look-ahead
    hx = df["HxG"].where(df["HxG"].notna(), df["HST"] * conv) if "HxG" in df else df["HST"] * conv
    ax = df["AxG"].where(df["AxG"].notna(), df["AST"] * conv) if "AxG" in df else df["AST"] * conv
    return hx, ax


def _team_feats(hist: deque, venue_hist: deque, date) -> dict:
    f = {}
    L = list(hist)
    for n in (5, WINDOW):
        last = L[-n:]
        if last:
            f[f"ppg{n}"] = np.mean([m["pts"] for m in last])
            f[f"gd{n}"] = np.mean([m["gf"] - m["ga"] for m in last])
        else:
            f[f"ppg{n}"] = f[f"gd{n}"] = np.nan
    last = L[-WINDOW:]

    def avg(k):
        v = [m[k] for m in last if m[k] is not None and np.isfinite(m[k])]
        return float(np.mean(v)) if v else np.nan
    f["xgf"], f["xga"] = avg("xgf"), avg("xga")
    f["xgd"] = f["xgf"] - f["xga"]
    f["sotd"] = avg("sotf") - avg("sota")
    f["cornd"] = avg("cf") - avg("ca")
    f["cards"] = avg("cards")
    V = list(venue_hist)[-8:]
    f["venue_ppg"] = np.mean([m["pts"] for m in V]) if V else np.nan
    f["venue_gd"] = np.mean([m["gf"] - m["ga"] for m in V]) if V else np.nan
    if L:
        f["rest"] = min((date - L[-1]["date"]).days, 30)
        f["g14"] = sum(1 for m in L[-6:] if (date - m["date"]).days <= 14)
    else:
        f["rest"], f["g14"] = np.nan, np.nan
    f["n_hist"] = min(len(L), 40)
    return f


def build(matches: pd.DataFrame) -> pd.DataFrame:
    """matches: football-data style rows sorted by Date (played rows have FTHG/FTAG; upcoming rows NaN).
    Returns a feature frame with the same index."""
    df = matches.copy()
    for c in ("HST", "AST", "HC", "AC", "HY", "AY", "HR", "AR", "HxG", "AxG"):
        if c not in df:
            df[c] = np.nan
    hx, ax = _xg(df)
    df["_hx"], df["_ax"] = hx, ax
    hist: dict = defaultdict(lambda: deque(maxlen=40))
    venue: dict = defaultdict(lambda: deque(maxlen=12))
    table: dict = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))  # (div, season) -> team -> [P, Pts, GD]
    rows = {}
    for date, day in df.groupby("Date", sort=True):
        for i, r in day.iterrows():
            ctry = LEAGUES.get(r.Div, ("", "?"))[1]
            kh, ka = (ctry, r.HomeTeam), (ctry, r.AwayTeam)
            fh = _team_feats(hist[kh], venue[(kh, "H")], date)
            fa = _team_feats(hist[ka], venue[(ka, "A")], date)
            t = table[(r.Div, r.Season)]
            n_teams = max(len(t), 2)
            ranked = sorted(t.items(), key=lambda kv: (-kv[1][1], -kv[1][2]))
            pos = {team: k + 1 for k, (team, _) in enumerate(ranked)}
            feat = {}
            for side, f, team in (("h", fh, r.HomeTeam), ("a", fa, r.AwayTeam)):
                for k, v in f.items():
                    feat[f"{side}_{k}"] = v
                P, pts, gd = t.get(team, [0, 0, 0]) if team in t else (0, 0, 0)
                feat[f"{side}_sppg"] = pts / P if P else np.nan
                feat[f"{side}_sgd"] = gd / P if P else np.nan
                feat[f"{side}_pos"] = pos.get(team, np.nan) / n_teams if team in pos else np.nan
            for k in ("ppg5", "ppg10", "gd10", "xgd", "sotd", "cornd", "venue_ppg", "rest", "sppg", "pos"):
                feat[f"d_{k}"] = feat[f"h_{k}"] - feat[f"a_{k}"]
            played = max((v[0] for v in t.values()), default=0)
            feat["season_progress"] = played / (2 * (n_teams - 1)) if n_teams > 1 else 0
            feat["tier"] = LEAGUES.get(r.Div, (0, 0, 3))[2]
            rows[i] = feat
        # update state with this date's results
        for i, r in day.iterrows():
            if pd.isna(r.FTHG) or pd.isna(r.FTAG):
                continue
            ctry = LEAGUES.get(r.Div, ("", "?"))[1]
            hg, ag = int(r.FTHG), int(r.FTAG)
            ph = 3 if hg > ag else 1 if hg == ag else 0
            pa = 3 if ag > hg else 1 if hg == ag else 0
            cards_h = (r.HY if np.isfinite(r.HY) else np.nan) + 2 * (r.HR if np.isfinite(r.HR) else 0)
            cards_a = (r.AY if np.isfinite(r.AY) else np.nan) + 2 * (r.AR if np.isfinite(r.AR) else 0)
            mh = dict(date=date, gf=hg, ga=ag, pts=ph, xgf=r._hx, xga=r._ax, sotf=r.HST, sota=r.AST, cf=r.HC, ca=r.AC, cards=cards_h)
            ma = dict(date=date, gf=ag, ga=hg, pts=pa, xgf=r._ax, xga=r._hx, sotf=r.AST, sota=r.HST, cf=r.AC, ca=r.HC, cards=cards_a)
            hist[(ctry, r.HomeTeam)].append(mh)
            hist[(ctry, r.AwayTeam)].append(ma)
            venue[((ctry, r.HomeTeam), "H")].append(mh)
            venue[((ctry, r.AwayTeam), "A")].append(ma)
            t = table[(r.Div, r.Season)]
            for team, gf, ga, p in ((r.HomeTeam, hg, ag, ph), (r.AwayTeam, ag, hg, pa)):
                t[team][0] += 1
                t[team][1] += p
                t[team][2] += gf - ga
    out = pd.DataFrame.from_dict(rows, orient="index").reindex(df.index)
    out = pd.concat([out, market_features(df)], axis=1)
    if "EloH" in df:
        out["elo_diff"] = df["EloH"] - df["EloA"]
    return out
