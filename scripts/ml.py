"""Machine-learning layer: gradient-boosted models on every available pre-match feature.

Targets: full-time result (H/D/A), over 2.5 goals, both teams to score.
Inputs: features.py (form, xG, shots, corners, venue form, rest, table, Elo, bookmaker probabilities,
Asian-handicap line) + the Dixon-Coles model's probabilities and expected goals.

    python scripts/ml.py --evaluate   # train on older seasons, test out-of-sample, write ml_report.md
    python scripts/ml.py --train      # fit on everything played so far -> data/ml_model.pkl
"""
from __future__ import annotations

import argparse
import datetime as dt
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import features
from common import DATA, ROOT, get_logger, load_params, season_code, write_json
from fetch import fetch_history, merge_understat
from model import Model, compute_elo

log = get_logger("ml")
MODEL_FILE = DATA / "ml_model.pkl"
DC_CACHE = DATA / "raw" / "dc_probs.csv"
DC_COLS = ["pH", "pD", "pA", "pO25", "pBTTS", "lh", "la"]


def all_history(today: dt.date) -> pd.DataFrame:
    seasons = [season_code(today, k) for k in range(-5, 1)]
    h = fetch_history(seasons)
    h = merge_understat(h)
    h, _ = compute_elo(h)
    return h


def dc_probs(hist: pd.DataFrame, seasons: list[str]) -> pd.DataFrame:
    """Walk-forward Dixon-Coles output for every match (weekly refits; cached for finished seasons)."""
    import backtest
    params = load_params()
    cached = pd.read_csv(DC_CACHE) if DC_CACHE.exists() else pd.DataFrame()
    current = season_code()
    frames = []
    model = Model(hist, params)
    for s in seasons:
        if len(cached) and s != current and (cached["Season"].astype(str).str.zfill(4) == s).any():
            frames.append(cached[cached["Season"].astype(str).str.zfill(4) == s])
            continue
        sim = backtest.simulate(model, s, params, collect_bets=False)
        sim["Season"] = s
        frames.append(sim[["Div", "Date", "home", "away", "Season"] + DC_COLS])
        log.info("Dixon-Coles walk-forward for %s: %d matches", s, len(sim))
    out = pd.concat(frames, ignore_index=True)
    out["Season"] = out["Season"].astype(str).str.zfill(4)
    out.to_csv(DC_CACHE, index=False)
    out["Date"] = pd.to_datetime(out["Date"])
    return out


def dataset(hist: pd.DataFrame, dc: pd.DataFrame) -> pd.DataFrame:
    X = features.build(hist)
    d = hist[["Div", "Date", "HomeTeam", "AwayTeam", "Season", "FTHG", "FTAG"]].copy()
    d = d.join(X)
    dc = dc.rename(columns={"home": "HomeTeam", "away": "AwayTeam"})
    d = d.merge(dc.drop(columns=["Season"]), on=["Div", "Date", "HomeTeam", "AwayTeam"], how="left")
    d["y_res"] = np.select([d.FTHG > d.FTAG, d.FTHG == d.FTAG], [0, 1], 2)
    d["y_o25"] = ((d.FTHG + d.FTAG) > 2.5).astype(int)
    d["y_btts"] = ((d.FTHG > 0) & (d.FTAG > 0)).astype(int)
    return d


def add_logits(X: pd.DataFrame) -> pd.DataFrame:
    """Log-odds versions of the bookmaker and Dixon-Coles probabilities (linear models need these)."""
    X = X.copy()
    for pre, cols in (("m", ["mH", "mD", "mA"]), ("dc", ["pH", "pD", "pA"])):
        if all(c in X for c in cols):
            P = np.clip(X[cols].to_numpy(float), 1e-4, 1)
            X[f"{pre}_lH"], X[f"{pre}_lA"] = np.log(P[:, 0] / P[:, 1]), np.log(P[:, 2] / P[:, 1])
    for pre, c in (("m", "mO25"), ("dc", "pO25")):
        if c in X:
            p = np.clip(X[c].to_numpy(float), 1e-4, 1 - 1e-4)
            X[f"{pre}_lO25"] = np.log(p / (1 - p))
    return X


def feature_cols(d: pd.DataFrame) -> list[str]:
    skip = {"Div", "Date", "HomeTeam", "AwayTeam", "Season", "FTHG", "FTAG", "y_res", "y_o25", "y_btts"}
    return [c for c in add_logits(d.head(1)).columns if c not in skip]


MARKET_BLEND = 0.5   # chosen on the 2024-25 validation season (see ml_report.md)


def make_clf():
    """Slow, heavily regularised boosting (won on the validation season)."""
    return HistGradientBoostingClassifier(learning_rate=0.02, max_iter=1500, max_leaf_nodes=8, min_samples_leaf=300,
                                          l2_regularization=5.0, early_stopping=True, validation_fraction=0.15,
                                          n_iter_no_change=40, random_state=7)


class LinearStacker:
    """Logistic regression on standardised features, median-imputed with missing-value indicators."""

    def fit(self, X: pd.DataFrame, y):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        self.cols = list(X.columns)
        self.med = X.median()
        self.miss = [c for c in self.cols if X[c].isna().mean() > 0.01]
        A = self._prep(X)
        self.scaler = StandardScaler().fit(A)
        self.clf = LogisticRegression(C=0.05, max_iter=3000).fit(self.scaler.transform(A), y)
        return self

    def _prep(self, X):
        X = X.reindex(columns=self.cols).astype(float)
        return pd.concat([X.fillna(self.med).fillna(0), X[self.miss].isna().astype(float).add_suffix("_na")], axis=1)

    def predict_proba(self, X):
        return self.clf.predict_proba(self.scaler.transform(self._prep(X)))


def fit(train: pd.DataFrame, cols: list[str]) -> dict:
    X = add_logits(train)[cols]
    models = {}
    for target in ("y_res", "y_o25", "y_btts"):
        y = train[target].to_numpy()
        hgb = make_clf().fit(X.to_numpy(float), y)
        lin = LinearStacker().fit(X, y)
        models[target] = (hgb, lin)
        log.info("trained %s on %d matches (boosting: %d iterations)", target, len(train), hgb.n_iter_)
    return {"models": models, "cols": cols, "market_blend": MARKET_BLEND}


def predict(bundle: dict, X: pd.DataFrame) -> pd.DataFrame:
    """Ensemble (boosting + logistic), then blended 50/50 with the bookmakers where their prices exist."""
    X = add_logits(X).reindex(columns=bundle["cols"])
    A = X.to_numpy(float)
    m, w = bundle["models"], bundle.get("market_blend", MARKET_BLEND)

    def ens(target):
        hgb, lin = m[target]
        return 0.5 * hgb.predict_proba(A) + 0.5 * lin.predict_proba(X)
    r = ens("y_res")
    mk = X[["mH", "mD", "mA"]].to_numpy(float)
    has = ~np.isnan(mk).any(axis=1)
    r[has] = (1 - w) * r[has] + w * mk[has]
    out = pd.DataFrame({"H": r[:, 0], "D": r[:, 1], "A": r[:, 2]}, index=X.index)
    o = ens("y_o25")[:, 1]
    mo = X["mO25"].to_numpy(float)
    out["O25"] = np.where(np.isnan(mo), o, (1 - w) * o + w * mo)
    out["BTTS"] = ens("y_btts")[:, 1]
    return out


# ---------------------------------------------------------------- evaluation
def _res_metrics(P: np.ndarray, y: np.ndarray) -> dict:
    ok = ~np.isnan(P).any(axis=1)
    P, y = P[ok], y[ok]
    Y = np.eye(3)[y]
    pick = P.argmax(axis=1)
    return {"n": int(len(y)), "accuracy": round(float((pick == y).mean()), 4),
            "log_loss": round(float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1)))), 4),
            "brier": round(float(np.mean(np.sum((P - Y) ** 2, axis=1))), 4)}


def _bin_metrics(p: np.ndarray, y: np.ndarray) -> dict:
    ok = ~np.isnan(p)
    p, y = p[ok], y[ok]
    return {"n": int(len(y)), "accuracy": round(float(((p >= 0.5) == y).mean()), 4),
            "log_loss": round(float(-np.mean(y * np.log(np.clip(p, 1e-12, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1)))), 4)}


def confidence_table(P: np.ndarray, y: np.ndarray) -> list[dict]:
    pmax, pick = P.max(axis=1), P.argmax(axis=1)
    low = P.argmin(axis=1)
    rows = []
    for t in (0.0, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75):
        s = pmax >= t
        if s.sum() < 20:
            continue
        rows.append({"threshold": t, "games": int(s.sum()), "share": round(float(s.mean()), 4),
                     "winner_hit": round(float((pick[s] == y[s]).mean()), 4),
                     "double_chance_hit": round(float((low[s] != y[s]).mean()), 4)})
    return rows


def evaluate_split(d: pd.DataFrame, train_seasons: list[str], test_season: str) -> dict:
    cols = feature_cols(d)
    tr = d[d.Season.isin(train_seasons)]
    te = d[d.Season == test_season].copy()
    bundle = fit(tr, cols)
    pr = predict(bundle, te)
    y = te["y_res"].to_numpy()
    dcP = te[["pH", "pD", "pA"]].to_numpy(float)
    mkP = te[["mH", "mD", "mA"]].to_numpy(float)
    mlP = pr[["H", "D", "A"]].to_numpy()
    both = ~np.isnan(mkP).any(axis=1) & ~np.isnan(dcP).any(axis=1)
    rep = {"train": train_seasons, "test": test_season, "n_train": int(len(tr)), "n_test": int(len(te)),
           "result": {"dixon_coles": _res_metrics(dcP[both], y[both]), "bookmakers": _res_metrics(mkP[both], y[both]),
                      "ml": _res_metrics(mlP[both], y[both]), "ml_all_games": _res_metrics(mlP, y)},
           "over25": {"dixon_coles": _bin_metrics(te["pO25"].to_numpy(float), te["y_o25"].to_numpy()),
                      "bookmakers": _bin_metrics(te["mO25"].to_numpy(float), te["y_o25"].to_numpy()),
                      "ml": _bin_metrics(pr["O25"].to_numpy(), te["y_o25"].to_numpy())},
           "btts": {"dixon_coles": _bin_metrics(te["pBTTS"].to_numpy(float), te["y_btts"].to_numpy()),
                    "ml": _bin_metrics(pr["BTTS"].to_numpy(), te["y_btts"].to_numpy())},
           "confidence_ml": confidence_table(mlP, y),
           "confidence_bookmakers": confidence_table(mkP[both], y[both])}
    # which features matter (permutation importance is slow; use a quick drop-in proxy: correlation of splits)
    return rep


def evaluate(today: dt.date | None = None) -> dict:
    today = today or dt.date.today()
    hist = all_history(today)
    seasons = sorted(hist["Season"].unique())
    dc = dc_probs(hist, seasons)
    d = dataset(hist, dc)
    d = d[d["FTHG"].notna()]
    cur, prev = season_code(today), season_code(today, -1)
    train = [s for s in seasons if s not in (seasons[0], prev, cur)]   # first season = warm-up only
    reports = [evaluate_split(d, train, prev)]
    if (d.Season == cur).sum() > 200:
        reports.append(evaluate_split(d, train + [prev], cur))
    write_json(DATA / "ml_eval.json", reports)
    write_markdown(reports)
    return reports


def pct(x):
    return f"{x * 100:.1f}%"


def write_markdown(reports: list[dict]) -> None:
    L = ["# Machine-learning layer: out-of-sample evaluation\n",
         "Gradient-boosted models trained on every pre-match feature (form, xG, shots, corners, venue form, rest, "
         "table, Elo, bookmaker probabilities, Asian-handicap line) plus the Dixon-Coles model's output. "
         "Each test season is never seen in training.\n"]
    for r in reports:
        L += [f"## Test season {r['test']} (trained on {', '.join(r['train'])}; {r['n_train']} training matches)\n",
              "### Result (1X2)\n", "| model | games | winner correct | log loss | Brier |", "|---|---|---|---|---|"]
        for k, name in (("dixon_coles", "Dixon-Coles (previous)"), ("bookmakers", "Bookmakers (no-vig)"), ("ml", "**ML layer**")):
            m = r["result"][k]
            L.append(f"| {name} | {m['n']} | {pct(m['accuracy'])} | {m['log_loss']} | {m['brier']} |")
        L += ["\n### Over/under 2.5 and BTTS\n", "| market | model | games | call correct | log loss |", "|---|---|---|---|---|"]
        for mk, lab in (("over25", "Over/under 2.5"), ("btts", "BTTS")):
            for k, v in r[mk].items():
                L.append(f"| {lab} | {k.replace('_', ' ')} | {v['n']} | {pct(v['accuracy'])} | {v['log_loss']} |")
        L += ["\n### Winner hit rate when only confident games are called\n",
              "| ML favourite at least | games called | share of games | winner correct | double chance correct |",
              "|---|---|---|---|---|"]
        L += [f"| {c['threshold']:.0%} | {c['games']} | {pct(c['share'])} | {pct(c['winner_hit'])} | {pct(c['double_chance_hit'])} |"
              for c in r["confidence_ml"]]
        L.append("")
    (ROOT / "ml_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def train_final(today: dt.date | None = None, before_season: str | None = None, path=MODEL_FILE) -> dict:
    """Fit on every played match (optionally only seasons before `before_season`, for honest replays)."""
    today = today or dt.date.today()
    hist = all_history(today)
    seasons = sorted(hist["Season"].unique())
    dc = dc_probs(hist, seasons)
    d = dataset(hist, dc)
    d = d[d["FTHG"].notna() & (d.Season != seasons[0])]
    if before_season:
        d = d[d.Season < before_season]
    bundle = fit(d, feature_cols(d))
    bundle["trained_at"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    bundle["trained_through"] = str(d["Date"].max().date())
    bundle["n_train"] = int(len(d))
    path.write_bytes(pickle.dumps(bundle))
    log.info("saved %s (%d matches, through %s)", path.name, len(d), bundle["trained_through"])
    return bundle


def load_bundle(path=MODEL_FILE) -> dict | None:
    try:
        return pickle.loads(path.read_bytes())
    except Exception:  # noqa: BLE001
        return None


def bundle_stale(days: int = 7) -> bool:
    b = load_bundle()
    if not b or "trained_at" not in b:
        return True
    return (pd.Timestamp.utcnow().tz_localize(None) - pd.Timestamp(b["trained_at"]).tz_localize(None)).days >= days


# ---------------------------------------------------------------- applying the model to games
STRONG, VERY_STRONG = 0.55, 0.60


def fixture_features(hist_elo: pd.DataFrame, rows: dict, elo: dict, season: str) -> pd.DataFrame:
    """Features for upcoming games (rows: {game_id: football-data style row}); history supplies the form."""
    from common import LEAGUES, TIER_ELO
    fx = pd.DataFrame.from_dict(rows, orient="index")
    fx["Date"] = pd.to_datetime(fx["Date"])
    fx["Season"] = season
    fx["FTHG"] = np.nan
    fx["FTAG"] = np.nan
    for side, col in (("HomeTeam", "EloH"), ("AwayTeam", "EloA")):
        if col not in fx or fx[col].isna().all():
            fx[col] = [elo.get((LEAGUES[r.Div][1], getattr(r, side)), TIER_ELO.get(LEAGUES[r.Div][2], 1300))
                       for r in fx.itertuples()]
    start = fx["Date"].min() if len(fx) else hist_elo["Date"].max()
    # form windows need ~40 games per team and the current season's table: 400 days is plenty
    base = hist_elo[(hist_elo["Date"] < start) & (hist_elo["Date"] >= start - pd.Timedelta(days=400))]
    allm = pd.concat([base, fx]).sort_values("Date", kind="stable")
    return features.build(allm).loc[fx.index]


def apply(bundle: dict, records: list[dict], feats: pd.DataFrame) -> None:
    """Replace each record's probabilities with the ML ensemble and add pick tiers (in place)."""
    ids = [r["id"] for r in records if r["id"] in feats.index]
    if not ids or not bundle:
        return
    X = feats.loc[ids].copy()
    by_id = {r["id"]: r for r in records}
    X["pH"] = [by_id[i]["probs"]["H"] for i in ids]
    X["pD"] = [by_id[i]["probs"]["D"] for i in ids]
    X["pA"] = [by_id[i]["probs"]["A"] for i in ids]
    X["pO25"] = [(by_id[i]["markets"].get("O/U 2.5") or {}).get("Over") for i in ids]
    X["pBTTS"] = [(by_id[i]["markets"].get("BTTS") or {}).get("Yes") for i in ids]
    X["lh"] = [by_id[i]["xg"]["home"] for i in ids]
    X["la"] = [by_id[i]["xg"]["away"] for i in ids]
    P = predict(bundle, X)
    for i in ids:
        r, p = by_id[i], P.loc[i]
        r["probs_dc"] = dict(r["probs"])
        r["probs"] = {"H": float(p.H), "D": float(p.D), "A": float(p.A)}
        r["markets"].setdefault("O/U 2.5", {}).update({"Over": float(p.O25), "Under": float(1 - p.O25)})
        r["markets"].setdefault("BTTS", {}).update({"Yes": float(p.BTTS), "No": float(1 - p.BTTS)})
        r["prob_source"] = "ml"
        r["pick"] = make_pick(r)


def make_pick(r: dict) -> dict:
    p = r["probs"]
    top = max("HDA", key=lambda s: p[s])
    low = min("HDA", key=lambda s: p[s])
    name = {"H": r["home"], "D": "Draw", "A": r["away"]}
    dc = [s for s in "HDA" if s != low]
    dc_label = {("H", "D"): f"{r['home']} or draw", ("D", "A"): f"{r['away']} or draw",
                ("H", "A"): f"{r['home']} or {r['away']}"}[tuple(dc)]
    tier = "very strong" if p[top] >= VERY_STRONG else "strong" if p[top] >= STRONG else None
    return {"selection": top, "label": name[top] if top == "D" else f"{name[top]} to win", "p": p[top], "tier": tier,
            "double_chance": {"selections": "".join(dc), "label": dc_label, "p": 1 - p[low]}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--train", action="store_true")
    a = ap.parse_args()
    import ml as _ml  # run through the importable module so pickled classes load from main.py too
    if a.evaluate:
        import json
        for r in _ml.evaluate():
            print(json.dumps({k: r[k] for k in ("test", "result", "over25", "btts", "confidence_ml")}, indent=1))
    if a.train:
        _ml.train_final()
