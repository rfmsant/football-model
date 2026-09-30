"""Offline tests: python -m unittest discover tests"""
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import backtest  # noqa: E402
import extras  # noqa: E402
import notify  # noqa: E402
import oddsapi  # noqa: E402
import summary  # noqa: E402
import track  # noqa: E402
from common import DEFAULT_PARAMS, match_name, norm_name  # noqa: E402
from model import (Model, ah_outcomes, best_bet, derive_markets, dixon_coles_matrix,  # noqa: E402
                   evaluate_markets, market_odds_from_row, no_vig)


def fake_history(n_weeks=30, seed=1):
    rng = np.random.default_rng(seed)
    teams = [f"Team {c}" for c in "ABCDEFGHIJ"]
    strength = {t: s for t, s in zip(teams, np.linspace(0.7, 1.4, len(teams)))}
    rows, d = [], pd.Timestamp("2026-01-03")
    for w in range(n_weeks):
        order = rng.permutation(teams)
        for i in range(0, len(order), 2):
            h, a = order[i], order[i + 1]
            hg, ag = rng.poisson(1.4 * strength[h] / strength[a]), rng.poisson(1.1 * strength[a] / strength[h])
            rows.append({"Div": "E0", "Season": "2526", "Date": d + pd.Timedelta(days=7 * w), "HomeTeam": h, "AwayTeam": a,
                         "FTHG": hg, "FTAG": ag, "HST": hg * 3 + 2, "AST": ag * 3 + 2, "HxG": np.nan, "AxG": np.nan,
                         "AvgH": 2.2, "AvgD": 3.4, "AvgA": 3.3, "MaxH": 2.3, "MaxD": 3.5, "MaxA": 3.4})
    return pd.DataFrame(rows)


class TestNames(unittest.TestCase):
    def test_matching(self):
        cands = ["Manchester City", "Manchester United", "Wolverhampton Wanderers", "Nottingham Forest"]
        self.assertEqual(match_name("Man City", cands), "Manchester City")
        self.assertEqual(match_name("Man United", cands), "Manchester United")
        self.assertEqual(match_name("Wolves", cands), "Wolverhampton Wanderers")
        self.assertEqual(match_name("Nott'm Forest", cands), "Nottingham Forest")
        self.assertIsNone(match_name("Real Madrid", cands))
        self.assertNotEqual(norm_name("Man City"), norm_name("Man United"))


class TestModel(unittest.TestCase):
    def test_matrix_and_markets(self):
        mat = dixon_coles_matrix(1.6, 1.1, -0.08)
        self.assertEqual(mat.shape, (7, 7))
        self.assertAlmostEqual(mat.sum(), 1.0, places=9)
        mk = derive_markets(mat, [-0.5, -0.25, 0.0])
        x = mk["1X2"]
        self.assertAlmostEqual(x["H"] + x["D"] + x["A"], 1.0, places=9)
        self.assertGreater(x["H"], x["A"])
        self.assertAlmostEqual(mk["AH -0.5"]["Home"], x["H"], places=9)   # -0.5 == home win
        self.assertAlmostEqual(mk["AH +0"]["Home"], x["H"] / (x["H"] + x["A"]), places=9)  # draw no bet
        self.assertGreater(mk["O/U 1.5"]["Over"], mk["O/U 2.5"]["Over"])
        w, p, l = ah_outcomes(mat, -0.25)
        self.assertAlmostEqual(w + p + l, 1.0, places=9)

    def test_no_vig_and_ev(self):
        f = no_vig([2.0, 3.5, 4.0])
        self.assertAlmostEqual(sum(f), 1.0)
        self.assertIsNone(no_vig([2.0, None, 3.0]))
        mk = {"1X2": {"H": 0.6, "D": 0.25, "A": 0.15}}
        odds = {"1X2": {"H": {"avg": 2.0, "max": 2.1}, "D": {"avg": 3.6, "max": 3.7}, "A": {"avg": 4.5, "max": 4.6}}}
        ev = evaluate_markets(mk, odds, 0.5)
        h = next(e for e in ev if e["selection"] == "H")
        self.assertGreater(h["edge"], 0)
        self.assertAlmostEqual(h["ev"], h["p_bet"] * 2.1 - 1, places=3)
        self.assertEqual(best_bet(ev, DEFAULT_PARAMS)["selection"], "H")

    def test_no_asian_handicap_bets(self):
        ev = [{"market": "AH -0.5", "selection": "Home", "ev": 0.2, "odds": 2.0},
              {"market": "O/U 2.5", "selection": "Over", "ev": 0.05, "odds": 1.9}]
        self.assertEqual(best_bet(ev, DEFAULT_PARAMS)["market"], "O/U 2.5")

    def test_price_cap(self):
        row = {"AvgH": 2.0, "AvgD": 3.4, "AvgA": 4.0, "MaxH": 3.5, "MaxD": 3.5, "MaxA": 4.1}
        mk = {"1X2": {"H": 0.5, "D": 0.27, "A": 0.23}}
        h = next(e for e in evaluate_markets(mk, market_odds_from_row(row), 0.5) if e["selection"] == "H")
        self.assertLessEqual(h["odds"], 2.0 * 1.07 + 1e-9)

    def test_fit_and_predict(self):
        hist = fake_history()
        m = Model(hist, dict(DEFAULT_PARAMS))
        ref = hist["Date"].max() + pd.Timedelta(days=1)
        strong = m.predict("E0", "Team J", "Team A", ref, "2526")
        weak = m.predict("E0", "Team A", "Team J", ref, "2526")
        self.assertGreater(strong["markets"]["1X2"]["H"], weak["markets"]["1X2"]["H"])
        unknown = m.predict("E0", "Nobody FC", "Team A", ref, "2526")   # unseen team must not crash
        self.assertIn("1X2", unknown["markets"])


class TestBacktest(unittest.TestCase):
    def test_settle(self):
        self.assertEqual(backtest.settle("1X2", "H", 2, 1, 2.0), 1.0)
        self.assertEqual(backtest.settle("O/U 2.5", "Under", 2, 1, 1.9), -1.0)
        self.assertEqual(backtest.settle("AH -1", "Home", 2, 1, 1.9), 0.0)          # push
        self.assertAlmostEqual(backtest.settle("AH -0.75", "Home", 1, 0, 2.0), 0.5)  # half win
        self.assertAlmostEqual(backtest.settle("AH -0.25", "Away", 1, 1, 2.0), 0.5)  # away +0.25 on a draw


def af_response(path, params):
    if path == "fixtures":
        return {"errors": [], "response": [{"fixture": {"id": 99}, "teams": {"home": {"id": 1, "name": "Manchester City"},
                                                                          "away": {"id": 2, "name": "Wolverhampton Wanderers"}}}]}
    if path == "injuries":
        return {"errors": [], "response": [
            {"player": {"id": 10, "name": "Striker Nine", "type": "Missing Fixture", "reason": "Knee Injury"}, "team": {"id": 1}},
            {"player": {"id": 11, "name": "Keeper One", "type": "Questionable", "reason": "Illness"}, "team": {"id": 2}}]}
    if path == "players":
        pid = params["id"]
        stats = {10: ("Attacker", 900, 8, 2), 11: ("Goalkeeper", 900, 0, 0)}[pid]
        return {"errors": [], "response": [{"statistics": [{"league": {"id": 39}, "games": {"position": stats[0], "minutes": stats[1]},
                                                             "goals": {"total": stats[2], "assists": stats[3]}}]}]}
    if path == "coachs":
        return {"errors": [], "response": [{"career": [{"team": {"id": params["team"]}, "start": "2026-09-01", "end": None}]}]}
    return {"errors": {"x": "unknown"}, "response": []}


class FakeResp:
    def __init__(self, data, status=200, headers=None):
        self._d, self.status_code, self.headers, self.text = data, status, headers or {}, ""

    def json(self):
        return self._d


def fake_http_get(url, params=None, headers=None, **kw):
    if "api-sports" in url:
        return FakeResp(af_response(url.rsplit("/", 1)[1], params or {}), headers={"x-ratelimit-requests-remaining": "80"})
    if "the-odds-api" in url:
        return FakeResp([{"home_team": "Manchester City", "away_team": "Wolverhampton Wanderers", "bookmakers": [
            {"key": "bk1", "markets": [
                {"key": "h2h", "outcomes": [{"name": "Manchester City", "price": 1.5}, {"name": "Draw", "price": 4.5},
                                            {"name": "Wolverhampton Wanderers", "price": 7.0}]},
                {"key": "totals", "outcomes": [{"name": "Over", "price": 1.3, "point": 1.5}, {"name": "Under", "price": 3.4, "point": 1.5}]}]},
            {"key": "betfair_ex_uk", "markets": [{"key": "h2h", "outcomes": [{"name": "Manchester City", "price": 9.9}]}]}]}])
    return None


class TestExtras(unittest.TestCase):
    def setUp(self):
        rows = []
        d = pd.Timestamp("2026-08-15")
        for k in range(6):
            rows.append({"Div": "E0", "Season": "2627", "Date": d + pd.Timedelta(days=7 * k), "HomeTeam": "Man City",
                         "AwayTeam": f"Opp{k}", "FTHG": 2, "FTAG": 0})
            rows.append({"Div": "E0", "Season": "2627", "Date": d + pd.Timedelta(days=7 * k + 1), "HomeTeam": f"Opp{k}",
                         "AwayTeam": "Wolves", "FTHG": 1, "FTAG": 1})
        rows.append({"Div": "E0", "Season": "2627", "Date": pd.Timestamp("2026-10-01"), "HomeTeam": "Wolves",
                     "AwayTeam": "Opp9", "FTHG": 0, "FTAG": 0})
        self.hist = pd.DataFrame(rows)
        self.games = [{"id": "g1", "league": "E0", "home": "Man City", "away": "Wolves", "date": "2026-10-03"}]

    def test_without_keys(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            out = extras.run(self.games, self.hist, "2627")
        g = out["games"]["g1"]
        self.assertFalse(g["injury_data"])
        self.assertEqual(g["away_context"]["rest_days"], 2)
        self.assertLess(g["lambda_mult"][1], 1.0)   # Wolves on short rest score less

    def test_with_mocked_apis(self):
        with mock.patch.dict(os.environ, {"APIFOOTBALL_KEY": "x", "ODDS_API_KEY": "y"}), \
                mock.patch.object(extras, "http_get", side_effect=fake_http_get), \
                mock.patch.object(oddsapi, "http_get", side_effect=fake_http_get):
            out = extras.run(self.games, self.hist, "2627")
        g = out["games"]["g1"]
        self.assertTrue(g["injury_data"])
        self.assertEqual(g["home_absences"][0]["name"], "Striker Nine")
        self.assertGreater(g["home_absences"][0]["goal_share"], 0.5)
        self.assertEqual(g["away_absences"][0]["status"], "doubtful")
        self.assertTrue(any("new manager" in f for f in g["factors"]))
        self.assertEqual(g["extra_odds"]["1X2"]["H"]["max"], 1.5)   # exchange price ignored
        self.assertIn("O/U 1.5", g["extra_odds"])
        merged = extras.merge_odds({}, g["extra_odds"])
        self.assertIn("O/U 1.5", merged)

    def test_api_error_is_graceful(self):
        def broken(url, **kw):
            return FakeResp({"errors": {"token": "Error/Missing application key"}, "response": []})
        with mock.patch.dict(os.environ, {"APIFOOTBALL_KEY": "bad"}), mock.patch.object(extras, "http_get", side_effect=broken):
            out = extras.run(self.games, self.hist, "2627")
        self.assertFalse(out["games"]["g1"]["injury_data"])
        self.assertIn("application key", out["api_football_error"])


class TestOddsBudget(unittest.TestCase):
    def test_runs_left(self):
        import datetime as dt
        self.assertEqual(oddsapi.runs_left_this_month(dt.date(2026, 10, 1)), 31)  # daily runs
        self.assertEqual(oddsapi.runs_left_this_month(dt.date(2026, 9, 30)), 1)   # last day of month

    def _client(self, remaining, today):
        import datetime as dt
        resp = FakeResp([], headers={"x-requests-remaining": str(remaining)})
        with mock.patch.object(oddsapi, "http_get", return_value=resp):
            return oddsapi.OddsClient("k", dt.date(*today))

    def test_budget_never_exceeds_month(self):
        c = self._client(500, (2026, 10, 1))
        self.assertEqual(c.budget, 15)                  # (500 - 15) // 31 days
        c = self._client(60, (2026, 10, 27))            # 5 days left, 45 usable credits -> 9 per run
        self.assertEqual(c.budget, 9)
        c = self._client(10, (2026, 10, 30))            # below the reserve -> nothing
        self.assertEqual(c.budget, 0)
        self.assertFalse(c.can_fetch())

    def test_league_cached_and_budgeted(self):
        c = self._client(500, (2026, 9, 30))
        c.budget = 3                                    # room for one 2-credit league
        with mock.patch.object(oddsapi, "http_get", side_effect=fake_http_get) as g:
            self.assertTrue(c.league("E0"))
            self.assertTrue(c.league("E0"))             # cached: no second request
            self.assertIsNone(c.league("D1"))           # over budget
            self.assertEqual(g.call_count, 1)
        self.assertEqual(c.used, 2)

    def test_discovery_adds_missing_games(self):
        import datetime as dt
        c = self._client(500, (2026, 10, 1))
        c.cache["E0"] = [{"home_team": "Manchester City", "away_team": "Wolverhampton Wanderers",
                          "commence_time": "2026-10-03T14:00:00Z", "bookmakers": fake_http_get("the-odds-api")._d[0]["bookmakers"]}]
        for lg in oddsapi.CORE_LEAGUES[1:]:
            c.cache[lg] = []
        hist = pd.DataFrame({"Div": ["E0"], "Season": ["2627"], "HomeTeam": ["Man City"], "AwayTeam": ["Wolves"]})
        with mock.patch.object(c, "events", side_effect=lambda lg: c.cache.get(lg, [])):
            fx = oddsapi.discover_fixtures(c, pd.DataFrame(columns=["Div", "HomeTeam", "AwayTeam"]), hist, "2627", dt.date(2026, 10, 1))
            self.assertEqual(len(fx), 1)
            r = fx.iloc[0]
            self.assertEqual((r.HomeTeam, r.AwayTeam, r.Time), ("Man City", "Wolves", "15:00"))
            self.assertEqual(r.AvgH, 1.5)
            again = oddsapi.discover_fixtures(c, fx, hist, "2627", dt.date(2026, 10, 1))   # no duplicates
            self.assertEqual(len(again), 1)

    def test_discovery_soonest_first_and_skips_listed_leagues(self):
        import datetime as dt
        c = self._client(500, (2026, 9, 30))
        c.budget = 2                                    # room for exactly one league
        ev = lambda h, a, t: {"home_team": h, "away_team": a, "commence_time": t, "bookmakers": []}
        events = {"E0": [ev("Arsenal", "Leeds", "2026-10-10T11:30:00Z")],
                  "E2": [ev("Barnsley", "Wigan", "2026-10-03T14:00:00Z")],
                  "SP2": [ev("Racing Santander", "Eibar", "2026-10-02T18:30:00Z")]}
        hist = pd.DataFrame({"Div": ["E0", "E2", "SP2"], "Season": "2627", "HomeTeam": ["Arsenal", "Barnsley", "Santander"],
                             "AwayTeam": ["Leeds", "Wigan", "Eibar"]})
        fetched = []
        with mock.patch.object(c, "events", side_effect=lambda lg: events.get(lg, [])),                 mock.patch.object(c, "league", side_effect=lambda lg: fetched.append(lg) or None):
            oddsapi.discover_fixtures(c, pd.DataFrame(columns=["Div", "HomeTeam", "AwayTeam"]), hist, "2627", dt.date(2026, 9, 30))
        self.assertEqual(fetched, ["SP2"])              # Friday's Segunda game first; stops when over budget
        listed = pd.DataFrame({"Div": ["SP2"], "HomeTeam": ["Santander"], "AwayTeam": ["Eibar"]})
        fetched.clear()
        with mock.patch.object(c, "events", side_effect=lambda lg: events.get(lg, [])),                 mock.patch.object(c, "league", side_effect=lambda lg: fetched.append(lg) or None):
            oddsapi.discover_fixtures(c, listed, hist, "2627", dt.date(2026, 9, 30))
        self.assertEqual(fetched[:1], ["E2"])           # SP2 already in fixtures.csv -> no credits spent on it


class TestTrackRecord(unittest.TestCase):
    def pred(self):
        return {"generated_at": "2026-10-03T07:00:00Z", "games": [{
            "league": "E2", "league_name": "League One", "date": "2026-10-03", "time": "15:00", "home": "Barnsley",
            "away": "Wigan", "probs": {"H": 0.5, "D": 0.27, "A": 0.23}, "likely_score": "1-0", "confidence": 55,
            "markets": {"O/U 2.5": {"Over": 0.45, "Under": 0.55}, "BTTS": {"Yes": 0.52, "No": 0.48}},
            "lean": {"market": "1X2", "selection": "H", "label": "Barnsley to win", "odds": 2.1, "model_p": 0.5,
                     "market_p": 0.46, "ev": 0.03, "edge": 0.04},
            "best_bet": {"market": "O/U 2.5", "selection": "Under", "label": "Under 2.5 goals", "odds": 1.9,
                         "model_p": 0.55, "market_p": 0.5, "ev": 0.04, "edge": 0.05}}]}

    def test_record_settle_summarise(self):
        import datetime as dt
        ledger = {"games": {}}
        self.assertEqual(track.record_predictions(ledger, self.pred(), dt.date(2026, 10, 3)), 1)
        self.assertEqual(track.record_predictions(ledger, self.pred(), dt.date(2026, 10, 4)), 0)  # past game: frozen
        hist = pd.DataFrame({"Div": ["E2"], "Date": [pd.Timestamp("2026-10-04")], "HomeTeam": ["Barnsley"],
                             "AwayTeam": ["Wigan"], "FTHG": [2], "FTAG": [0]})   # played a day late
        self.assertEqual(track.settle_results(ledger, hist), 1)
        e = next(iter(ledger["games"].values()))
        o = e["outcome"]
        self.assertTrue(o["winner_ok"] and o["ou_ok"])
        self.assertFalse(o["score_ok"])
        self.assertFalse(o["btts_ok"])                       # predicted yes, Wigan didn't score
        self.assertAlmostEqual(o["lean_pl"], 1.1)
        self.assertAlmostEqual(o["value_pl"], 0.9)
        self.assertEqual(o["value_type"], "Goals (over/under)")
        s = track.summarise(ledger)
        self.assertEqual(s["winner"]["pct"], 1.0)
        self.assertEqual(s["value_bets"]["All"]["profit"], 0.9)
        self.assertEqual(track.settle_results(ledger, hist), 0)   # already settled

    def test_settle_btts(self):
        self.assertEqual(backtest.settle("BTTS", "Yes", 1, 1, 1.8), 0.8)
        self.assertEqual(backtest.settle("BTTS", "Yes", 1, 0, 1.8), -1.0)


class TestML(unittest.TestCase):
    def test_make_pick_tiers(self):
        import ml
        r = {"home": "A", "away": "B", "probs": {"H": 0.62, "D": 0.22, "A": 0.16}}
        p = ml.make_pick(r)
        self.assertEqual((p["label"], p["tier"]), ("A to win", "very strong"))
        self.assertEqual(p["double_chance"]["label"], "A or draw")
        self.assertAlmostEqual(p["double_chance"]["p"], 0.84)
        r["probs"] = {"H": 0.30, "D": 0.33, "A": 0.37}
        p = ml.make_pick(r)
        self.assertIsNone(p["tier"])
        self.assertEqual(p["double_chance"]["label"], "B or draw")

    def test_features_no_lookahead(self):
        import features
        h = fake_history(n_weeks=6)
        f1 = features.build(h)
        h2 = h.copy()
        last = h2["Date"] == h2["Date"].max()
        h2.loc[last, ["FTHG", "FTAG"]] = [9, 9]          # change only the final day's results
        f2 = features.build(h2)
        pd.testing.assert_frame_equal(f1[last.values], f2[last.values])   # that day's features can't see them


class TestDeepDive(unittest.TestCase):
    def bet(self, **kw):
        b = {"rank": 1, "match": "A v B", "league": "E2", "kickoff": "2026-10-03 15:00", "market": "Draw no bet",
             "selection": "A", "label": "B draw no bet", "model_p": 0.61, "final_p": 0.72, "odds_seen": 1.55, "min_odds": 1.4,
             "meets_bar": True, "adjustments": [{"factor": "injuries", "delta": 0.11, "reason": "x"}],
             "analysis": {"injuries": "x"}, "verdict": "BET"}
        b.update(kw)
        return b

    def test_validate(self):
        import deep
        self.assertEqual(deep.validate({"date": "2026-10-01", "bets": [self.bet()]}), [])
        errs = deep.validate({"date": "2026-10-01", "bets": [self.bet(final_p=0.80), self.bet(market="AH -0.5"),
                                                             self.bet(odds_seen=1.3), self.bet(verdict="MAYBE")]})
        self.assertEqual(len(errs), 4)   # sum mismatch, bad market, meets_bar wrong, bad verdict

    def test_settle_new_markets(self):
        self.assertAlmostEqual(backtest.settle("Double chance", "1X", 1, 1, 1.5), 0.5)
        self.assertEqual(backtest.settle("Double chance", "X2", 2, 1, 1.5), -1.0)
        self.assertEqual(backtest.settle("Draw no bet", "A", 1, 1, 1.8), 0.0)
        self.assertAlmostEqual(backtest.settle("Draw no bet", "A", 0, 2, 1.8), 0.8)
        self.assertAlmostEqual(backtest.settle("O/U 1.5", "Over", 1, 1, 1.3), 0.3)

    def test_market_lambda_fit(self):
        import shortlist
        from model import dixon_coles_matrix
        lh, la = shortlist.fit_market_lambdas(0.55, 0.20, 0.55)
        m = shortlist._mk(dixon_coles_matrix(lh, la, -0.08))
        self.assertAlmostEqual(m["H"], 0.55, delta=0.02)
        self.assertAlmostEqual(m["O25"], 0.55, delta=0.03)


class TestOutput(unittest.TestCase):
    def game(self):
        return {"home": "A", "away": "B", "probs": {"H": 0.5, "D": 0.27, "A": 0.23}, "xg": {"home": 1.7, "away": 1.0},
                "likely_score": "1-0", "confidence": 62, "has_xg": False, "injury_data": False, "factors": [],
                "best_bet": {"market": "AH -0.25", "selection": "Home", "odds": 1.95, "model_p": 0.58, "market_p": 0.52,
                             "edge": 0.06, "ev": 0.08, "label": "A -0.25 (Asian handicap)"},
                "league_name": "Premier League", "date": "2026-10-03", "time": "15:00"}

    def test_summary(self):
        s = summary.generate(self.game())
        self.assertIn("A -0.25", s)
        self.assertIn("no xG", s)

    def test_daily_overview(self):
        import datetime as dt
        base = self.game()
        g1 = dict(base, id="a", markets={"O/U 2.5": {"Over": 0.6}}, evals=[], lean=None, best_bet=None)
        g2 = dict(base, id="b", home="C", away="D", probs={"H": 0.34, "D": 0.32, "A": 0.34}, xg={"home": 1.0, "away": 0.9},
                  markets={}, evals=[{"market": "1X2", "selection": "H", "edge": 0.08, "model_p": 0.34, "market_p": 0.26}])
        ov = summary.daily_overview("2026-10-03", [g1, g2], dt.date(2026, 10, 3))
        self.assertTrue(ov["text"].startswith("Today, 03 Oct: 2 games"))
        kinds = {h["kind"]: h["game"] for h in ov["highlights"]}
        self.assertEqual(kinds["Strongest favourite"], "a")
        self.assertEqual(kinds["Closest call"], "b")
        self.assertEqual(kinds["Model vs bookies"], "b")
        self.assertEqual(ov["n_value"], 1)

    def test_notify(self):
        pred = {"top_bets": [self.game()] * 7, "n_games": 100, "n_leagues": 20}
        payload = notify.build_payload(pred, "https://x.github.io/football-model/")
        self.assertEqual(len(payload["embeds"][0]["fields"]), 5)
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(notify.run(pred))
        with mock.patch.dict(os.environ, {"DISCORD_WEBHOOK": "https://discord.invalid/x"}), \
                mock.patch.object(notify.requests, "post", return_value=FakeResp({}, 204)) as post:
            self.assertTrue(notify.run(pred))
            self.assertEqual(post.call_args.kwargs["json"]["username"], "football-model")


if __name__ == "__main__":
    unittest.main()
