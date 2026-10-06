"""Offline tests for scripts/intl.py (results top-up, calibration, market blend): python -m unittest discover tests"""
import datetime as dt
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import intl  # noqa: E402
import shortlist  # noqa: E402

TEAMS = {"England", "Croatia", "Republic of Ireland", "Israel", "Czech Republic", "Ukraine", "Hungary", "Turkey"}


def fx(home, away, gh, ga, league=5, status="FT", date="2026-10-03T18:45:00+00:00"):
    return {"fixture": {"date": date, "status": {"short": status}, "venue": {"city": "X"}}, "league": {"id": league},
            "teams": {"home": {"name": home}, "away": {"name": away}}, "goals": {"home": gh, "away": ga}}


def results_frame(rows):
    df = pd.DataFrame(rows, columns=intl.COLUMNS)
    df["date"] = pd.to_datetime(df["date"])
    return df


class TestParseFixtures(unittest.TestCase):
    def test_keeps_finished_national_team_games_and_maps_names(self):
        rows = intl.parse_fixtures([
            fx("Croatia", "England", 0, 7),
            fx("Ireland", "Israel", 1, 1),                       # alias -> Republic of Ireland
            fx("Ukraine", "Hungary", 1, 2),                      # exiled home side -> neutral
            fx("Czechia", "Türkiye", 2, 0, status="NS"),         # not finished
            fx("Arsenal", "Chelsea", 1, 0, league=39),           # club league
            fx("Atlantis", "England", 0, 1),                     # unknown team
        ], TEAMS)
        self.assertEqual([(r["home_team"], r["away_team"], r["home_score"], r["away_score"]) for r in rows],
                         [("Croatia", "England", 0, 7), ("Republic of Ireland", "Israel", 1, 1), ("Ukraine", "Hungary", 1, 2)])
        self.assertEqual(rows[0]["date"], "2026-10-03")
        self.assertEqual(rows[0]["tournament"], "UEFA Nations League")
        self.assertFalse(rows[0]["neutral"])
        self.assertTrue(rows[2]["neutral"])


class TestMergeResults(unittest.TestCase):
    def test_top_up_rows_dropped_once_upstream_has_them(self):
        base = results_frame([["2026-08-26", "England", "Croatia", 1, 0, "Friendly", "", "", False],
                              ["2026-10-03", "Croatia", "England", 0, 7, "UEFA Nations League", "", "", False]])
        extra = results_frame([["2026-10-04", "Croatia", "England", 0, 7, "UEFA Nations League", "", "", False],  # 1 day apart
                               ["2026-10-05", "Ukraine", "Hungary", 1, 2, "UEFA Nations League", "", "", True]])
        df = intl.merge_results(base, extra)
        self.assertEqual(len(df), 3)
        self.assertEqual(df.iloc[-1].home_team, "Ukraine")
        self.assertTrue(df.date.is_monotonic_increasing)


class TestTopUp(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        base = results_frame([["2026-09-28", "England", "Croatia", 1, 0, "Friendly", "", "", False],
                              ["2026-09-28", "Ukraine", "Hungary", 0, 0, "Friendly", "", "", False],
                              ["2026-09-28", "Republic of Ireland", "Israel", 0, 0, "Friendly", "", "", False]])
        (self.tmp / "intl").mkdir()
        base.assign(date=base.date.dt.strftime("%Y-%m-%d")).to_csv(self.tmp / "intl" / "results.csv", index=False)
        for name, val in (("CACHE", self.tmp / "intl"), ("EXTRA_RESULTS", self.tmp / "extra.csv"), ("FETCHED", self.tmp / "fetched.json")):
            p = mock.patch.object(intl, name, val)
            p.start()
            self.addCleanup(p.stop)

    def test_newest_first_capped_and_days_marked_done(self):
        calls = []

        class Api:
            disabled, error = False, None

            def get(self, path, date):
                calls.append(date)
                return [fx("Croatia", "England", 0, 7, date=date + "T18:45:00+00:00")] if date == "2026-10-03" else []

        added = intl.update_recent(Api(), today=dt.date(2026, 10, 6), max_calls=4)
        self.assertEqual(calls, ["2026-10-05", "2026-10-04", "2026-10-03", "2026-10-02"])
        self.assertEqual(added, 1)
        extra = pd.read_csv(intl.EXTRA_RESULTS)
        self.assertEqual(extra.iloc[0].home_team, "Croatia")
        done = json.loads(intl.FETCHED.read_text())
        self.assertNotIn("2026-10-05", done)            # yesterday: may still change, refetch next run
        self.assertIn("2026-10-04", done)
        calls.clear()
        self.assertEqual(intl.update_recent(Api(), today=dt.date(2026, 10, 6), max_calls=3), 0)   # no duplicates
        self.assertEqual(calls, ["2026-10-05", "2026-10-01", "2026-09-30"])

    def test_skipped_without_key(self):
        class Api:
            disabled, error = True, "APIFOOTBALL_KEY not set"
        self.assertEqual(intl.update_recent(Api(), today=dt.date(2026, 10, 6)), 0)


class TestCalibration(unittest.TestCase):
    def test_platt_recovers_known_miscalibration(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0.3, 0.9, 20000)
        true = 1 / (1 + np.exp(-(-0.2 + 0.7 * np.log(p / (1 - p)))))
        y = rng.uniform(size=p.size) < true
        a, b = intl.fit_logistic(p, y)
        self.assertAlmostEqual(a, -0.2, delta=0.06)
        self.assertAlmostEqual(b, 0.7, delta=0.06)
        self.assertAlmostEqual(intl.apply_cal(0.73, (a, b)), float(1 / (1 + np.exp(-(a + b * np.log(0.73 / 0.27))))), places=6)
        self.assertEqual(intl.apply_cal(0.73, None), 0.73)


class TestBlend(unittest.TestCase):
    ODDS = {"1X2": {"H": {"avg": 1.5}, "D": {"avg": 4.2}, "A": {"avg": 6.5}},
            "O/U 2.5": {"Over": {"avg": 1.8}, "Under": {"avg": 2.0}}}
    BETA = np.array([0.12, 0.737, 0.24, -0.059])

    def test_full_market_weight_returns_market(self):
        gp = intl.game_probs(self.BETA, {}, 1900, 1700, self.ODDS, w=1.0)
        self.assertAlmostEqual(sum(gp["probs"].values()), 1.0, places=6)
        self.assertAlmostEqual(gp["probs"]["H"], gp["market_1x2"]["H"], places=9)
        self.assertAlmostEqual(gp["goals"]["O25"], (1 / 1.8) / (1 / 1.8 + 1 / 2.0), places=9)
        self.assertAlmostEqual(gp["goals"]["O15"], gp["market_goals"]["O15"], places=9)

    def test_zero_market_weight_returns_model(self):
        gp = intl.game_probs(self.BETA, {}, 1900, 1700, self.ODDS, w=0.0)
        self.assertAlmostEqual(gp["probs"]["H"], gp["model_1x2"]["H"], places=9)
        self.assertAlmostEqual(gp["goals"]["O35"], gp["model_goals"]["O35"], places=9)

    def test_no_1x2_prices_no_game(self):
        self.assertIsNone(intl.game_probs(self.BETA, {}, 1900, 1700, {"O/U 2.5": {}}))

    def test_shortlist_uses_supplied_goal_probs(self):
        g = {"xg": {"home": 2.0, "away": 0.8}, "markets": {"O/U 2.5": {"Over": 0.55}, "BTTS": {"Yes": 0.45}},
             "goal_probs": {"O15": 0.71, "O35": 0.28}}
        self.assertEqual(shortlist.model_goal_probs(g), {"O15": 0.71, "O25": 0.55, "O35": 0.28, "BTTS": 0.45})
        del g["goal_probs"]                                   # club games: unchanged behaviour
        self.assertAlmostEqual(shortlist.model_goal_probs(g)["O25"], 0.55, places=2)


class TestTuneBlend(unittest.TestCase):
    def test_needs_enough_games(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(intl, "PRED_LOG", Path(d) / "none.csv"):
            self.assertEqual(intl.tune_blend()["games"], 0)


if __name__ == "__main__":
    unittest.main()
