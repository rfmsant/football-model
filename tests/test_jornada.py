"""Offline tests for the Liga Portugal round report and the consistent goal lines: python -m unittest discover tests"""
import datetime as dt
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import jornada  # noqa: E402
import main  # noqa: E402
from model import dixon_coles_matrix, evaluate_markets, goal_lines_matching  # noqa: E402

UTC = dt.timezone.utc


class TestGoalLines(unittest.TestCase):
    def test_matches_target_over25(self):
        for target in (0.40, 0.55, 0.70):
            gl = goal_lines_matching(1.5, 1.1, target, -0.08)
            self.assertAlmostEqual(gl["O25"], target, places=4)
            self.assertGreater(gl["O15"], gl["O25"])
            self.assertLess(gl["O35"], gl["O25"])

    def test_3_5_follows_the_2_5_number(self):
        # Heerenveen v Excelsior case: goal model 1.67 + 1.21, ML over 2.5 = 66%. The raw goal model said
        # Under 3.5 = 68%; consistent with 66% over 2.5 it must be clearly lower.
        i, j = __import__("numpy").indices((7, 7))
        raw_u35 = float(dixon_coles_matrix(1.67, 1.21, -0.08)[(i + j) <= 3].sum())
        gl = goal_lines_matching(1.67, 1.21, 0.66, -0.08)
        self.assertGreater(raw_u35, 0.66)
        self.assertLess(1 - gl["O35"], 0.60)


class TestRescoreBlend(unittest.TestCase):
    def test_per_market_weight(self):
        mk = {"O/U 3.5": {"Over": 0.30, "Under": 0.70}, "O/U 2.5": {"Over": 0.60, "Under": 0.40}}
        odds = {"O/U 3.5": {"Over": {"avg": 2.0, "max": 2.0}, "Under": {"avg": 2.0, "max": 2.0}},
                "O/U 2.5": {"Over": {"avg": 2.0, "max": 2.0}, "Under": {"avg": 2.0, "max": 2.0}}}
        ev = {(e["market"], e["selection"]): e for e in evaluate_markets(mk, odds, {"O/U 3.5": 0.8})}
        self.assertAlmostEqual(ev[("O/U 3.5", "Under")]["p_bet"], 0.2 * 0.70 + 0.8 * 0.5, places=4)
        self.assertAlmostEqual(ev[("O/U 2.5", "Over")]["p_bet"], 0.60, places=4)   # not listed: no blend

    def test_rescore_blends_goal_model_lines_only(self):
        rec = {"home": "A", "away": "B", "probs": {"H": 0.5, "D": 0.25, "A": 0.25},
               "markets": {"O/U 2.5": {"Over": 0.6, "Under": 0.4}, "O/U 3.5": {"Over": 0.3, "Under": 0.7}}}
        odds = {"O/U 3.5": {"Over": {"avg": 2.0, "max": 2.0}, "Under": {"avg": 2.0, "max": 2.0}},
                "O/U 2.5": {"Over": {"avg": 2.0, "max": 2.0}, "Under": {"avg": 2.0, "max": 2.0}}}
        main.rescore(rec, odds, {"market_weight": 0.8, "min_ev": 0.04, "min_odds": 1.3, "max_odds": 3.5})
        ev = {(e["market"], e["selection"]): e for e in rec["evals"]}
        self.assertAlmostEqual(ev[("O/U 3.5", "Under")]["p_bet"], 0.54, places=4)
        self.assertAlmostEqual(ev[("O/U 2.5", "Over")]["p_bet"], 0.60, places=4)


class TestFocusFlagging(unittest.TestCase):
    def test_liga_portugal_games_always_flagged(self):
        recs = [{"id": f"E1-{k}", "league": "E1", "date": "2026-10-20", "max_ev": 0.1, "evals": []} for k in range(30)]
        recs += [{"id": "P1-1", "league": "P1", "date": "2026-10-23", "max_ev": None, "evals": []},
                 {"id": "P1-2", "league": "P1", "date": "2026-11-02", "max_ev": None, "evals": []}]
        flagged = main.select_flagged(recs, n=5, today=dt.date(2026, 10, 20))
        self.assertIn("P1-1", flagged)
        self.assertNotIn("P1-2", flagged)          # beyond FOCUS_DAYS
        self.assertEqual(len(flagged), 6)


def game(home, away, date, time, league="P1"):
    return {"id": f"{league}-{home}", "league": league, "date": date, "time": time, "home": home, "away": away}


class TestRound(unittest.TestCase):
    def test_next_round_window_and_started(self):
        gs = [game("Vitoria", "Maritimo", "2026-10-23", "20:45"), game("Gil Vicente", "Porto", "2026-10-24", "20:30"),
              game("Santa Clara", "Benfica", "2026-10-26", "19:45"), game("Porto", "Braga", "2026-10-31", "20:30"),
              game("Wolves", "Bolton", "2026-10-23", "19:45", league="E1")]
        r = jornada.next_round(gs, dt.datetime(2026, 10, 20, 12, tzinfo=UTC))
        self.assertEqual([g["home"] for g in r], ["Vitoria", "Gil Vicente", "Santa Clara"])
        # once Friday's game has started, the round is the rest of the weekend
        r = jornada.next_round(gs, dt.datetime(2026, 10, 23, 21, tzinfo=UTC))
        self.assertEqual([g["home"] for g in r], ["Gil Vicente", "Santa Clara"])

    def test_due_once_within_post_days(self):
        rep = {"games": [{"date": "2026-10-23", "time": "20:45"}]}
        self.assertFalse(jornada.due(rep, dt.datetime(2026, 10, 19, 12, tzinfo=UTC)))
        self.assertTrue(jornada.due(rep, dt.datetime(2026, 10, 20, 20, tzinfo=UTC)))
        self.assertFalse(jornada.due(dict(rep, posted_at="2026-10-20T06:30:00Z"), dt.datetime(2026, 10, 21, tzinfo=UTC)))

    def test_one_bet_per_market_family(self):
        cs = [{"market": "1X2", "v": 1}, {"market": "Draw no bet", "v": 2}, {"market": "O/U 2.5", "v": 3},
              {"market": "O/U 3.5", "v": 4}, {"market": "BTTS", "v": 5}]
        self.assertEqual([c["v"] for c in jornada._one_per_outcome(cs)], [1, 3, 5])

    def test_payload_without_games(self):
        self.assertIn("No Liga Portugal games", jornada.discord_payload({"games": []})["embeds"][0]["description"])

    def test_payload_lists_bets_and_min_price(self):
        rep = {"round_start": "2026-10-23", "round_end": "2026-10-26", "games": [{
            "match": "Gil Vicente v Porto", "date": "2026-10-24", "time": "20:30",
            "chances": {"H": 0.15, "D": 0.22, "A": 0.63, "over25": 0.55, "btts": 0.45},
            "prices": {"H": 6.5, "D": 4.2, "A": 1.55, "over25": 1.85, "under25": 1.95},
            "best": [{"market": "O/U 2.5", "selection": "Over", "label": "Over 2.5 goals", "p_model": 0.58, "odds": 1.85,
                      "odds_source": "real", "value": 0.073, "min_odds": 1.69}], "closest": None}]}
        d = jornada.discord_payload(rep)["embeds"][0]
        self.assertIn("1 value bet", d["title"])
        self.assertIn("Over 2.5", d["description"])
        self.assertIn("min 1.69", d["description"])

    def test_game_bets_skips_pass_and_requires_value(self):
        g = game("Gil Vicente", "Porto", "2026-10-24", "20:30")
        cands = [{"market": "O/U 2.5", "selection": "Over", "label": "Over 2.5 goals", "p_model": 0.58, "odds": 1.85,
                  "odds_source": "real", "match": "Gil Vicente v Porto"},
                 {"market": "1X2", "selection": "A", "label": "Porto to win", "p_model": 0.63, "odds": 1.50,
                  "odds_source": "real", "match": "Gil Vicente v Porto"},
                 {"market": "BTTS", "selection": "No", "label": "Both teams NOT to score", "p_model": 0.60, "odds": 1.90,
                  "odds_source": "real", "match": "Gil Vicente v Porto"}]
        deep = {"bets": [{"match": "Gil Vicente v Porto", "kickoff": "2026-10-24 20:30", "market": "BTTS",
                          "selection": "No", "verdict": "PASS", "final_p": 0.6}]}
        with mock.patch.object(jornada.shortlist, "candidates_for", return_value=[dict(c) for c in cands]), \
                mock.patch.object(jornada, "read_json", return_value=deep):
            out = jornada.game_bets(g)
        self.assertEqual([b["label"] for b in out["best"]], ["Over 2.5 goals"])   # Porto win: no value; BTTS no: PASS


if __name__ == "__main__":
    unittest.main()
