"""Offline tests for the weekly Liga Portugal deep dive: python -m unittest discover tests"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import portugal  # noqa: E402
import portugal_live  # noqa: E402


def _game(**kw):
    g = {"match": "Braga v Gil Vicente", "kickoff": "2026-10-19 20:15", "confidence": "medium",
         "outcomes": {"H": 0.57, "D": 0.25, "A": 0.18, "over15": 0.72, "over25": 0.45, "over35": 0.23, "btts": 0.44},
         "best_bet": {"market": "Double chance", "selection": "1X", "label": "Braga or draw", "p": 0.82,
                      "odds_seen": None, "min_odds": 1.24},
         "why": "w", "factors": {k: "..." for k in portugal.FACTORS}, "risks": ["r"], "sources": []}
    g.update(kw)
    return g


def _rep(*games):
    return {"date": "2026-10-18", "week_start": "2026-10-19", "week_end": "2026-10-25", "summary": "s", "games": list(games)}


class TestValidate(unittest.TestCase):
    def test_valid_report(self):
        self.assertEqual(portugal.validate(_rep(_game())), [])

    def test_wrong_min_odds_and_market(self):
        bad = _game(best_bet={"market": "Asian handicap", "selection": "H", "label": "x", "p": 0.82, "min_odds": 1.40})
        errs = portugal.validate(_rep(bad))
        self.assertTrue(any("unsupported market" in e for e in errs))
        self.assertTrue(any("min_odds should be 1.24" in e for e in errs))

    def test_outcomes_must_add_up(self):
        bad = _game(outcomes={"H": 0.6, "D": 0.3, "A": 0.3, "over15": 0.7, "over25": 0.4, "over35": 0.2, "btts": 0.4})
        self.assertTrue(any("H+D+A" in e for e in portugal.validate(_rep(bad))))

    def test_payload_splits_long_reports(self):
        p = portugal.discord_payload(_rep(*[_game() for _ in range(30)]))
        self.assertGreater(len(p["embeds"]), 1)
        self.assertTrue(all(len(e["description"]) <= 4096 for e in p["embeds"]))


class TestLive(unittest.TestCase):
    def test_club_keys_match_across_sources(self):
        k = portugal_live.key
        self.assertEqual(k("SC Braga"), k("Sp Braga"))
        self.assertEqual(k("Vitória SC"), k("Vitoria SC"))
        self.assertEqual(k("Vitória de Guimaraes"), k("Guimaraes"))
        self.assertEqual(k("Sporting CP"), k("Sp Lisbon"))
        self.assertEqual(k("Académico"), k("Academico de Viseu"))
        self.assertEqual(k("Marítimo M."), k("Maritimo"))
        self.assertEqual(portugal_live.pair("Gil Vicente v FC Porto"), ("gil", "porto"))
        self.assertTrue(all(key in portugal_live.STADIUMS for key in ("benfica", "santa", "nacional", "academico", "casa")))

    def test_bet_price(self):
        p = {"H": 1.60, "D": 4.00, "A": 5.75, "over25": 2.05, "under25": 1.75}
        bp = portugal_live.bet_price
        self.assertEqual(bp({"market": "1X2", "selection": "H"}, p), {"price": 1.60, "derived": False})
        self.assertEqual(bp({"market": "O/U 2.5", "selection": "Under"}, p)["price"], 1.75)
        self.assertEqual(bp({"market": "Double chance", "selection": "1X"}, p), {"price": 1.14, "derived": True})
        self.assertEqual(bp({"market": "Draw no bet", "selection": "H"}, p)["price"], 1.2)
        self.assertIsNone(bp({"market": "BTTS", "selection": "No"}, p))
        self.assertIsNone(bp({"market": "1X2", "selection": "H"}, None))

    def test_american_odds(self):
        self.assertEqual(portugal_live._dec(-190), 1.53)
        self.assertEqual(portugal_live._dec(300), 4.0)
        self.assertIsNone(portugal_live._dec(None))


if __name__ == "__main__":
    unittest.main()
