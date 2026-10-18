"""Offline tests for the weekly Liga Portugal deep dive: python -m unittest discover tests"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import portugal  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
