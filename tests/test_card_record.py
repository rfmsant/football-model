"""Offline tests for card settlement + per-leg tracking and the estimated-price haircut in scripts/card.py."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import card  # noqa: E402


def leg(match, market, selection, p, odds, date="2026-10-04", source="estimated"):
    return {"match": match, "date": date, "market": market, "selection": selection, "label": f"{selection}",
            "p_model": p, "odds": odds, "odds_source": source}


class TestSettle(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "cards").mkdir()
        for name, val in (("CARDS", self.tmp / "cards"), ("DATA", self.tmp)):
            p = mock.patch.object(card, name, val)
            p.start()
            self.addCleanup(p.stop)

    def write_card(self, date, main, alts=()):
        (self.tmp / "cards" / f"{date}.json").write_text(json.dumps(
            {"date": date, "main": {"odds": 2.0, "legs": main}, "alternatives": [{"odds": 2.0, "legs": a} for a in alts]}))

    def test_combos_and_legs(self):
        self.write_card("2026-10-04",
                        [leg("Malta v Andorra", "O/U 1.5", "Over", 0.71, 1.49), leg("Azerbaijan v Lithuania", "Double chance", "1X", 0.73, 1.36)],
                        [[leg("Greece v Germany", "O/U 3.5", "Under", 0.69, 1.49),
                          leg("Republic of Ireland v Israel", "Draw no bet", "H", 0.65, 1.52)],
                         [leg("Croatia v England", "O/U 3.5", "Under", 0.68, 1.46), leg("Spain v Czech Republic", "BTTS", "No", 0.68, 1.75)],
                         [leg("Malta v Andorra", "O/U 1.5", "Over", 0.71, 1.49), leg("Wales v Denmark", "O/U 2.5", "Under", 0.66, 1.6)]])
        res = {("Malta", "Andorra", "2026-10-04"): (2, 1), ("Azerbaijan", "Lithuania", "2026-10-04"): (1, 0),
               ("Greece", "Germany", "2026-10-04"): (0, 0), ("Republic of Ireland", "Israel", "2026-10-04"): (1, 1),
               ("Croatia", "England", "2026-10-04"): (0, 7)}   # Spain v Czech + Wales v Denmark not in results
        with mock.patch.object(card, "_results", return_value=res):
            out = card.settle_cards()
        st = [(c["kind"], c["status"], c["profit"]) for c in out["cards"]]
        self.assertEqual(st[0], ("main", "won", round(1.49 * 1.36 - 1, 3)))
        self.assertEqual(st[1], ("alternative", "won", 0.49))          # DNB refunded: 1.49 x 1
        self.assertEqual(st[2], ("alternative", "lost", -1.0))          # lost leg beats the pending one
        self.assertEqual(st[3], ("alternative", "pending", None))
        legs = {l["match"]: l["result"] for l in out["legs"]}
        self.assertEqual(len(out["legs"]), 7)                           # Malta leg counted once
        self.assertEqual(legs["Republic of Ireland v Israel"], "void")
        self.assertEqual(legs["Croatia v England"], "lost")
        self.assertEqual(legs["Spain v Czech Republic"], "pending")
        s = out["legs_summary"]["all"]
        self.assertEqual((s["settled"], s["won"], s["void"]), (4, 3, 1))
        self.assertIn("O/U 3.5", out["legs_summary"]["by_market"])
        self.assertTrue((self.tmp / "card_record.json").exists())


class TestHaircut(unittest.TestCase):
    def test_estimated_prices_cut_real_and_researched_kept(self):
        legs = [dict(leg("A v B", "O/U 1.5", "Over", 0.80, 1.40), date="2026-10-10", time="20:00"),
                dict(leg("C v D", "O/U 2.5", "Over", 0.80, 1.40, source="real"), date="2026-10-10", time="20:00"),
                dict(leg("E v F", "BTTS", "Yes", 0.80, 1.40), date="2026-10-10", time="20:00", researched=True)]
        files = {"predictions.json": {"games": []}, "latest.json": {"bets": []},
                 "intl.json": {"games": [{"date": "2026-10-10", "time": "20:00", "hard": False, "candidates": legs}]}}
        with mock.patch.object(card, "read_json", side_effect=lambda p, d=None: files.get(Path(p).name, d)):
            try:
                out = card.pool("2026-10-10", now=card.kickoff_utc("2026-10-10", "12:00"))   # newer pool(date, now)
            except (TypeError, AttributeError):
                out = card.pool("2026-10-10")
        odds = {c["match"]: c["odds"] for c in out}
        self.assertEqual(odds["A v B"], round(1.40 * card.ESTIMATED_PRICE_HAIRCUT, 2))
        self.assertEqual(odds["C v D"], 1.40)
        self.assertEqual(odds["E v F"], 1.40)


if __name__ == "__main__":
    unittest.main()
