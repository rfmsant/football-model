"""Offline tests for scripts/card.py leg pool and card date: python -m unittest discover tests"""
import datetime as dt
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import card  # noqa: E402

UTC = dt.timezone.utc


def leg(home, away, date, time, market="O/U 1.5", selection="Over", p=0.70, odds=1.50):
    return {"market": market, "selection": selection, "label": f"{selection} {market}", "p_model": p, "odds": odds,
            "odds_source": "real", "match": f"{home} v {away}", "date": date, "time": time}


def intl_game(home, away, date, time, cands=None):
    return {"id": f"INT-{date}-{home}-{away}", "date": date, "time": time, "home": home, "away": away, "hard": False,
            "candidates": cands if cands is not None else [leg(home, away, date, time)]}


def club_game(home, away, date, time):
    return {"id": f"T-{date}-{home}", "date": date, "time": time, "home": home, "away": away}


class CardTest(unittest.TestCase):
    """Patches card.read_json / shortlist so pool() sees only the fixtures set on self.files."""

    def setUp(self):
        self.files = {"predictions.json": {"games": []}, "intl.json": {"games": []}, "latest.json": {"bets": []}}
        p = [mock.patch.object(card, "read_json", side_effect=lambda path, default=None: self.files.get(Path(path).name, default)),
             mock.patch.object(card.shortlist, "is_hard", return_value=False),
             mock.patch.object(card.shortlist, "candidates_for",
                               side_effect=lambda g: [leg(g["home"], g["away"], g["date"], g["time"])])]
        for x in p:
            x.start()
            self.addCleanup(x.stop)


class TestKickoff(unittest.TestCase):
    def test_uk_time_converted_to_utc(self):
        self.assertEqual(card.kickoff_utc("2026-10-06", "15:00"), dt.datetime(2026, 10, 6, 14, 0, tzinfo=UTC))   # BST
        self.assertEqual(card.kickoff_utc("2026-12-06", "15:00"), dt.datetime(2026, 12, 6, 15, 0, tzinfo=UTC))   # GMT
        self.assertIsNone(card.kickoff_utc("2026-10-06", None))

    def test_started(self):
        now = dt.datetime(2026, 10, 6, 15, 0, tzinfo=UTC)
        self.assertTrue(card.started("2026-10-06", "15:00", now))      # 14:00 UTC
        self.assertFalse(card.started("2026-10-06", "19:45", now))
        self.assertTrue(card.started("2026-10-05", None, now))
        self.assertFalse(card.started("2026-10-06", None, now))


class TestPool(CardTest):
    def test_skips_games_already_kicked_off(self):
        self.files["intl.json"] = {"games": [intl_game("Kazakhstan", "Faroe Islands", "2026-10-06", "15:00"),
                                             intl_game("Scotland", "Slovenia", "2026-10-06", "19:45")]}
        self.files["predictions.json"] = {"games": [club_game("Early FC", "Morning Utd", "2026-10-06", "12:30"),
                                                    club_game("Late FC", "Night Utd", "2026-10-06", "20:00")]}
        legs = card.pool("2026-10-06", now=dt.datetime(2026, 10, 6, 15, 0, tzinfo=UTC))
        self.assertEqual(sorted(l["match"] for l in legs), ["Late FC v Night Utd", "Scotland v Slovenia"])

    def test_game_kicking_off_later_in_uk_time_is_kept(self):
        # 15:00 UK = 14:00 UTC; at 13:59 UTC it hasn't started
        self.files["intl.json"] = {"games": [intl_game("Kazakhstan", "Faroe Islands", "2026-10-06", "15:00")]}
        self.assertEqual(len(card.pool("2026-10-06", now=dt.datetime(2026, 10, 6, 13, 59, tzinfo=UTC))), 1)
        self.assertEqual(len(card.pool("2026-10-06", now=dt.datetime(2026, 10, 6, 14, 0, tzinfo=UTC))), 0)

    def test_pass_verdict_removes_leg(self):
        d = "2026-10-10"
        self.files["predictions.json"] = {"games": [club_game("Paderborn", "Stuttgart", d, "14:30")]}
        self.files["intl.json"] = {"games": [intl_game("Alanyaspor", "Erzurumspor", d, "14:00",
                                                       [leg("Alanyaspor", "Erzurumspor", d, "14:00", "O/U 1.5", "Over"),
                                                        leg("Alanyaspor", "Erzurumspor", d, "14:00", "BTTS", "Yes")])]}
        self.files["latest.json"] = {"bets": [
            {"match": "Paderborn v Stuttgart", "kickoff": f"{d} 14:30", "market": "O/U 1.5", "selection": "Over",
             "verdict": "PASS", "final_p": 0.75},
            {"match": "Alanyaspor v Erzurumspor", "kickoff": f"{d} 14:00", "market": "O/U 1.5", "selection": "Over",
             "verdict": "PASS", "final_p": 0.75}]}
        legs = card.pool(d, now=dt.datetime(2026, 10, 6, 15, 0, tzinfo=UTC))
        self.assertEqual([(l["match"], l["market"]) for l in legs], [("Alanyaspor v Erzurumspor", "BTTS")])

    def test_non_pass_verdict_overrides_probability(self):
        d = "2026-10-10"
        self.files["predictions.json"] = {"games": [club_game("Maritimo", "Porto", d, "18:00")]}
        self.files["latest.json"] = {"bets": [
            {"match": "Maritimo v Porto", "kickoff": f"{d} 18:00", "market": "O/U 1.5", "selection": "Over",
             "verdict": "BET SMALL", "final_p": 0.72, "odds_seen": 1.55, "adjustments": [{"reason": "form"}]}]}
        (l,) = card.pool(d, now=dt.datetime(2026, 10, 6, 15, 0, tzinfo=UTC))
        self.assertTrue(l["researched"])
        self.assertEqual((l["p_model"], l["odds"]), (0.72, 1.55))


class TestDefaultDate(CardTest):
    def setUp(self):
        super().setUp()
        self.files["intl.json"] = {"games": [intl_game("Scotland", "Slovenia", "2026-10-06", "19:45")]}
        self.files["predictions.json"] = {"games": [club_game(f"Home{i}", f"Away{i}", "2026-10-09", "19:00") for i in range(2)]
                                          + [club_game(f"Home{i}", f"Away{i}", "2026-10-10", "15:00") for i in range(2, 7)]}

    def test_today_while_it_has_upcoming_legs(self):
        self.assertEqual(card.default_date(dt.datetime(2026, 10, 6, 15, 0, tzinfo=UTC)), "2026-10-06")

    def test_skips_empty_tomorrow_to_next_day_with_enough_legs(self):
        # after Scotland v Slovenia kicks off (18:45 UTC): 7 Oct and 8 Oct are empty, 9 Oct has 2 legs, 10 Oct has 5
        self.assertEqual(card.default_date(dt.datetime(2026, 10, 6, 19, 0, tzinfo=UTC)), "2026-10-10")

    def test_falls_back_to_next_date_with_any_legs(self):
        self.files["predictions.json"]["games"] = self.files["predictions.json"]["games"][:2]   # only 9 Oct, 2 legs
        self.assertEqual(card.default_date(dt.datetime(2026, 10, 6, 19, 0, tzinfo=UTC)), "2026-10-09")

    def test_tomorrow_when_nothing_ahead(self):
        self.files["predictions.json"]["games"] = []
        self.assertEqual(card.default_date(dt.datetime(2026, 10, 6, 19, 0, tzinfo=UTC)), "2026-10-07")

    def test_uses_uk_date_just_after_midnight(self):
        # 23:30 UTC on 9 Oct is 00:30 UK on 10 Oct
        self.assertEqual(card.default_date(dt.datetime(2026, 10, 9, 23, 30, tzinfo=UTC)), "2026-10-10")


if __name__ == "__main__":
    unittest.main()
