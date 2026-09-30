"""Template-generated written analysis for each game (no paid APIs, no LLM)."""
from __future__ import annotations

SEL_LABEL = {"H": "home win", "D": "draw", "A": "away win", "Over": "over", "Under": "under",
             "Yes": "yes", "No": "no", "Home": "home", "Away": "away"}


def bet_label(b: dict, home: str, away: str) -> str:
    m, s = b["market"], b["selection"]
    if m == "1X2":
        return {"H": f"{home} to win", "D": "Draw", "A": f"{away} to win"}[s]
    if m.startswith("O/U"):
        return f"{s} {m.split()[1]} goals"
    if m == "BTTS":
        return f"Both teams to score: {s}"
    if m.startswith("AH"):
        line = float(m.split()[1])
        team, ln = (home, line) if s == "Home" else (away, -line)
        return f"{team} {ln + 0:+g} (Asian handicap)"
    return f"{m} {s}"


def _form_words(form: str) -> str:
    if not form:
        return "no recent league form available"
    pts = sum(3 if c == "W" else 1 if c == "D" else 0 for c in form)
    mood = "excellent" if pts >= 12 else "good" if pts >= 9 else "mixed" if pts >= 5 else "poor"
    return f"{mood} form ({form})"


def _conf_word(c: int) -> str:
    return "high" if c >= 70 else "moderate" if c >= 50 else "low"


def generate(g: dict) -> str:
    home, away = g["home"], g["away"]
    p = g["probs"]
    xh, xa = g["xg"]["home"], g["xg"]["away"]
    parts = []
    fav = max(("H", "D", "A"), key=lambda k: p[k])
    if fav == "D" or max(p["H"], p["A"]) < 0.42:
        parts.append(f"The model sees a finely balanced game ({p['H']:.0%} / {p['D']:.0%} / {p['A']:.0%}).")
    else:
        team = home if fav == "H" else away
        strength = "clear favourites" if p[fav] >= 0.6 else "favourites"
        parts.append(f"{team} are {strength} at {p[fav]:.0%}, with the draw at {p['D']:.0%}.")
    tot = xh + xa
    tempo = "a high-scoring game" if tot >= 3.1 else "a low-scoring game" if tot <= 2.3 else "an average number of goals"
    parts.append(f"Expected goals {xh:.2f}-{xa:.2f} point to {tempo}; the most likely score is {g['likely_score']}.")

    ctx_h, ctx_a = g.get("context", {}).get("home", {}), g.get("context", {}).get("away", {})
    if ctx_h.get("form") or ctx_a.get("form"):
        parts.append(f"{home} come in with {_form_words(ctx_h.get('form', ''))}, "
                     f"{away} with {_form_words(ctx_a.get('form', ''))}.")
    elo = g.get("elo_diff")
    if elo is not None and abs(elo) >= 120:
        stronger = home if elo > 0 else away
        parts.append(f"Elo ratings favour {stronger} by {abs(elo):.0f} points.")

    abs_txt = []
    for side, team in (("home", home), ("away", away)):
        a = [x for x in g.get("absences", {}).get(side, []) if x["status"] == "out"]
        if a:
            names = ", ".join(x["name"] for x in a[:3])
            more = f" and {len(a) - 3} more" if len(a) > 3 else ""
            abs_txt.append(f"{team} are without {names}{more}")
    if abs_txt:
        parts.append("; ".join(abs_txt) + ".")
    other = [f for f in g.get("factors", []) if "missing key players" not in f]
    if other:
        parts.append("Also factored in: " + "; ".join(other[:3]) + ".")

    b = g.get("best_bet")
    if b:
        parts.append(f"Best value: {bet_label(b, home, away)} at {b['odds']:.2f}. The model gives it "
                     f"{b['model_p']:.0%} against a market {b['market_p']:.0%} (EV {b['ev']:+.1%}).")
    else:
        parts.append("No bet clears the value threshold: the market price looks fair.")

    notes = []
    if not g.get("has_xg"):
        notes.append("no xG data for this league (shots on target used instead)")
    if not g.get("injury_data"):
        notes.append("team news not checked")
    conf = g["confidence"]
    parts.append(f"Confidence: {_conf_word(conf)} ({conf}/100)" + (f", {', '.join(notes)}." if notes else "."))
    return " ".join(parts)
