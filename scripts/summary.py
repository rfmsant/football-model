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

    b, lean = g.get("best_bet"), g.get("lean")
    if b:
        parts.append(f"Value bet: {bet_label(b, home, away)} at {b['odds']:.2f}. The model gives it "
                     f"{b['model_p']:.0%} against a market {b['market_p']:.0%} (EV {b['ev']:+.1%}).")
    elif lean:
        verdict = ("close to fair value" if lean["ev"] > -0.02 else "no value at current prices")
        parts.append(f"Model's lean: {bet_label(lean, home, away)} at {lean['odds']:.2f} "
                     f"({lean['model_p']:.0%} model vs {lean['market_p']:.0%} market, EV {lean['ev']:+.1%}), "
                     f"{verdict}; not a recommended bet.")
    else:
        parts.append("No bookmaker prices yet, so there's no comparison with the market.")

    notes = []
    if not g.get("has_xg"):
        notes.append("no xG data for this league (shots on target used instead)")
    if not g.get("injury_data"):
        notes.append("team news not checked")
    conf = g["confidence"]
    parts.append(f"Confidence: {_conf_word(conf)} ({conf}/100)" + (f", {', '.join(notes)}." if notes else "."))
    return " ".join(parts)


def _match(g: dict) -> str:
    return f"{g['home']} v {g['away']}"


def daily_overview(date: str, games: list[dict], today=None) -> dict:
    """Template-generated overview of one day's fixtures."""
    import datetime as dt
    d = dt.date.fromisoformat(date)
    today = today or dt.date.today()
    leagues: dict = {}
    for g in games:
        leagues[g["league_name"]] = leagues.get(g["league_name"], 0) + 1
    ranked = sorted(leagues.items(), key=lambda kv: -kv[1])
    lg_txt = ", ".join(f"{n} {k}" for k, n in ranked[:5]) + (f" and {len(ranked) - 5} more leagues" if len(ranked) > 5 else "")
    fav = sorted(games, key=lambda g: -max(g["probs"]["H"], g["probs"]["A"]))
    close = sorted(games, key=lambda g: abs(g["probs"]["H"] - g["probs"]["A"]))
    goals = sorted(games, key=lambda g: -(g["xg"]["home"] + g["xg"]["away"]))
    tight = goals[::-1]
    priced = [g for g in games if any(e.get("market") == "1X2" and "edge" in e for e in g.get("evals", []))]

    def gap(g):
        return max(abs(e["edge"]) for e in g["evals"] if e.get("market") == "1X2" and "edge" in e)
    disagree = sorted(priced, key=lambda g: -gap(g))
    value = [g for g in games if g.get("best_bet")]

    hl = []
    if fav and max(fav[0]["probs"]["H"], fav[0]["probs"]["A"]) >= 0.45:
        g = fav[0]
        side, p = ("home", g["probs"]["H"]) if g["probs"]["H"] >= g["probs"]["A"] else ("away", g["probs"]["A"])
        hl.append({"kind": "Strongest favourite", "game": g["id"],
                   "text": f"{g[side]} ({p:.0%}) in {_match(g)}, {g['league_name']}"})
    if len(close) > 1:
        g = close[0]
        hl.append({"kind": "Closest call", "game": g["id"],
                   "text": f"{_match(g)}: {g['probs']['H']:.0%} / {g['probs']['D']:.0%} / {g['probs']['A']:.0%}"})
    if goals:
        g = goals[0]
        o25 = g["markets"].get("O/U 2.5", {}).get("Over")
        hl.append({"kind": "Most goals expected", "game": g["id"],
                   "text": f"{_match(g)}: xG {g['xg']['home']:.1f}-{g['xg']['away']:.1f}"
                           + (f", over 2.5 at {o25:.0%}" if o25 else "")})
    if len(tight) > 1:
        g = tight[0]
        hl.append({"kind": "Tightest game", "game": g["id"],
                   "text": f"{_match(g)}: xG {g['xg']['home']:.1f}-{g['xg']['away']:.1f}, likely {g['likely_score']}"})
    if disagree and gap(disagree[0]) >= 0.04:
        g = disagree[0]
        e = max((e for e in g["evals"] if e.get("market") == "1X2" and "edge" in e), key=lambda e: abs(e["edge"]))
        who = {"H": g["home"], "D": "the draw", "A": g["away"]}[e["selection"]]
        more = "more" if e["edge"] > 0 else "less"
        hl.append({"kind": "Model vs bookies", "game": g["id"],
                   "text": f"{_match(g)}: model rates {who} {more} highly ({e['model_p']:.0%} vs {e['market_p']:.0%})"})

    day = "Today" if d == today else "Tomorrow" if d == today + dt.timedelta(days=1) else d.strftime("%A")
    text = f"{day}, {d.strftime('%d %b')}: {len(games)} game{'s' if len(games) != 1 else ''} ({lg_txt}). "
    if value:
        text += (f"{len(value)} price{'s' if len(value) != 1 else ''} clear{'' if len(value) != 1 else 's'} the value threshold: "
                 + "; ".join(f"{_match(g)}, {bet_label(g['best_bet'], g['home'], g['away'])} @ {g['best_bet']['odds']:.2f}"
                            for g in value[:3]) + (" and more" if len(value) > 3 else "") + ".")
    elif priced:
        text += "No price clears the value threshold; the bookmakers look about right on this slate."
    else:
        text += "No bookmaker prices yet; model probabilities only."
    return {"date": date, "n_games": len(games), "leagues": leagues, "text": text, "highlights": hl,
            "n_value": len(value)}
