# Weekly Liga Portugal deep dive: procedure

Runs **every Sunday evening**. Covers every Liga Portugal Betclic game that kicks off from the next day (Monday)
to the following Sunday, including rearranged games. For **each game**, give the single best bet, whatever its
odds, from the deepest analysis possible, and a probability for every outcome. The output goes to Discord
automatically when it is pushed.

## Fixtures
`data/predictions.json` often does not have the next round yet. Get the fixture list from the web (ligaportugal.pt
calendar, A Bola, Record, O Jogo, ESPN) and check every game: date, Lisbon kick-off time, venue, and that it is not
postponed. List games played on neutral ground or moved to another stadium.

## For every game, research and record (Portuguese press first: A Bola, Record, O Jogo, Maisfutebol, zerozero)
**Check every article's date.** Use only information from the current or previous week. Results from earlier
seasons regularly show up in searches. If nothing current exists, say so and make no adjustment.

1. **Injuries and suspensions** for both teams: who is out or doubtful, expected return, the replacement and how
   much the player matters (starter? top scorer? captain?). Yellow-card accumulation (Liga Portugal: a ban after
   5 yellows). Players ineligible for rearranged games (not registered on the original date).
2. **Fatigue and squad drain**: games in the last 7 and 14 days, European games midweek (Champions League,
   Europa League, Conference League) before or after, Taça de Portugal or Taça da Liga games and extra time,
   international call-ups, travel (Madeira and the Azores are flights; European away trips), days of rest
   compared with the opponent.
3. **Line-ups and rotation**: predicted XIs, the manager's habits when there is a big game next.
4. **Form**: last 5–6 league games with scores, shots and xG if available; home form for the home team and away
   form for the away team; goals for and against; clean sheets; whether recent results flatter them.
5. **Home advantage**: home and away record this season, crowd, pitch (synthetic or poor pitches, small grounds),
   long trips for the away team.
6. **Manager and news**: new manager (bounce), manager under pressure or sacked, dressing-room problems, money
   problems, fan protests, statements in press conferences.
7. **Motivation**: title race, Europe places, relegation (the bottom two go down, 16th plays a play-off), derby,
   a big game next, a dead rubber.
8. **Referee**: the appointment (Conselho de Arbitragem announces it about two days before; if not out yet, say
   so) and their cards, penalties and home/away tendencies.
9. **Weather and pitch**: forecast rain, wind, heat for kick-off; past waterlogged pitches.
10. **Tactics**: style matchup, set pieces, pace against a high line, how each team plays when ahead or behind.
11. **Market**: current odds at Betclic or a mainstream bookmaker, and how they have moved. Many games won't have
    prices on Sunday. Then write `odds_seen: null` and rely on `min_odds`.
12. **Head-to-head**: recent meetings, only as a small factor.

## Probabilities
For each game give your **final probabilities for all outcomes** (`outcomes`): home/draw/away, over 1.5/2.5/3.5
and BTTS yes. Start from the model's numbers when the game is in `data/predictions.json`, otherwise from the
bookmaker prices or your own estimate from the table, form and goals. Record each change as
`{factor, delta, reason}`. Every probability must be honest: do not inflate it to make a bet look better.

## Picking the best bet per game
- Consider every market Betclic offers (1X2, double chance, draw no bet, O/U 1.5/2.5/3.5, BTTS) and choose the bet
  that is **most likely to win** after the research, while avoiding near-certain bets nobody can price (for
  example, over 0.5). Odds do not matter for the choice.
- Also give up to two `alternatives`, for example a value bet at better odds.
- `min_odds` = `round(1 / p + 0.02, 2)`: the lowest price at which the bet is still worth taking.
- `confidence`: high / medium / low, depending on how much current information exists.
- If a game is genuinely a coin flip, still give the most likely bet, mark confidence low and say so in `risks`.

## Output: `data/portugal/<run date YYYY-MM-DD>.json`
```json
{
  "date": "2026-10-18",
  "generated_at": "2026-10-18T20:30:00Z",
  "week_start": "2026-10-19",
  "week_end": "2026-10-25",
  "summary": "Two or three sentences: the week, main themes (Europe midweek, cup fatigue, sackings).",
  "games": [
    {
      "match": "Braga v Gil Vicente",
      "kickoff": "2026-10-19 20:15",
      "venue": "Estadio Municipal de Braga",
      "outcomes": {"H": 0.58, "D": 0.25, "A": 0.17, "over15": 0.72, "over25": 0.45, "over35": 0.23, "btts": 0.44},
      "best_bet": {"market": "Double chance", "selection": "1X", "label": "Braga or draw", "p": 0.82,
                   "odds_seen": 1.22, "odds_source": "bookmaker average, 2026-10-18", "min_odds": 1.24},
      "alternatives": [{"market": "O/U 2.5", "selection": "Under", "label": "Under 2.5 goals", "p": 0.55,
                        "odds_seen": 1.75, "odds_source": "...", "min_odds": 1.84}],
      "confidence": "medium",
      "adjustments": [{"factor": "injuries", "delta": -0.03, "reason": "..."}],
      "factors": {"injuries": "...", "fatigue": "...", "lineups": "...", "form": "...", "home_away": "...",
                  "manager_news": "...", "motivation": "...", "referee": "...", "weather": "...", "tactics": "...",
                  "market": "...", "h2h": "..."},
      "risks": ["..."],
      "sources": ["https://..."]
    }
  ],
  "skipped": [{"match": "...", "reason": "postponed"}]
}
```
`market`/`selection` use the same names as DEEP_DIVE.md (`1X2` H/D/A, `Double chance` 1X/X2/12, `Draw no bet`
H/A, `O/U 1.5|2.5|3.5` Over/Under, `BTTS` Yes/No).

## Publish
```bash
python scripts/portugal.py publish data/portugal/<date>.json   # validates and writes latest.json
git add data/portugal && git commit -m "Liga Portugal weekly <date>" && git pull --rebase && git push
```
The push sends the report to Discord (workflow "Liga Portugal weekly to Discord").
