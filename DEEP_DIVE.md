# Daily deep dive: procedure

Goal: **10 bets for the day, each with a probability of at least 70% and odds of at least 1.40**, backed by the
deepest analysis possible. The statistical model is anchored to the bookmakers, so a 70%+ bet at 1.40+ only
exists where research finds something the price has not absorbed. The research exists to find that gap, or to
show there isn't one.

## Inputs
- `data/shortlist.json`: about 20 candidate bets (model probability 60%+, odds 1.40+, hard-to-call games already
  removed). Each candidate's game has a full `dossiers[game_id]`:
  - model and market: probabilities, likely scores, prices, bookmaker spread, line movement;
  - table and context: position and gaps, rest, games in 14/30 days, travel, Elo trend;
  - form: last 6 matches in detail, last 5 and 10 against the season, home/away splits;
  - patterns: clean sheets, BTTS, overs, first/second-half goals, half-time leads and comebacks;
  - head-to-head, referee stats, kick-off weather;
  - top 5 leagues only: Understat style (situations, game state, timing, formation, PPDA, xPts luck) and each
    key player's season numbers and last-5-match form (xG, xA, shots, key passes, minutes, hot/cold).
- `data/predictions.json`: every upcoming game, if more candidates are needed.

## For every candidate, research (web search) and record
1. **Injuries and suspensions** for both teams, confirmed where possible: who is out or doubtful, expected
   return, and the replacement's quality. Weight each absence by the player's minutes share and share of team
   xG/xA from the dossier.
2. **Line-ups**: predicted XIs (official site previews, local press). Rotation risk from cup or European games
   either side. Use confirmed XIs if within about an hour of kick-off.
3. **Individual form**: key players' last 5 matches (dossier for the top 5 leagues; FotMob, Sofascore or WhoScored
   pages for the other leagues): goals, assists, ratings, minutes, and whether they are nursing a knock.
4. **Manager and news**: new manager, pressure, press-conference quotes, dressing-room issues, signings,
   off-field events.
5. **Motivation**: derby, title/Europe/relegation stakes, dead rubber, the next fixture's importance.
6. **Referee**: appointment and tendencies. The dossier covers English and Scottish leagues; search for others.
7. **Conditions**: weather (dossier), pitch, travel, altitude, kick-off time.
8. **Market**: current odds at a mainstream European bookmaker (search "<match> odds"). Note whether the price
   has moved since the model ran and in which direction.
9. **Tactical matchup**: formation and style clash (pressing vs build-up, set-piece strength vs set-piece
   weakness, pace vs high line).

## Adjusting the probability
Start from the model's `p_model`. Apply explicit, justified adjustments of usually ±1 to 6 points each, with a
total rarely beyond ±12. Every adjustment needs a factor, a signed delta and a one-line reason citing evidence.
Do not double-count: the model already includes form, xG, table, Elo, rest and the bookmakers' prices. Only
**new information** moves the number (absences, line-ups, news, motivation, matchups the stats miss).

`final_p = p_model + sum(deltas)`, clipped to [0.05, 0.95].

## Choosing the 10
- **Meets the bar**: `final_p >= 0.70` and a real price of `odds_seen >= 1.40`.
- Rank by value, `final_p * odds_seen - 1`, then by confidence in the evidence.
- If fewer than 10 meet the bar, look at more games: other markets on the same games, later candidates in
  `predictions.json`, or the next day. If there still aren't 10, fill the list with the best remaining bets,
  marked `"meets_bar": false`, so it is always clear which ones are genuine.
- At most 2 bets per game. Never include a game that is essentially a coin flip.
- `min_odds` = the lowest price at which the bet is still worth it: `max(1.40, round(1 / final_p + 0.02, 2))`.

## Output: `data/deep/<YYYY-MM-DD>.json` (the date the analysis was done)
```json
{
  "date": "2026-10-01",
  "generated_at": "2026-10-01T09:30:00Z",
  "summary": "Two-three sentences on the day: how many meet the bar, main themes.",
  "bets": [
    {
      "rank": 1,
      "game_id": "E2-20261003-12",
      "match": "Blackpool v Leicester",
      "league": "E2",
      "league_name": "League One",
      "kickoff": "2026-10-03 15:00",
      "market": "Draw no bet",
      "selection": "A",
      "label": "Leicester draw no bet",
      "model_p": 0.61,
      "final_p": 0.72,
      "odds_seen": 1.55,
      "odds_source": "bet365 / oddschecker, 2026-10-01 09:10",
      "min_odds": 1.40,
      "meets_bar": true,
      "confidence": "high",
      "adjustments": [
        {"factor": "injuries", "delta": 0.05, "reason": "Blackpool without first-choice CB X (90% minutes) and top scorer Y (35% of team xG)"},
        {"factor": "lineups", "delta": 0.02, "reason": "Leicester unchanged XI expected; no midweek game"}
      ],
      "analysis": {
        "injuries": "...", "lineups": "...", "players": "...", "form": "...", "tactics": "...",
        "referee": "...", "weather": "...", "motivation": "...", "market": "...", "h2h": "..."
      },
      "risks": ["..."],
      "verdict": "BET",
      "sources": ["https://..."]
    }
  ],
  "rejected": [{"match": "...", "label": "...", "reason": "..."}]
}
```
`market` and `selection` must be one of the following, so results can be settled automatically:
- `1X2`: H, D or A
- `Double chance`: 1X, X2 or 12
- `Draw no bet`: H or A
- `O/U 1.5`, `O/U 2.5` or `O/U 3.5`: Over or Under
- `BTTS`: Yes or No

`verdict` is one of: BET, BET SMALL, WAIT FOR LINE-UPS, PASS.

## Publish
```bash
python scripts/deep.py publish data/deep/<date>.json   # validates, updates latest.json and index.json
git add data/deep && git commit -m "Deep dive <date>" && git push
```
A push to `data/deep/**` redeploys the site: the **Top 10 bets** tab. Results are settled automatically as
games finish and appear under **Results & record → Deep-dive bets**.
