# football-model

A free, fully automated football betting model. Twice a week it scans every fixture in 22 European
leagues, prices each game with a Dixon-Coles model, and ranks bets by **expected value** (not by
how likely they are). It publishes the results to GitHub Pages and posts the top 5 picks to Discord.

It runs only on free services: GitHub Actions, GitHub Pages, football-data.co.uk, ClubElo, Understat,
and the free tiers of API-Football and The Odds API.

**Site:** `https://<your-user>.github.io/football-model/` (add `?demo` to see a replay of a past week)
**Backtest:** [backtest.md](backtest.md)

## How it works

| Step | Script | What it does |
|---|---|---|
| 1 | `scripts/fetch.py`, `scripts/oddsapi.py` | Downloads the next 7 days of fixtures and odds (`fixtures.csv`, topped up with top-5-league games from The Odds API), results, shots, xG and closing odds for the current and previous season, Understat xG for the top 5 leagues, and ClubElo ratings. |
| 2 | `scripts/model.py` | Fits time-weighted attack/defence ratings per league. Inputs are xG where available, otherwise shots on target × league conversion, blended with goals. It blends these with Elo (a built-in cross-division Elo, averaged with ClubElo when that API is up), builds a Dixon-Coles 0-0 to 6-6 scoreline matrix, derives 1X2, O/U 1.5/2.5/3.5 and BTTS (bets are only recommended in European markets: 1X2, over/under, BTTS; Asian handicap is excluded via `bet_markets` in `common.py`), removes the bookmaker margin, and computes edge and EV. |
| 3 | `scripts/main.py` (stage 1) | Scans every game and flags 25: the top games by EV, plus the games where model and market (or bookmakers among themselves) disagree most. |
| 4 | `scripts/extras.py` | For flagged games only: injuries and suspensions from API-Football, weighted by each player's minutes share and goal contribution. Also rest days, fixture congestion, manager change, league-table motivation, and extra bookmaker prices from The Odds API. |
| 5 | `scripts/backtest.py` | Walk-forward backtest. Tunes weights on the season two years back, then reports calibration, Brier score, log loss, ROI and closing line value out-of-sample on last season. Writes `data/params.json` and `backtest.md`. |
| 6 | `scripts/summary.py` | Writes a short analysis for each game from templates (no paid APIs). |
| 7 | `scripts/notify.py` | Posts the top 5 bets to Discord. |
| 8 | `scripts/main.py` | Runs everything and writes `data/predictions.json`. |

The site (`index.html`) is one static file. It reads `data/predictions.json` and shows win/draw/loss %,
most likely score, expected goals, best bet, edge, EV, confidence and the summary for each game. You
can sort by edge, EV, confidence or kick-off and filter by league. It works on mobile and has dark and
light modes.

### Key definitions
- **Edge** = model probability − bookmaker no-vig probability (from the average odds).
- **EV** = p × best price − 1. Here p blends the model with the market, weighted by the tuned
  `market_weight`. Best prices more than 7% above the average are treated as stale and capped.
- **Confidence** (5–95) goes down when xG is missing, when team news wasn't checked, early in the
  season, when there are no odds, and when the model disagrees with the market by an implausible amount.

## Setup

1. **Fork or clone** this repo, then enable Pages under **Settings → Pages → Source: GitHub Actions**.
2. **Add the three secrets** under **Settings → Secrets and variables → Actions → New repository secret**:

| Secret | Where to get it | Used for |
|---|---|---|
| `APIFOOTBALL_KEY` | Sign up at [dashboard.api-football.com](https://dashboard.api-football.com/register) (free plan: 100 requests/day). Copy the key from *Account → My Access*. | Injuries, suspensions, player minutes and goals, manager changes (flagged games only; capped at 90 requests per run). |
| `ODDS_API_KEY` | Sign up at [the-odds-api.com](https://the-odds-api.com/#get-access) (free plan: 500 credits/month). The key arrives by email. | Upcoming games and odds for the top 5 leagues (so the site has games even between football-data refreshes), plus extra bookmaker odds for flagged games. Budgeted so it never exceeds the free 500 credits: see below. |
| `DISCORD_WEBHOOK` | In Discord: *Server Settings → Integrations → Webhooks → New Webhook*. Pick a channel and *Copy Webhook URL*. | Posting the top 5 bets. |

   **Odds API credit budget** (`scripts/oddsapi.py`): each league request costs 2 credits (1X2 + totals; no Asian handicap). Before
   every run the pipeline reads the credits left (a free call) and allows at most
   `(credits left - 15) / scheduled runs left this month`, capped at 40 per run (20 leagues). The Premier
   League, La Liga, Bundesliga, Serie A and Ligue 1 are always fetched first (12-day window, so the next
   round shows up even during international breaks). Leftover budget goes to the leagues of the flagged
   games, most valuable first. Each league is fetched at most once per run. A normal month uses at most
   about 360 credits.

   All three are optional. Without them the pipeline still runs: it skips injuries, extra odds or
   Discord and lowers confidence where team news is missing.
3. **Run it**: go to *Actions → Weekly predictions → Run workflow*. After that it runs automatically
   every **Tuesday and Friday at 07:00 UTC**. football-data.co.uk refreshes `fixtures.csv` on those days.
   Tick *backtest* to force a re-tune. Otherwise the backtest re-runs by itself when `params.json` is
   more than 30 days old (about 12 minutes).

## Run locally

```bash
pip install -r requirements.txt
python -m unittest discover tests          # offline tests (APIs are mocked)
python scripts/main.py --no-notify         # live run -> data/predictions.json
python scripts/main.py --demo 2026-09-19   # replay a past week -> data/predictions_demo.json
python scripts/backtest.py                 # tune + backtest -> backtest.md, data/params.json
python -m http.server 8000                 # then open http://localhost:8000/
```

Set `APIFOOTBALL_KEY`, `ODDS_API_KEY` and `DISCORD_WEBHOOK` as environment variables to use them
locally.

## Robustness
- Every source is optional. A failed download is logged and the run continues. ClubElo falls back
  to the built-in Elo, missing xG falls back to shots on target, and games without odds are still
  predicted.
- Team names are fuzzy-matched within the same league or country, with an alias table
  (`common.py`) for the usual suspects (Man United, Nott'm Forest, M'gladbach and so on).
- A single failing game is skipped rather than aborting the run. The API-Football request budget is
  enforced and stops early when the daily quota is nearly used up.

## Limitations
- Rest days and congestion only count league games, because the free data doesn't include cups or
  European matches.
- API-Football's free plan may restrict which seasons it serves. If it does, the run logs the plan
  error, carries on without team news and lowers confidence.
- Treat backtest ROI over one season as noise-dominated. Closing line value is the better signal.

This is a statistical model, not financial advice. 18+. Please gamble responsibly.
