# Daily Options Briefing System

A hybrid Python + Claude AI pipeline that delivers a professional pre-market options briefing every weekday at 8:00 AM ET. Python handles all data fetching, quantitative calculations, screening, scoring, and risk pre-computation. Claude AI writes the narrative interpretation only.

---

## Architecture

```
7:30 AM  Phase 1  data_engine.py         → raw_market_data.json
7:35 AM  Phase 2  quant_engine.py        → quant_signals.json
7:39 AM  Phase 3  screening_engine.py    → screened_candidates.json
7:43 AM  Phase 4  risk_manager.py        → top_candidates.json (+ trade mgmt)
7:45 AM  Phase 5  scenario_classifier.py → top_candidates.json (+ scenarios)
7:46 AM  Phase 6  claude_interpreter.py  → Daily Briefing markdown
8:00 AM  Phase 7  delivery.py            → YYYY-MM-DD_OptionsBrief.md
```

---

## Supported Strategies

| Code | Strategy | Type |
|---|---|---|
| `bull_put_spread` | Bull Put Spread | Credit spread |
| `bear_call_spread` | Bear Call Spread | Credit spread |
| `iron_condor` | Iron Condor | Credit spread |
| `bull_call_spread` | Bull Call Spread | Debit spread |
| `bear_put_spread` | Bear Put Spread | Debit spread |
| `long_straddle` | Long Straddle | Debit vol play |
| `long_strangle` | Long Strangle | Debit vol play |
| `cash_secured_put` | Cash-Secured Put | Credit, cash required |

**Covered call**: Claude flags manually when conditions are met. Not in automated selection.

---

## Prerequisites

- Python 3.11+
- macOS (for launchd scheduling)
- Free API accounts (see below)

---

## API Keys Setup

### Required

| Service | Where to Get | `.env` Key |
|---|---|---|
| Alpaca Markets | [alpaca.markets](https://alpaca.markets) — free paper account | `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` |
| Financial Modeling Prep | [financialmodelingprep.com](https://financialmodelingprep.com) — free tier (250 req/day) | `FMP_API_KEY` |
| Finnhub | [finnhub.io](https://finnhub.io) — free account (60 req/min) | `FINNHUB_API_KEY` |
| FRED (St. Louis Fed) | [fred.stlouisfed.org/docs/api](https://fred.stlouisfed.org/docs/api/api_key.html) — free | `FRED_API_KEY` |
| NewsAPI | [newsapi.org](https://newsapi.org) — free developer plan (100 req/day) | `NEWS_API_KEY` |
| Anthropic | [console.anthropic.com](https://console.anthropic.com) — pay-per-token | `ANTHROPIC_API_KEY` |

### Optional Upgrade

| Service | Where to Get | `.env` Key | Benefit |
|---|---|---|---|
| Tradier | [developer.tradier.com](https://developer.tradier.com) — free developer sandbox | `TRADIER_TOKEN` | More reliable options chains; includes broker-quoted Greeks. Leave blank to use yfinance instead. |

> **Options data source**: yfinance is the default (no account needed). Set `TRADIER_TOKEN` to upgrade to Tradier automatically — the system detects the token and switches sources with no other changes required.

---

## Installation

```bash
# 1. Navigate to project directory
cd "options-briefing"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up API keys
cp .env.example .env
# Edit .env and fill in your API keys

# 4. Seed the IV history database (one-time, ~5 minutes)
python db/seed_iv_history.py
```

---

## Scheduling — macOS launchd

The pipeline runs automatically at 7:30 AM ET on weekdays via macOS launchd. It will fire on wake even if the Mac was asleep at 7:30 AM.

```bash
# 1. Edit the plist file — update paths for your machine
open com.gg.options-briefing.plist

# 2. Copy to LaunchAgents
cp com.gg.options-briefing.plist ~/Library/LaunchAgents/

# 3. Load the job
launchctl load ~/Library/LaunchAgents/com.gg.options-briefing.plist

# 4. Verify it is loaded
launchctl list | grep options-briefing

# To unload / stop:
launchctl unload ~/Library/LaunchAgents/com.gg.options-briefing.plist
```

---

## Manual Run

Run the pipeline immediately regardless of schedule:

```bash
python main.py
```

---

## Output

Briefing files are saved to:
```
output/briefings/YYYY-MM-DD_OptionsBrief.md
```

Each file contains:
- Market environment (VIX, SPY trend, sector rotation, macro events)
- Up to 10 options trade setups with full B-S pricing tables
- Trade management dates (21 DTE, profit target, stop loss)
- Portfolio exposure check
- Pre-trade checklist

---

## Logs

Daily log files at `logs/YYYY-MM-DD.log`. Each entry includes a structured error code:

| Range | Area |
|---|---|
| E1xxx | Data pipeline (universe, market data, options, earnings, sentiment) |
| E2xxx | Quant engine (Black-Scholes, volatility, strike selection) |
| E3xxx | Screening and scoring |
| E4xxx | Output (Claude API, delivery, JSON serialization) |
| E5xxx | Scheduler (market holidays, pipeline timeout) |

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Phase by phase
pytest tests/test_universe_manager.py tests/test_market_data.py tests/test_technicals.py tests/test_db_manager.py -v
pytest tests/test_volatility.py tests/test_black_scholes.py tests/test_structure_selector.py tests/test_options_data.py -v
pytest tests/test_data_engine.py -v
pytest tests/test_screens.py tests/test_scorer.py tests/test_composition.py tests/test_screening_engine.py -v
pytest tests/test_risk_manager.py tests/test_scenario_classifier.py tests/test_claude_interpreter.py tests/test_delivery.py -v
```

---

## Important Caveats

- **Black-Scholes outputs are theoretical only.** Always verify live mid-price and IV Rank on your broker (ThinkorSwim or IBKR) before placing any trade.
- **IV Rank proxy**: For the first 30 days after setup, IV Rank is computed from a HV30 proxy. The briefing header shows `⚠️ IV RANK PROXY — N/30 days` until real IV data accumulates.
- **IV Rank scale**: Barchart IV Rank ≠ ThinkorSwim IV Percentile. Both are valid but use the same source consistently.

---

## Oracle Cloud Upgrade Path

When you decide to move from Mac to Oracle Cloud (recommended: Ampere A1, 4 cores, 24 GB RAM — always free):

1. Provision an Oracle Cloud Always Free ARM VM
2. `sudo apt update && sudo apt install python3.11 python3-pip -y`
3. Clone / copy the project, `pip install -r requirements.txt`
4. Copy your `.env` to the VM (never commit secrets to git)
5. Replace launchd with crontab: `crontab -e` → add `30 7 * * 1-5 /usr/bin/python3 /path/to/main.py`
6. Add Gmail SMTP email delivery to `delivery.py`:
   - Add `DELIVERY_EMAIL_ENABLED=true`, `GMAIL_ADDRESS`, `GMAIL_APP_PASSWORD` to `.env`
7. Run `python db/seed_iv_history.py` once on the VM
8. Test: `python main.py` — verify briefing is emailed before enabling cron

---

## Disclaimer

This system is for personal educational use only. It is not financial advice. All options trade setups are theoretical. Always verify data independently and consult a licensed financial advisor before trading.
