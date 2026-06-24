# Daily Options Briefing System

A hybrid Python + Claude AI pipeline that delivers a professional pre-market options briefing every weekday at 7:30 AM ET. Python handles all data fetching, quantitative calculations, screening, scoring, and risk pre-computation. Claude AI writes the narrative interpretation only — it does zero math.

---

## Architecture

```
7:30 AM  Phase 1  data_engine.py         → raw_market_data.json
         Phase 2A quant_engine.py        → lightweight signals (all 515 tickers)
         Phase 2B screening_engine.py    → score + filter top 15 candidates
         Phase 2C quant_engine.py        → deep fetch: options chain + B-S pricing
         Phase 4  risk_manager.py        → trade management dates, PoP quality, sizing
         Phase 5  scenario_classifier.py → active scenario flags (FOMC, earnings, VIX)
         Phase 6  claude_interpreter.py  → Claude writes narrative only
~7:37 AM Phase 7  delivery.py            → YYYY-MM-DD_OptionsBrief.txt
```

Total runtime: ~7 minutes (dominated by Phase 2C — 15 deep option chain fetches with 1-second Finnhub rate-limit sleep).

---

## Supported Strategies

| Code | Strategy | Type | When Selected |
|---|---|---|---|
| `bull_put_spread` | Bull Put Spread | Credit | IV Rank > 50%, Bullish |
| `bear_call_spread` | Bear Call Spread | Credit | IV Rank > 50%, Bearish |
| `iron_condor` | Iron Condor | Credit | IV Rank > 50%, Neutral |
| `earnings_credit_spread` | Earnings Credit Spread | Credit | Earnings ≤7 days, implied move > hist avg |
| `cash_secured_put` | Cash-Secured Put | Credit | Bullish income bias |
| `bull_call_spread` | Bull Call Spread | Debit | IV Rank < 30%, Bullish |
| `bear_put_spread` | Bear Put Spread | Debit | IV Rank < 30%, Bearish |
| `long_straddle` | Long Straddle | Debit vol | IV Rank < 30%, Neutral or pre-earnings |
| `long_strangle` | Long Strangle | Debit vol | IV Rank < 30%, wide strangle bias |

**PoP quality gates are structure-aware:**
- Credit spreads: 60% floor (half-size 60–70%, full-size 70%+)
- Debit spreads: 40% floor (half-size 40–50%, full-size 50%+) — ATM entry ≈ 50% is expected for debit structures

**Covered call**: flagged in the briefing when conditions are met, not in automated selection (requires ownership status).

---

## Data Sources

| Data | Source | Notes |
|---|---|---|
| S&P 500 / Nasdaq-100 universe | Wikipedia via `pd.read_html` | Falls back to local cache if offline |
| Market caps | yfinance `Ticker.fast_info.market_cap` | ThreadPoolExecutor(8) |
| Options chain | yfinance (primary), Tradier (if token set) | Pre-market: raw strikes used for B-S; bid/ask=0 is fine |
| VIX, SPY, sector ETFs | yfinance batch download | |
| T-bill rate (risk-free) | FRED `DTB3` series | |
| Fear & Greed index | CNN endpoint → VIX proxy fallback | CNN blocks intermittently |
| Put/Call ratio | CBOE date-specific CSV → SPY OI → VIX-based proxy | 3-tier fallback |
| Earnings calendar | Finnhub `/calendar/earnings` | |
| Macro calendar | FRED releases API + Federal Reserve FOMC page + BEA schedule | No Finnhub needed — multi-source, degrades gracefully |
| Analyst ratings / news | Finnhub per-ticker | Per candidate in Phase 2C |
| Unusual activity | Barchart scraper (BeautifulSoup) | Fragile — may break on HTML changes |
| IV history | SQLite (`db/iv_history.db`) | Auto-seeded from HV20 proxy on cold start; real IV accumulates daily |

---

## Prerequisites

- Python 3.11+
- macOS (for launchd scheduling; Linux users substitute `crontab`)

---

## API Keys Setup

Create a `.env` file in the project root:

```
FINNHUB_API_KEY=your_key_here
FRED_API_KEY=your_key_here
NEWS_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here

# Optional
TRADIER_TOKEN=your_token_here
```

### Required

| Service | Where to Get | Key |
|---|---|---|
| Finnhub | [finnhub.io](https://finnhub.io) — free account (60 req/min) | `FINNHUB_API_KEY` |
| FRED (St. Louis Fed) | [fred.stlouisfed.org/docs/api](https://fred.stlouisfed.org/docs/api/api_key.html) — free | `FRED_API_KEY` |
| NewsAPI | [newsapi.org](https://newsapi.org) — free developer plan (100 req/day) | `NEWS_API_KEY` |
| Anthropic | [console.anthropic.com](https://console.anthropic.com) — pay-per-token | `ANTHROPIC_API_KEY` |

### Optional

| Service | Key | Benefit |
|---|---|---|
| Tradier (sandbox) | `TRADIER_TOKEN` | More reliable options chains vs yfinance. Leave blank to use yfinance — no other changes needed. |

> **FMP API is not used.** Financial Modeling Prep v3 endpoints return 403 for accounts created after August 2025. All data is sourced from Wikipedia, yfinance, Finnhub, FRED, and direct scraping.

> **Alpaca API is not used.** Pre-market prices fall back to yfinance automatically.

---

## Installation

```bash
# 1. Clone / navigate to the project
cd "options-briefing"

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up API keys
cp .env.example .env
# Edit .env and fill in your keys

# 5. Run the pipeline (IV history is seeded automatically on first run)
python main.py
```

> IV Rank is computed from a rolling HV20 proxy on cold start and auto-seeds the database for all 515 tickers. No manual seeding step required. Real IV accumulates daily; IV Rank becomes fully reliable after ~30 trading days.

---

## Scheduling — macOS launchd

The pipeline runs automatically at 7:30 AM ET on weekdays. It fires on wake if the Mac was asleep at 7:30 AM.

```bash
# 1. Edit the plist file — update the path to match your machine
nano com.gg.options-briefing.plist

# 2. Copy to LaunchAgents
cp com.gg.options-briefing.plist ~/Library/LaunchAgents/

# 3. Load the job
launchctl load ~/Library/LaunchAgents/com.gg.options-briefing.plist

# 4. Verify it is loaded
launchctl list | grep options-briefing

# To unload:
launchctl unload ~/Library/LaunchAgents/com.gg.options-briefing.plist
```

---

## Manual Run

```bash
cd "options-briefing"
.venv/bin/python main.py
```

---

## Output

Briefing files are saved to:
```
output/briefings/YYYY-MM-DD_OptionsBrief.txt
```

Each file contains:
- Market environment (VIX regime, SPY trend, sector rotation, macro event calendar)
- Up to 10 options trade setups with full B-S theoretical pricing tables
- IV Data Quality row per candidate (real IV days / 30 threshold)
- Trade management dates (21 DTE exit, profit target, stop loss)
- PoP quality label — structure-aware (credit vs debit thresholds)
- Portfolio exposure check and active scenario flags
- Pre-trade checklist

---

## Logs

| File | Contents |
|---|---|
| `logs/YYYY-MM-DD.log` | Structured pipeline log with error codes |
| `logs/launchd_stderr.log` | Cumulative stderr from launchd runs |

Error code ranges:

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
.venv/bin/python -m pytest tests/ -q

# By phase
pytest tests/test_universe_manager.py tests/test_market_data.py tests/test_macro_data.py tests/test_technicals.py tests/test_db_manager.py -v
pytest tests/test_volatility.py tests/test_black_scholes.py tests/test_structure_selector.py tests/test_options_data.py -v
pytest tests/test_data_engine.py -v
pytest tests/test_screens.py tests/test_scorer.py tests/test_quant_engine.py tests/test_screening_engine.py -v
pytest tests/test_risk_manager.py tests/test_scenario_classifier.py tests/test_claude_interpreter.py tests/test_delivery.py -v
```

> 8 pre-existing test failures in `test_claude_interpreter`, `test_db_manager`, and `test_options_data` (Tradier mock / SQLite edge cases). All pipeline logic tests pass.

---

## Important Caveats

- **Black-Scholes outputs are theoretical only.** Always verify live mid-price and IV Rank on your broker before placing any trade.
- **IV Rank proxy**: For the first ~30 trading days after setup, IV Rank is computed from a rolling HV20 proxy. The briefing shows `⚠️ Proxy only` in the IV Data Quality row for each candidate until real IV accumulates.
- **Pre-market pricing**: The pipeline runs at 7:30 AM ET, before the options market opens at 9:30 AM. Bid/ask quotes are zero pre-open; the pipeline uses raw strike prices and B-S theoretical pricing instead. Always verify the live mid-price on your broker before entering.
- **Degenerate spreads**: If a strike selection produces a zero-width spread (e.g., long strike = short strike due to limited chain data), the candidate is silently dropped before reaching Claude rather than shown as invalid.

---

## Oracle Cloud Upgrade Path

To move from Mac launchd to Oracle Cloud Always Free (4 ARM cores, 24 GB RAM):

1. Provision an Oracle Cloud Always Free ARM VM
2. `sudo apt update && sudo apt install python3.11 python3-pip -y`
3. Clone the project and `pip install -r requirements.txt`
4. Copy `.env` to the VM (never commit secrets to git)
5. Replace launchd with crontab: `crontab -e` → `30 7 * * 1-5 cd /path/to/options-briefing && .venv/bin/python main.py`
6. Optionally add Gmail SMTP delivery to `delivery.py` (`GMAIL_ADDRESS`, `GMAIL_APP_PASSWORD` in `.env`)
7. Test with `python main.py` before enabling cron

---

## Disclaimer

This system is for personal educational use only. It is not financial advice. All options trade setups are theoretical. Always verify data independently and consult a licensed financial advisor before trading.
