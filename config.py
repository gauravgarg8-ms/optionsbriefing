import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ────────────────────────────────────────────────────────────────
ALPACA_API_KEY              = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY           = os.getenv("ALPACA_SECRET_KEY", "")
TRADIER_TOKEN               = os.getenv("TRADIER_TOKEN", "")
FMP_API_KEY                 = os.getenv("FMP_API_KEY", "")
FINNHUB_API_KEY             = os.getenv("FINNHUB_API_KEY", "")
FRED_API_KEY                = os.getenv("FRED_API_KEY", "")
NEWS_API_KEY                = os.getenv("NEWS_API_KEY", "")
ANTHROPIC_API_KEY           = os.getenv("ANTHROPIC_API_KEY", "")

# ── API Endpoints ────────────────────────────────────────────────────────────
FMP_BASE          = "https://financialmodelingprep.com/api/v3"
FMP_SP500_URL     = f"{FMP_BASE}/sp500_constituent"
FMP_NASDAQ100_URL = f"{FMP_BASE}/nasdaq_constituent"
FMP_EARNINGS_URL  = f"{FMP_BASE}/earning_calendar"
FMP_MACRO_URL     = f"{FMP_BASE}/economic_calendar"

TRADIER_BASE      = "https://sandbox.tradier.com/v1"
FINNHUB_BASE      = "https://finnhub.io/api/v1"
FRED_BASE         = "https://api.stlouisfed.org/fred/series/observations"
NEWS_API_BASE     = "https://newsapi.org/v2"
CBOE_PC_URL       = "https://cdn.cboe.com/data/us/options/market_statistics/daily/"
CNN_FG_URL        = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
BARCHART_UA_URL   = "https://www.barchart.com/options/unusual-activity/stocks"

# ── Macro Calendar Sources ────────────────────────────────────────────────────
FED_FOMC_URL      = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
BLS_SCHEDULE_URL  = "https://www.bls.gov/schedule/news_release/"
BEA_SCHEDULE_URL  = "https://www.bea.gov/news/schedule"

# ── FRED Series ─────────────────────────────────────────────────────────────
FRED_TBILL_SERIES = "DTB3"

# ── Universe Pre-filter ──────────────────────────────────────────────────────
UNIVERSE_MIN_PRICE      = 10.0
UNIVERSE_MIN_MARKET_CAP = 2_000_000_000  # $2B

# ── IV / Volatility ─────────────────────────────────────────────────────────
IV_RANK_CREDIT_MIN               = 50
IV_RANK_DEBIT_MAX                = 30
IV_RV_MIDDLE_ZONE_CREDIT_THRESHOLD = 1.2   # middle zone (30-50% IV Rank): ≥1.2 → credit
IV_HISTORY_MIN_DAYS              = 30      # days before IV Rank is considered reliable
IV_PROXY_WARNING_DAYS            = 30      # show ⚠️ proxy warning until this many real days

# ── Screening ────────────────────────────────────────────────────────────────
MAX_CANDIDATES_PER_POOL = {
    "high_iv":      15,
    "earnings":      6,
    "low_iv_trend":  8,
    "bearish":       4,
}
SCREEN_HIGH_IV_MIN_PRICE      = 15.0
SCREEN_HIGH_IV_MIN_OPTIONS_VOL = 500
SCREEN_EARNINGS_MIN_OPTIONS_VOL = 1000
SCREEN_EARNINGS_MIN_IMPLIED_MOVE = 0.05  # 5%
SCREEN_EARNINGS_DAYS_AHEAD      = 7

# ── Scoring ──────────────────────────────────────────────────────────────────
SCORE_FLOOR             = 45
SCORE_HIGH_CONFIDENCE   = 65
MIN_CANDIDATES_FOR_TRADE = 5

# ── Golden Rules ─────────────────────────────────────────────────────────────
POP_FLOOR               = 0.60   # credit spreads: min PoP to trade
POP_HALF_SIZE_THRESHOLD = 0.70   # credit spreads: PoP below this → half size
POP_FLOOR_DEBIT               = 0.40   # debit spreads: min PoP to trade (ATM entry ≈ 50% is normal)
POP_HALF_SIZE_DEBIT_THRESHOLD = 0.50   # debit spreads: PoP below this → half size
LIQUIDITY_MAX_BID_ASK_PCT = 0.10   # 10%
LIQUIDITY_MIN_OI          = 500

# ── Portfolio Limits ──────────────────────────────────────────────────────────
MAX_SECTOR_POSITIONS = 2
MAX_EARNINGS_PLAYS   = 4
MAX_NET_DELTA        = 20
MAX_SECTOR_PCT       = 0.30

# ── VIX Regimes ───────────────────────────────────────────────────────────────
VIX_REGIMES = [
    (15,   "calm",     1.00),
    (25,   "normal",   1.00),
    (35,   "elevated", 0.75),
    (float("inf"), "crisis", 0.50),
]

# ── Options Deep Fetch ────────────────────────────────────────────────────────
DEEP_FETCH_DTE_MIN    = 25
DEEP_FETCH_DTE_MAX    = 45
DEEP_FETCH_SLEEP_SECS = 1   # between tickers — Finnhub 60 req/min guard

# ── Pinned Tickers (always included in briefing regardless of screens) ────────
PINNED_TICKERS = [{"symbol": "SPY", "sector": "Index ETF", "name": "SPDR S&P 500 ETF", "force_dte_0": True}]

# ── Repeat Ticker History ─────────────────────────────────────────────────────
TICKER_HISTORY_WINDOW_DAYS = 7  # days to look back when flagging repeat appearances

# ── Half-Size Score Override ──────────────────────────────────────────────────
POP_HALF_SIZE_SCORE_OVERRIDE = 70  # full size allowed if score >= this, even at low PoP

# ── Claude / Output ───────────────────────────────────────────────────────────
CLAUDE_MODEL        = "claude-sonnet-4-6"
CLAUDE_MAX_TOKENS   = 32000
CLAUDE_MAX_RETRIES  = 3
PIPELINE_TIMEOUT_MINS = 30

OUTPUT_FILENAME_PATTERN = "{date}_OptionsBrief.txt"
OUTPUT_BRIEFINGS_DIR    = "output/briefings"
