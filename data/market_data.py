import math
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger

from config import VIX_REGIMES
from errors import ErrorCode

SECTOR_ETFS = ["XLK", "XLE", "XLF", "XLV", "XLU", "XLI", "XLB", "XLP"]

# Maps SPDR sector ETF tickers to the GICS sector name strings used by Wikipedia
# (which is the source for candidate.sector in universe_manager.py).
_ETF_TO_GICS: dict[str, set[str]] = {
    "XLK": {"Information Technology", "Technology", "EDP Services", "Semiconductors"},
    "XLF": {"Financials", "Financial Services", "Banks"},
    "XLV": {"Health Care", "Healthcare", "Biotechnology", "Pharmaceuticals"},
    "XLU": {"Utilities"},
    "XLI": {"Industrials"},
    "XLE": {"Energy"},
    "XLB": {"Materials"},
    "XLP": {"Consumer Staples"},
    "XLY": {"Consumer Discretionary"},
    "XLC": {"Communication Services", "Telecommunication Services"},
    "XLRE": {"Real Estate"},
}


def fetch_vix_spy() -> dict:
    """
    Returns VIX level and SPY with 50d/200d MAs.
    {vix, spy_price, spy_ma50, spy_ma200, spy_hv20}
    """
    try:
        vix_data = yf.download("^VIX", period="5d", interval="1d", progress=False, auto_adjust=True)
        spy_data = yf.download("SPY", period="12mo", interval="1d", progress=False, auto_adjust=True)

        # squeeze() on a single-row DataFrame can return a scalar — guard with Series cast
        vix_raw   = vix_data["Close"].squeeze()
        vix_close = vix_raw if hasattr(vix_raw, "iloc") else pd.Series([float(vix_raw)])
        spy_raw   = spy_data["Close"].squeeze()
        spy_close = spy_raw if hasattr(spy_raw, "iloc") else pd.Series([float(spy_raw)])

        vix       = float(vix_close.iloc[-1])
        spy_price = float(spy_close.iloc[-1])
        spy_ma50  = float(spy_close.rolling(50).mean().iloc[-1])
        spy_ma200 = float(spy_close.rolling(200).mean().iloc[-1])

        log_returns = np.log(spy_close / spy_close.shift(1))
        spy_hv20 = float(log_returns.rolling(20).std().iloc[-1] * np.sqrt(252))

        prior_close  = float(spy_close.iloc[-2]) if len(spy_close) >= 2 else spy_price
        five_day_high = float(spy_data["High"].tail(5).max())
        five_day_low  = float(spy_data["Low"].tail(5).min())

        result = {
            "vix":          round(vix, 2),
            "vix_regime":   classify_vix(vix)[0],
            "vix_sizing":   classify_vix(vix)[1],
            "spy_price":    round(spy_price, 2),
            "spy_prior_close": round(prior_close, 2),
            "spy_5d_high":  round(five_day_high, 2),
            "spy_5d_low":   round(five_day_low, 2),
            "spy_ma50":     round(spy_ma50, 2),
            "spy_ma200":    round(spy_ma200, 2),
            "spy_trend":    classify_spy_trend(spy_price, spy_ma50, spy_ma200),
            "spy_hv20":     round(spy_hv20 * 100, 2),
        }
        logger.info(f"VIX={vix:.1f} ({result['vix_regime']}), SPY={spy_price:.2f} ({result['spy_trend']})")
        return result

    except Exception as e:
        logger.error(f"[{ErrorCode.E1003}] VIX/SPY fetch failed: {e}")
        return {
            "vix": 20.0, "vix_regime": "normal", "vix_sizing": 1.0,
            "spy_price": 0.0, "spy_prior_close": 0.0,
            "spy_5d_high": 0.0, "spy_5d_low": 0.0,
            "spy_ma50": 0.0, "spy_ma200": 0.0,
            "spy_trend": "unknown", "spy_hv20": 0.0,
        }


def fetch_sector_rotation() -> dict:
    """
    Fetch 20-day returns for all sector ETFs.
    Returns {sector_returns, leading_sectors, lagging_sectors}.
    """
    try:
        data = yf.download(
            SECTOR_ETFS, period="2mo", interval="1d", progress=False, auto_adjust=True
        )
        closes = data["Close"]
        if closes.empty:
            raise ValueError("Empty sector ETF data")

        returns = {}
        for etf in SECTOR_ETFS:
            if etf in closes.columns:
                series = closes[etf].dropna()
                if len(series) >= 21:
                    ret_20d = float((series.iloc[-1] / series.iloc[-21] - 1) * 100)
                    returns[etf] = round(ret_20d, 2)

        if not returns:
            raise ValueError("No sector return data computed")

        sorted_etfs  = sorted(returns, key=returns.get, reverse=True)
        leading      = sorted_etfs[:3]
        lagging      = sorted_etfs[-3:]

        leading_names = set().union(*[_ETF_TO_GICS.get(e, set()) for e in leading])
        lagging_names = set().union(*[_ETF_TO_GICS.get(e, set()) for e in lagging])

        logger.info(f"Sector rotation — leading: {leading}, lagging: {lagging}")
        return {
            "sector_returns":       returns,
            "leading_sectors":      leading,
            "lagging_sectors":      lagging,
            "leading_sector_names": leading_names,
            "lagging_sector_names": lagging_names,
        }

    except Exception as e:
        logger.error(f"[{ErrorCode.E1004}] Sector ETF fetch failed: {e}")
        return {
            "sector_returns":       {},
            "leading_sectors":      [],
            "lagging_sectors":      [],
            "leading_sector_names": set(),
            "lagging_sector_names": set(),
        }


def fetch_premarket_prices(tickers: list[str]) -> dict[str, float]:
    """
    Fetch latest prices via yfinance (fallback when Alpaca not configured).
    Returns {ticker: price}.
    """
    if not tickers:
        return {}
    try:
        data = yf.download(tickers, period="2d", interval="1d", progress=False, auto_adjust=True)
        closes = data["Close"] if hasattr(data["Close"], "columns") else data["Close"].to_frame()
        prices = {}
        for t in tickers:
            if t in closes.columns:
                val = closes[t].dropna()
                if not val.empty:
                    prices[t] = round(float(val.iloc[-1]), 2)
        logger.info(f"Fetched prices for {len(prices)}/{len(tickers)} tickers")
        return prices
    except Exception as e:
        logger.error(f"[{ErrorCode.E1003}] Premarket price fetch failed: {e}")
        return {}


def fetch_market_caps(tickers: list[str], max_workers: int = 8) -> dict[str, float]:
    """
    Fetch market caps via yfinance fast_info in parallel.
    Returns {ticker: market_cap_float}. Missing/failed tickers are omitted.
    """
    if not tickers:
        return {}

    def _get_cap(t: str) -> tuple[str, float]:
        try:
            cap = yf.Ticker(t).fast_info.market_cap
            return t, float(cap) if cap else 0.0
        except Exception:
            return t, 0.0

    result = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_get_cap, t): t for t in tickers}
        for future in as_completed(futures):
            t, cap = future.result()
            if cap > 0:
                result[t] = cap

    logger.info(f"Fetched market caps for {len(result)}/{len(tickers)} tickers")
    return result


def classify_vix(vix: float) -> tuple[str, float]:
    """
    Returns (regime_label, size_multiplier) based on VIX level.
    Regimes: calm (<15), normal (15-25), elevated (25-35), crisis (>35).
    """
    for threshold, label, multiplier in VIX_REGIMES:
        if vix < threshold:
            return label, multiplier
    return "crisis", 0.50


def classify_spy_trend(price: float, ma50: float, ma200: float) -> str:
    """
    Returns one of: strong_uptrend, moderate_uptrend, short_term_weak, bear_risk.
    """
    if price > ma50 and price > ma200:
        return "strong_uptrend"
    if price > ma200:
        return "moderate_uptrend"
    if price > ma200 * 0.95:
        return "short_term_weak"
    return "bear_risk"


def compute_spy_0dte_setup(vix_spy: dict, put_call: float, macro_events: list) -> dict:
    """
    Compute today's SPY 0DTE setup: expected move, key S/R levels, and structure recommendation.
    Pure computation — no additional API calls. Uses VIX as IV proxy.
    Called in Phase 1 after vix_spy, put_call, and macro_events are all available.
    """
    spy_price    = vix_spy.get("spy_price", 0.0)
    vix          = vix_spy.get("vix", 20.0)
    vix_regime   = vix_spy.get("vix_regime", "normal")
    spy_trend    = vix_spy.get("spy_trend", "unknown")
    ma50         = vix_spy.get("spy_ma50", 0.0)
    ma200        = vix_spy.get("spy_ma200", 0.0)
    prior_close  = vix_spy.get("spy_prior_close", spy_price)
    five_day_high = vix_spy.get("spy_5d_high", spy_price)
    five_day_low  = vix_spy.get("spy_5d_low", spy_price)

    if spy_price <= 0:
        return {"error": "SPY price unavailable"}

    # 1-SD daily expected move: price × (VIX/100) / sqrt(252)
    daily_1sd = spy_price * (vix / 100.0) / math.sqrt(252)

    em_low  = round(spy_price - daily_1sd, 2)
    em_high = round(spy_price + daily_1sd, 2)

    # Key S/R levels (rounded to nearest $1 for cleanliness)
    below_round5 = round((spy_price // 5) * 5, 2)      # nearest $5 below
    above_round5 = round(below_round5 + 5, 2)           # nearest $5 above

    raw_support = sorted({
        round(prior_close, 2),
        round(ma50, 2),
        round(five_day_low, 2),
        round(below_round5, 2),
        round(em_low, 2),
    }, reverse=True)

    raw_resistance = sorted({
        round(five_day_high, 2),
        round(above_round5, 2),
        round(ma50 * 1.005, 2) if spy_price > ma50 else round(ma50, 2),
        round(em_high, 2),
    })

    # Remove levels too close to current price (< $1 away) to reduce noise
    support    = [s for s in raw_support    if s < spy_price - 1.0][:4]
    resistance = [r for r in raw_resistance if r > spy_price + 1.0][:4]

    # Detect high-impact macro event today
    major_macro_today = any(
        e.get("days_away", 99) == 0 and e.get("is_high_impact", False)
        for e in (macro_events or [])
    )
    macro_events_today = [
        e.get("event", "") for e in (macro_events or [])
        if e.get("days_away", 99) == 0 and e.get("is_high_impact", False)
    ]

    # Structure recommendation
    if major_macro_today:
        structure   = "skip"
        skip_reason = f"High-impact macro event today ({', '.join(macro_events_today)}) — unpredictable price action, elevated gap risk"
    elif vix_regime in ("elevated", "crisis"):
        structure   = "skip"
        skip_reason = f"VIX {vix:.1f} ({vix_regime} regime) — spreads too wide, risk/reward unfavorable for 0DTE"
    elif spy_trend in ("strong_uptrend", "moderate_uptrend") and put_call < 1.0 and vix < 20:
        structure   = "bull_put_spread"
        skip_reason = None
    elif spy_trend == "bear_risk" and put_call > 1.3:
        structure   = "bear_call_spread"
        skip_reason = None
    else:
        structure   = "iron_condor"
        skip_reason = None

    # Suggested strikes for the recommended structure
    spread_width = 5.0   # $5 wide spreads are standard for SPY 0DTE
    suggested_strikes: dict = {}
    if structure == "bull_put_spread":
        short_put = round(em_low - 1, 0)    # sell just below 1-SD low
        long_put  = round(short_put - spread_width, 0)
        suggested_strikes = {"sell_put": short_put, "buy_put": long_put}
    elif structure == "bear_call_spread":
        short_call = round(em_high + 1, 0)  # sell just above 1-SD high
        long_call  = round(short_call + spread_width, 0)
        suggested_strikes = {"sell_call": short_call, "buy_call": long_call}
    elif structure == "iron_condor":
        short_put  = round(em_low - 1, 0)
        long_put   = round(short_put - spread_width, 0)
        short_call = round(em_high + 1, 0)
        long_call  = round(short_call + spread_width, 0)
        suggested_strikes = {
            "sell_put": short_put, "buy_put": long_put,
            "sell_call": short_call, "buy_call": long_call,
        }

    overnight_chg_pct = round((spy_price - prior_close) / prior_close * 100, 2) if prior_close else 0.0

    return {
        "spy_price":              spy_price,
        "prior_close":            round(prior_close, 2),
        "overnight_change_pct":   overnight_chg_pct,
        "vix":                    vix,
        "vix_regime":             vix_regime,
        "spy_trend":              spy_trend,
        "ma50":                   round(ma50, 2),
        "ma200":                  round(ma200, 2),
        "five_day_high":          round(five_day_high, 2),
        "five_day_low":           round(five_day_low, 2),
        "expected_move_1sd":      round(daily_1sd, 2),
        "expected_move_range":    [em_low, em_high],
        "expected_move_pct":      round(daily_1sd / spy_price * 100, 2),
        "key_levels": {
            "support":    support,
            "resistance": resistance,
        },
        "recommended_structure":  structure,
        "skip_reason":            skip_reason,
        "suggested_strikes":      suggested_strikes,
        "spread_width":           spread_width,
        "major_macro_today":      major_macro_today,
        "iv_source":              "VIX proxy — verify live ATM IV on broker before trading",
        "put_call_ratio":         put_call,
        "expiry":                 date.today().isoformat(),
    }
