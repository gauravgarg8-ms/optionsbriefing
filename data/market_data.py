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

        result = {
            "vix":       round(vix, 2),
            "vix_regime": classify_vix(vix)[0],
            "vix_sizing": classify_vix(vix)[1],
            "spy_price":  round(spy_price, 2),
            "spy_ma50":   round(spy_ma50, 2),
            "spy_ma200":  round(spy_ma200, 2),
            "spy_trend":  classify_spy_trend(spy_price, spy_ma50, spy_ma200),
            "spy_hv20":   round(spy_hv20 * 100, 2),
        }
        logger.info(f"VIX={vix:.1f} ({result['vix_regime']}), SPY={spy_price:.2f} ({result['spy_trend']})")
        return result

    except Exception as e:
        logger.error(f"[{ErrorCode.E1003}] VIX/SPY fetch failed: {e}")
        return {
            "vix": 20.0, "vix_regime": "normal", "vix_sizing": 1.0,
            "spy_price": 0.0, "spy_ma50": 0.0, "spy_ma200": 0.0,
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
