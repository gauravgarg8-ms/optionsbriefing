"""
Ticker universe — S&P 500 + Nasdaq-100 from Wikipedia (free, no API key).
FMP constituent endpoints were deprecated for new accounts after Aug 2025.
"""
import io
import json
from pathlib import Path

import pandas as pd
import requests
from loguru import logger

from config import UNIVERSE_MIN_PRICE, UNIVERSE_MIN_MARKET_CAP
from errors import ErrorCode

CACHE_PATH = Path(__file__).parent.parent / "output" / "universe_cache.json"

_SP500_URL  = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_NDX_URL    = "https://en.wikipedia.org/wiki/Nasdaq-100"


def _wiki_tables(url: str) -> list:
    """Fetch Wikipedia page with browser headers and parse tables."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    return pd.read_html(io.StringIO(resp.text))


def get_sp500() -> list[dict]:
    """Fetch S&P 500 constituents from Wikipedia."""
    try:
        tables = _wiki_tables(_SP500_URL)
        df     = tables[0]
        # Wikipedia columns: "Symbol", "Security", "GICS Sector", ...
        symbol_col = next((c for c in df.columns if "symbol" in c.lower() or "ticker" in c.lower()), df.columns[0])
        sector_col = next((c for c in df.columns if "sector" in c.lower()), None)
        result = []
        for _, row in df.iterrows():
            ticker = str(row[symbol_col]).strip().replace(".", "-")
            sector = str(row[sector_col]).strip() if sector_col else "Unknown"
            if ticker and len(ticker) <= 6:
                result.append({"symbol": ticker, "sector": sector, "name": ""})
        logger.info(f"Wikipedia S&P 500: {len(result)} constituents")
        return result
    except Exception as e:
        logger.error(f"[{ErrorCode.E1001}] Wikipedia S&P 500 fetch failed: {e}")
        return []


def get_nasdaq100() -> list[dict]:
    """Fetch Nasdaq-100 constituents from Wikipedia."""
    try:
        tables = _wiki_tables(_NDX_URL)
        df = None
        for t in tables:
            cols = [str(c).lower() for c in t.columns]
            if any("ticker" in c or "symbol" in c for c in cols):
                df = t
                break
        if df is None:
            raise ValueError("Nasdaq-100 table not found on Wikipedia page")

        symbol_col = next((c for c in df.columns if "ticker" in str(c).lower() or "symbol" in str(c).lower()), df.columns[0])
        sector_col = next((c for c in df.columns if "sector" in str(c).lower()), None)
        result = []
        for _, row in df.iterrows():
            ticker = str(row[symbol_col]).strip().replace(".", "-")
            sector = str(row[sector_col]).strip() if sector_col else "Unknown"
            if ticker and len(ticker) <= 6:
                result.append({"symbol": ticker, "sector": sector, "name": ""})
        logger.info(f"Wikipedia Nasdaq-100: {len(result)} constituents")
        return result
    except Exception as e:
        logger.error(f"[{ErrorCode.E1001}] Wikipedia Nasdaq-100 fetch failed: {e}")
        return []


def build_universe() -> list[dict]:
    """
    Fetch S&P 500 + Nasdaq-100 from Wikipedia, deduplicate, cache.
    Falls back to last cached result on failure.
    """
    sp500  = get_sp500()
    nasdaq = get_nasdaq100()

    combined = {r["symbol"]: r for r in sp500}
    for r in nasdaq:
        combined.setdefault(r["symbol"], r)

    if not combined:
        logger.error(f"[{ErrorCode.E1001}] Both Wikipedia feeds empty — loading fallback cache")
        return _load_cache()

    universe = list(combined.values())
    _save_cache(universe)
    logger.info(f"Universe built: {len(universe)} unique tickers")
    return universe


def prefilter_universe(universe: list[dict], price_data: dict[str, dict]) -> list[dict]:
    """
    Filter to liquid, tradeable candidates.
    price_data: {ticker: {price: float, market_cap: float}}
    Criteria: price > $10 AND market_cap > $2B
    """
    passed = []
    for entry in universe:
        ticker = entry.get("symbol", "")
        pdata  = price_data.get(ticker, {})
        price  = pdata.get("price", 0.0)
        mcap   = pdata.get("market_cap", 0.0)
        if price >= UNIVERSE_MIN_PRICE and mcap >= UNIVERSE_MIN_MARKET_CAP:
            passed.append({**entry, "price": price, "market_cap": mcap})

    if not passed:
        logger.error(f"[{ErrorCode.E1002}] Pre-filter returned 0 tickers — check price_data")
    else:
        logger.info(
            f"Universe pre-filter: {len(universe)} → {len(passed)} tickers "
            f"(price>${UNIVERSE_MIN_PRICE}, mcap>${UNIVERSE_MIN_MARKET_CAP/1e9:.0f}B)"
        )
    return passed


def _save_cache(universe: list[dict]) -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(universe, indent=2))
    except OSError as e:
        logger.warning(f"Could not write universe cache: {e}")


def _load_cache() -> list[dict]:
    if CACHE_PATH.exists():
        try:
            data = json.loads(CACHE_PATH.read_text())
            logger.info(f"Loaded {len(data)} tickers from universe cache (fallback)")
            return data
        except Exception as e:
            logger.error(f"Failed to load universe cache: {e}")
    return []
