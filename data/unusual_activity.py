import json
import time
from datetime import date
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from loguru import logger

from config import BARCHART_UA_URL
from errors import ErrorCode

_CACHE_PATH = Path(__file__).parent.parent / "output" / "unusual_activity_cache.json"
_HEADERS    = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
}


def fetch_unusual_activity() -> list[dict]:
    """
    Scrape Barchart unusual options activity.
    Returns list of {ticker, volume, oi_change, sentiment}.
    Caches result per calendar day — no re-scraping within same day.
    """
    today = date.today().isoformat()

    # Return cached result if already fetched today
    cached = _load_cache()
    if cached and cached.get("fetched_date") == today:
        logger.info(f"Unusual activity: loaded {len(cached.get('tickers', []))} tickers from cache")
        return cached.get("tickers", [])

    results = _scrape_barchart()
    _save_cache(results, today)
    return results


def _scrape_barchart() -> list[dict]:
    try:
        resp = requests.get(BARCHART_UA_URL, headers=_HEADERS, timeout=20)
        if resp.status_code == 403:
            logger.warning(f"[{ErrorCode.E1010}] Barchart returned 403 — scraping blocked")
            return []
        resp.raise_for_status()

        soup   = BeautifulSoup(resp.text, "html.parser")
        table  = soup.find("table")
        if not table:
            logger.warning(f"[{ErrorCode.E1010}] Barchart: no table found in response")
            return []

        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        results = []
        for row in table.find_all("tr")[1:]:   # skip header row
            cells = [td.get_text(strip=True) for td in row.find_all("td")]
            if len(cells) < 2:
                continue
            try:
                ticker = cells[0].upper().strip()
                if not ticker or len(ticker) > 6:
                    continue
                # Try to extract volume and sentiment from available columns
                vol = _safe_int(cells[1]) if len(cells) > 1 else 0
                # Sentiment: look for Put/Call column or infer from context
                sentiment = _infer_sentiment(cells, headers)
                results.append({"ticker": ticker, "volume": vol, "sentiment": sentiment})
            except Exception:
                continue

        logger.info(f"Barchart unusual activity: scraped {len(results)} tickers")
        return results[:50]   # cap at 50

    except Exception as e:
        logger.error(f"[{ErrorCode.E1010}] Barchart scrape failed: {e}")
        return []


def _infer_sentiment(cells: list[str], headers: list[str]) -> str:
    """Infer sentiment from cell values using header names."""
    for i, h in enumerate(headers):
        if i >= len(cells):
            break
        if "put" in h or "bearish" in h.lower():
            return "BEARISH"
        if "call" in h or "bullish" in h.lower():
            return "BULLISH"
    # Fallback: scan cell values
    row_text = " ".join(cells).lower()
    if "put" in row_text:
        return "BEARISH"
    if "call" in row_text:
        return "BULLISH"
    return "NEUTRAL"


def _safe_int(s: str) -> int:
    try:
        return int(s.replace(",", "").replace("K", "000").replace("M", "000000").split(".")[0])
    except Exception:
        return 0


def _load_cache() -> dict:
    try:
        if _CACHE_PATH.exists():
            return json.loads(_CACHE_PATH.read_text())
    except Exception as e:
        logger.debug(f"Unusual activity cache load failed: {e}")
    return {}


def _save_cache(tickers: list[dict], fetched_date: str) -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps({
            "fetched_date": fetched_date,
            "tickers":      tickers,
            "source":       "barchart",
        }, indent=2))
    except Exception as e:
        logger.debug(f"Unusual activity cache save failed: {e}")
