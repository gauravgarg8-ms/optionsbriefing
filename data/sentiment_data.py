import csv
import io
import json
from datetime import date, datetime
from pathlib import Path

import requests
import yfinance as yf
from loguru import logger

from config import (CNN_FG_URL, CBOE_PC_URL, FINNHUB_API_KEY, FINNHUB_BASE,
                    NEWS_API_KEY, NEWS_API_BASE)
from errors import ErrorCode

# Keywords that trigger adverse_headline flag
ADVERSE_KEYWORDS = {"ban", "sanction", "tariff", "lawsuit", "recall", "fraud",
                    "investigation", "downgrade", "warning", "halt", "suspend"}

_FG_CACHE: dict = {}
_FG_CACHE_DATE: str = ""


_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://www.cnn.com/markets/fear-and-greed",
    "Origin":          "https://www.cnn.com",
}

# Separate headers for CBOE — CNN Referer/Origin causes 403 on cboe.com
_CBOE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://www.cboe.com/us/options/market_statistics/daily/",
}


def fetch_fear_greed() -> dict:
    """
    Fetch Fear & Greed Index. Tries CNN first (with full browser headers),
    then falls back to a VIX-based proxy so the briefing always has a value.
    Returns {score: int, label: str}. Cached per day.
    """
    global _FG_CACHE, _FG_CACHE_DATE
    today = date.today().isoformat()
    if _FG_CACHE_DATE == today and _FG_CACHE:
        return _FG_CACHE

    # ── Primary: CNN endpoint ────────────────────────────────────────────────
    try:
        resp = requests.get(CNN_FG_URL, timeout=10, headers=_BROWSER_HEADERS)
        resp.raise_for_status()
        data  = resp.json()
        score = int(float(data["fear_and_greed"]["score"]))
        label = data["fear_and_greed"]["rating"].title()
        _FG_CACHE      = {"score": score, "label": label, "source": "cnn"}
        _FG_CACHE_DATE = today
        logger.info(f"Fear & Greed (CNN): {score} ({label})")
        return _FG_CACHE
    except Exception as e:
        logger.warning(f"[{ErrorCode.E1007}] CNN Fear & Greed failed: {e} — trying VIX proxy")

    # ── Fallback: VIX-based proxy ─────────────────────────────────────────────
    try:
        vix_data = yf.download("^VIX", period="2d", interval="1d",
                               progress=False, auto_adjust=True)
        if not vix_data.empty:
            vix = float(vix_data["Close"].squeeze().iloc[-1])
            if vix < 15:    score, label = 75, "Extreme Greed"
            elif vix < 18:  score, label = 62, "Greed"
            elif vix < 22:  score, label = 50, "Neutral"
            elif vix < 28:  score, label = 35, "Fear"
            else:           score, label = 20, "Extreme Fear"
            result = {"score": score, "label": label, "source": "vix_proxy"}
            _FG_CACHE      = result
            _FG_CACHE_DATE = today
            logger.info(f"Fear & Greed (VIX proxy, VIX={vix:.1f}): {score} ({label})")
            return result
    except Exception as e2:
        logger.warning(f"[{ErrorCode.E1007}] VIX proxy also failed: {e2}")

    return {"score": 50, "label": "Neutral", "source": "default"}


def fetch_put_call_ratio() -> float:
    """
    Fetch equity put/call ratio.
    1. CBOE date-specific CSV (last 3 trading days).
    2. SPY options open-interest ratio via yfinance (free fallback).
    3. Hard default 0.9 if both fail.
    """
    from datetime import timedelta

    # ── Primary: CBOE date-specific CSV ─────────────────────────────────────
    today = date.today()
    cboe_urls = []
    for days_back in range(5):
        d = today - timedelta(days=days_back)
        if d.weekday() < 5:   # Mon–Fri only
            cboe_urls.append(
                f"https://cdn.cboe.com/data/us/options/market_statistics/daily/"
                f"{d.isoformat()}_options_volume.csv"
            )

    for url in cboe_urls:
        try:
            resp = requests.get(url, timeout=15, headers=_CBOE_HEADERS)
            if resp.status_code in (403, 404):
                logger.debug(f"CBOE {resp.status_code} for {url[-14:-4]}, trying next date")
                continue
            resp.raise_for_status()
            reader = csv.DictReader(io.StringIO(resp.text))
            rows   = list(reader)
            if not rows:
                continue
            last = rows[-1]
            # Column names vary; look for equity put/call first
            for col in last:
                if "equity" in col.lower() and ("put" in col.lower() or "p/c" in col.lower()):
                    val = float(last[col])
                    logger.info(f"CBOE equity P/C ratio ({url[-14:-4]}): {val}")
                    return round(val, 3)
            # Fallback: first numeric value in 0.3–2.0 range
            for col, val in last.items():
                try:
                    v = float(val)
                    if 0.3 <= v <= 2.0:
                        return round(v, 3)
                except (ValueError, TypeError):
                    continue
        except Exception as e:
            logger.debug(f"CBOE fetch error for {url[-14:-4]}: {e}")
            continue

    # ── Fallback 1: SPY options open-interest ratio ──────────────────────────
    try:
        spy      = yf.Ticker("SPY")
        expiries = spy.options
        if expiries:
            chain   = spy.option_chain(expiries[0])
            put_oi  = int(chain.puts["openInterest"].sum())
            call_oi = int(chain.calls["openInterest"].sum())
            if call_oi > 0:
                ratio = round(put_oi / call_oi, 3)
                logger.info(f"P/C ratio (SPY OI proxy, {expiries[0]}): {ratio}")
                return ratio
    except Exception as e:
        logger.warning(f"[{ErrorCode.E1007}] SPY P/C proxy failed: {e}")

    # ── Fallback 2: VIX-based proxy (batch endpoint, rate-limit resilient) ───
    try:
        vix_data = yf.download("^VIX", period="2d", interval="1d",
                               progress=False, auto_adjust=True)
        if not vix_data.empty:
            vix = float(vix_data["Close"].squeeze().iloc[-1])
            # Higher VIX → more put buying → higher P/C ratio
            if vix < 15:    ratio = 0.65
            elif vix < 18:  ratio = 0.75
            elif vix < 22:  ratio = 0.90
            elif vix < 28:  ratio = 1.05
            else:           ratio = 1.20
            logger.info(f"P/C ratio (VIX={vix:.1f} proxy): {ratio}")
            return ratio
    except Exception as e:
        logger.debug(f"VIX P/C proxy failed: {e}")

    logger.warning(f"[{ErrorCode.E1007}] CBOE P/C ratio unavailable — defaulting to 0.9")
    return 0.9


def fetch_market_news() -> list[dict]:
    """
    Fetch top market headlines from NewsAPI.
    Returns list of {headline, source, sentiment}.
    """
    try:
        resp = requests.get(
            f"{NEWS_API_BASE}/top-headlines",
            params={
                "category": "business",
                "language": "en",
                "pageSize": 20,
                "apiKey":   NEWS_API_KEY,
            },
            timeout=10,
        )
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        results  = []
        for a in articles[:10]:
            headline  = a.get("title", "")
            source    = a.get("source", {}).get("name", "")
            sentiment = _classify_headline_sentiment(headline)
            results.append({"headline": headline, "source": source, "sentiment": sentiment})
        return results
    except Exception as e:
        logger.warning(f"[{ErrorCode.E1007}] NewsAPI market news failed: {e}")
        return []


def fetch_company_news(ticker: str) -> list[dict]:
    """
    Fetch company-specific news from Finnhub.
    Returns list of {headline, source, url, sentiment}.
    """
    try:
        today      = date.today().isoformat()
        week_ago   = date.fromordinal(date.today().toordinal() - 7).isoformat()
        resp = requests.get(
            f"{FINNHUB_BASE}/company-news",
            params={"symbol": ticker, "from": week_ago, "to": today, "token": FINNHUB_API_KEY},
            timeout=10,
        )
        resp.raise_for_status()
        articles = resp.json()
        if not isinstance(articles, list):
            return []
        results = []
        for a in articles[:3]:
            headline  = a.get("headline", "")
            source    = a.get("source", "")
            sentiment = _classify_headline_sentiment(headline)
            results.append({
                "headline": headline,
                "source":   source,
                "url":      a.get("url", ""),
                "sentiment": sentiment,
            })
        return results
    except Exception as e:
        logger.warning(f"[{ErrorCode.E1007}] Company news failed for {ticker}: {e}")
        return []


def fetch_sector_news(sectors: list[str]) -> dict[str, dict]:
    """
    Fetch Finnhub general news and filter by sector name keyword.
    Returns {sector: {headlines: [...], adverse_headline: bool}}.
    """
    sector_results = {}
    try:
        resp = requests.get(
            f"{FINNHUB_BASE}/news",
            params={"category": "general", "token": FINNHUB_API_KEY},
            timeout=10,
        )
        resp.raise_for_status()
        articles = resp.json()
        if not isinstance(articles, list):
            return {s: {"headlines": [], "adverse_headline": False} for s in sectors}

        # Map sector ETF codes to readable names for keyword matching
        sector_names = {
            "XLK": ["technology", "tech", "semiconductor", "software", "AI", "chip"],
            "XLE": ["energy", "oil", "gas", "petroleum", "crude"],
            "XLF": ["financial", "bank", "finance", "insurance", "credit"],
            "XLV": ["health", "pharma", "biotech", "medical", "drug"],
            "XLU": ["utility", "utilities", "electric", "power grid"],
            "XLI": ["industrial", "manufacturing", "aerospace", "defense"],
            "XLB": ["material", "chemical", "mining", "metal", "commodity"],
            "XLP": ["consumer staple", "retail", "food", "beverage", "household"],
        }

        for sector in sectors:
            keywords     = sector_names.get(sector, [sector.lower()])
            relevant     = []
            adverse_flag = False
            for a in articles:
                headline = (a.get("headline", "") + " " + a.get("summary", "")).lower()
                if any(kw.lower() in headline for kw in keywords):
                    relevant.append({
                        "headline":  a.get("headline", ""),
                        "source":    a.get("source", ""),
                        "sentiment": _classify_headline_sentiment(a.get("headline", "")),
                    })
                    if any(bad.lower() in headline for bad in ADVERSE_KEYWORDS):
                        adverse_flag = True
            sector_results[sector] = {
                "headlines":       relevant[:3],
                "adverse_headline": adverse_flag,
            }
    except Exception as e:
        logger.warning(f"[{ErrorCode.E1007}] Sector news fetch failed: {e}")
        for s in sectors:
            sector_results.setdefault(s, {"headlines": [], "adverse_headline": False})

    return sector_results


def classify_market_sentiment(fear_greed: dict, put_call_ratio: float,
                               news_signals: list[dict]) -> dict:
    """
    Combine F&G, P/C ratio, and news to produce overall BULLISH/BEARISH/NEUTRAL.
    Returns {market_sentiment, structure_bias, warning_flags}.
    """
    score   = fear_greed.get("score", 50)
    warning_flags = []

    # Count bullish/bearish news headlines
    bull_count  = sum(1 for n in news_signals if n.get("sentiment") == "BULLISH")
    bear_count  = sum(1 for n in news_signals if n.get("sentiment") == "BEARISH")
    news_signal = "BULLISH" if bull_count > bear_count else ("BEARISH" if bear_count > bull_count else "NEUTRAL")

    # Composite scoring
    bull_score = 0
    bear_score = 0

    if score >= 60:    bull_score += 1
    elif score <= 40:  bear_score += 1

    if put_call_ratio > 1.2:    bull_score += 1   # contrarian: high P/C → put buying exhausted
    elif put_call_ratio < 0.7:  bear_score += 1   # complacency → lean cautious

    if news_signal == "BULLISH":   bull_score += 1
    elif news_signal == "BEARISH": bear_score += 1

    if bull_score > bear_score:
        sentiment = "BULLISH"
        structure_bias = "NEUTRAL_TO_DEBIT" if score < 76 else "CREDIT"
    elif bear_score > bull_score:
        sentiment = "BEARISH"
        structure_bias = "CREDIT"
    else:
        sentiment = "NEUTRAL"
        structure_bias = "NEUTRAL"

    # Warning flags
    if score >= 76:
        warning_flags.append("extreme_greed_caution_on_debit")
    if score <= 25:
        warning_flags.append("extreme_fear_credit_on_quality")
    if put_call_ratio > 1.2:
        warning_flags.append("high_put_call_contrarian_bullish")
    if put_call_ratio < 0.7:
        warning_flags.append("low_put_call_lean_credit")

    return {
        "market_sentiment": sentiment,
        "structure_bias":   structure_bias,
        "warning_flags":    warning_flags,
        "news_signal":      news_signal,
    }


def _classify_headline_sentiment(headline: str) -> str:
    """Simple keyword-based sentiment for headlines."""
    if not headline:
        return "NEUTRAL"
    h = headline.lower()
    bullish_words = {"surge", "beat", "rally", "strong", "upgrade", "buy", "growth",
                     "record", "gain", "rise", "boost", "outperform", "bull"}
    bearish_words = {"fall", "drop", "miss", "weak", "downgrade", "sell", "loss",
                     "decline", "crash", "fear", "concern", "risk", "cut", "layoff",
                     "tariff", "sanction", "lawsuit", "recall"}
    bull = sum(1 for w in bullish_words if w in h)
    bear = sum(1 for w in bearish_words if w in h)
    if bull > bear:
        return "BULLISH"
    if bear > bull:
        return "BEARISH"
    return "NEUTRAL"
