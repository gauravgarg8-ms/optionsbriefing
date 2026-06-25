"""
Phase 1 orchestrator: assembles all market data from sub-modules.

Two-phase fetch pattern:
  Phase 1 (universe-wide): market prices, macro, sentiment, unusual activity
  deep_fetch(tickers):     options chain + earnings per ticker (called by screening_engine)
                           includes 1-second sleep between tickers (Finnhub rate limit guard)
"""

import json
import time
from datetime import date
from pathlib import Path

from loguru import logger

from data.universe_manager import build_universe, prefilter_universe
from data.market_data import fetch_vix_spy, fetch_sector_rotation, fetch_premarket_prices, fetch_market_caps
from data.macro_data import fetch_macro_calendar, fetch_rates
from data.sentiment_data import (
    fetch_fear_greed, fetch_put_call_ratio, fetch_market_news,
    fetch_company_news, fetch_sector_news, classify_market_sentiment,
)
from data.unusual_activity import fetch_unusual_activity
from data.options_data import fetch_options_chain, fetch_0dte_chain, filter_liquid_strikes
from data.earnings_data import (
    fetch_earnings_calendar, fetch_earnings_history, fetch_analyst_ratings,
    fetch_insider_transactions, compute_implied_move, compute_hist_avg_move,
    classify_earnings_candidate,
)
from config import DEEP_FETCH_SLEEP_SECS, PINNED_TICKERS

_FORCE_0DTE_SYMBOLS = {entry["symbol"] for entry in PINNED_TICKERS if entry.get("force_dte_0")}
from errors import ErrorCode

OUTPUT_RAW = Path(__file__).parent.parent / "output" / "raw_market_data.json"


def run_data_collection() -> dict:
    """
    Phase 1: collect all universe-wide data.
    Returns the raw_market_data dict and writes it to disk.
    """
    today = date.today().isoformat()
    logger.info(f"[Phase 1] Starting data collection for {today}")

    # ── Universe ─────────────────────────────────────────────────────────────
    universe_raw = build_universe()

    # Force-add pinned tickers (e.g., SPY) if not already in the universe
    existing_symbols = {r.get("symbol", r.get("ticker", "")) for r in universe_raw}
    for entry in PINNED_TICKERS:
        if entry["symbol"] not in existing_symbols:
            universe_raw.append(entry)

    price_data   = {}
    if universe_raw:
        tickers      = [r.get("symbol", r.get("ticker", "")) for r in universe_raw]
        valid_tickers = [t for t in tickers if t]
        raw_prices   = fetch_premarket_prices(valid_tickers)
        market_caps  = fetch_market_caps(valid_tickers)
        for entry in universe_raw:
            t  = entry.get("symbol", entry.get("ticker", ""))
            mc = market_caps.get(t) or entry.get("marketCap", entry.get("market_cap", 0)) or 0
            price_data[t] = {"price": raw_prices.get(t, 0.0), "market_cap": float(mc)}
    universe = prefilter_universe(universe_raw, price_data)
    logger.info(f"[Phase 1] Universe: {len(universe_raw)} → {len(universe)} tickers after pre-filter")

    # ── Market environment ───────────────────────────────────────────────────
    vix_spy      = fetch_vix_spy()
    sector_rot   = fetch_sector_rotation()
    rates        = fetch_rates()
    macro_events = fetch_macro_calendar()
    fear_greed   = fetch_fear_greed()
    put_call     = fetch_put_call_ratio()
    market_news  = fetch_market_news()
    unusual      = fetch_unusual_activity()

    leading_sectors  = sector_rot.get("leading_sectors", [])
    lagging_sectors  = sector_rot.get("lagging_sectors", [])
    sector_news      = fetch_sector_news(leading_sectors[:2])
    sentiment_result = classify_market_sentiment(fear_greed, put_call, market_news)

    market_env = {
        **vix_spy,
        **rates,
        "fear_greed_score":  fear_greed.get("score", 50),
        "fear_greed_label":  fear_greed.get("label", "Neutral"),
        "put_call_ratio":    put_call,
        "market_sentiment":  sentiment_result["market_sentiment"],
        "structure_bias":    sentiment_result["structure_bias"],
        "sentiment_warning_flags": sentiment_result.get("warning_flags", []),
        "news_signal":       sentiment_result.get("news_signal", "NEUTRAL"),
        "leading_sectors":   leading_sectors,
        "lagging_sectors":   lagging_sectors,
        "sector_returns":    sector_rot.get("sector_returns", {}),
        "sector_news":       sector_news,
        "universe_size":     len(universe),
    }

    # ── Earnings calendar (universe-wide) ────────────────────────────────────
    earnings_calendar = fetch_earnings_calendar(days_ahead=14)

    result = {
        "date":              today,
        "universe":          universe,
        "market_environment": market_env,
        "macro_events":      macro_events,
        "earnings_calendar": earnings_calendar,
        "unusual_activity":  unusual,
    }

    _save_json(result, OUTPUT_RAW)
    logger.info(f"[Phase 1] Complete — raw_market_data.json written")
    return result


def deep_fetch(tickers: list[str], earnings_calendar: list[dict]) -> dict[str, dict]:
    """
    Phase 2C: per-ticker deep fetch for top ~15 candidates.
    Includes 1-second sleep between tickers (Finnhub 60 req/min guard — Decision #4).
    Returns {ticker: {chain, earnings, analyst_ratings, insider, company_news}}.
    """
    results = {}
    logger.info(f"[Phase 2C] Deep fetch for {len(tickers)} tickers")

    for i, ticker in enumerate(tickers):
        ticker_data = {}
        try:
            # Options chain — 0DTE path for pinned tickers (e.g. SPY), standard otherwise
            if ticker in _FORCE_0DTE_SYMBOLS:
                chain_raw = fetch_0dte_chain(ticker)
            else:
                chain_raw = fetch_options_chain(ticker)
            raw       = chain_raw.get("options", [])
            liquid    = filter_liquid_strikes(raw)
            ticker_data["chain"] = {
                "expiry":      chain_raw.get("expiry"),
                "options":     liquid,
                "raw_options": raw,      # pre-filter; impliedVolatility valid pre-market
                "source":      chain_raw.get("source"),
            }
            # Earnings history + analyst ratings + insider
            ticker_data["earnings_history"]   = fetch_earnings_history(ticker)
            ticker_data["analyst_ratings"]    = fetch_analyst_ratings(ticker)
            ticker_data["insider_transactions"] = fetch_insider_transactions(ticker)
            ticker_data["company_news"]        = fetch_company_news(ticker)
            ticker_data["earnings_info"]       = classify_earnings_candidate(ticker, earnings_calendar)
        except Exception as e:
            logger.warning(f"[{ErrorCode.E3002}] Deep fetch partial failure for {ticker}: {e}")

        results[ticker] = ticker_data

        if i < len(tickers) - 1:
            time.sleep(DEEP_FETCH_SLEEP_SECS)   # Finnhub rate limit guard

    logger.info(f"[Phase 2C] Deep fetch complete: {len(results)}/{len(tickers)} tickers")
    return results


def _save_json(data: dict, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str))
    except Exception as e:
        logger.error(f"[{ErrorCode.E4004}] Failed to write {path}: {e}")
