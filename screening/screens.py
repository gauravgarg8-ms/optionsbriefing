"""
Screens 1–4: pool builders for candidate selection.
Step 0 (prefilter) is handled by universe_manager.prefilter_universe().
Screens operate on the lightweight quant_signals dict.
"""
from loguru import logger
from config import (
    MAX_CANDIDATES_PER_POOL,
    SCREEN_HIGH_IV_MIN_PRICE, SCREEN_HIGH_IV_MIN_OPTIONS_VOL,
    SCREEN_EARNINGS_MIN_OPTIONS_VOL, SCREEN_EARNINGS_MIN_IMPLIED_MOVE,
    SCREEN_EARNINGS_DAYS_AHEAD,
    IV_RANK_CREDIT_MIN, IV_RANK_DEBIT_MAX,
)


def screen_high_iv(candidates: list[dict]) -> list[dict]:
    """
    Screen 1: High IV Pool.
    Criteria: IV Rank > 50%, price > $15, avg options vol > 500/day.
    Returns top MAX_CANDIDATES_PER_POOL['high_iv'] by IV Rank descending.
    """
    max_count = MAX_CANDIDATES_PER_POOL["high_iv"]
    passed = [
        c for c in candidates
        if c.get("iv_rank", 0) > IV_RANK_CREDIT_MIN
        and c.get("price", 0) > SCREEN_HIGH_IV_MIN_PRICE
        and c.get("avg_options_vol", 0) > SCREEN_HIGH_IV_MIN_OPTIONS_VOL
    ]
    passed.sort(key=lambda x: x.get("iv_rank", 0), reverse=True)
    result = passed[:max_count]
    logger.info(f"Screen 1 (High IV): {len(passed)} → {len(result)} candidates")
    return [_tag_pool(c, "high_iv") for c in result]


def screen_earnings(candidates: list[dict], earnings_calendar: list[dict]) -> list[dict]:
    """
    Screen 2: Earnings Pool.
    Criteria: earnings within 7 days, implied move > 5%, avg options vol > 1000/day.
    """
    max_count = MAX_CANDIDATES_PER_POOL["earnings"]
    earnings_tickers = {
        e["ticker"]: e for e in earnings_calendar
        if e.get("days_away", 99) <= SCREEN_EARNINGS_DAYS_AHEAD
    }
    passed = []
    for c in candidates:
        ticker = c.get("ticker", "")
        if ticker not in earnings_tickers:
            continue
        if c.get("avg_options_vol", 0) <= SCREEN_EARNINGS_MIN_OPTIONS_VOL:
            continue
        implied = c.get("implied_move_pct", 0) or 0
        if implied < SCREEN_EARNINGS_MIN_IMPLIED_MOVE:
            continue
        passed.append({**c, "earnings_days_away": earnings_tickers[ticker]["days_away"],
                       "earnings_date": earnings_tickers[ticker]["date"]})

    passed.sort(key=lambda x: x.get("avg_options_vol", 0), reverse=True)
    result = passed[:max_count]
    logger.info(f"Screen 2 (Earnings): {len(passed)} → {len(result)} candidates")
    return [_tag_pool(c, "earnings") for c in result]


def screen_low_iv_trend(candidates: list[dict], market_env: dict) -> list[dict]:
    """
    Screen 3: Low IV / Trend Pool.
    Criteria: IV Rank < 30%, above 50d+200d MA, positive RS vs SPY, in leading sectors.
    """
    max_count      = MAX_CANDIDATES_PER_POOL["low_iv_trend"]
    leading_sectors = market_env.get("leading_sectors", [])
    passed = [
        c for c in candidates
        if c.get("iv_rank", 100) < IV_RANK_DEBIT_MAX
        and c.get("above_50ma", False)
        and c.get("above_200ma", False)
        and c.get("rs_20d", 0) > 0
        and c.get("sector", "") in leading_sectors
    ]
    passed.sort(key=lambda x: x.get("rs_20d", 0), reverse=True)
    result = passed[:max_count]
    logger.info(f"Screen 3 (Low IV/Trend): {len(passed)} → {len(result)} candidates")
    return [_tag_pool(c, "low_iv_trend") for c in result]


def screen_bearish(candidates: list[dict], market_env: dict) -> list[dict]:
    """
    Screen 4: Bearish Pool.
    Criteria: below 50d MA, in lagging sectors, negative RS vs SPY.
    """
    max_count      = MAX_CANDIDATES_PER_POOL["bearish"]
    lagging_sectors = market_env.get("lagging_sectors", [])
    passed = [
        c for c in candidates
        if not c.get("above_50ma", True)
        and c.get("sector", "") in lagging_sectors
        and c.get("rs_20d", 0) < 0
    ]
    passed.sort(key=lambda x: x.get("rs_20d", 0))   # most negative RS first
    result = passed[:max_count]
    logger.info(f"Screen 4 (Bearish): {len(passed)} → {len(result)} candidates")
    return [_tag_pool(c, "bearish") for c in result]


def merge_pools(pools: list[list[dict]]) -> list[dict]:
    """
    Merge screen pools, deduplicating by ticker.
    First occurrence (highest priority pool) wins if ticker appears in multiple pools.
    """
    seen   = set()
    merged = []
    for pool in pools:
        for c in pool:
            ticker = c.get("ticker", "")
            if ticker and ticker not in seen:
                seen.add(ticker)
                merged.append(c)
    return merged


def _tag_pool(candidate: dict, pool_name: str) -> dict:
    return {**candidate, "screen_pool": pool_name}
