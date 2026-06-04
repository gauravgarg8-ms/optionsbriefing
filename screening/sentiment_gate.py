"""
Screen 5 — Sentiment Gate.
Applies disqualifiers, amplifiers, and warning flags per Section 10 of spec.
"""
from loguru import logger


def apply_sentiment_gate(candidate: dict, market_env: dict) -> tuple[dict, bool]:
    """
    Apply sentiment gate to a single candidate.
    Returns (enriched_candidate, is_disqualified).
    Modifies candidate in-place with sentiment_flags, warning_flags, covered_call_opportunity.
    """
    bonus_flags   = list(candidate.get("sentiment_flags",  []))
    warning_flags = list(candidate.get("warning_flags",   []))

    direction  = candidate.get("direction", "Bullish")
    is_bullish = direction == "Bullish"
    is_bearish = direction == "Bearish"
    is_debit   = candidate.get("structure", "").lower() in {
        "bull_call_spread", "bear_put_spread", "long_straddle", "long_strangle"
    }

    iv_rank         = candidate.get("iv_rank", 0)
    fg_score        = market_env.get("fear_greed_score", 50)
    pc_ratio        = market_env.get("put_call_ratio", 0.9)
    mkt_sentiment   = market_env.get("market_sentiment", "NEUTRAL")
    leading_sectors = market_env.get("leading_sectors", [])
    lagging_sectors = market_env.get("lagging_sectors", [])
    ticker_sector   = candidate.get("sector", "")

    # ── DISQUALIFIERS ────────────────────────────────────────────────────────
    # Analyst downgrade within 3 days + bullish setup
    if candidate.get("recent_downgrade_days", 99) <= 3 and is_bullish:
        logger.info(f"[SentimentGate] {candidate.get('ticker')} DISQUALIFIED: recent downgrade + bullish")
        return candidate, True

    # Major negative news today + bearish news signal
    if candidate.get("news_signal") == "BEARISH" and candidate.get("major_negative_event"):
        logger.info(f"[SentimentGate] {candidate.get('ticker')} DISQUALIFIED: major negative event")
        return candidate, True

    # Sector adverse headline + bullish setup
    sector_news = market_env.get("sector_news", {})
    sector_data = sector_news.get(ticker_sector, {})
    if sector_data.get("adverse_headline") and is_bullish:
        logger.info(f"[SentimentGate] {candidate.get('ticker')} DISQUALIFIED: sector adverse headline + bullish")
        return candidate, True

    # ── AMPLIFIERS (bonus flags → +pts in scorer) ────────────────────────────
    if candidate.get("recent_upgrade_days", 99) <= 7:
        bonus_flags.append("analyst_upgrade_7d")

    if candidate.get("unusual_options_aligned"):
        bonus_flags.append("unusual_options_aligned")

    if mkt_sentiment == "BULLISH" and ticker_sector in leading_sectors and is_bullish:
        bonus_flags.append("bullish_sentiment_leading_sector")

    if mkt_sentiment == "BEARISH" and ticker_sector in lagging_sectors and is_bearish:
        bonus_flags.append("bearish_sentiment_lagging_sector")

    if fg_score <= 25 and iv_rank >= 60 and is_bullish:
        bonus_flags.append("extreme_fear_panic_premium")

    if candidate.get("positive_company_news"):
        bonus_flags.append("positive_company_news")

    if candidate.get("insider_buying"):
        bonus_flags.append("insider_buying")

    # ── WARNING FLAGS (keep, flag for half-size) ─────────────────────────────
    if fg_score >= 76 and is_bullish and is_debit:
        warning_flags.append("extreme_greed_debit_half_size")

    if pc_ratio > 1.2 and is_bullish and is_debit:
        warning_flags.append("high_put_call_debit_half_size")

    # ── COVERED CALL OPPORTUNITY FLAG (Decision #3) ──────────────────────────
    # Not an automated strategy — Claude flags it in briefing narrative
    covered_call_opp = is_bullish and iv_rank > 40

    enriched = {
        **candidate,
        "sentiment_flags":         bonus_flags,
        "warning_flags":           warning_flags,
        "covered_call_opportunity": covered_call_opp,
    }
    return enriched, False


def apply_gate_to_pool(candidates: list[dict], market_env: dict) -> list[dict]:
    """
    Apply sentiment gate to all candidates in a pool.
    Returns only non-disqualified candidates with enriched flags.
    """
    results     = []
    disqualified = 0
    for c in candidates:
        enriched, is_disq = apply_sentiment_gate(c, market_env)
        if is_disq:
            disqualified += 1
        else:
            results.append(enriched)
    if disqualified:
        logger.info(f"[SentimentGate] Removed {disqualified} disqualified candidates")
    return results
