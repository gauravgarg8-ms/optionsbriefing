"""
Phase 3 orchestrator: 2A → 2B → 2C pipeline.

2A: Lightweight screening of all ~200 tickers (no API calls).
2B: Score top candidates with 0–100 rubric using lightweight data.
2C: Deep fetch for top 15 → options chain + earnings → full quant + final scoring.
"""
import json
from pathlib import Path

from loguru import logger

from screening.screens import screen_high_iv, screen_earnings, screen_low_iv_trend, screen_bearish, merge_pools
from screening.sentiment_gate import apply_gate_to_pool
from screening.scorer import filter_by_score, compute_confidence
from screening.composition import compose_final_output
from quant.quant_engine import run_deep_quant
from config import PINNED_TICKERS
from errors import ErrorCode

OUTPUT_SCREENED = Path(__file__).parent.parent / "output" / "screened_candidates.json"


def run_screening(quant_data: dict, data_engine=None) -> dict:
    """
    Full 2A → 2B → 2C pipeline.
    quant_data: output of run_quant_calculations() — contains quant_signals + market_environment.
    data_engine: optional module override for testing (defaults to data.data_engine).
    """
    market_env       = quant_data.get("market_environment", {})
    quant_signals    = quant_data.get("quant_signals", {})
    earnings_calendar = quant_data.get("earnings_calendar", [])
    tbill_rate       = quant_data.get("tbill_rate", 0.051)

    candidates = list(quant_signals.values())
    logger.info(f"[Phase 3 — 2A] Screening {len(candidates)} tickers")

    # ── Step 2A: Run 4 screens ───────────────────────────────────────────────
    pool_high_iv   = screen_high_iv(candidates)
    pool_earnings  = screen_earnings(candidates, earnings_calendar)
    pool_low_trend = screen_low_iv_trend(candidates, market_env)
    pool_bearish   = screen_bearish(candidates, market_env)

    all_pools = merge_pools([pool_high_iv, pool_earnings, pool_low_trend, pool_bearish])

    if not all_pools:
        logger.error(f"[{ErrorCode.E3001}] All screening pools empty")
        return _empty_output()

    # ── Step 2A — Apply sentiment gate ──────────────────────────────────────
    after_gate = apply_gate_to_pool(all_pools, market_env)

    # ── Step 2B: Initial scoring with lightweight data ───────────────────────
    logger.info(f"[Phase 3 — 2B] Scoring {len(after_gate)} candidates")
    pre_scored = filter_by_score(after_gate, market_env)
    top_15     = pre_scored[:15]

    if not top_15:
        logger.error(f"[{ErrorCode.E3001}] No candidates passed score floor in 2B")
        return _empty_output()

    # ── Force-inject pinned tickers (e.g. SPY) into deep-fetch pool ──────────
    for entry in PINNED_TICKERS:
        sym = entry["symbol"]
        if sym in quant_signals and not any(c.get("ticker") == sym for c in top_15):
            pinned_c = {**quant_signals[sym], "ticker": sym, "screen_pool": "pinned"}
            top_15.insert(0, pinned_c)
            logger.info(f"[Phase 3 — 2C] Pinned ticker {sym} force-injected into deep-fetch pool")

    # ── Step 2C: Deep fetch + full quant for top 15 ──────────────────────────
    logger.info(f"[Phase 3 — 2C] Deep fetch for {len(top_15)} candidates")
    engine = data_engine
    if engine is None:
        from data import data_engine as _de
        engine = _de

    tickers    = [c["ticker"] for c in top_15]
    deep_data  = engine.deep_fetch(tickers, earnings_calendar)

    # Enrich with deep quant + re-score
    enriched   = []
    for c in top_15:
        ticker  = c["ticker"]
        td      = deep_data.get(ticker, {})
        chain   = td.get("chain", {})
        # Merge earnings info
        earn_info = td.get("earnings_info", {})
        c.update(earn_info)
        # Run deep B-S calculations — always call so raw_options fallback works pre-market
        # (run_deep_quant returns lightweight unchanged if expiry or raw_options are missing)
        if chain.get("expiry") and (chain.get("options") or chain.get("raw_options")):
            c = run_deep_quant(ticker, c, chain, tbill_rate=tbill_rate)
        # Attach analyst/news/insider signals for sentiment gate pass 2
        c["recent_downgrade_days"] = _check_downgrade(td.get("analyst_ratings", []))
        c["recent_upgrade_days"]   = _check_upgrade(td.get("analyst_ratings", []))
        c["positive_company_news"] = _has_positive_news(td.get("company_news", []))
        c["insider_buying"]        = _has_insider_buy(td.get("insider_transactions", []))
        c["news_signal"]           = _top_news_sentiment(td.get("company_news", []))
        c["_leading_sectors"]      = market_env.get("leading_sectors", [])
        enriched.append(c)

    # Drop candidates flagged as degenerate spreads (zero-width, zero-value) by run_deep_quant
    valid_enriched = [c for c in enriched if not c.get("invalid_spread")]
    skipped = len(enriched) - len(valid_enriched)
    if skipped:
        logger.warning(f"[Phase 3 — 2C] Dropped {skipped} candidate(s) with degenerate spreads")
    enriched = valid_enriched

    # Re-apply sentiment gate with full data
    after_gate2 = apply_gate_to_pool(enriched, market_env)

    # Apply liquidity gate (bid/ask > 10% OR OI < 500 already handled in deep quant)
    # Re-score with full options data
    final_scored = filter_by_score(after_gate2, market_env)

    # Add confidence scores
    for c in final_scored:
        conf = compute_confidence(c)
        c["confidence_score"]  = conf["score_string"]
        c["confidence_label"]  = conf["label"]

    if not final_scored:
        logger.error(f"[{ErrorCode.E3003}] Liquidity gate eliminated all candidates")
        return _empty_output()

    # ── Compose final output ─────────────────────────────────────────────────
    output = compose_final_output(final_scored, market_env)
    output["market_environment"] = market_env
    output["earnings_calendar"]  = earnings_calendar
    output["tbill_rate"]         = tbill_rate

    _save_json(output)
    logger.info(
        f"[Phase 3] Complete — {len(output['candidates'])} candidates, "
        f"reduced_opportunity={output['reduced_opportunity_day']}"
    )
    return output


# ── Signal helpers ────────────────────────────────────────────────────────────

def _check_downgrade(ratings: list[dict]) -> int:
    """Returns days since most recent downgrade, or 99 if none."""
    from datetime import date, datetime
    today = date.today()
    for r in ratings:
        period = r.get("period", "")
        try:
            d = datetime.strptime(period, "%Y-%m-%d").date()
            if r.get("sell", 0) > 0 or r.get("strong_sell", 0) > 0:
                return (today - d).days
        except Exception:
            continue
    return 99


def _check_upgrade(ratings: list[dict]) -> int:
    """Returns days since most recent upgrade (strong_buy or buy increase), or 99."""
    from datetime import date, datetime
    today = date.today()
    for r in ratings:
        period = r.get("period", "")
        try:
            d = datetime.strptime(period, "%Y-%m-%d").date()
            if r.get("strong_buy", 0) > 0 or r.get("buy", 0) > 0:
                return (today - d).days
        except Exception:
            continue
    return 99


def _has_positive_news(news: list[dict]) -> bool:
    return any(n.get("sentiment") == "BULLISH" for n in news)


def _has_insider_buy(transactions: list[dict]) -> bool:
    return any(
        "buy" in (t.get("transaction_type") or "").lower()
        for t in transactions
    )


def _top_news_sentiment(news: list[dict]) -> str:
    if not news:
        return "NEUTRAL"
    bull = sum(1 for n in news if n.get("sentiment") == "BULLISH")
    bear = sum(1 for n in news if n.get("sentiment") == "BEARISH")
    if bull > bear:    return "BULLISH"
    if bear > bull:    return "BEARISH"
    return "NEUTRAL"


def _empty_output() -> dict:
    return {
        "candidates":              [],
        "no_trade_day":            True,
        "reduced_opportunity_day": False,
        "portfolio_check":         {},
        "market_environment":      {},
        "earnings_calendar":       [],
        "tbill_rate":              0.051,
    }


def _save_json(data: dict) -> None:
    try:
        OUTPUT_SCREENED.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_SCREENED.write_text(json.dumps(data, indent=2, default=str))
    except Exception as e:
        logger.error(f"[{ErrorCode.E4004}] Failed to write screened_candidates.json: {e}")
