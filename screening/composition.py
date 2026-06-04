"""
Final candidate selection + portfolio exposure check.
"""
from loguru import logger
from config import (
    SCORE_HIGH_CONFIDENCE, MIN_CANDIDATES_FOR_TRADE,
    MAX_SECTOR_POSITIONS, MAX_EARNINGS_PLAYS, MAX_NET_DELTA, MAX_SECTOR_PCT,
)
from errors import ErrorCode


def select_final_10(scored_candidates: list[dict]) -> list[dict]:
    """
    Select up to 10 candidates from scored pool (already sorted by score desc).
    All candidates must have score >= SCORE_FLOOR (enforced by scorer.filter_by_score).
    """
    return scored_candidates[:10]


def check_reduced_opportunity(candidates: list[dict]) -> bool:
    """
    Returns True if fewer than MIN_CANDIDATES_FOR_TRADE score >= SCORE_HIGH_CONFIDENCE.
    Triggers REDUCED OPPORTUNITY DAY — do NOT force 10 setups.
    """
    high_score_count = sum(1 for c in candidates if c.get("score", 0) >= SCORE_HIGH_CONFIDENCE)
    reduced = high_score_count < MIN_CANDIDATES_FOR_TRADE
    if reduced:
        logger.warning(
            f"[Composition] REDUCED OPPORTUNITY DAY: only {high_score_count} candidates "
            f"score ≥{SCORE_HIGH_CONFIDENCE} (need {MIN_CANDIDATES_FOR_TRADE})"
        )
    return reduced


def enforce_earnings_limit(candidates: list[dict]) -> list[dict]:
    """
    If more than MAX_EARNINGS_PLAYS candidates have earnings < 7 days,
    trim to the top MAX_EARNINGS_PLAYS by score.
    """
    earnings = [c for c in candidates if (c.get("earnings_days_away") or 99) <= 7]
    non_earn = [c for c in candidates if (c.get("earnings_days_away") or 99) > 7]

    if len(earnings) > MAX_EARNINGS_PLAYS:
        logger.warning(
            f"[Composition] {len(earnings)} earnings plays → trimming to {MAX_EARNINGS_PLAYS}"
        )
        earnings = sorted(earnings, key=lambda x: x.get("score", 0), reverse=True)[:MAX_EARNINGS_PLAYS]

    result = earnings + non_earn
    result.sort(key=lambda x: x.get("score", 0), reverse=True)
    return result


def compute_portfolio_exposure(candidates: list[dict]) -> dict:
    """
    Compute net portfolio delta, vega, sector concentration, earnings plays.
    Returns portfolio_check dict with warnings list.
    """
    warnings        = []
    sector_counts   = {}
    earnings_count  = 0
    net_delta       = 0.0
    net_vega        = 0.0

    for c in candidates:
        # Sector counts
        sector = c.get("sector", "UNKNOWN")
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

        # Earnings plays count
        if (c.get("earnings_days_away") or 99) <= 7:
            earnings_count += 1

        # Delta and vega aggregation
        bs     = c.get("bs", c.get("spread_pricing", {}))
        delta  = float(bs.get("delta", 0) or 0)
        vega   = float(bs.get("vega", 0)  or 0)
        net_delta += delta
        net_vega  += vega

    # Concentrated sectors
    concentrated = [s for s, count in sector_counts.items() if count > MAX_SECTOR_POSITIONS]

    # Build warnings
    if concentrated:
        warnings.append(f"Sector concentration: {concentrated}")
    if abs(net_delta) > MAX_NET_DELTA:
        warnings.append(f"Net delta {net_delta:+.1f} exceeds ±{MAX_NET_DELTA} limit")
    if earnings_count > MAX_EARNINGS_PLAYS:
        warnings.append(f"{earnings_count} earnings plays exceeds limit of {MAX_EARNINGS_PLAYS}")

    # Vega direction
    vega_direction = "Net Short Vega" if net_vega < 0 else ("Net Long Vega" if net_vega > 0 else "Vega Neutral")

    return {
        "net_portfolio_delta":  round(net_delta, 2),
        "net_portfolio_vega":   round(net_vega, 2),
        "vega_direction":       vega_direction,
        "sector_counts":        sector_counts,
        "concentrated_sectors": concentrated,
        "earnings_plays_count": earnings_count,
        "portfolio_warnings":   warnings,
    }


def compose_final_output(scored_candidates: list[dict], market_env: dict) -> dict:
    """
    Full composition pipeline:
    1. Enforce earnings limit
    2. Select top 10
    3. Compute portfolio exposure
    4. Check reduced opportunity flag
    Returns final composition dict.
    """
    if not scored_candidates:
        logger.error(f"[{ErrorCode.E3001}] No candidates passed screening — emitting NO-TRADE-DAY")
        return {
            "candidates":           [],
            "no_trade_day":         True,
            "reduced_opportunity_day": False,
            "portfolio_check":      {},
        }

    after_earnings  = enforce_earnings_limit(scored_candidates)
    final_10        = select_final_10(after_earnings)
    portfolio_check = compute_portfolio_exposure(final_10)
    reduced         = check_reduced_opportunity(scored_candidates)

    return {
        "candidates":              final_10,
        "no_trade_day":            False,
        "reduced_opportunity_day": reduced,
        "portfolio_check":         portfolio_check,
    }
