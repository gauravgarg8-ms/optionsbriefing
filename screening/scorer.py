"""
0–100 scoring rubric + 7-signal confidence score.
Section 9 of the architecture spec.
"""
import math

from loguru import logger
from config import (
    SCORE_FLOOR, POP_FLOOR, LIQUIDITY_MAX_BID_ASK_PCT, LIQUIDITY_MIN_OI,
    IV_RANK_CREDIT_MIN, IV_RANK_DEBIT_MAX, IV_RV_MIDDLE_ZONE_CREDIT_THRESHOLD,
)
from errors import ErrorCode


def compute_score(candidate: dict, market_env: dict) -> int:
    """
    0–100 scoring rubric.
    Returns integer score; 0 if golden rules disqualify the candidate.
    """
    structure  = candidate.get("structure", "")
    is_credit  = structure in {"bull_put_spread", "bear_call_spread", "iron_condor",
                               "cash_secured_put", "earnings_credit_spread"}
    is_debit   = structure in {"bull_call_spread", "bear_put_spread",
                               "long_straddle", "long_strangle"}
    is_earnings = structure == "earnings_credit_spread"
    direction   = candidate.get("direction", "Bullish")
    is_bearish  = direction == "Bearish"

    iv_rank  = candidate.get("iv_rank", 50)
    iv_rv    = candidate.get("iv_rv_ratio", 1.0)
    hv20     = candidate.get("hv20", 0)
    bid_ask  = candidate.get("bid_ask_pct", 0)
    oi       = candidate.get("oi_target", candidate.get("open_interest", 0))
    avg_vol  = candidate.get("avg_options_vol", 0)
    pop      = candidate.get("bs", {}).get("pop", candidate.get("pop", 0))
    ev       = candidate.get("bs", {}).get("ev", candidate.get("ev", 0))
    score    = 0

    # ── Golden rules — disqualify before scoring ─────────────────────────────
    if pop > 0 and pop < POP_FLOOR:
        logger.debug(f"[Scorer] {candidate.get('ticker')} excluded: PoP={pop:.2f} < {POP_FLOOR}")
        return 0
    if ev != 0 and ev <= 0:
        logger.debug(f"[Scorer] {candidate.get('ticker')} excluded: EV={ev:.2f} ≤ 0")
        return 0
    if bid_ask > LIQUIDITY_MAX_BID_ASK_PCT or (oi > 0 and oi < LIQUIDITY_MIN_OI):
        logger.debug(f"[Scorer] {candidate.get('ticker')} excluded: illiquid bid/ask={bid_ask:.1%} OI={oi}")
        return 0

    # ── 1. IV Rank Alignment (20 pts) ────────────────────────────────────────
    if is_earnings:
        score += 20 if iv_rank >= 60 else (10 if iv_rank >= 40 else 5)
    elif is_credit:
        if iv_rank >= 70:   score += 20
        elif iv_rank >= 60: score += 15
        elif iv_rank >= 50: score += 10
        else:               score += 0
    elif is_debit:
        if iv_rank < 20:    score += 20
        elif iv_rank < 30:  score += 13
        elif iv_rank < 40:  score += 7
        else:               score += 0

    # ── 2. Trend & RS Alignment (20 pts) ─────────────────────────────────────
    above_50  = candidate.get("above_50ma", False)
    above_200 = candidate.get("above_200ma", False)
    _rs       = candidate.get("rs_20d", 0)
    rs_20d    = 0.0 if (_rs is None or (isinstance(_rs, float) and math.isnan(_rs))) else _rs
    if not is_bearish:
        if above_50 and above_200 and rs_20d > 0:   score += 20
        elif above_50 and above_200:                  score += 14
        elif not above_50:                            score += 10
        elif not above_200:                           score += 3
    else:
        if not above_50 and rs_20d < 0:              score += 20
        elif not above_50:                            score += 14
        elif not above_200:                           score += 10
        else:                                         score += 3

    # ── 3. Liquidity Quality (20 pts) ────────────────────────────────────────
    if avg_vol > 10_000 and bid_ask < 0.03 and oi > 1_000:
        score += 20
    elif avg_vol >= 1_000 and bid_ask < 0.05 and oi >= 500:
        score += 13
    elif avg_vol >= 500 and bid_ask < 0.08:
        score += 7
    else:
        score += 0   # already caught by golden rule above for extreme cases

    # ── 4. Catalyst + Sentiment (20 pts) ─────────────────────────────────────
    # Catalyst (10 pts)
    earn_days = candidate.get("earnings_days_away")
    implied_move = candidate.get("implied_move_pct", 0) or 0
    if earn_days is not None and earn_days <= 7 and implied_move > 0.10:
        score += 10
    elif earn_days is not None and earn_days <= 14:
        score += 7
    else:
        score += 3

    # Sentiment (10 pts)
    bonus_flags = candidate.get("sentiment_flags", [])
    if len(bonus_flags) >= 2:   score += 10
    elif len(bonus_flags) == 1: score += 7
    elif not candidate.get("warning_flags"):
        score += 4   # neutral — no flags either direction
    else:
        score += 2   # has warning flags

    # ── 5. Sector + Relative Strength (10 pts) ───────────────────────────────
    leading  = market_env.get("leading_sectors", [])
    lagging  = market_env.get("lagging_sectors", [])
    sector   = candidate.get("sector", "")
    if not is_bearish:
        if sector in leading:  score += 10
        elif sector in lagging: score += 2
        else:                   score += 5
    else:
        if sector in lagging:  score += 10
        elif sector in leading: score += 2
        else:                   score += 5

    # ── 6. IV/RV Ratio (10 pts) ──────────────────────────────────────────────
    if is_credit:
        if iv_rv >= 2.0:    score += 10
        elif iv_rv >= 1.5:  score += 7
        elif iv_rv >= 1.0:  score += 4
        else:               score += 0   # misaligned: selling when IV < RV
    elif is_debit:
        if iv_rv < 0.7:     score += 10
        elif iv_rv < 0.9:   score += 7
        elif iv_rv <= 1.0:  score += 4
        else:               score += 0   # misaligned: buying when IV > RV

    # Sanity check
    if not (0 <= score <= 100):
        logger.error(f"[{ErrorCode.E3004}] Score={score} out of range for {candidate.get('ticker')}")
        score = max(0, min(100, score))

    return int(score)


def compute_confidence(candidate: dict) -> dict:
    """
    7-signal confidence score.
    Returns {label, count, details}.
    """
    structure  = candidate.get("structure", "")
    is_credit  = structure in {"bull_put_spread", "bear_call_spread", "iron_condor",
                               "cash_secured_put", "earnings_credit_spread"}
    is_bullish = candidate.get("direction", "Bullish") == "Bullish"

    iv_rank  = candidate.get("iv_rank", 50)
    iv_rv    = candidate.get("iv_rv_ratio", 1.0)
    above_50 = candidate.get("above_50ma", False)
    above_200 = candidate.get("above_200ma", False)
    rs_20d   = candidate.get("rs_20d", 0)
    news     = candidate.get("news_signal", "NEUTRAL")
    leading  = candidate.get("sector", "") in candidate.get("_leading_sectors", [])
    ev       = candidate.get("bs", {}).get("ev", candidate.get("ev", 0))

    signals = [
        # Signal 1: IV Rank direction
        (is_credit and iv_rank > 50) or (not is_credit and iv_rank < 30),
        # Signal 2: IV/RV ratio direction
        (is_credit and iv_rv > 1.5) or (not is_credit and iv_rv < 1.0),
        # Signal 3: Trend alignment
        (is_bullish and above_50 and above_200) or (not is_bullish and not above_50),
        # Signal 4: Relative Strength
        (is_bullish and rs_20d > 0) or (not is_bullish and rs_20d < 0),
        # Signal 5: News sentiment
        (is_bullish and news == "BULLISH") or (not is_bullish and news == "BEARISH"),
        # Signal 6: Sector rotation alignment
        leading,
        # Signal 7: EV quality
        ev > 0,
    ]

    count = sum(signals)
    if count >= 5:
        label = "High"
    elif count >= 3:
        label = "Medium"
    else:
        label = "Low"

    return {"label": label, "count": count, "score_string": f"{label} ({count}/7)"}


def filter_by_score(candidates: list[dict], market_env: dict,
                    floor: int = SCORE_FLOOR) -> list[dict]:
    """
    Score all candidates, attach score, filter out score < floor.
    Returns sorted list (highest score first).
    Pass floor=0 for the 2B ranking pass (before deep fetch sets structure).
    """
    scored = []
    for c in candidates:
        s = compute_score(c, market_env)
        if s >= floor:
            scored.append({**c, "score": s})

    scored.sort(key=lambda x: x["score"], reverse=True)
    logger.info(f"[Scorer] {len(candidates)} → {len(scored)} candidates after score filter (floor={floor})")
    return scored
