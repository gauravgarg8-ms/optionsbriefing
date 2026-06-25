"""
Phase 4: Per-trade management pre-computation.
Adds trade management fields to each candidate so Claude only narrates — never calculates.
"""
from datetime import date, datetime, timedelta
from loguru import logger

from config import (
    POP_FLOOR, POP_HALF_SIZE_THRESHOLD,
    POP_FLOOR_DEBIT, POP_HALF_SIZE_DEBIT_THRESHOLD,
    POP_HALF_SIZE_SCORE_OVERRIDE,
)

_DEBIT_STRUCTURES = {"bull_call_spread", "bear_put_spread", "long_straddle", "long_strangle"}
from errors import ErrorCode


def compute_trade_management(candidate: dict) -> dict:
    """
    Add trade management fields to a single candidate dict.
    Returns enriched copy with date_21_dte, avoid_hold_past,
    profit_target_usd, stop_loss_usd, pop_quality, pop_half_size.
    """
    structure  = candidate.get("structure", "")
    expiry_str = candidate.get("expiry") or candidate.get("strikes", {}).get("expiry")
    pricing    = candidate.get("spread_pricing", {})
    bs         = candidate.get("bs", {})
    pop        = float(bs.get("pop", pricing.get("pop", 0)) or 0)

    # ── Expiry-based date fields ─────────────────────────────────────────────
    date_21_dte    = None
    avoid_hold_past = None
    if expiry_str:
        try:
            expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
            is_0dte     = (expiry_date == date.today())
            if is_0dte:
                # 0DTE: 21-DTE concept doesn't apply — exit intraday
                date_21_dte     = "0DTE — exit before market close today"
                avoid_hold_past = expiry_date.isoformat()
            else:
                date_21_dte     = (expiry_date - timedelta(days=21)).isoformat()
                avoid_hold_past = (expiry_date - timedelta(days=7)).isoformat()
        except ValueError as e:
            logger.warning(f"[{ErrorCode.E2002}] Bad expiry format for {candidate.get('ticker')}: {e}")

    # ── Profit target and stop loss ──────────────────────────────────────────
    is_credit  = structure in {"bull_put_spread", "bear_call_spread", "iron_condor",
                               "cash_secured_put", "earnings_credit_spread"}
    is_debit   = structure in {"bull_call_spread", "bear_put_spread",
                               "long_straddle", "long_strangle"}

    max_profit  = float(pricing.get("max_profit", 0)  or 0)
    max_loss    = float(pricing.get("max_loss",   0)  or 0) if pricing.get("max_loss") != "unlimited" else 0
    net_credit  = float(pricing.get("net_credit", 0)  or 0)
    net_debit   = float(pricing.get("net_debit",  0)  or 0)

    profit_target_usd = 0.0
    stop_loss_usd     = 0.0

    if is_credit:
        profit_target_usd = round(max_profit * 0.50, 2)
        stop_loss_usd     = round(net_credit * 2 * 100, 2)
    elif is_debit:
        profit_target_usd = round(net_debit * 100, 2)   # exit when spread doubles
        stop_loss_usd     = round(net_debit * 100, 2)   # 100% of premium paid
    elif structure == "cash_secured_put":
        profit_target_usd = round(max_profit * 0.80, 2)
        stop_loss_usd     = round(max_loss, 2)

    # ── PoP quality label (structure-aware floors) ───────────────────────────
    is_debit  = structure in _DEBIT_STRUCTURES
    pop_floor = POP_FLOOR_DEBIT if is_debit else POP_FLOOR
    pop_half_threshold = POP_HALF_SIZE_DEBIT_THRESHOLD if is_debit else POP_HALF_SIZE_THRESHOLD

    if pop >= 0.80:
        pop_quality = "High"
    elif pop >= 0.70:
        pop_quality = "Good"
    elif pop >= pop_floor:
        pop_quality = "Acceptable"
    elif pop > 0:
        pop_quality = "EXCLUDE"
    else:
        pop_quality = "N/A"

    score = float(candidate.get("score", 0) or 0)
    pop_half_size = (pop_floor <= pop < pop_half_threshold) and (score < POP_HALF_SIZE_SCORE_OVERRIDE)

    return {
        **candidate,
        "date_21_dte":       date_21_dte,
        "avoid_hold_past":   avoid_hold_past,
        "profit_target_usd": profit_target_usd,
        "stop_loss_usd":     stop_loss_usd,
        "pop_quality":       pop_quality,
        "pop_half_size":     pop_half_size,
    }


def compute_all_trade_management(screened_output: dict) -> dict:
    """
    Apply trade management to every candidate in the screened output.
    Returns enriched copy of screened_output.
    """
    candidates = screened_output.get("candidates", [])
    enriched   = [compute_trade_management(c) for c in candidates]
    logger.info(f"[Phase 4] Trade management computed for {len(enriched)} candidates")
    return {**screened_output, "candidates": enriched}
