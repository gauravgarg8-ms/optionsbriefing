"""
14-row structure decision matrix + 1-SD OTM strike selection.

Decision #7 tie-breaking: in the middle IV zone (30–50% IV Rank):
  IV/RV ≥ 1.2 → credit structure
  IV/RV <  1.2 → debit structure

Golden rules (enforced here):
  IV Rank > 50% AND tries debit → override to credit
  IV Rank < 20% AND tries credit → override to debit
"""

import math
from datetime import date, datetime, timedelta
from loguru import logger

from config import IV_RANK_CREDIT_MIN, IV_RANK_DEBIT_MAX, IV_RV_MIDDLE_ZONE_CREDIT_THRESHOLD
from errors import ErrorCode


def select_structure(
    iv_rank: float,
    iv_rv_ratio: float,
    bias: str,         # "Bullish" | "Bearish" | "Neutral"
    earnings_flag: bool = False,
    implied_move_pct: float | None = None,
    hist_avg_move_pct: float | None = None,
    holds_shares: bool = False,
) -> str:
    """
    Returns a strategy code string.
    """
    bias = bias.capitalize()

    # ── Earnings override (highest priority) ─────────────────────────────
    if earnings_flag and implied_move_pct is not None and hist_avg_move_pct is not None:
        if implied_move_pct > hist_avg_move_pct:
            return "earnings_credit_spread"
        else:
            return "long_straddle"

    # ── Post-earnings IV crush (flagged externally) ───────────────────────
    # Handled by scenario_classifier — structure_selector doesn't see it

    # ── Cash-secured put (income bias, any IV regime) ─────────────────────
    if bias == "Bullish income":
        return "cash_secured_put"

    # ── High IV zone (IV Rank > 50%) ──────────────────────────────────────
    if iv_rank > IV_RANK_CREDIT_MIN:
        if bias == "Bullish":
            return "bull_put_spread"
        if bias == "Bearish":
            return "bear_call_spread"
        return "iron_condor"

    # ── Low IV zone (IV Rank < 30%) ───────────────────────────────────────
    if iv_rank < IV_RANK_DEBIT_MAX:
        if bias == "Bullish":
            return "bull_call_spread"
        if bias == "Bearish":
            return "bear_put_spread"
        return "long_straddle"

    # ── Middle zone (30–50% IV Rank) — Decision #7 tie-breaking ──────────
    if iv_rv_ratio >= IV_RV_MIDDLE_ZONE_CREDIT_THRESHOLD:
        # Lean credit
        if bias == "Bullish":
            return "bull_put_spread"
        if bias == "Bearish":
            return "bear_call_spread"
        return "iron_condor"
    else:
        # Lean debit
        if bias == "Bullish":
            return "bull_call_spread"
        if bias == "Bearish":
            return "bear_put_spread"
        return "long_straddle"


def apply_golden_rules(structure: str, iv_rank: float) -> str:
    """
    Enforce golden rules post-selection:
      - Debit with IV Rank > 50% → override to credit equivalent
      - Credit with IV Rank < 20% → override to debit equivalent
    """
    debit_structures  = {"bull_call_spread", "bear_put_spread", "long_straddle", "long_strangle"}
    credit_structures = {"bull_put_spread", "bear_call_spread", "iron_condor", "cash_secured_put",
                         "earnings_credit_spread"}

    if structure in debit_structures and iv_rank > IV_RANK_CREDIT_MIN:
        # Map debit → credit equivalent
        override_map = {
            "bull_call_spread": "bull_put_spread",
            "bear_put_spread":  "bear_call_spread",
            "long_straddle":    "iron_condor",
            "long_strangle":    "iron_condor",
        }
        new_structure = override_map.get(structure, "bull_put_spread")
        logger.info(
            f"Golden rule override: {structure} → {new_structure} "
            f"(IV Rank={iv_rank:.0f}% > 50% — no debit when IV rich)"
        )
        return new_structure

    if structure in credit_structures and iv_rank < 20:
        override_map = {
            "bull_put_spread":       "bull_call_spread",
            "bear_call_spread":      "bear_put_spread",
            "iron_condor":           "long_straddle",
            "cash_secured_put":      "bull_call_spread",
            "earnings_credit_spread":"long_straddle",
        }
        new_structure = override_map.get(structure, "bull_call_spread")
        logger.info(
            f"Golden rule override: {structure} → {new_structure} "
            f"(IV Rank={iv_rank:.0f}% < 20% — no credit when IV cheap)"
        )
        return new_structure

    return structure


def select_strikes(
    S: float,
    structure: str,
    options: list[dict],
    iv30: float,
    T_days: int,
    support: float,
    resistance: float,
) -> dict:
    """
    Select strikes for the chosen structure using 1-SD OTM formula.
    Returns {k_short, k_long, k_call (for condor/strangle), expiry, dte}.
    """
    if not options or T_days <= 0:
        logger.warning(f"[{ErrorCode.E2004}] select_strikes: no options or T_days={T_days}")
        return {}

    T_years  = T_days / 365.0
    sd_move  = S * iv30 * math.sqrt(T_years)
    expiry   = options[0].get("expiration_date", "")
    strikes  = sorted({float(o["strike"]) for o in options})

    def nearest_strike(target: float) -> float:
        return min(strikes, key=lambda x: abs(x - target))

    try:
        if structure == "bull_put_spread":
            k_short = nearest_strike(S - sd_move)
            k_long  = nearest_strike(k_short - 5)
            k_long  = min(k_long, k_short - (strikes[1] - strikes[0]) if len(strikes) > 1 else k_short - 5)
            return {"k_short": k_short, "k_long": k_long, "expiry": expiry, "dte": T_days}

        if structure == "bear_call_spread":
            stride  = (strikes[-1] - strikes[-2]) if len(strikes) > 1 else 5
            k_short = nearest_strike(S + sd_move)
            k_long  = nearest_strike(k_short + 5)
            k_long  = max(k_long, k_short + stride)   # mirror bull_put guard: ensure k_long > k_short
            return {"k_short": k_short, "k_long": k_long, "expiry": expiry, "dte": T_days}

        if structure == "iron_condor":
            put_stride  = (strikes[1]  - strikes[0])  if len(strikes) > 1 else 5
            call_stride = (strikes[-1] - strikes[-2]) if len(strikes) > 1 else 5
            k_put_short  = nearest_strike(S - sd_move)
            k_put_long   = nearest_strike(k_put_short - 5)
            k_put_long   = min(k_put_long, k_put_short - put_stride)    # ensure k_put_long < k_put_short
            k_call_short = nearest_strike(S + sd_move)
            k_call_long  = nearest_strike(k_call_short + 5)
            k_call_long  = max(k_call_long, k_call_short + call_stride) # ensure k_call_long > k_call_short
            return {"k_put_short": k_put_short, "k_put_long": k_put_long,
                    "k_call_short": k_call_short, "k_call_long": k_call_long,
                    "expiry": expiry, "dte": T_days}

        if structure == "bull_call_spread":
            k_long  = nearest_strike(S)       # ATM call
            k_short = nearest_strike(resistance if resistance > S else S + sd_move * 0.5)
            if k_short <= k_long:
                k_short = nearest_strike(k_long + 5)
            return {"k_long": k_long, "k_short": k_short, "expiry": expiry, "dte": T_days}

        if structure == "bear_put_spread":
            k_long  = nearest_strike(S)       # ATM put
            k_short = nearest_strike(support if support < S else S - sd_move * 0.5)
            if k_short >= k_long:
                k_short = nearest_strike(k_long - 5)
            return {"k_long": k_long, "k_short": k_short, "expiry": expiry, "dte": T_days}

        if structure == "long_straddle":
            k = nearest_strike(S)
            return {"k": k, "expiry": expiry, "dte": T_days}

        if structure == "long_strangle":
            k_put  = nearest_strike(S - sd_move * 0.5)
            k_call = nearest_strike(S + sd_move * 0.5)
            return {"k_put": k_put, "k_call": k_call, "expiry": expiry, "dte": T_days}

        if structure in ("cash_secured_put", "earnings_credit_spread"):
            k_short = nearest_strike(S - sd_move)
            k_long  = nearest_strike(k_short - 5)
            return {"k_short": k_short, "k_long": k_long, "expiry": expiry, "dte": T_days}

    except Exception as e:
        logger.error(f"[{ErrorCode.E2004}] select_strikes failed for {structure}: {e}")

    return {}
