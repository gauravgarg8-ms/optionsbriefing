"""
Phase 5: Deterministic 7-scenario classification (Python only — not Claude).
Classifies which market scenarios are active today.

S1  Earnings Trade         — any candidate with earnings_days_away ≤ 7
S2  FOMC/CPI/Macro Event   — any macro_event with days_away ≤ 7
S3  Trending No Catalyst   — no earnings, no macro within 7 days, VIX < 25
S4  High VIX / Selloff     — VIX > 25
S5  Sector Rotation        — leading_sectors non-empty (always active — shows flow)
S6  Geopolitical Event     — geopolitical_risk_flag in market_env
S7  Post-Earnings Recovery — any candidate with earnings_yesterday = True
"""
from loguru import logger


def classify_scenarios(market_env: dict, candidates: list[dict]) -> list[dict]:
    """
    Returns list of active scenario dicts: [{code, name, ...extra_fields}].
    Multiple scenarios can be active simultaneously.
    """
    active = []
    vix    = market_env.get("vix", 20.0)

    # ── S1: Earnings Trade ───────────────────────────────────────────────────
    earnings_candidates = [
        c for c in candidates
        if (c.get("earnings_days_away") or 99) <= 7
    ]
    if earnings_candidates:
        tickers = [c["ticker"] for c in earnings_candidates]
        active.append({
            "code":    "S1",
            "name":    "Earnings Trade",
            "tickers": tickers,
        })

    # ── S2: FOMC/CPI/Macro Event ─────────────────────────────────────────────
    macro_events     = market_env.get("macro_events", [])
    imminent_macros  = [e for e in macro_events if e.get("days_away", 99) <= 7]
    if imminent_macros:
        active.append({
            "code":   "S2",
            "name":   "FOMC/CPI/Macro Event",
            "events": [e["event"] for e in imminent_macros],
        })

    # ── S3: Trending Market, No Catalyst ─────────────────────────────────────
    has_imminent_earnings = bool(earnings_candidates)
    has_imminent_macro    = bool(imminent_macros)
    if not has_imminent_earnings and not has_imminent_macro and vix < 25:
        active.append({
            "code": "S3",
            "name": "Trending Market No Catalyst",
        })

    # ── S4: High VIX / Market Selloff ────────────────────────────────────────
    if vix > 25:
        active.append({
            "code": "S4",
            "name": "High VIX / Market Selloff",
            "vix":  round(vix, 1),
        })

    # ── S5: Sector Rotation ──────────────────────────────────────────────────
    leading = market_env.get("leading_sectors", [])
    lagging = market_env.get("lagging_sectors", [])
    active.append({
        "code":    "S5",
        "name":    "Sector Rotation",
        "leading": leading,
        "lagging": lagging,
    })

    # ── S6: Geopolitical Event ───────────────────────────────────────────────
    if market_env.get("geopolitical_risk_flag"):
        active.append({
            "code": "S6",
            "name": "Geopolitical Event",
        })

    # ── S7: Post-Earnings Recovery ───────────────────────────────────────────
    post_earn = [c for c in candidates if c.get("earnings_yesterday")]
    if post_earn:
        active.append({
            "code":    "S7",
            "name":    "Post-Earnings Recovery",
            "tickers": [c["ticker"] for c in post_earn],
        })

    codes = [s["code"] for s in active]
    logger.info(f"[Phase 5] Active scenarios: {codes}")
    return active
