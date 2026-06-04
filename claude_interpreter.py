"""
Phase 6: Claude AI narrative interpretation.
Sends the fully pre-computed top_candidates.json to Claude and receives the briefing.
Claude writes the narrative only — all numbers are pre-computed by Python.
"""
import json
import time
from pathlib import Path

import anthropic
from loguru import logger

from config import CLAUDE_MODEL, CLAUDE_MAX_TOKENS, CLAUDE_MAX_RETRIES, ANTHROPIC_API_KEY
from errors import ErrorCode

SYSTEM_PROMPT = """
You are a professional pre-market options analyst generating a daily briefing for GG,
an IT Consultant building expertise in stock analysis and options trading.

You receive a structured JSON payload with ALL quant metrics pre-computed.
YOUR ONLY JOB is to narrate, explain, and format — do NOT re-fetch or re-calculate.

JSON structure you receive:
  market_environment: VIX regime, SPY trend, Fear&Greed, Put/Call, sectors, macro events
  active_scenarios: list of Scenario 1–7 codes that apply today
  portfolio_check: net delta, vega direction, sector concentration, warnings
  no_trade_day: boolean
  reduced_opportunity_day: boolean
  candidates: top N setups with ALL fields pre-computed

YOUR RESPONSIBILITIES:

1. **Market Environment** — interpret VIX/SPY/sentiment, state structure bias in 2-3 sentences.

2. **Sector Rotation** — explain money flow: which sectors are leading/lagging and why it matters today.

3. **Scenario statement** — name each active scenario (S1–S7), 1-sentence market character description.

4. **For EACH candidate**, write:
   a. Trade thesis (2 sentences: why this stock, why this structure, why today)
   b. Full data table using JSON values verbatim — do NOT change any numbers
   c. B-S theoretical trade setup table — ALWAYS prefix with: ⚠️ B-S THEORETICAL — verify live mid-price on broker before entry
   d. Greeks, Risk/Reward & Expectancy — show d1/d2 values, PoP formula, EV calculation
   e. Trade management section using pre-computed dates/prices from JSON verbatim
   f. Educational "Why this structure" sentence referencing IV Rank + IV/RV from JSON
   g. If covered_call_opportunity=true: add a note "💡 If you hold 100+ shares: a covered call at [resistance] may also be appropriate. Requires 100 shares — not an automated setup."

5. **Quick Reference Summary Table** — one row per setup: ticker, structure, net credit/debit, PoP, EV, 21-DTE date, confidence

6. **Portfolio Exposure Check** — narrate portfolio_check.portfolio_warnings. If none, state "Portfolio exposure within limits."

7. **Golden Rule sentence** — one line combining today's IV Rank regime + IV/RV interpretation.

8. **Pre-trade checklist** (always include):
   ☐ Confirm IV Rank on broker (Barchart ≠ TOS — different scale)
   ☐ IV/RV agrees with structure direction?
   ☐ Bid/ask < 5% of mid at target strikes?
   ☐ Live mid-price confirmed from broker chain (NOT B-S estimate)
   ☐ PoP ≥ 65% (60–65% = half size)? EV positive?
   ☐ Score ≥ 45/100? Max loss ≤ 2–5% of portfolio?
   ☐ Profit target + stop loss written before entry? 21 DTE date noted?

If no_trade_day=true OR reduced_opportunity_day=true:
  - Report market environment, sector rotation, and scenario statement ONLY.
  - For no_trade_day: display "🚫 NO-TRADE DAY — fewer than 5 setups meet minimum quality threshold."
  - For reduced_opportunity_day: display "⚠️ REDUCED OPPORTUNITY DAY — present available setups only. Do not force 10 setups."

TONE: Professional but educational. Always explain the WHY behind every number.
Show d1/d2 and the PoP formula used. Never invent numbers — use the JSON values exactly.
If the IV RANK PROXY warning is present in the JSON, include it prominently.
"""


def run_claude_briefing(payload: dict) -> str:
    """
    Send payload to Claude and return the briefing markdown text.
    Raises after max retries are exhausted.
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    user_message = (
        f"Generate today's daily options briefing based on this pre-computed data:\n\n"
        f"{json.dumps(payload, indent=2, default=str)}"
    )

    for attempt in range(1, CLAUDE_MAX_RETRIES + 1):
        try:
            with client.messages.stream(
                model      = CLAUDE_MODEL,
                max_tokens = CLAUDE_MAX_TOKENS,
                system     = SYSTEM_PROMPT,
                messages   = [{"role": "user", "content": user_message}],
            ) as stream:
                text = stream.get_final_text()
                stop_reason = stream.get_final_message().stop_reason
            logger.info(f"[Phase 6] Claude briefing generated ({len(text)} chars, stop_reason={stop_reason})")
            if stop_reason == "max_tokens":
                logger.error(
                    f"[{ErrorCode.E4001}] Claude hit max_tokens ({CLAUDE_MAX_TOKENS}) — "
                    f"briefing is TRUNCATED. Raise CLAUDE_MAX_TOKENS in config.py."
                )
                text += (
                    "\n\n> ⚠️ **BRIEFING TRUNCATED** — Claude hit the max_tokens limit. "
                    "Increase `CLAUDE_MAX_TOKENS` in config.py and re-run."
                )
            return text

        except anthropic.RateLimitError as e:
            wait = 60 * attempt
            logger.warning(f"[{ErrorCode.E4002}] Claude rate limited (attempt {attempt}): {e} — sleeping {wait}s")
            if attempt < CLAUDE_MAX_RETRIES:
                time.sleep(wait)

        except anthropic.APIError as e:
            logger.warning(f"[{ErrorCode.E4001}] Claude API error (attempt {attempt}): {e}")
            if attempt < CLAUDE_MAX_RETRIES:
                time.sleep(10)

    logger.error(f"[{ErrorCode.E4001}] Claude API failed after {CLAUDE_MAX_RETRIES} attempts")
    raise RuntimeError(f"Claude API failed after {CLAUDE_MAX_RETRIES} retries")


def run_with_retry(top_candidates_path: str = "output/top_candidates.json") -> str:
    """
    Load top_candidates.json and call Claude. Convenience wrapper used by main.py.
    """
    payload = json.loads(Path(top_candidates_path).read_text())
    return run_claude_briefing(payload)
