"""
Phase 7: Write the daily briefing file to disk.
Output: output/briefings/YYYY-MM-DD_OptionsBrief.md
Appends a metadata footer: date, pipeline duration, candidate count, model, IV proxy status.
"""
import json
from datetime import date, datetime
from pathlib import Path

from loguru import logger

from config import OUTPUT_BRIEFINGS_DIR, OUTPUT_FILENAME_PATTERN, CLAUDE_MODEL
from errors import ErrorCode


def write_briefing(
    briefing_text: str,
    top_candidates: dict,
    pipeline_start: datetime,
    iv_proxy_days: int = 0,
) -> Path:
    """
    Write briefing markdown to output/briefings/YYYY-MM-DD_OptionsBrief.md.
    Returns the path of the written file.
    Logs E4003 and attempts fallback to current directory on write failure.
    """
    today          = date.today().isoformat()
    filename       = OUTPUT_FILENAME_PATTERN.format(date=today)
    briefings_dir  = Path(OUTPUT_BRIEFINGS_DIR)
    output_path    = briefings_dir / filename

    duration_secs  = (datetime.now() - pipeline_start).total_seconds()
    candidate_count = len(top_candidates.get("candidates", []))
    is_no_trade    = top_candidates.get("no_trade_day", False)
    is_reduced     = top_candidates.get("reduced_opportunity_day", False)

    # Build metadata footer
    proxy_warning = ""
    if iv_proxy_days < 30:
        proxy_warning = (
            f"\n> ⚠️ **IV RANK PROXY** — {iv_proxy_days}/30 days of real IV data accumulated. "
            f"IV Rank values are approximated from HV30 history. "
            f"Verify IV Rank independently on Barchart/TOS before trading."
        )

    day_type = "NO-TRADE DAY" if is_no_trade else ("REDUCED OPPORTUNITY" if is_reduced else "Standard")
    footer = (
        f"\n\n---\n"
        f"*Generated: {today} | "
        f"Pipeline: {duration_secs:.0f}s | "
        f"Setups: {candidate_count} | "
        f"Day type: {day_type} | "
        f"Model: {CLAUDE_MODEL}*"
        f"{proxy_warning}\n"
    )

    full_content = briefing_text + footer

    # Primary write
    try:
        briefings_dir.mkdir(parents=True, exist_ok=True)
        output_path.write_text(full_content, encoding="utf-8")
        logger.info(f"[Phase 7] Briefing written: {output_path}")
        return output_path
    except OSError as e:
        logger.error(f"[{ErrorCode.E4003}] Primary write failed: {e}")

    # Fallback: write to current directory
    fallback_path = Path(filename)
    try:
        fallback_path.write_text(full_content, encoding="utf-8")
        logger.warning(f"[Phase 7] Briefing written to fallback path: {fallback_path}")
        return fallback_path
    except OSError as e2:
        logger.error(f"[{ErrorCode.E4003}] Fallback write also failed: {e2}")
        raise
