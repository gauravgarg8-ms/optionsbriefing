"""
Daily Options Briefing System — 7-phase pipeline orchestrator.
Run manually: python main.py
Scheduled:    launchd fires this Mon–Fri 7:30 AM ET via com.gg.options-briefing.plist
"""
import json
import resource
import sys
from datetime import datetime, date, timedelta
from pathlib import Path

# Raise soft fd limit before any imports open SQLite/network connections.
# launchd default is 256; yfinance opens one SQLite tz-cache file per Ticker call.
_soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
if _soft < 4096:
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(4096, _hard), _hard))

import pandas_market_calendars as mcal
from loguru import logger

from errors import ErrorCode
from config import PIPELINE_TIMEOUT_MINS, TICKER_HISTORY_WINDOW_DAYS

# Configure loguru: daily rotating log file + console
logger.remove()
logger.add(
    f"logs/{date.today().isoformat()}.log",
    rotation="1 day", retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)
logger.add(sys.stderr, level="INFO",
           format="{time:HH:mm:ss} | {level} | {message}")

# Output paths
OUTPUT_DIR     = Path("output")
RAW_PATH       = OUTPUT_DIR / "raw_market_data.json"
QUANT_PATH     = OUTPUT_DIR / "quant_signals.json"
SCREENED_PATH  = OUTPUT_DIR / "screened_candidates.json"
TOP_PATH       = OUTPUT_DIR / "top_candidates.json"
HISTORY_PATH   = OUTPUT_DIR / "ticker_history.json"


# ── Repeat ticker history helpers ─────────────────────────────────────────────

def _load_ticker_history() -> dict:
    if not HISTORY_PATH.exists():
        return {}
    try:
        return json.loads(HISTORY_PATH.read_text())
    except Exception:
        return {}


def _save_ticker_history(history: dict, today: str, tickers: list) -> None:
    cutoff  = (date.today() - timedelta(days=TICKER_HISTORY_WINDOW_DAYS)).isoformat()
    history = {d: t for d, t in history.items() if d >= cutoff}
    history[today] = tickers
    try:
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        HISTORY_PATH.write_text(json.dumps(history, indent=2))
    except Exception as e:
        logger.warning(f"Failed to write ticker history: {e}")


def _enrich_repeat_flags(candidates: list, history: dict) -> None:
    """Attach repeat_days_ago (int or None) to each candidate in-place."""
    for c in candidates:
        ticker   = c.get("ticker", "")
        days_ago = None
        for i in range(1, TICKER_HISTORY_WINDOW_DAYS + 1):
            d = (date.today() - timedelta(days=i)).isoformat()
            if d in history and ticker in history[d]:
                days_ago = i
                break
        c["repeat_days_ago"] = days_ago


# ── Excluded ticker split helpers ─────────────────────────────────────────────

def _is_excluded(c: dict) -> bool:
    if c.get("pop_quality") == "EXCLUDE":
        return True
    pricing    = c.get("spread_pricing", {})
    net_credit = float(pricing.get("net_credit") or 0)
    net_debit  = float(pricing.get("net_debit")  or 0)
    if net_credit == 0 and net_debit == 0:
        return True
    return False


def _get_exclude_reason(c: dict) -> str:
    if c.get("pop_quality") == "EXCLUDE":
        bs      = c.get("bs", {})
        pricing = c.get("spread_pricing", {})
        pop     = float(bs.get("pop", pricing.get("pop", 0)) or 0)
        return f"PoP {pop:.1%} below minimum floor"
    return "Degenerate spread (zero-width or zero-value strikes)"


def _save_json(data: dict, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str))
    except Exception as e:
        logger.error(f"[{ErrorCode.E4004}] Failed to write {path.name}: {e}")


def _is_market_holiday() -> bool:
    nyse   = mcal.get_calendar("NYSE")
    today  = date.today().isoformat()
    valid  = nyse.valid_days(start_date=today, end_date=today)
    return len(valid) == 0


def run_daily_briefing() -> None:
    pipeline_start = datetime.now()
    today          = date.today().isoformat()
    logger.info(f"{'='*60}")
    logger.info(f"Daily Options Briefing — {today}")
    logger.info(f"{'='*60}")

    # ── Market holiday guard (Decision #8) ──────────────────────────────────
    if _is_market_holiday():
        logger.info(f"[{ErrorCode.E5001}] {today} is a market holiday — skipping pipeline")
        return

    # ── Timeout guard ────────────────────────────────────────────────────────
    def _check_timeout():
        elapsed = (datetime.now() - pipeline_start).total_seconds() / 60
        if elapsed > PIPELINE_TIMEOUT_MINS:
            logger.error(f"[{ErrorCode.E5002}] Pipeline timeout ({elapsed:.1f} min > {PIPELINE_TIMEOUT_MINS} min)")
            return True
        return False

    # ── Phase 1: Data Collection ─────────────────────────────────────────────
    phase_start = datetime.now()
    logger.info("[Phase 1] Data collection starting...")
    from data.data_engine import run_data_collection
    raw_data = run_data_collection()
    _save_json(raw_data, RAW_PATH)
    logger.info(f"[Phase 1] Complete in {(datetime.now()-phase_start).total_seconds():.1f}s")
    if _check_timeout(): return

    # ── Phase 2: Quant Calculations ──────────────────────────────────────────
    phase_start = datetime.now()
    logger.info("[Phase 2] Quant calculations starting...")
    from quant.quant_engine import run_quant_calculations
    quant_data = run_quant_calculations(raw_data)
    # Carry earnings calendar + tbill rate from raw_data into quant_data
    quant_data["earnings_calendar"] = raw_data.get("earnings_calendar", [])
    quant_data["tbill_rate"]        = quant_data.get("market_environment", {}).get("tbill_3m", 0.051)
    _save_json(quant_data, QUANT_PATH)
    logger.info(f"[Phase 2] Complete in {(datetime.now()-phase_start).total_seconds():.1f}s")
    if _check_timeout(): return

    # ── Phase 3: Screening ───────────────────────────────────────────────────
    phase_start = datetime.now()
    logger.info("[Phase 3] Screening starting...")
    from screening.screening_engine import run_screening
    screened = run_screening(quant_data)
    _save_json(screened, SCREENED_PATH)
    logger.info(f"[Phase 3] Complete in {(datetime.now()-phase_start).total_seconds():.1f}s")
    if _check_timeout(): return

    # ── Phase 4: Trade Management Pre-Computation ────────────────────────────
    phase_start = datetime.now()
    logger.info("[Phase 4] Trade management starting...")
    from risk_manager import compute_all_trade_management
    screened = compute_all_trade_management(screened)
    logger.info(f"[Phase 4] Complete in {(datetime.now()-phase_start).total_seconds():.1f}s")

    # ── Repeat ticker flagging ───────────────────────────────────────────────
    ticker_history = _load_ticker_history()
    _enrich_repeat_flags(screened.get("candidates", []), ticker_history)
    repeats = [c["ticker"] for c in screened.get("candidates", []) if c.get("repeat_days_ago")]
    if repeats:
        logger.info(f"[Repeat flags] {len(repeats)} repeat ticker(s): {repeats}")

    # ── Excluded ticker split ────────────────────────────────────────────────
    all_candidates = screened.get("candidates", [])
    actionable     = [c for c in all_candidates if not _is_excluded(c)]
    excluded       = [c for c in all_candidates if _is_excluded(c)]
    for c in excluded:
        c["exclude_reason"] = _get_exclude_reason(c)
    screened["candidates"]          = actionable
    screened["excluded_candidates"] = excluded
    if excluded:
        logger.info(f"[Excluded split] {len(excluded)} candidate(s) moved to appendix: "
                    f"{[c['ticker'] for c in excluded]}")

    # ── Phase 5: Scenario Classification ────────────────────────────────────
    phase_start = datetime.now()
    logger.info("[Phase 5] Scenario classification starting...")
    from scenario_classifier import classify_scenarios
    market_env   = screened.get("market_environment", {})
    # Merge macro_events into market_env for scenario classifier
    market_env["macro_events"] = screened.get("macro_events", raw_data.get("macro_events", []))
    active_scenarios = classify_scenarios(market_env, screened.get("candidates", []))
    screened["active_scenarios"] = active_scenarios
    screened["date"]             = today

    # Attach per-candidate real_iv_days before Claude sees the payload.
    from db.db_manager import DBManager
    db = DBManager()
    for c in screened.get("candidates", []):
        c["real_iv_days"] = db.get_real_iv_days(c.get("ticker", ""))

    _save_json(screened, TOP_PATH)
    logger.info(f"[Phase 5] Complete in {(datetime.now()-phase_start).total_seconds():.1f}s")
    if _check_timeout(): return

    # Carry SPY 0DTE setup into the payload Claude receives
    screened["spy_0dte"] = raw_data.get("spy_0dte", {})

    # ── Phase 6: Claude Interpretation ──────────────────────────────────────
    phase_start = datetime.now()
    logger.info("[Phase 6] Claude interpretation starting...")
    from claude_interpreter import run_claude_briefing
    briefing_text = run_claude_briefing(screened)
    logger.info(f"[Phase 6] Complete in {(datetime.now()-phase_start).total_seconds():.1f}s")

    # ── Phase 7: Delivery ────────────────────────────────────────────────────
    phase_start = datetime.now()
    logger.info("[Phase 7] Writing briefing to disk...")
    from delivery import write_briefing
    output_path = write_briefing(
        briefing_text  = briefing_text,
        top_candidates = screened,
        pipeline_start = pipeline_start,
    )
    logger.info(f"[Phase 7] Complete — briefing at {output_path}")

    # ── Save ticker history (actionable only — excluded shouldn't seed next day's repeats) ──
    _save_ticker_history(
        ticker_history, today,
        [c["ticker"] for c in screened.get("candidates", [])],
    )

    total = (datetime.now() - pipeline_start).total_seconds()
    logger.info(f"{'='*60}")
    logger.info(f"Pipeline complete in {total:.1f}s ({total/60:.1f} min)")
    logger.info(f"Briefing: {output_path}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    run_daily_briefing()
