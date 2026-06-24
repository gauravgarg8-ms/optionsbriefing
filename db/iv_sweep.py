"""
End-of-day IV sweep — populates real IV30 for the full 516-ticker universe.

Run after market close (scheduled at 4:15 PM ET Mon–Fri via launchd).
Fetches yfinance options chains without the liquidity filter so impliedVolatility
(computed from last-traded prices) is available even when bid/ask spreads are wide.

Usage:
    .venv/bin/python db/iv_sweep.py
"""

import resource
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

# Raise soft fd limit before any imports open SQLite/network connections.
# launchd default is 256; 20 parallel yfinance+SQLite threads exhaust this fast.
_soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
if _soft < 4096:
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(4096, _hard), _hard))

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from data.options_data import _fetch_yfinance_chain
from data.universe_manager import _load_cache
from db.db_manager import DBManager
from quant.volatility import compute_iv30

_LOG_PATH = Path(__file__).parent.parent / "logs" / f"iv_sweep_{date.today().isoformat()}.log"
logger.add(str(_LOG_PATH), rotation="1 day", retention="30 days", level="INFO")

_MAX_WORKERS = 20
_db_lock     = threading.Lock()   # serialize SQLite writes; fetches remain parallel


def _sweep_ticker(ticker: str, db: DBManager) -> tuple[str, bool]:
    """Fetch raw yfinance chain, compute IV30, write to DB. Returns (ticker, success)."""
    try:
        chain = _fetch_yfinance_chain(ticker)
        options = chain.get("options", [])
        if not options:
            return ticker, False

        iv30 = compute_iv30(options)
        if iv30 <= 0:
            return ticker, False

        with _db_lock:
            db.upsert_iv_with_source(ticker, date.today().isoformat(), iv30, source="real")
        return ticker, True
    except Exception as e:
        logger.warning(f"iv_sweep failed for {ticker}: {e}")
        return ticker, False


def run_sweep() -> None:
    universe = _load_cache()
    if not universe:
        logger.error("Universe cache empty — run main pipeline first to populate cache")
        sys.exit(1)

    tickers = [entry["symbol"] for entry in universe]
    logger.info(f"IV sweep starting: {len(tickers)} tickers")

    db = DBManager()
    success, failed = 0, 0

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        futures = {pool.submit(_sweep_ticker, t, db): t for t in tickers}
        for fut in as_completed(futures):
            ticker, ok = fut.result()
            if ok:
                success += 1
            else:
                failed += 1

    logger.info(
        f"IV sweep complete: {success}/{len(tickers)} tickers written "
        f"({failed} skipped — no valid IV)"
    )


if __name__ == "__main__":
    run_sweep()
