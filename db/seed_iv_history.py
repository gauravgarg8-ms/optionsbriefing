"""
One-time IV history seeding script.

Run once at project setup:
    python db/seed_iv_history.py

Two-step seeding (Decision #1):
  Step 1 — Barchart scrape: fetch today's IV Rank for all universe tickers.
            Derive a proxy iv30 value from IV Rank + local HV30 scale anchor.
  Step 2 — HV30 backfill: for each ticker, fetch 252 days of OHLCV from yfinance
            and compute rolling 30-day realised vol as a proxy iv30 series.
            Real IV30 from Tradier overwrites these as daily accumulation runs.

After ~30 days of live operation, real Tradier IV values dominate the database.
"""

import sys
import time
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from loguru import logger

# Allow running from project root or db/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from db.db_manager import DBManager
from errors import ErrorCode

logger.add("logs/seed_{time:YYYY-MM-DD}.log", rotation="1 day", retention="30 days")


# ── Step 1: Barchart IV Rank scrape ─────────────────────────────────────────

def scrape_barchart_iv_rank(tickers: list[str]) -> dict[str, float]:
    """
    Scrape current IV Rank from Barchart for a list of tickers.
    Returns {ticker: iv_rank_0_to_100}. Missing tickers are omitted.
    """
    iv_ranks = {}
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    logger.info(f"Barchart scrape: fetching IV Rank for {len(tickers)} tickers")
    for i, ticker in enumerate(tickers):
        try:
            url = f"https://www.barchart.com/stocks/quotes/{ticker}/volatility-greeks"
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                logger.warning(f"Barchart {ticker}: HTTP {resp.status_code} — skipping")
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            # IV Rank is in a data table — find the row labelled "IV Rank"
            for row in soup.find_all("tr"):
                cells = row.find_all("td")
                if len(cells) >= 2 and "IV Rank" in cells[0].get_text():
                    raw = cells[1].get_text(strip=True).replace("%", "")
                    iv_ranks[ticker] = float(raw)
                    break
            time.sleep(0.5)  # polite scraping
            if (i + 1) % 25 == 0:
                logger.info(f"  ... scraped {i+1}/{len(tickers)}")
        except Exception as e:
            logger.warning(f"[{ErrorCode.E1010}] Barchart scrape failed for {ticker}: {e}")
    logger.info(f"Barchart scrape complete: {len(iv_ranks)}/{len(tickers)} tickers retrieved")
    return iv_ranks


# ── Step 2: HV30 backfill proxy ──────────────────────────────────────────────

def compute_hv30_series(ticker: str) -> pd.Series | None:
    """
    Fetch 252+ days of daily OHLCV and compute rolling 30-day realised vol
    (annualised) for each day. Returns a Series indexed by date string.
    """
    try:
        df = yf.download(ticker, period="14mo", interval="1d", progress=False, auto_adjust=True)
        if df.empty or len(df) < 35:
            return None
        close = df["Close"].squeeze()
        log_returns = np.log(close / close.shift(1))
        hv30 = log_returns.rolling(30).std() * np.sqrt(252)
        hv30 = hv30.dropna()
        hv30.index = pd.to_datetime(hv30.index).strftime("%Y-%m-%d")
        return hv30
    except Exception as e:
        logger.warning(f"HV30 backfill failed for {ticker}: {e}")
        return None


def seed_ticker(db: DBManager, ticker: str, barchart_iv_rank: float | None) -> int:
    """
    Seed iv_history for one ticker. Returns number of rows written.
    """
    hv30_series = compute_hv30_series(ticker)
    if hv30_series is None or hv30_series.empty:
        logger.warning(f"No price history for {ticker} — skipping")
        return 0

    rows_written = 0
    for as_of_date, hv30_val in hv30_series.items():
        if pd.isna(hv30_val) or hv30_val <= 0:
            continue
        try:
            db.upsert_iv_with_source(ticker, as_of_date, round(float(hv30_val), 4), source="proxy")
            rows_written += 1
        except Exception:
            pass  # already logged in upsert

    # Overwrite today's value with Barchart-derived IV if available
    if barchart_iv_rank is not None:
        # Back-derive proxy iv30 from IV Rank using today's HV30 as scale anchor
        today_hv30 = float(hv30_series.iloc[-1]) if not hv30_series.empty else 0.25
        # IV Rank = (iv30 - 52wk_low) / (52wk_high - 52wk_low)
        # Approximate: use HV30 * (1 + iv_rank/100) as today's iv30 proxy
        today_iv30 = today_hv30 * (1.0 + barchart_iv_rank / 100.0)
        today_str = date.today().isoformat()
        try:
            db.upsert_iv_with_source(ticker, today_str, round(today_iv30, 4), source="barchart")
            rows_written += 1
        except Exception:
            pass

    return rows_written


# ── Main entry point ──────────────────────────────────────────────────────────

def main():
    db = DBManager()

    # Load universe from cache or FMP
    cache_path = Path(__file__).parent.parent / "output" / "universe_cache.json"
    if cache_path.exists():
        import json
        universe = json.loads(cache_path.read_text())
        tickers = [t["symbol"] for t in universe if "symbol" in t]
        logger.info(f"Loaded {len(tickers)} tickers from universe cache")
    else:
        logger.warning("universe_cache.json not found — run data/universe_manager.py first to populate it")
        logger.info("Using a small default set for testing")
        tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "V", "UNH"]

    # Step 1: Barchart scrape for today's IV Rank
    logger.info("=== Step 1: Barchart IV Rank scrape ===")
    barchart_ranks = scrape_barchart_iv_rank(tickers)

    # Step 2: HV30 backfill for full 252-day proxy history
    logger.info("=== Step 2: HV30 backfill proxy seeding ===")
    total_rows = 0
    for i, ticker in enumerate(tickers):
        iv_rank = barchart_ranks.get(ticker)
        rows = seed_ticker(db, ticker, iv_rank)
        total_rows += rows
        if (i + 1) % 10 == 0:
            logger.info(f"  Seeded {i+1}/{len(tickers)} tickers — {total_rows} rows so far")
        time.sleep(0.1)  # avoid hammering yfinance

    coverage = db.get_db_coverage()
    logger.info(
        f"Seeding complete: {len(coverage)} tickers, {total_rows} total rows written. "
        f"Real IV accumulation will begin on next pipeline run."
    )


if __name__ == "__main__":
    main()
