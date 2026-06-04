"""
Manual IV history importer — loads a CSV of real IV data into the DB.

Usage:
    .venv/bin/python db/import_iv_csv.py path/to/iv_data.csv

CSV format (header row required):
    ticker,date,iv30

    - ticker : stock symbol  (e.g. AAPL)
    - date   : YYYY-MM-DD    (e.g. 2026-05-01)
    - iv30   : implied vol   — accept EITHER decimal (0.2543) OR percent (25.43)
               Values > 2.0 are assumed to be percentages and divided by 100.

Example rows:
    AAPL,2026-05-01,25.43       <- percent form   → stored as 0.2543
    MSFT,2026-05-01,0.2310      <- decimal form   → stored as 0.2310
    NVDA,2026-05-01,68.7        <- percent form   → stored as 0.6870

The script tags all imported rows as source='real' and skips duplicates
(same ticker+date already in DB with source='real').
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from db.db_manager import DBManager


def _parse_iv30(raw: str) -> float | None:
    """Parse iv30 from string; normalise percent → decimal if value > 2.0."""
    try:
        val = float(raw.strip().replace("%", ""))
        if val <= 0:
            return None
        return round(val / 100.0, 6) if val > 2.0 else round(val, 6)
    except ValueError:
        return None


def import_csv(csv_path: str) -> None:
    path = Path(csv_path)
    if not path.exists():
        logger.error(f"File not found: {csv_path}")
        sys.exit(1)

    db = DBManager()

    # Load existing real rows so we can skip duplicates
    existing: set[tuple[str, str]] = set()
    try:
        import sqlite3
        with sqlite3.connect(str(Path(__file__).parent / "iv_history.db")) as conn:
            rows = conn.execute(
                "SELECT ticker, date FROM iv_history WHERE source='real'"
            ).fetchall()
            existing = {(r[0].upper(), r[1]) for r in rows}
    except Exception:
        pass  # if it fails, we'll just overwrite via upsert

    written = skipped = errors = 0

    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        # Normalise column names: strip whitespace, lowercase
        if reader.fieldnames is None:
            logger.error("CSV has no header row")
            sys.exit(1)
        reader.fieldnames = [c.strip().lower() for c in reader.fieldnames]

        required = {"ticker", "date", "iv30"}
        missing  = required - set(reader.fieldnames)
        if missing:
            logger.error(f"CSV missing required columns: {missing}. Found: {reader.fieldnames}")
            sys.exit(1)

        for i, row in enumerate(reader, start=2):
            ticker = row.get("ticker", "").strip().upper()
            date   = row.get("date", "").strip()
            iv_raw = row.get("iv30", "").strip()

            if not ticker or not date or not iv_raw:
                logger.warning(f"Row {i}: empty field(s) — skipping")
                errors += 1
                continue

            # Validate date format
            try:
                from datetime import datetime
                datetime.strptime(date, "%Y-%m-%d")
            except ValueError:
                logger.warning(f"Row {i}: invalid date '{date}' — expected YYYY-MM-DD")
                errors += 1
                continue

            iv30 = _parse_iv30(iv_raw)
            if iv30 is None:
                logger.warning(f"Row {i}: invalid iv30 '{iv_raw}' for {ticker} — skipping")
                errors += 1
                continue

            if (ticker, date) in existing:
                skipped += 1
                continue

            db.upsert_iv_with_source(ticker, date, iv30, source="real")
            existing.add((ticker, date))
            written += 1

    logger.info(
        f"Import complete: {written} rows written, {skipped} skipped (already real), "
        f"{errors} errors"
    )
    print(f"\nDone — {written} rows written, {skipped} skipped, {errors} errors")
    print(f"Log: logs/  (check for warnings)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    import_csv(sys.argv[1])
