import sqlite3
from pathlib import Path
from datetime import date, timedelta
from loguru import logger
from errors import ErrorCode

DEFAULT_DB_PATH = Path(__file__).parent / "iv_history.db"


class DBManager:
    def __init__(self, db_path=None):
        self.db_path = str(db_path or DEFAULT_DB_PATH)
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        try:
            with self._connect() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS iv_history (
                        ticker TEXT NOT NULL,
                        date   TEXT NOT NULL,
                        iv30   REAL NOT NULL,
                        PRIMARY KEY (ticker, date)
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_ticker ON iv_history(ticker)")
        except sqlite3.Error as e:
            logger.error(f"[{ErrorCode.E1012}] Failed to initialise iv_history DB: {e}")
            raise

    def upsert_iv(self, ticker: str, as_of_date: str, iv30: float) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO iv_history (ticker, date, iv30) VALUES (?, ?, ?)",
                    (ticker.upper(), as_of_date, float(iv30)),
                )
        except sqlite3.Error as e:
            logger.error(f"[{ErrorCode.E1012}] upsert_iv failed for {ticker} {as_of_date}: {e}")
            raise

    def get_iv_history(self, ticker: str, days: int = 365) -> list[tuple[str, float]]:
        cutoff = (date.today() - timedelta(days=days)).isoformat()
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT date, iv30 FROM iv_history WHERE ticker=? AND date>=? ORDER BY date",
                    (ticker.upper(), cutoff),
                ).fetchall()
            return rows
        except sqlite3.Error as e:
            logger.error(f"[{ErrorCode.E1012}] get_iv_history failed for {ticker}: {e}")
            return []

    def get_52wk_high_low(self, ticker: str) -> tuple[float | None, float | None]:
        rows = self.get_iv_history(ticker, days=365)
        if len(rows) < 30:
            logger.warning(
                f"[{ErrorCode.E1011}] IV history cold start for {ticker}: "
                f"{len(rows)} days available (need 30+) — using default IV Rank 50"
            )
            return None, None
        values = [iv for _, iv in rows]
        return max(values), min(values)

    def get_db_coverage(self) -> dict[str, int]:
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT ticker, COUNT(*) as days FROM iv_history GROUP BY ticker"
                ).fetchall()
            return {ticker: days for ticker, days in rows}
        except sqlite3.Error as e:
            logger.error(f"[{ErrorCode.E1012}] get_db_coverage failed: {e}")
            return {}

    def get_real_iv_days(self, ticker: str) -> int:
        """Returns count of rows tagged as real (non-proxy) IV data."""
        try:
            with self._connect() as conn:
                # Real data flagged by source column if present; fall back to total count
                cols = [r[1] for r in conn.execute("PRAGMA table_info(iv_history)").fetchall()]
                if "source" in cols:
                    count = conn.execute(
                        "SELECT COUNT(*) FROM iv_history WHERE ticker=? AND source='real'",
                        (ticker.upper(),),
                    ).fetchone()[0]
                else:
                    count = conn.execute(
                        "SELECT COUNT(*) FROM iv_history WHERE ticker=?",
                        (ticker.upper(),),
                    ).fetchone()[0]
            return count
        except sqlite3.Error as e:
            logger.error(f"[{ErrorCode.E1012}] get_real_iv_days failed for {ticker}: {e}")
            return 0

    def upsert_iv_with_source(self, ticker: str, as_of_date: str, iv30: float, source: str = "real") -> None:
        """Upsert a single row with source tag."""
        self.bulk_upsert_iv_proxy(ticker, [(as_of_date, iv30)], source=source)

    def bulk_upsert_iv_proxy(self, ticker: str, rows: list[tuple[str, float]],
                             source: str = "proxy") -> None:
        """
        Insert multiple (date, iv30) rows in a single connection+transaction.
        Replaces 252 individual upsert_iv_with_source calls with one executemany,
        avoiding file-descriptor exhaustion when seeding the full universe cold.
        """
        if not rows:
            return
        try:
            with self._connect() as conn:
                cols = [r[1] for r in conn.execute("PRAGMA table_info(iv_history)").fetchall()]
                if "source" not in cols:
                    conn.execute("ALTER TABLE iv_history ADD COLUMN source TEXT DEFAULT 'real'")
                conn.executemany(
                    "INSERT OR REPLACE INTO iv_history (ticker, date, iv30, source) VALUES (?, ?, ?, ?)",
                    [(ticker.upper(), d, float(iv), source) for d, iv in rows],
                )
        except sqlite3.Error as e:
            logger.error(f"[{ErrorCode.E1012}] bulk_upsert_iv_proxy failed for {ticker}: {e}")
