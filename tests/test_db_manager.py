import sqlite3
from datetime import date, timedelta
from unittest.mock import patch

import pytest


class TestDBManagerUpsertRetrieve:
    def test_upsert_and_retrieve(self, memory_db):
        memory_db.upsert_iv("AAPL", "2026-01-15", 0.32)
        rows = memory_db.get_iv_history("AAPL", days=365)
        assert len(rows) == 1
        assert rows[0][0] == "2026-01-15"
        assert abs(rows[0][1] - 0.32) < 1e-6

    def test_upsert_overwrites_same_primary_key(self, memory_db):
        memory_db.upsert_iv("AAPL", "2026-01-15", 0.32)
        memory_db.upsert_iv("AAPL", "2026-01-15", 0.45)  # overwrite
        rows = memory_db.get_iv_history("AAPL", days=365)
        assert len(rows) == 1
        assert abs(rows[0][1] - 0.45) < 1e-6

    def test_ticker_normalised_to_uppercase(self, memory_db):
        memory_db.upsert_iv("aapl", "2026-01-15", 0.30)
        rows = memory_db.get_iv_history("AAPL", days=365)
        assert len(rows) == 1

    def test_multiple_tickers_isolated(self, memory_db):
        memory_db.upsert_iv("AAPL", "2026-01-15", 0.30)
        memory_db.upsert_iv("MSFT", "2026-01-15", 0.25)
        assert len(memory_db.get_iv_history("AAPL")) == 1
        assert len(memory_db.get_iv_history("MSFT")) == 1


class TestGet52wkHighLow:
    def test_cold_start_returns_none_with_warning(self, memory_db):
        for i in range(10):
            memory_db.upsert_iv("NVDA", f"2026-01-{i+1:02d}", 0.30 + i * 0.01)
        high, low = memory_db.get_52wk_high_low("NVDA")
        assert high is None
        assert low is None

    def test_sufficient_data_returns_correct_values(self, memory_db):
        # Insert 365 rows anchored to today so the 52-week window always covers them
        base = date.today() - timedelta(days=364)
        for i in range(365):
            d = (base + timedelta(days=i)).isoformat()
            iv = 0.20 + (i % 100) * 0.003  # oscillates between 0.20 and 0.497
            memory_db.upsert_iv("SPY", d, round(iv, 4))
        # Overwrite two rows within the window with known extremes
        memory_db.upsert_iv("SPY", (base + timedelta(days=10)).isoformat(), 0.70)  # known max
        memory_db.upsert_iv("SPY", (base + timedelta(days=11)).isoformat(), 0.05)  # known min
        high, low = memory_db.get_52wk_high_low("SPY")
        assert high is not None
        assert low is not None
        assert high >= 0.70 - 1e-4
        assert low <= 0.05 + 1e-4

    def test_unknown_ticker_returns_none(self, memory_db):
        high, low = memory_db.get_52wk_high_low("UNKNOWN")
        assert high is None
        assert low is None


class TestDBCoverage:
    def test_get_db_coverage(self, memory_db):
        for i in range(5):
            memory_db.upsert_iv("AAPL", f"2026-01-{i+1:02d}", 0.30)
        for i in range(3):
            memory_db.upsert_iv("MSFT", f"2026-01-{i+1:02d}", 0.25)
        coverage = memory_db.get_db_coverage()
        assert coverage.get("AAPL") == 5
        assert coverage.get("MSFT") == 3


class TestUpsertWithSource:
    def test_source_tag_stored_and_queryable(self, memory_db):
        memory_db.upsert_iv_with_source("AAPL", "2026-01-15", 0.32, source="proxy")
        memory_db.upsert_iv_with_source("AAPL", "2026-01-16", 0.34, source="real")
        real_days = memory_db.get_real_iv_days("AAPL")
        assert real_days == 1
