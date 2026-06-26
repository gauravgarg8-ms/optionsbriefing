import numpy as np
import pandas as pd
import pytest
from datetime import date, timedelta

from quant.volatility import compute_hv20, compute_iv_rank, compute_iv_percentile, compute_iv_rv_ratio


class TestComputeHV20:
    def test_flat_prices_returns_zero(self):
        prices = pd.Series([100.0] * 30)
        assert compute_hv20(prices) == 0.0

    def test_known_returns_series(self):
        np.random.seed(0)
        n = 30
        log_rets = np.random.randn(n) * 0.01
        prices   = pd.Series(100.0 * np.exp(np.cumsum(log_rets)))
        result   = compute_hv20(prices)
        # pandas .std() uses ddof=1; match that in expected
        expected = float(np.std(log_rets[-20:], ddof=1) * np.sqrt(252))
        assert abs(result - expected) < 0.001

    def test_insufficient_data_returns_zero(self):
        prices = pd.Series([100.0] * 20)
        assert compute_hv20(prices) == 0.0

    def test_result_is_annualised(self):
        # Use genuinely varying returns so std > 0 (constant returns → std=0)
        np.random.seed(42)
        n      = 50
        rets   = np.random.randn(n) * 0.01   # ~1% daily vol
        prices = pd.Series(100.0 * np.exp(np.cumsum(rets)))
        result = compute_hv20(prices)
        assert result > 0.05   # annualised vol should be well above 5%


class TestComputeIVRank:
    def test_correct_calculation(self, memory_db):
        # Anchor to today so all rows fall within the 52-week window
        base = date.today() - timedelta(days=364)
        for i in range(365):
            d  = (base + timedelta(days=i)).isoformat()
            iv = 0.20 + (i / 364) * 0.40   # range 0.20 → 0.60
            memory_db.upsert_iv("AAPL", d, iv)
        rank = compute_iv_rank("AAPL", 0.50, db=memory_db)
        # IV Rank = (0.50 - 0.20) / (0.60 - 0.20) * 100 = 75
        assert abs(rank - 75.0) < 2.0

    def test_cold_start_returns_50(self, memory_db):
        rank = compute_iv_rank("UNKNOWN", 0.30, db=memory_db)
        assert rank == 50.0

    def test_fewer_than_30_rows_returns_50(self, memory_db):
        for i in range(10):
            memory_db.upsert_iv("TINY", f"2026-01-{i+1:02d}", 0.30)
        rank = compute_iv_rank("TINY", 0.35, db=memory_db)
        assert rank == 50.0

    def test_rank_clipped_0_to_100(self, memory_db):
        base = date(2025, 5, 29)
        # Insert varying IV (0.20 → 0.50) so high != low, giving a valid range
        for i in range(100):
            iv = 0.20 + (i / 99) * 0.30   # range 0.20 → 0.50
            memory_db.upsert_iv("CLIP", (base + timedelta(days=i)).isoformat(), round(iv, 4))
        # current_iv=1.0 is way above 52wk high (0.50) → rank should clip to 100
        rank = compute_iv_rank("CLIP", 1.0, db=memory_db)
        assert rank == 100.0


class TestComputeIVPercentile:
    def test_above_all_history_returns_100(self, memory_db):
        base = date(2025, 5, 29)
        for i in range(100):
            memory_db.upsert_iv("PCTTEST", (base + timedelta(days=i)).isoformat(), 0.25)
        pct = compute_iv_percentile("PCTTEST", 0.50, db=memory_db)
        assert pct == 100.0

    def test_below_all_history_returns_0(self, memory_db):
        base = date(2025, 5, 29)
        for i in range(100):
            memory_db.upsert_iv("PCTTEST2", (base + timedelta(days=i)).isoformat(), 0.40)
        pct = compute_iv_percentile("PCTTEST2", 0.10, db=memory_db)
        assert pct == 0.0

    def test_cold_start_returns_50(self, memory_db):
        pct = compute_iv_percentile("NODATA", 0.30, db=memory_db)
        assert pct == 50.0


class TestComputeIVRVRatio:
    def test_standard_calculation(self):
        assert abs(compute_iv_rv_ratio(0.42, 0.28) - 1.5) < 0.01

    def test_zero_hv20_returns_1(self):
        assert compute_iv_rv_ratio(0.30, 0.0) == 1.0

    def test_iv_equal_hv_returns_1(self):
        assert compute_iv_rv_ratio(0.25, 0.25) == 1.0
