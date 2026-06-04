import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path for all tests
sys.path.insert(0, str(Path(__file__).parent.parent))

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def synthetic_ohlcv():
    """
    200-day synthetic OHLCV DataFrame.
    Prices drift upward with random noise — realistic enough for ATR/RSI/support/resistance.
    """
    np.random.seed(42)
    n = 200
    dates = pd.bdate_range(end="2026-05-29", periods=n)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    close = np.clip(close, 10, None)
    high  = close + np.abs(np.random.randn(n)) * 0.8
    low   = close - np.abs(np.random.randn(n)) * 0.8
    open_ = close + np.random.randn(n) * 0.3
    volume = (np.random.randint(1_000_000, 5_000_000, n)).astype(float)
    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
    return df


@pytest.fixture
def flat_ohlcv():
    """200-day OHLCV with completely flat prices — for testing edge cases."""
    n = 200
    dates = pd.bdate_range(end="2026-05-29", periods=n)
    price = np.full(n, 100.0)
    df = pd.DataFrame(
        {"Open": price, "High": price + 0.01, "Low": price - 0.01,
         "Close": price, "Volume": np.full(n, 1_000_000.0)},
        index=dates,
    )
    return df


@pytest.fixture
def sample_tradier_chain():
    return json.loads((FIXTURES / "sample_chain_nvda.json").read_text())


@pytest.fixture
def sample_market_env():
    return json.loads((FIXTURES / "sample_market_env.json").read_text())


@pytest.fixture
def sample_top_candidates():
    return json.loads((FIXTURES / "sample_top_candidates.json").read_text())


@pytest.fixture
def memory_db(tmp_path):
    """
    In-memory SQLite DBManager instance for tests — no disk writes.
    Uses a temp file path to avoid shared state between tests.
    """
    from db.db_manager import DBManager
    db = DBManager(db_path=tmp_path / "test_iv.db")
    return db
