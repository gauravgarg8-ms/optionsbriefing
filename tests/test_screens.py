import pytest
from screening.screens import screen_high_iv, screen_earnings, screen_low_iv_trend, screen_bearish, merge_pools
from config import MAX_CANDIDATES_PER_POOL


def _make_candidate(ticker, iv_rank=60, price=50, avg_vol=5000,
                    above_50=True, above_200=True, rs_20d=2.0, sector="XLK",
                    implied_move=0.0):
    return {
        "ticker": ticker, "iv_rank": iv_rank, "price": price,
        "avg_options_vol": avg_vol, "above_50ma": above_50, "above_200ma": above_200,
        "rs_20d": rs_20d, "sector": sector, "implied_move_pct": implied_move,
    }


MARKET_ENV = {"leading_sectors": ["XLK", "XLI"], "lagging_sectors": ["XLU", "XLP"]}


class TestScreenHighIV:
    def test_passes_high_iv_candidates(self):
        candidates = [_make_candidate("AAPL", iv_rank=70, price=150, avg_vol=10000)]
        result = screen_high_iv(candidates)
        assert len(result) == 1
        assert result[0]["screen_pool"] == "high_iv"

    def test_excludes_low_iv_rank(self):
        candidates = [_make_candidate("XYZ", iv_rank=40)]
        assert screen_high_iv(candidates) == []

    def test_excludes_low_price(self):
        candidates = [_make_candidate("XYZ", iv_rank=70, price=10)]  # price ≤ $15
        assert screen_high_iv(candidates) == []

    def test_excludes_low_options_vol(self):
        candidates = [_make_candidate("XYZ", iv_rank=70, price=50, avg_vol=300)]
        assert screen_high_iv(candidates) == []

    def test_respects_max_pool_size(self):
        candidates = [_make_candidate(f"T{i}", iv_rank=70+i, price=50, avg_vol=5000)
                      for i in range(20)]
        result = screen_high_iv(candidates)
        assert len(result) <= MAX_CANDIDATES_PER_POOL["high_iv"]

    def test_sorted_by_iv_rank_descending(self):
        candidates = [
            _make_candidate("LOW",  iv_rank=55, price=50, avg_vol=5000),
            _make_candidate("HIGH", iv_rank=80, price=50, avg_vol=5000),
        ]
        result = screen_high_iv(candidates)
        assert result[0]["ticker"] == "HIGH"


class TestScreenEarnings:
    def _make_earn_cal(self, ticker, days_away):
        return [{"ticker": ticker, "date": "2026-06-04", "days_away": days_away}]

    def test_passes_earnings_candidate(self):
        candidates = [_make_candidate("NVDA", price=150, avg_vol=50000, implied_move=0.08)]
        result = screen_earnings(candidates, self._make_earn_cal("NVDA", 5))
        assert len(result) == 1
        assert result[0]["screen_pool"] == "earnings"

    def test_excludes_earnings_outside_7_days(self):
        candidates = [_make_candidate("NVDA", price=150, avg_vol=50000, implied_move=0.08)]
        result = screen_earnings(candidates, self._make_earn_cal("NVDA", 10))
        assert len(result) == 0

    def test_excludes_low_implied_move(self):
        candidates = [_make_candidate("NVDA", price=150, avg_vol=50000, implied_move=0.03)]
        result = screen_earnings(candidates, self._make_earn_cal("NVDA", 5))
        assert len(result) == 0

    def test_excludes_low_options_vol(self):
        candidates = [_make_candidate("NVDA", price=150, avg_vol=800, implied_move=0.08)]
        result = screen_earnings(candidates, self._make_earn_cal("NVDA", 5))
        assert len(result) == 0

    def test_different_ticker_not_matched(self):
        candidates = [_make_candidate("AAPL", price=150, avg_vol=50000, implied_move=0.08)]
        result = screen_earnings(candidates, self._make_earn_cal("NVDA", 5))
        assert len(result) == 0


class TestScreenLowIVTrend:
    def test_passes_low_iv_trend_candidate(self):
        candidates = [_make_candidate("MSFT", iv_rank=20, above_50=True,
                                       above_200=True, rs_20d=3.0, sector="XLK")]
        result = screen_low_iv_trend(candidates, MARKET_ENV)
        assert len(result) == 1
        assert result[0]["screen_pool"] == "low_iv_trend"

    def test_excludes_high_iv_rank(self):
        candidates = [_make_candidate("MSFT", iv_rank=40, sector="XLK")]
        assert screen_low_iv_trend(candidates, MARKET_ENV) == []

    def test_excludes_below_50ma(self):
        candidates = [_make_candidate("MSFT", iv_rank=20, above_50=False, sector="XLK")]
        assert screen_low_iv_trend(candidates, MARKET_ENV) == []

    def test_excludes_negative_rs(self):
        candidates = [_make_candidate("MSFT", iv_rank=20, above_50=True,
                                       above_200=True, rs_20d=-1.0, sector="XLK")]
        assert screen_low_iv_trend(candidates, MARKET_ENV) == []

    def test_excludes_non_leading_sector(self):
        candidates = [_make_candidate("XOM", iv_rank=20, above_50=True,
                                       above_200=True, rs_20d=2.0, sector="XLE")]
        assert screen_low_iv_trend(candidates, MARKET_ENV) == []


class TestScreenBearish:
    def test_passes_bearish_candidate(self):
        candidates = [_make_candidate("D", above_50=False, rs_20d=-3.0, sector="XLU")]
        result = screen_bearish(candidates, MARKET_ENV)
        assert len(result) == 1
        assert result[0]["screen_pool"] == "bearish"

    def test_excludes_above_50ma(self):
        candidates = [_make_candidate("D", above_50=True, rs_20d=-3.0, sector="XLU")]
        assert screen_bearish(candidates, MARKET_ENV) == []

    def test_excludes_positive_rs(self):
        candidates = [_make_candidate("D", above_50=False, rs_20d=2.0, sector="XLU")]
        assert screen_bearish(candidates, MARKET_ENV) == []

    def test_excludes_non_lagging_sector(self):
        candidates = [_make_candidate("D", above_50=False, rs_20d=-2.0, sector="XLK")]
        assert screen_bearish(candidates, MARKET_ENV) == []


class TestMergePools:
    def test_deduplicates_by_ticker(self):
        c1 = {"ticker": "AAPL", "screen_pool": "high_iv"}
        c2 = {"ticker": "AAPL", "screen_pool": "earnings"}  # duplicate
        c3 = {"ticker": "NVDA", "screen_pool": "high_iv"}
        result = merge_pools([[c1, c3], [c2]])
        tickers = [r["ticker"] for r in result]
        assert tickers.count("AAPL") == 1
        assert len(result) == 2

    def test_first_pool_wins_on_duplicate(self):
        c1 = {"ticker": "AAPL", "screen_pool": "high_iv"}
        c2 = {"ticker": "AAPL", "screen_pool": "earnings"}
        result = merge_pools([[c1], [c2]])
        assert result[0]["screen_pool"] == "high_iv"
