import pytest
from scenario_classifier import classify_scenarios

BASE_ENV = {
    "vix": 18.4,
    "leading_sectors": ["XLK", "XLI"],
    "lagging_sectors": ["XLU", "XLP"],
    "macro_events": [],
    "geopolitical_risk_flag": False,
}


def _env(**kwargs):
    return {**BASE_ENV, **kwargs}


def _candidate(ticker, earnings_days=99, earnings_yesterday=False):
    return {"ticker": ticker, "earnings_days_away": earnings_days,
            "earnings_yesterday": earnings_yesterday}


def _codes(scenarios):
    return [s["code"] for s in scenarios]


class TestS1EarningsTrade:
    def test_candidate_within_7d_triggers_s1(self):
        candidates = [_candidate("NVDA", earnings_days=5)]
        result = classify_scenarios(_env(), candidates)
        assert "S1" in _codes(result)

    def test_s1_contains_ticker(self):
        candidates = [_candidate("NVDA", earnings_days=5)]
        result = classify_scenarios(_env(), candidates)
        s1 = next(s for s in result if s["code"] == "S1")
        assert "NVDA" in s1["tickers"]

    def test_no_earnings_no_s1(self):
        candidates = [_candidate("NVDA", earnings_days=30)]
        result = classify_scenarios(_env(), candidates)
        assert "S1" not in _codes(result)


class TestS2MacroEvent:
    def test_macro_within_7d_triggers_s2(self):
        env = _env(macro_events=[{"event": "CPI", "days_away": 4, "alert": "WATCH"}])
        result = classify_scenarios(env, [])
        assert "S2" in _codes(result)

    def test_macro_at_day_7_triggers_s2(self):
        env = _env(macro_events=[{"event": "FOMC", "days_away": 7, "alert": "WATCH"}])
        result = classify_scenarios(env, [])
        assert "S2" in _codes(result)

    def test_macro_at_day_8_no_s2(self):
        env = _env(macro_events=[{"event": "GDP", "days_away": 8, "alert": "MONITOR"}])
        result = classify_scenarios(env, [])
        assert "S2" not in _codes(result)

    def test_no_macros_no_s2(self):
        result = classify_scenarios(_env(macro_events=[]), [])
        assert "S2" not in _codes(result)


class TestS3TrendingNoContext:
    def test_vix_18_no_earn_no_macro_triggers_s3(self):
        result = classify_scenarios(_env(vix=18.0, macro_events=[]), [])
        assert "S3" in _codes(result)

    def test_vix_26_no_s3(self):
        result = classify_scenarios(_env(vix=26.0), [])
        assert "S3" not in _codes(result)

    def test_earnings_present_no_s3(self):
        candidates = [_candidate("NVDA", earnings_days=5)]
        result = classify_scenarios(_env(vix=18.0), candidates)
        assert "S3" not in _codes(result)


class TestS4HighVIX:
    def test_vix_above_25_triggers_s4(self):
        result = classify_scenarios(_env(vix=28.0), [])
        assert "S4" in _codes(result)

    def test_vix_exactly_25_no_s4(self):
        result = classify_scenarios(_env(vix=25.0), [])
        assert "S4" not in _codes(result)

    def test_vix_18_no_s4(self):
        result = classify_scenarios(_env(vix=18.0), [])
        assert "S4" not in _codes(result)


class TestS5SectorRotation:
    def test_s5_always_present(self):
        result = classify_scenarios(_env(), [])
        assert "S5" in _codes(result)

    def test_s5_contains_leading_lagging(self):
        result = classify_scenarios(_env(), [])
        s5 = next(s for s in result if s["code"] == "S5")
        assert "XLK" in s5["leading"]
        assert "XLU" in s5["lagging"]


class TestS6Geopolitical:
    def test_geo_flag_triggers_s6(self):
        result = classify_scenarios(_env(geopolitical_risk_flag=True), [])
        assert "S6" in _codes(result)

    def test_no_geo_flag_no_s6(self):
        result = classify_scenarios(_env(geopolitical_risk_flag=False), [])
        assert "S6" not in _codes(result)


class TestS7PostEarnings:
    def test_earnings_yesterday_triggers_s7(self):
        candidates = [_candidate("AAPL", earnings_yesterday=True)]
        result = classify_scenarios(_env(), candidates)
        assert "S7" in _codes(result)

    def test_no_earnings_yesterday_no_s7(self):
        candidates = [_candidate("AAPL", earnings_yesterday=False)]
        result = classify_scenarios(_env(), candidates)
        assert "S7" not in _codes(result)


class TestMultipleScenarios:
    def test_s1_and_s2_both_active(self):
        env        = _env(macro_events=[{"event": "CPI", "days_away": 3}])
        candidates = [_candidate("NVDA", earnings_days=4)]
        result     = classify_scenarios(env, candidates)
        codes      = _codes(result)
        assert "S1" in codes
        assert "S2" in codes

    def test_s4_and_s5_both_active(self):
        result = classify_scenarios(_env(vix=30.0), [])
        codes  = _codes(result)
        assert "S4" in codes
        assert "S5" in codes
