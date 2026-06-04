import json
from unittest.mock import patch, MagicMock

import pytest
import responses as resp_lib

from config import CNN_FG_URL, CBOE_PC_URL, FINNHUB_BASE, NEWS_API_BASE
from data.sentiment_data import (
    fetch_fear_greed, fetch_put_call_ratio, fetch_market_news,
    fetch_sector_news, classify_market_sentiment, _classify_headline_sentiment,
)

CNN_MOCK_RESPONSE = {
    "fear_and_greed": {"score": 61, "rating": "greed", "timestamp": "2026-05-29"}
}

CBOE_CSV_MOCK = (
    "DATE,PUT/CALL RATIO,EQUITY PUT/CALL RATIO,INDEX PUT/CALL RATIO,VIX PUT/CALL RATIO\n"
    "05/28/2026,0.91,0.82,1.12,0.45\n"
    "05/29/2026,0.89,0.79,1.08,0.42\n"
)


class TestFetchFearGreed:
    @resp_lib.activate
    def test_parses_score_and_label(self):
        resp_lib.add(resp_lib.GET, CNN_FG_URL, json=CNN_MOCK_RESPONSE, status=200)
        result = fetch_fear_greed()
        assert result["score"] == 61
        assert result["label"] == "Greed"

    @resp_lib.activate
    def test_failure_returns_neutral_50(self):
        """When both CNN and VIX proxy fail, defaults to score=50 Neutral."""
        resp_lib.add(resp_lib.GET, CNN_FG_URL, json={}, status=404)
        with patch("data.sentiment_data._FG_CACHE", {}), \
             patch("data.sentiment_data._FG_CACHE_DATE", ""), \
             patch("data.sentiment_data.yf.download", side_effect=Exception("no VIX")):
            result = fetch_fear_greed()
        assert result["score"] == 50
        assert result["label"] == "Neutral"

    @resp_lib.activate
    def test_vix_proxy_used_when_cnn_fails(self):
        """When CNN fails but VIX is available, VIX proxy score is used."""
        import pandas as pd
        import numpy as np
        resp_lib.add(resp_lib.GET, CNN_FG_URL, json={}, status=404)
        dates  = pd.bdate_range(end="2026-05-29", periods=2)
        vix_df = pd.DataFrame({"Close": [30.0, 30.0]}, index=dates)  # VIX=30 → Extreme Fear
        with patch("data.sentiment_data._FG_CACHE", {}), \
             patch("data.sentiment_data._FG_CACHE_DATE", ""), \
             patch("data.sentiment_data.yf.download", return_value=vix_df):
            result = fetch_fear_greed()
        assert result["score"] == 20
        assert result["label"] == "Extreme Fear"
        assert result["source"] == "vix_proxy"


class TestFetchPutCallRatio:
    @resp_lib.activate
    def test_parses_equity_ratio(self):
        from datetime import date
        today = date.today().isoformat()
        resp_lib.add(resp_lib.GET, f"{CBOE_PC_URL}{today}_options_volume.csv",
                     body=CBOE_CSV_MOCK, status=200, content_type="text/csv")
        result = fetch_put_call_ratio()
        assert isinstance(result, float)
        assert 0.3 <= result <= 2.0

    @resp_lib.activate
    def test_failure_returns_default_0_9(self):
        # All three sources blocked (CBOE 403, SPY Ticker, VIX download) → hard default 0.9
        with patch("data.sentiment_data.yf.Ticker",   side_effect=Exception("no SPY")), \
             patch("data.sentiment_data.yf.download",  side_effect=Exception("no VIX")):
            result = fetch_put_call_ratio()
        assert result == 0.9

    @resp_lib.activate
    def test_network_error_returns_default(self):
        # All three sources blocked → hard default 0.9
        with patch("data.sentiment_data.yf.Ticker",   side_effect=Exception("no SPY")), \
             patch("data.sentiment_data.yf.download",  side_effect=Exception("no VIX")):
            result = fetch_put_call_ratio()
        assert result == 0.9


class TestClassifyHeadlineSentiment:
    def test_bullish_keywords(self):
        assert _classify_headline_sentiment("Stock surges on record earnings beat") == "BULLISH"

    def test_bearish_keywords(self):
        assert _classify_headline_sentiment("Tariff concerns cause market drop") == "BEARISH"

    def test_neutral_headline(self):
        assert _classify_headline_sentiment("Company announces quarterly results") == "NEUTRAL"

    def test_empty_headline(self):
        assert _classify_headline_sentiment("") == "NEUTRAL"


class TestFetchSectorNews:
    @resp_lib.activate
    def test_adverse_keyword_sets_flag(self):
        articles = [
            {"headline": "US imposes new tariff on tech imports", "source": "Reuters",
             "summary": "Technology sector faces headwinds from new tariff policy"}
        ]
        resp_lib.add(resp_lib.GET, f"{FINNHUB_BASE}/news", json=articles, status=200)
        result = fetch_sector_news(["XLK"])
        assert result["XLK"]["adverse_headline"] is True

    @resp_lib.activate
    def test_no_adverse_keyword_false_flag(self):
        articles = [
            {"headline": "AI chip demand remains strong", "source": "CNBC",
             "summary": "Technology stocks continue to outperform"}
        ]
        resp_lib.add(resp_lib.GET, f"{FINNHUB_BASE}/news", json=articles, status=200)
        result = fetch_sector_news(["XLK"])
        assert result["XLK"]["adverse_headline"] is False

    @resp_lib.activate
    def test_failure_returns_defaults(self):
        resp_lib.add(resp_lib.GET, f"{FINNHUB_BASE}/news", body=Exception("network"))
        result = fetch_sector_news(["XLK", "XLE"])
        assert "XLK" in result
        assert result["XLK"]["adverse_headline"] is False


class TestClassifyMarketSentiment:
    def test_bullish_signals_produce_bullish(self):
        fg     = {"score": 70, "label": "Greed"}
        news   = [{"sentiment": "BULLISH"}] * 3
        result = classify_market_sentiment(fg, 0.75, news)
        assert result["market_sentiment"] == "BULLISH"

    def test_extreme_greed_adds_warning(self):
        fg     = {"score": 80, "label": "Extreme Greed"}
        news   = [{"sentiment": "BULLISH"}] * 3
        result = classify_market_sentiment(fg, 0.65, news)
        assert "extreme_greed_caution_on_debit" in result["warning_flags"]

    def test_extreme_fear_adds_warning(self):
        fg     = {"score": 20, "label": "Extreme Fear"}
        news   = [{"sentiment": "BEARISH"}] * 3
        result = classify_market_sentiment(fg, 1.3, news)
        assert "extreme_fear_credit_on_quality" in result["warning_flags"]

    def test_high_put_call_adds_contrarian_flag(self):
        fg     = {"score": 50, "label": "Neutral"}
        news   = []
        result = classify_market_sentiment(fg, 1.4, news)
        assert "high_put_call_contrarian_bullish" in result["warning_flags"]

    def test_bearish_signals_produce_bearish(self):
        fg     = {"score": 30, "label": "Fear"}
        news   = [{"sentiment": "BEARISH"}] * 5
        result = classify_market_sentiment(fg, 0.65, news)
        assert result["market_sentiment"] == "BEARISH"
