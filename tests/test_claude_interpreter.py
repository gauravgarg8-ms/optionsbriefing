import json
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import anthropic
import pytest

from claude_interpreter import run_claude_briefing, run_with_retry, SYSTEM_PROMPT

FIXTURES = Path(__file__).parent / "fixtures"

SAMPLE_PAYLOAD = json.loads((FIXTURES / "sample_top_candidates.json").read_text())


def _make_mock_message(text="Daily briefing content here."):
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    return msg


class TestRunClaudeBriefing:
    @patch("claude_interpreter.anthropic.Anthropic")
    def test_calls_claude_with_system_prompt(self, mock_anthropic_cls):
        mock_client  = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_mock_message()

        result = run_claude_briefing(SAMPLE_PAYLOAD)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == SYSTEM_PROMPT
        assert "Generate today's daily options briefing" in call_kwargs["messages"][0]["content"]

    @patch("claude_interpreter.anthropic.Anthropic")
    def test_returns_text_from_response(self, mock_anthropic_cls):
        mock_client  = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_mock_message("Briefing result text")

        result = run_claude_briefing(SAMPLE_PAYLOAD)
        assert result == "Briefing result text"

    @patch("claude_interpreter.time.sleep")
    @patch("claude_interpreter.anthropic.Anthropic")
    def test_retries_on_rate_limit(self, mock_anthropic_cls, mock_sleep):
        mock_client  = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = [
            anthropic.RateLimitError.__new__(anthropic.RateLimitError),
            _make_mock_message("Success after retry"),
        ]

        result = run_claude_briefing(SAMPLE_PAYLOAD)
        assert result == "Success after retry"
        mock_sleep.assert_called_once()   # slept once between attempts

    @patch("claude_interpreter.time.sleep")
    @patch("claude_interpreter.anthropic.Anthropic")
    def test_raises_after_max_retries(self, mock_anthropic_cls, mock_sleep):
        mock_client  = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.side_effect = anthropic.RateLimitError.__new__(
            anthropic.RateLimitError
        )

        with pytest.raises(RuntimeError, match="Claude API failed"):
            run_claude_briefing(SAMPLE_PAYLOAD)

    @patch("claude_interpreter.anthropic.Anthropic")
    def test_no_trade_day_payload_still_calls_claude(self, mock_anthropic_cls):
        mock_client  = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_mock_message("NO-TRADE DAY report")

        payload = {**SAMPLE_PAYLOAD, "no_trade_day": True, "candidates": []}
        result  = run_claude_briefing(payload)
        assert result == "NO-TRADE DAY report"
        mock_client.messages.create.assert_called_once()

    @patch("claude_interpreter.anthropic.Anthropic")
    def test_json_payload_included_in_user_message(self, mock_anthropic_cls):
        mock_client  = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_mock_message()

        run_claude_briefing(SAMPLE_PAYLOAD)

        call_content = mock_client.messages.create.call_args[1]["messages"][0]["content"]
        # Verify key fields from the payload appear in the user message
        assert "NVDA" in call_content or "candidates" in call_content


class TestRunWithRetry:
    @patch("claude_interpreter.run_claude_briefing")
    def test_loads_json_and_calls_briefing(self, mock_briefing, tmp_path):
        payload_path = tmp_path / "top_candidates.json"
        payload_path.write_text(json.dumps(SAMPLE_PAYLOAD))
        mock_briefing.return_value = "Briefing text"

        result = run_with_retry(str(payload_path))
        assert result == "Briefing text"
        mock_briefing.assert_called_once_with(SAMPLE_PAYLOAD)
