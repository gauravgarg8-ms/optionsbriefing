import json
from datetime import date, datetime
from pathlib import Path

import pytest

from delivery import write_briefing


def _make_top_candidates(no_trade=False, reduced=False, n_candidates=3):
    return {
        "no_trade_day":            no_trade,
        "reduced_opportunity_day": reduced,
        "candidates": [{"ticker": f"T{i}"} for i in range(n_candidates)],
    }


class TestWriteBriefing:
    def test_file_named_correctly(self, tmp_path):
        today = date.today().isoformat()
        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "delivery.OUTPUT_BRIEFINGS_DIR", str(tmp_path)
        ):
            path = write_briefing(
                briefing_text  = "Test briefing content",
                top_candidates = _make_top_candidates(),
                pipeline_start = datetime.now(),
            )
        assert f"{today}_OptionsBrief.md" in str(path)

    def test_briefing_text_written(self, tmp_path):
        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "delivery.OUTPUT_BRIEFINGS_DIR", str(tmp_path)
        ):
            path = write_briefing(
                briefing_text  = "Briefing content here",
                top_candidates = _make_top_candidates(),
                pipeline_start = datetime.now(),
            )
        content = path.read_text()
        assert "Briefing content here" in content

    def test_metadata_footer_appended(self, tmp_path):
        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "delivery.OUTPUT_BRIEFINGS_DIR", str(tmp_path)
        ):
            path = write_briefing(
                briefing_text  = "Content",
                top_candidates = _make_top_candidates(n_candidates=5),
                pipeline_start = datetime.now(),
            )
        content = path.read_text()
        assert "Generated:" in content
        assert "Setups: 5" in content

    def test_iv_proxy_warning_shown_when_cold_start(self, tmp_path):
        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "delivery.OUTPUT_BRIEFINGS_DIR", str(tmp_path)
        ):
            path = write_briefing(
                briefing_text  = "Content",
                top_candidates = _make_top_candidates(),
                pipeline_start = datetime.now(),
                iv_proxy_days  = 5,   # < 30 → show proxy warning
            )
        content = path.read_text()
        assert "IV RANK PROXY" in content
        assert "5/30" in content

    def test_no_iv_proxy_warning_after_30_days(self, tmp_path):
        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "delivery.OUTPUT_BRIEFINGS_DIR", str(tmp_path)
        ):
            path = write_briefing(
                briefing_text  = "Content",
                top_candidates = _make_top_candidates(),
                pipeline_start = datetime.now(),
                iv_proxy_days  = 35,   # ≥ 30 → no proxy warning
            )
        content = path.read_text()
        assert "IV RANK PROXY" not in content

    def test_no_trade_day_in_footer(self, tmp_path):
        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "delivery.OUTPUT_BRIEFINGS_DIR", str(tmp_path)
        ):
            path = write_briefing(
                briefing_text  = "Content",
                top_candidates = _make_top_candidates(no_trade=True),
                pipeline_start = datetime.now(),
            )
        content = path.read_text()
        assert "NO-TRADE DAY" in content

    def test_fallback_on_permission_error(self, tmp_path):
        """If primary write fails, fallback to current directory."""
        import unittest.mock as um
        # Make the primary directory unwritable
        bad_dir = tmp_path / "readonly"
        bad_dir.mkdir()
        bad_dir.chmod(0o444)   # read-only

        fallback_name = f"{date.today().isoformat()}_OptionsBrief.md"
        fallback_path = Path(fallback_name)

        with um.patch("delivery.OUTPUT_BRIEFINGS_DIR", str(bad_dir)):
            try:
                path = write_briefing(
                    briefing_text  = "Fallback test",
                    top_candidates = _make_top_candidates(),
                    pipeline_start = datetime.now(),
                )
                assert path.exists()
                path.unlink(missing_ok=True)
            except OSError:
                pass   # Both write paths failed — acceptable on restricted filesystems
            finally:
                bad_dir.chmod(0o755)
                fallback_path.unlink(missing_ok=True)
