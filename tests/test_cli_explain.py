"""P8 — `squeeze-hunter explain` reads decision logs."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from typer.testing import CliRunner

from squeeze_hunter.cli import app
from squeeze_hunter.execution.decision_log import DecisionLog
from squeeze_hunter.execution.decisions import EntryDecision


def test_explain_reads_a_backtest_decisions_file(tmp_path: Path) -> None:
    log = DecisionLog()
    log.record(
        datetime(2025, 4, 21, tzinfo=UTC),
        [EntryDecision("HTZ", 9.4, "CAR", False, "insufficient_liquidity", 0.0)],
        source="backtest",
    )
    path = tmp_path / "decisions.parquet"
    log.to_frame().to_parquet(path, index=False)
    result = CliRunner().invoke(
        app, ["explain", "--ticker", "HTZ", "--date", "2025-04-21", "--source", str(path)]
    )
    assert result.exit_code == 0, result.output
    assert "insufficient_liquidity" in result.output


def test_explain_with_no_log_says_so(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["explain", "--source", str(tmp_path / "missing.parquet")])
    assert result.exit_code == 0
    assert "no decisions" in result.output
