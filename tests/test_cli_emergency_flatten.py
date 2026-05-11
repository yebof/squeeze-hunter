"""R7.C8: emergency-flatten command must validate --mode before any broker work."""

from typer.testing import CliRunner

from squeeze_hunter.cli import app

runner = CliRunner()


def test_emergency_flatten_rejects_invalid_mode() -> None:
    """R7.C8 regression: --mode must be 'paper' or 'live'. Other values
    (including 'sim' or typos) used to fall silently into the live IBKR
    branch — dangerous on a real account.
    """
    result = runner.invoke(app, ["emergency-flatten", "--mode", "sim", "--confirm"])
    assert result.exit_code == 2
    assert "paper" in result.output
    assert "live" in result.output


def test_emergency_flatten_requires_confirm() -> None:
    """A separate guard: without --confirm the command must refuse even if
    --mode is valid. R7.C8 keeps this behavior.
    """
    result = runner.invoke(app, ["emergency-flatten", "--mode", "paper"])
    assert result.exit_code == 2
