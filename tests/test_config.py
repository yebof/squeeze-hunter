from pathlib import Path

import pytest

from squeeze_hunter.config import load_settings


def test_load_settings_from_yaml(tmp_path: Path) -> None:
    cfg = tmp_path / "s.yml"
    cfg.write_text(
        """
score:
  threshold: 8.0
  weights:
    f1_si_pct: 2.0
    f2_days_to_cover: 1.0
risk:
  kelly_fraction: 0.20
  position_cap: 0.08
  max_positions: 6
""",
        encoding="utf-8",
    )
    s = load_settings(cfg)
    assert s.score.threshold == 8.0
    assert s.risk.kelly_fraction == 0.20
    assert s.score.weights["f1_si_pct"] == 2.0


def test_settings_env_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = tmp_path / "s.yml"
    cfg.write_text("score:\n  threshold: 7.0\n", encoding="utf-8")
    monkeypatch.setenv("SH_SCORE__THRESHOLD", "9.5")
    s = load_settings(cfg)
    assert s.score.threshold == 9.5
