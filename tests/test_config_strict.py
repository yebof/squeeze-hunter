"""Round-12: nested YAML typos must fail loud, like top-level ones already do."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from squeeze_hunter.config import Settings, load_settings


def test_nested_config_typo_fails_loud() -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate({"risk": {"kelly_fraction": 0.2, "monthly_drawdown_kil": 0.1}})
    with pytest.raises(ValidationError):
        Settings.model_validate({"stops": {"hard_stop_pct": -0.12}})


def test_example_yaml_loads_under_strict_nested_models() -> None:
    yaml_path = Path(__file__).resolve().parents[1] / "config" / "settings.example.yml"
    assert yaml_path.is_file(), yaml_path
    settings = load_settings(yaml_path)
    assert settings.score.weights
    assert settings.score.setup_thresholds == {"strong": 4.0, "mixed_floor": 3.0}
