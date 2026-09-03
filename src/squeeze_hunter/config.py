"""Pydantic Settings + YAML loader.

Layered config: yaml file → env overrides via SH_*__* (double underscore = nesting).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from squeeze_hunter.logging_setup import get_logger

_log = get_logger("config")


class _StrictSection(BaseModel):
    # Round-12: Settings itself forbids unknown top-level keys, but the nested
    # sections silently ignored typos (`monthly_drawdown_kil`, `hard_stop_pct`)
    # — the operator believed a risk knob was tuned when it wasn't.
    model_config = ConfigDict(extra="forbid")


class ScoreCfg(_StrictSection):
    threshold: float = 8.0
    weights: dict[str, float] = Field(default_factory=dict)
    # Classifier cutoffs (score/classifier.py): CAR needs A >= strong and
    # B < mixed_floor (GME mirrors); Mixed needs both >= mixed_floor. Round-12:
    # previously dead config — the classifier hardcoded 4.0 / 3.0.
    setup_thresholds: dict[str, float] = Field(
        default_factory=lambda: {"strong": 4.0, "mixed_floor": 3.0}
    )


class RiskCfg(_StrictSection):
    kelly_fraction: float = 0.20
    position_cap: float = 0.08
    max_positions: int = 6
    max_new_per_day: int = 3
    max_gross_exposure: float = 0.90
    monthly_drawdown_kill: float = 0.10
    bayes_prior_n: int = 30


class StopsCfg(_StrictSection):
    hard_stop: float = -0.12
    # R9.1: trailing stops are stored as the negative threshold (e.g., -0.20 =
    # exit when 20% below the peak). evaluate_stops takes a positive magnitude;
    # callers must pass `abs(settings.stops.trailing_car)` etc.
    trailing_car: float = -0.20
    trailing_gme: float = -0.25
    trailing_mixed: float = -0.22
    time_stop_days: int = 21
    signal_decay_halve: float = 0.50
    signal_decay_exit: float = 0.75


class UniverseCfg(_StrictSection):
    min_market_cap: float = 200_000_000
    max_market_cap: float = 10_000_000_000
    min_price: float = 5.0
    min_days_listed: int = 30


class DataCfg(_StrictSection):
    # R11: FINRA disseminates a settlement-date short-interest report ~8 US
    # business days LATER (settlement dates are the 15th & last business day;
    # the bulk file is published ~T+8). The backtest reveals each SI record on
    # settlement_date + this many business days so it cannot act on short
    # interest before it was public (lookahead → inflated Gate 1). Set 0 to
    # reveal on the settlement date (legacy behavior).
    finra_publication_lag_bdays: int = 8


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SH_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )
    score: ScoreCfg = Field(default_factory=ScoreCfg)
    risk: RiskCfg = Field(default_factory=RiskCfg)
    stops: StopsCfg = Field(default_factory=StopsCfg)
    universe: UniverseCfg = Field(default_factory=UniverseCfg)
    data: DataCfg = Field(default_factory=DataCfg)

    @classmethod
    def settings_customise_sources(
        cls: type[Settings],
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Priority (highest first): env > init (yaml) > defaults
        return env_settings, init_settings


def load_settings(yaml_path: Path | None = None) -> Settings:
    """Load settings: yaml file (if given) merged with env overrides.

    R5.M1: log when the YAML file is requested but missing — previously this
    fell silently through to defaults, which include an empty score.weights
    dict. That produced all-zero scores in scan/backtest with no warning.
    """
    base: dict[str, Any] = {}
    if yaml_path is not None:
        if yaml_path.exists():
            base = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
            _log.info("config_loaded", path=str(yaml_path))
        else:
            _log.warning("config_file_missing", path=str(yaml_path))
    settings = Settings(**base)
    if not settings.score.weights:
        _log.warning(
            "config_score_weights_empty",
            note="scan/backtest will produce all-zero scores. Set score.weights in YAML.",
        )
    return settings
