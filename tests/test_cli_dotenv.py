"""Round-12: the CLI must load `.env` (README says to create it; nothing read it)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from squeeze_hunter.cli import _load_dotenv


def test_cli_loads_dotenv_from_cwd_without_overriding_real_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text("SH_DOTENV_PROBE=from_file\nSH_DOTENV_EXISTING=from_file\n")
    monkeypatch.delenv("SH_DOTENV_PROBE", raising=False)
    monkeypatch.setenv("SH_DOTENV_EXISTING", "from_env")

    nested = tmp_path / "sub" / "dir"
    nested.mkdir(parents=True)
    loaded = _load_dotenv(nested)  # walks up to the repo-style root

    assert loaded == tmp_path / ".env"
    assert os.environ["SH_DOTENV_PROBE"] == "from_file"
    assert os.environ["SH_DOTENV_EXISTING"] == "from_env"


def test_cli_dotenv_is_optional(tmp_path: Path) -> None:
    assert _load_dotenv(tmp_path) is None
