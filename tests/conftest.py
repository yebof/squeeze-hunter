"""Shared fixtures for squeeze-hunter tests."""

import pytest


@pytest.fixture
def repo_root(tmp_path_factory):
    """Empty temp directory for filesystem-touching tests."""
    return tmp_path_factory.mktemp("repo")
