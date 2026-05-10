import pandas as pd
import pytest

from squeeze_hunter.signals.normalize import cross_sectional_z


def test_z_zero_when_at_mean() -> None:
    s = pd.Series([1.0, 2.0, 3.0])
    z = cross_sectional_z(s)
    assert z.iloc[1] == pytest.approx(0.0)


def test_z_clipped_to_window() -> None:
    s = pd.Series([0.0, 0.0, 0.0, 0.0, 100.0])
    z = cross_sectional_z(s, clip=3.0)
    assert z.max() == pytest.approx(2.0)


def test_z_handles_zero_std() -> None:
    s = pd.Series([5.0, 5.0, 5.0])
    z = cross_sectional_z(s)
    assert (z == 0.0).all()
