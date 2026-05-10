import numpy as np
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


def test_z_inf_does_not_pollute_other_tickers() -> None:
    """I1 regression: a single inf (e.g., days-to-cover with ADV20=0)
    must not zero out z-scores for the rest of the universe.

    Before the fix: mean=inf, std=NaN, the if-guard returned all zeros.
    After: inf entry gets z=0, but other entries get meaningful z.
    """
    # Use values where none coincide with the mean (avoid spurious zero z).
    s = pd.Series([1.0, 2.0, 4.0, float("inf")], index=["A", "B", "C", "D"])
    z = cross_sectional_z(s)
    # All three finite entries get non-zero z (none equal the mean of 7/3).
    assert z.loc["A"] != 0.0
    assert z.loc["B"] != 0.0
    assert z.loc["C"] != 0.0
    assert z.loc["D"] == 0.0  # the inf entry itself contributes nothing
    # Sanity: z's are correctly signed
    assert z.loc["A"] < 0.0  # 1 < mean
    assert z.loc["C"] > 0.0  # 4 > mean


def test_z_negative_inf_treated_same_as_inf() -> None:
    s = pd.Series([1.0, 2.0, 3.0, float("-inf")])
    z = cross_sectional_z(s)
    assert z.iloc[3] == 0.0


def test_z_nan_entries_get_zero_and_dont_pollute() -> None:
    s = pd.Series([1.0, 2.0, 3.0, np.nan, 4.0])
    z = cross_sectional_z(s)
    assert not z.isna().any()
    assert z.iloc[3] == 0.0
    # The valid entries should still have non-zero z (mean ≈ 2.5)
    assert z.iloc[0] < 0.0  # 1.0 below mean
    assert z.iloc[4] > 0.0  # 4.0 above mean


def test_z_all_inf_returns_zeros() -> None:
    s = pd.Series([float("inf"), float("inf"), float("inf")])
    z = cross_sectional_z(s)
    assert (z == 0.0).all()


def test_z_mixed_inf_and_zero_std_returns_zeros() -> None:
    """All non-inf entries identical → std=0 after filtering → fall back to zeros."""
    s = pd.Series([5.0, 5.0, 5.0, float("inf")])
    z = cross_sectional_z(s)
    assert (z == 0.0).all()


def test_z_preserves_index_and_length() -> None:
    s = pd.Series([1.0, 2.0, float("inf"), 3.0], index=["x", "y", "z", "w"])
    z = cross_sectional_z(s)
    assert list(z.index) == ["x", "y", "z", "w"]
    assert len(z) == 4
