import numpy as np
import pandas as pd

from squeeze_hunter.backtest.shuffle_test import random_shuffle_pvalue


def test_shuffle_real_strategy_significant() -> None:
    np.random.seed(0)
    n = 200
    rets = np.zeros(n)
    rets[::5] = 0.03  # consistent positive jumps every 5 days → real edge
    eq = (1 + pd.Series(rets)).cumprod() * 100_000
    eq.index = pd.date_range("2024-01-01", periods=n, freq="B")
    p = random_shuffle_pvalue(eq, n_permutations=200, seed=0)
    assert p < 0.05
