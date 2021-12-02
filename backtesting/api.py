import numpy as np

from .core import Backtesting, TradingStrategy, rank_sort, rescale_weights


def quick_sort(signals, assets, legsize=None, n_portfolios=None, rank=False,
               hml=True):
    """Calculate return on (legs of) a sorted strategy.

    Parameters
    ----------
    signals : pandas.DataFrame
        must be properly aligned with `y`, as no shifting will be performed
    assets : pandas.DataFrame
    legsize : int
    n_portfolios : int
    rank : bool
    hml : bool

    Returns
    -------
    pandas.Series

    """
    if rank:
        strat = TradingStrategy.long_short_rank(sort_values=signals,
                                                leverage="net")
        res = Backtesting(r_long=assets).backtest(strat, breakdown=False)
        return res

    # sort
    w = rank_sort(signals, n_portfolios, legsize) \
        .where(signals.notnull())

    # returns of all portfolios
    res = w.mul(assets, axis=1, level=1) \
        .where(assets.notnull()) \
        .sum(axis=1, level=0, min_count=1) / \
        w.sum(axis=1, level=0, min_count=1)

    if hml:
        res = res.assign(hml=(res["p_high"] - res["p_low"]))

    return res
