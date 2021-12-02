import pandas as pd
import numpy as np


def distance_sort(signals):
    """Sort based on the distance to the mean signal.

    Parameters
    ----------
    signals : pandas.DataFrame

    Returns
    -------
    pandas.DataFrame

    """
    ws = signals.sub(signals.mean(axis=1), axis=0)

    # rescale weights
    ws_short = ws.where(ws < 0).div(
        ws.where(ws < 0).sum(axis=1, min_count=1).abs(),
        axis=0
    )
    ws_long = ws.where(ws >= 0).div(
        ws.where(ws >= 0).sum(axis=1, min_count=1).abs(), axis=0)
    ws = ws_short.fillna(ws_long)

    return ws


def rank_sort(signals, n_portfolios=None, legsize=None):
    """

    Parameters
    ----------
    signals : pandas.DataFrame
    n_portfolios : int
    legsize : int
        no of assets in the extreme legs

    Returns
    -------
    pandas.DataFrame

    """
    # dropna, align ---------------------------------------------------------
    # signals = signals.dropna(how="all")
    if signals.columns.name is None:
        signals.columns.name = "asset"

    # -----------------------------------------------------------------------
    # one portfolio only
    if n_portfolios == 1:
        pass

    if legsize is not None:
        # number of assets
        n = signals.shape[1]

        if n_portfolios is not None:
            raise ValueError("Only one of legsize and n_portfolios can live!")

        # number of portfolios will be e.g. np.ceil(9/4) = 3
        n_portfolios = int(np.ceil(n / legsize))

        # to achieve this, we need this many fakes to be added:
        signals_ = add_fake_signal(signals,
                                   n=max(n_portfolios * legsize - n, 0),
                                   fillna=True)

        res = rank_sort(signals_, n_portfolios)

        # remove fakes
        # res = res.loc[:, (slice(None), signals.columns)] \
        #     .where(signals.notnull(), axis=1, level=1)\
        #     .fillna(0)

        res = pd.concat(
            {k: v[k][signals.columns].where(signals.notnull())
             for k, v in res.groupby(axis=1, level=0)},
            axis=1
        )

        return res

    # Get signal ranks row by row
    signal_ranks = signals.rank(axis=1, numeric_only=True, pct=True,
                                method="first")

    # -----------------------------------------------------------------------
    # init space for bins
    bins = signal_ranks*np.nan

    # start caching!
    cache_bins = {}

    # loop over rows; for each calculate # of assets, construct bins, cache
    #   them; result is a DataFrame of bin numbers
    for idx, row in signal_ranks.iterrows():
        # drop nans: needed for digitize later
        this_row = row.dropna()

        # Get number of assets available in the row xs
        n_assets = this_row.count()

        # cache bins
        if n_assets not in cache_bins:
            # Generate quantile bins, applying rule specified in 'custom_bins'
            cache_bins[n_assets] = custom_bins(n_assets, n_portfolios)

        # cut into bins
        bins.loc[idx, this_row.index] = np.digitize(
            this_row, cache_bins[n_assets])

    # allocate
    portfolios = dict()

    for p in range(n_portfolios):
        # where bins are equal to 1,2,...
        this_p = signals.where(bins == p).notnull().astype(float)

        # write each portfolio's constituent assets
        portfolios["p" + str(p+1)] = this_p

    res = pd.concat(portfolios, axis=1,
                    names=["portfolio", signals.columns.name])

    res = res.rename(
        columns={"p1": "p_low",
                 "p{:d}".format(n_portfolios): "p_high"}
    )

    res = res.fillna(0).where(signals.notnull())

    return res


def add_fake_signal(signals, n=1, fillna=False):
    """Add fake series to signal: the median in each row.

    Parameters
    ----------
    signals : panda.DataFrame
    n : int
    fillna : bool
        True to fill nans with row median values

    Returns
    -------

    """
    # calculate median across rows
    med = signals.median(axis=1, skipna=True)

    # concat
    if n > 0:
        to_concat = pd.concat([med, ]*n, axis=1,
                              keys=["fake_{}".format(n_) for n_ in range(n)])
    else:
        to_concat = None

    res = pd.concat((signals, to_concat), axis=1)

    if fillna:
        res = res.fillna(
            pd.concat([med, ] * res.shape[1], axis=1, keys=res.columns)
        )

    res = res.rename_axis(index=signals.index.name,
                          columns=signals.columns.name)

    return res


def custom_bins(n_assets, n_portf, epsilon=1e-05):
    """ Create bin ranges with Dmitry's rule for np.digitize to work on them.

    Example with 4 portfolios
    The idea is to have [0, 0.25+e), [0.25+e, 0.50+2e) etc.

    Parameters
    ----------
    n_assets : int
        number of assets
    n_portf : int
        number of portfolios
    epsilon : float
        such that epsilon*n_portf is less than 1/n_portf

    Returns
    -------
    bins : list
        with bin numbers

    Esxample
    --------
    custom_bins(3, 4) -> [1/3+e, 1/3+2e, 2/3+3e, 3/3+4e]
    """
    # number of assets in each portfolio
    nass = assets_in_each(n_assets, n_portf)

    # bin ranges
    bins = (np.array(nass)/n_assets+epsilon).cumsum()

    return bins


def assets_in_each(n_assets, n_portf):
    """ Calculate number of assets in each portfolio with Dmitry's rule.

    Example with 7 assets, 5 portfolios: having assigned 2 assets to the 5th
    portfolio, the problem is to assign 5 assets to 4 portfolios, but in
    reversed order (2 to the first). The rest is careful handling of quotients.

    Parameters
    ----------
    n_assets : int
        number of assets
    n_portf : int
        number of portfolios

    Returns
    -------
    res : list
        with numbers of assets in portfolios

    Esxample
    --------
    assets_in_each(5, 4) -> [1,1,1,2]

    """
    # quotient is the integer part of division, remainder is... well, you know
    quot = n_assets // n_portf
    rem = n_assets % n_portf

    # if n_assets is a multiple of n_portf, just assign them equally
    if rem == 0:
        return [quot, ] * n_portf

    # else start with the last portfolio
    init = [quot+1]

    # continue with the assigning fewer assets into fewer portfolios
    return assets_in_each(n_assets-quot-1, n_portf-1)[::-1] + init


if __name__ == '__main__':
    import pandas.util.testing as tm
    tm.N, tm.K = 15, 11

    df = tm.makeTimeDataFrame(freq='M')
    df.iloc[2, 8:] = np.nan
    df.iloc[5:, :6] = np.nan
    df.iloc[-1, :] = np.nan

    lol = rank_sort(df, legsize=2)

    print(lol.sum(axis=1, level=0))
