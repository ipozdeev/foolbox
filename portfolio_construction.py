import pandas as pd
from pandas.tseries.offsets import DateOffset
import numpy as np
import pickle


def rank_sort(returns, signals, n_portfolios=None, legsize=None,
              add_fake=False):
    """

    Parameters
    ----------
    returns : pandas.DataFrame
    signals : pandas.DataFrame
    n_portfolios : int
    legsize : int
        no of assets in the extreme legs
    add_fake : bool
        True to add a 'fake' column with median signal (won't affect
        composition of the exteme portfolios) to `signals` and a column
        full of np.nan to `returns`

    Returns
    -------

    """
    # write the list's contents into output dictionary
    portfolios = dict()

    # dropna, align ---------------------------------------------------------
    returns = returns.dropna(how="all")
    signals = signals.dropna(how="all")

    # align two frames to ensure the index is the same
    returns, signals = returns.align(signals, axis=0, join="inner")

    # -----------------------------------------------------------------------
    # one portfolio only
    if n_portfolios == 1:
        sig_notnull = signals.notnull()
        portfolios["portfolio1"] = sig_notnull.div(sig_notnull.sum(axis=1),
                                                   axis=0)
        portfolios["pf1"] = returns.mean(axis=1)

        return portfolios

    if legsize is not None:
        # number of assets
        n = signals.shape[1]

        if n_portfolios is not None:
            raise ValueError("Only one of legsize and n_portfolios can live!")

        # number of portfolios will be e.g. np.ceil(9/4) = 3
        n_portfolios = int(np.ceil(signals.shape[1] / legsize))

        # to achieve this, we need this many fakes to be added:
        for f in range(max(n % legsize - 1, 0)):
            returns, signals = add_fake_signal(returns, signals)

        return rank_sort(returns, signals, n_portfolios)

    # else sort as usually --------------------------------------------------
    # add fake column if needed
    if add_fake:
        returns, signals = add_fake_signal(returns, signals)

    # Get signal ranks row by row
    signal_ranks = signals.rank(
        axis=1,
        numeric_only=True,
        pct=True,
        method="average")

    # -----------------------------------------------------------------------
    # init space for bins
    bins = signal_ranks*np.nan

    # start cacheing!
    cache_bins = {}

    # loop over rows; for each calculate # of assets, construct bins, cache
    #   them; result is a DataFrame of bin numbers
    for idx, row in signal_ranks.iterrows():
        # drop nans: needed for digitize later
        this_row = row.dropna()
        # Get number of assets available in the row xs
        n_assets = this_row.count()

        # hash bins
        if n_assets not in cache_bins:
            # Generate quantile bins, applying rule specified in 'custom_bins'
            cache_bins[n_assets] = custom_bins(n_assets, n_portfolios)

        # cut into bins
        bins.loc[idx, this_row.index] = np.digitize(
            this_row, cache_bins[n_assets])

    # allocate
    for p in range(n_portfolios):
        # where bins are equal to 1,2,...
        this_portf = returns.where(bins == p)
        # write each portfolio's constituent assets
        portfolios["portfolio" + str(p+1)] = this_portf.drop("fake", axis=1,
                                                             errors="ignore")
        # get the equally-weighted return on each portfolio
        portfolios["p" + str(p+1)] = this_portf.mean(axis=1)
        portfolios["p" + str(p+1)].name = "p" + str(p+1)

    return portfolios


def get_factor_portfolios(portfolios, hml=False, hml_ascending=True):
    """
    Utility function returning a dataframe of returns to factor portfolios,
    i.e. portfolios p1, p2, ..., pP from the dictionary created by rank_sort or
    default_bas_adjustment functions.
    function.

    Parameters
    ----------
    portfolios: dictionary
        of dataframes created by rank_sort
    hml: bool
        include the high-minus-low portfolio column, default is False
    hml_ascending: bool
        if the hml column is included, should it be high-minus-low, i.e. last
        minus first portfolio, or vice versa, default is True, i.e. last minus
        first

    Returns
    -------
    factor_portfolios: pandas.DataFrame
        of returns to portfolos p1, p2, ..., pP from the dictionary created by
        rank_sort

    """

    # Infer the number of portfolios:
    n_portfolios = int(len(portfolios) / 2)  # two dataframes for a portfolio

    temp_list = []                 # create a list to collect dataframes
    for p in range(n_portfolios):  # iterate through number of portfolis
        temp_list.append(portfolios["p" + str(p+1)])
                                   # add dataframe to the list

    factor_portfolios = pd.concat(temp_list, axis=1)
                                   # concatenate list elements into dataframe
    if hml:
        if hml_ascending:
            factor_portfolios["hml"] = -factor_portfolios["p1"] +\
                                    factor_portfolios["p"+str(n_portfolios)]
        else:
            factor_portfolios["hml"] = factor_portfolios["p1"] -\
                                    factor_portfolios["p"+str(n_portfolios)]

    return factor_portfolios


def add_fake_signal(ret, sig):
    """Add fake series to signal: the median in each row.
    """
    r, s = ret.copy(), sig.copy()

    # calculate median across rows
    fake_sig = sig.apply(np.nanmedian, axis=1)

    # reinstall
    s.loc[:, "fake"] = fake_sig
    r.loc[:, "fake"] = np.nan

    return r, s


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


def get_holdings(portfolio):
    """Returnin holdings in `portfolio` of the output of 'rank_sort()'.

    Parameters
    ----------
    portfolio: pandas.Data.Frame
        corresponding to for example 'portfolio3' key of the 'rank_sort()'
        function, i.e. returns for currencies which are in the protfolio, and
        zeroes otherwise

    Returns
    -------
    holdings: pd.DataFrame
        of columns names of assets present in portfolio at each date. For
        example "2012-12-31: ['AUD', 'NZD', 'CAD']"

    """
    holdings = pd.DataFrame(index=portfolio.index, columns=["holdings"],
                            dtype="object")

    # Loop over the portfolio's rows, filling holdings with lists
    for date, row in portfolio.iterrows():
        holdings.loc[date, "holdings"] = portfolio.ix[date].dropna().index\
            .tolist()

    return holdings


def upsample_portfolios(portfolios, hfq_returns):
    """Given rank_sort() output for low frequency returns, and dataframe of
    higher frequency returns, upsamples the former, using higher frequency data
    from the latter

    Parameters
    ----------
    portfolios: dict
        corresponding to output of the 'rank_sort()' family of functions
    hfq_returns: pandas.DataFrame
        of returns to be used for ubpsampling. For example daily re

    Returns
    -------
    upsampled: dict
        of upsampled portfolios in 'portfolios'

    """
    # Identify number of portfolios
    n_portfolios = int(len(portfolios)/2)

    upsampled = {}
    for p in np.arange(1, n_portfolios+1, 1):
        # Upsample the holdings
        tmp = portfolios["portfolio"+str(p)].copy()
        hfq_holdings = hfq_returns.where(tmp.reindex(hfq_returns.index,
                                         method="bfill").notnull())
        upsampled["portfolio"+str(p)] = hfq_holdings

        # Get the corresponding factor portfolios
        upsampled["p"+str(p)] = hfq_holdings.mean(axis=1)
        upsampled["p"+str(p)].name = "p" + str(p)

    return upsampled


def bas_adjustment(portfolio, spot_mid, spot_bid, spot_ask, fwd_mid, fwd_bid,
                   fwd_ask, long=True):
    """Computes transaction costs-adjusted returns for a currency portfolio
    following the approach of Menkhoff et al. (2012).

    Adjustment scheme:

    Positions are established in the beginning of the sample and liquidiated at
    the end. There are following possible cases (rhs variables are in logs):

    I. Security enters a portfolio at t and exits it at t+1

        1.) Long position:  rx_{t+1} = bid_forward_{t} - ask_spot{t+1}
        2.) Short position: rx_{t+1} = -ask_forward_{t} + bid_spot{t+1}

    II. Security enters a portfolio at t and stays in it at t+1

        1.) Long position:  rx_{t+1} = bid_forward_{t} - mid_spot{t+1}
        2.) Short position: rx_{t+1} = -ask_forward_{t} + mid_spot{t+1}

    III. Security already was in a portfolio at t-1 and exits it at t+1

        1.) Long position:  rx_{t+1} = mid_forward_{t} - ask_spot{t+1}
        2.) Short position: rx_{t+1} = -mid_forward_{t} + bid_spot{t+1}

    IV. Security already was in a portfolio and stays in it at t+1

        1.) Long position:  rx_{t+1} = mid_forward_{t} - mid_spot{t+1}
        2.) Short position: rx_{t+1} = -mid_forward_{t} + mid_spot{t+1}

    NOTE:   for short positions, the function returns long portfolios adjusted
            for short transaction costs, this behavior is imposed to maintain
            uniform "long-minus-short" approach, i.e. use minus when shorting

    Parameters
    ----------
    portfolio: pandas.DataFrame
        of returns on individual currencies constituting this portfolio,
        supplied by the portfolio_construction.rank_sort(), namely by the
        portfolio1, portfolio2, ..., portfolioN keys from dictionary.
        Alternatively a DataFrame, reflecting portfolio constituents at each
        pont in time with non NaN values
    spot_mid: pandas.DataFrame
        of corresponding spot exchange rates midpoint quotes
    spot_bid: pandas.DataFrame
        of corresponding spot exchange rates bid
    spot_ask: pandas.DataFrame
        of corresponding spot exchange rates ask quotes
    fwd_mid: pandas.DataFrame
        of corresponding forward exchange rates midpoint quotes
    fwd_bid: pandas.DataFrame
        of corresponding forward exchange rates bid
    fwd_ask: pandas.DataFrame
        of corresponding forward exchange rates ask quotes
    long: bool
        long position in the portfolio if True, short if False

    Returns
    -------
    adj_portfolio: pandas.DataFrame
        of transaction costs-adjusted returns of individual currencies

    """
    # Create the output DataFrame
    adj_portfolio = pd.DataFrame(index=portfolio.index,
                                 columns=portfolio.columns, dtype="float")

    # Create Boolean DataFrames, covering the possible cases
    enters_exits = pd.notnull(portfolio) & pd.isnull(portfolio.shift(1)) &\
                                                pd.isnull(portfolio.shift(-1))
    enters_stays = pd.notnull(portfolio) & pd.isnull(portfolio.shift(1)) &\
                                                pd.notnull(portfolio.shift(-1))
    was_exits = pd.notnull(portfolio) & pd.notnull(portfolio.shift(1)) &\
                                                pd.isnull(portfolio.shift(-1))
    was_stays = pd.notnull(portfolio) & pd.notnull(portfolio.shift(1)) &\
                                                pd.notnull(portfolio.shift(-1))

    # Transaction costs adjustments, covering Cases I-IV:
    # The long position adjustment:
    if long:
        adj_portfolio[enters_exits] = np.log(fwd_bid.shift(1) / spot_ask)
        adj_portfolio[enters_stays] = np.log(fwd_bid.shift(1) / spot_mid)
        adj_portfolio[was_exits] = np.log(fwd_mid.shift(1) / spot_ask)
        adj_portfolio[was_stays] = np.log(fwd_mid.shift(1) / spot_mid)

    # The short position adjustment (absence of minus before logs intended):
    else:
        adj_portfolio[enters_exits] = np.log(fwd_ask.shift(1) / spot_bid)
        adj_portfolio[enters_stays] = np.log(fwd_ask.shift(1) / spot_mid)
        adj_portfolio[was_exits] = np.log(fwd_mid.shift(1) / spot_bid)
        adj_portfolio[was_stays] = np.log(fwd_mid.shift(1) / spot_mid)

    # Get the output
    return adj_portfolio


def default_bas_adjustment(portfolios, spot_mid, spot_bid, spot_ask,
                           fwd_mid, fwd_bid, fwd_ask, ascending=True):
    """A utility function adjusting the output of the rank_sort() function for
    transaction costs, using the Menkhoff et al. (2012) scheme. Applies the
    bas_adjustment() function to portfolio1, portfolio2, ..., portfolioN keys
    of portfolios dictionary supplied by rank_sort(), assuming by default short
    positions in 'low portfolios', e.g. portfolio1, and long positions in 'high
    portfolios' (controlled by parameter 'ascending').

    If the number of portfolios is odd, the median portfolio is not adjusted.

    The output preserves the dictionary structure of the input.

    Parameters
    ----------
    portfolios: dictionary
        of pandas.DataFrames generated by the rank_sort() function
    spot_mid: pandas.DataFrame
        of corresponding spot exchange rates midpoint quotes
    spot_bid: pandas.DataFrame
        of corresponding spot exchange rates bid
    spot_ask: pandas.DataFrame
        of corresponding spot exchange rates ask quotes
    fwd_mid: pandas.DataFrame
        of corresponding forward exchange rates midpoint quotes
    fwd_bid: pandas.DataFrame
        of corresponding forward exchange rates bid
    fwd_ask: pandas.DataFrame
        of corresponding forward exchange rates ask quotes
    ascending: bool
        portfolio1 is shorted, default is True

    Returns
    -------
    adj_protfolios: dictionary
        of dataframes. Keys p1, p2, ..., pN, where N is number of portfolios,
        contain transaction costs-adjusted returns on equally weighted
        portfolios from low to high. Keys portfolio1, portfolio2, ...,
        portfolioN contain returns of individual currencies in the
        corresponding portfolio

    """
    # Create output dictionary
    adj_portfolios = {}

    # Infer number of portfolios from the input dictionary, for each portfolio
    # there are two keys: equally weighted return, and returns of constituents
    n_portfolios = int(len(portfolios.keys()) / 2)

    # Get the number of portfolios to adjust for long and for short positions,
    # e.g. if there are 3 or 2 portfolios than n_to_adjust = 1 (1 for long and
    # 1 for short), for 5 or 4 portfolios, n_to_adjust = 2, and so on.
    n_to_adjust = n_portfolios // 2

    # If the number of portfolios is odd, the median portfolio is not adjusted
    if n_portfolios % 2 == 1:
        adj_portfolios["portfolio"+str(n_to_adjust+1)] =\
                                    portfolios["portfolio"+str(n_to_adjust+1)]

    # Adjust portfolios for transaction costs
    if ascending:
        # Adjust portfolios 1, 2, ... for short positions
        for p in np.arange(1, 1+n_to_adjust):
            portfolio = portfolios["portfolio"+str(p)]
            adj_portfolios["portfolio"+str(p)] =\
                                    bas_adjustment(portfolio, spot_mid,
                                                   spot_bid, spot_ask,
                                                   fwd_mid, fwd_bid, fwd_ask,
                                                   long=False)
        # Adjust portfolios N, N-1, ... for long positions
        for p in np.arange(n_portfolios, n_portfolios-n_to_adjust,-1):
            portfolio = portfolios["portfolio"+str(p)]
            adj_portfolios["portfolio"+str(p)] =\
                                    bas_adjustment(portfolio, spot_mid,
                                                   spot_bid, spot_ask,
                                                   fwd_mid, fwd_bid, fwd_ask,
                                                   long=True)

    else:
        # Adjust portfolios 1, 2, ... for long positions
        for p in np.arange(1, 1+n_to_adjust):
            portfolio = portfolios["portfolio"+str(p)]
            adj_portfolios["portfolio"+str(p)] =\
                                    bas_adjustment(portfolio, spot_mid,
                                                   spot_bid, spot_ask,
                                                   fwd_mid, fwd_bid, fwd_ask,
                                                   long=True)
        # Adjust portfolios N, N-1, ... for short positions
        for p in np.arange(n_portfolios, n_portfolios-n_to_adjust,-1):
            portfolio = portfolios["portfolio"+str(p)]
            adj_portfolios["portfolio"+str(p)] =\
                                    bas_adjustment(portfolio, spot_mid,
                                                   spot_bid, spot_ask,
                                                   fwd_mid, fwd_bid, fwd_ask,
                                                   long=False)
    # Add returns of equally-weighted portfolios
    for p in np.arange(1, n_portfolios+1):
        adj_portfolios["p" + str(p)] =\
                                adj_portfolios["portfolio"+str(p)].mean(axis=1)
        adj_portfolios["p" + str(p)].name = "p" + str(p)

    return adj_portfolios


def spot_bas_adjustment(portfolio, spot_mid, spot_bid, spot_ask, long=True):
    """Computes transaction costs-adjusted returns for a currency portfolio
    following the approach of Menkhoff et al. (2012), but using spot rates only

    Adjustment scheme:

    Positions are established in the beginning of the sample and liquidiated at
    the end. There are following possible cases (rhs variables are in logs):

    I. Security enters a portfolio at t and exits it at t+1

        1.) Long position:  ds_{t+1} = bid_spot_{t} - ask_spot{t+1}
        2.) Short position: ds_{t+1} = -ask_spot_{t} + bid_spot{t+1}

    II. Security enters a portfolio at t and stays in it at t+1

        1.) Long position:  ds_{t+1} = bid_spot_{t} - mid_spot{t+1}
        2.) Short position: ds_{t+1} = -ask_spot_{t} + mid_spot{t+1}

    III. Security already was in a portfolio at t-1 and exits it at t+1

        1.) Long position:  ds_{t+1} = mid_spot_{t} - ask_spot{t+1}
        2.) Short position: ds_{t+1} = -mid_spot_{t} + bid_spot{t+1}

    IV. Security already was in a portfolio and stays in it at t+1

        1.) Long position:  rx_{t+1} = mid_spot_{t} - mid_spot{t+1}
        2.) Short position: rx_{t+1} = -mid_spot_{t} + mid_spot{t+1}

    NOTE:   for short positions, the function returns long portfolios adjusted
            for short transaction costs, this behavior is imposed to maintain
            uniform "long-minus-short" approach, i.e. use minus when shorting

    Parameters
    ----------
    portfolio: pandas.DataFrame
        of returns on individual currencies constituting this portfolio,
        supplied by the portfolio_construction.rank_sort(), namely by the
        portfolio1, portfolio2, ..., portfolioN keys from dictionary.
        Alternatively a DataFrame, reflecting portfolio constituents at each
        pont in time with non NaN values
    spot_mid: pandas.DataFrame
        of corresponding spot exchange rates midpoint quotes
    spot_bid: pandas.DataFrame
        of corresponding spot exchange rates bid
    spot_ask: pandas.DataFrame
        of corresponding spot exchange rates ask quotes
    long: bool
        long position in the portfolio if True, short if False

    Returns
    -------
    adj_portfolio: pandas.DataFrame
        of transaction costs-adjusted returns of individual currencies

    """
    # Create the output DataFrame
    adj_portfolio = pd.DataFrame(index=portfolio.index,
                                 columns=portfolio.columns, dtype="float")

    # Create Boolean DataFrames, covering the possible cases
    enters_exits = pd.notnull(portfolio) & pd.isnull(portfolio.shift(1)) &\
                                                pd.isnull(portfolio.shift(-1))
    enters_stays = pd.notnull(portfolio) & pd.isnull(portfolio.shift(1)) &\
                                                pd.notnull(portfolio.shift(-1))
    was_exits = pd.notnull(portfolio) & pd.notnull(portfolio.shift(1)) &\
                                                pd.isnull(portfolio.shift(-1))
    was_stays = pd.notnull(portfolio) & pd.notnull(portfolio.shift(1)) &\
                                                pd.notnull(portfolio.shift(-1))

    # Transaction costs adjustments, covering Cases I-IV:
    # The long position adjustment:
    if long:
        adj_portfolio[enters_exits] = np.log(spot_bid.shift(1) / spot_ask)
        adj_portfolio[enters_stays] = np.log(spot_bid.shift(1) / spot_mid)
        adj_portfolio[was_exits] = np.log(spot_mid.shift(1) / spot_ask)
        adj_portfolio[was_stays] = np.log(spot_mid.shift(1) / spot_mid)

    # The short position adjustment (absence of minus before logs intended):
    else:
        adj_portfolio[enters_exits] = np.log(spot_ask.shift(1) / spot_bid)
        adj_portfolio[enters_stays] = np.log(spot_ask.shift(1) / spot_mid)
        adj_portfolio[was_exits] = np.log(spot_mid.shift(1) / spot_bid)
        adj_portfolio[was_stays] = np.log(spot_mid.shift(1) / spot_mid)

    # Get the output
    return adj_portfolio


def default_spot_bas_adjustment(portfolios, spot_bid, spot_mid, spot_ask,
                                ascending=True):
    """A utility function adjusting the output of the rank_sort() function for
    transaction costs, using the Menkhoff et al. (2012) scheme. Applies the
    bas_adjustment() function to portfolio1, portfolio2, ..., portfolioN keys
    of portfolios dictionary supplied by rank_sort(), assuming by default short
    positions in 'low portfolios', e.g. portfolio1, and long positions in 'high
    portfolios' (controlled by parameter 'ascending').

    If the number of portfolios is odd, the median portfolio is not adjusted.

    The output preserves the dictionary structure of the input.


    The output preserves the dictionary structure of the input.

    Parameters
    ----------
    portfolios: dictionary
        of pandas.DataFrames generated by the rank_sort() function
    spot_mid: pandas.DataFrame
        of corresponding spot exchange rates midpoint quotes
    spot_bid: pandas.DataFrame
        of corresponding spot exchange rates bid
    spot_ask: pandas.DataFrame
        of corresponding spot exchange rates ask quotes
    ascending: bool
        portfolio1 is shorted, default is True

    Returns
    -------
    adj_protfolios: dictionary
        of dataframes. Keys p1, p2, ..., pN, where N is number of portfolios,
        contain transaction costs-adjusted spot returns on equally weighted
        portfolios from low to high. Keys portfolio1, portfolio2, ...,
        portfolioN contain returns of individual currencies in the
        corresponding portfolio
    """
    # Create output dictionary
    adj_portfolios = {}

    # Infer number of portfolios from the input dictionary, for each portfolio
    # there are two keys: equally weighted return, and returns of constituents
    n_portfolios = int(len(portfolios.keys()) / 2)

    # Get the number of portfolios to adjust for long and for short positions,
    # e.g. if there are 3 or 2 portfolios than n_to_adjust = 1 (1 for long and
    # 1 for short), for 5 or 4 portfolios, n_to_adjust = 2, and so on.
    n_to_adjust = n_portfolios // 2

    # If the number of portfolios is odd, the median portfolio is not adjusted
    if n_portfolios % 2 == 1:
        adj_portfolios["portfolio"+str(n_to_adjust+1)] =\
                                    portfolios["portfolio"+str(n_to_adjust+1)]

    # Adjust portfolios for transaction costs
    if ascending:
        # Adjust portfolios 1, 2, ... for short positions
        for p in np.arange(1, 1+n_to_adjust):
            portfolio = portfolios["portfolio"+str(p)]
            adj_portfolios["portfolio"+str(p)] =\
                            spot_bas_adjustment(portfolio, spot_mid,
                                                spot_bid, spot_ask, long=False)
        # Adjust portfolios N, N-1, ... for long positions
        for p in np.arange(n_portfolios, n_portfolios-n_to_adjust,-1):
            portfolio = portfolios["portfolio"+str(p)]
            adj_portfolios["portfolio"+str(p)] =\
                            spot_bas_adjustment(portfolio, spot_mid,
                                                spot_bid, spot_ask, long=True)

    else:
        # Adjust portfolios 1, 2, ... for long positions
        for p in np.arange(1, 1+n_to_adjust):
            portfolio = portfolios["portfolio"+str(p)]
            adj_portfolios["portfolio"+str(p)] =\
                            spot_bas_adjustment(portfolio, spot_mid,
                                                spot_bid, spot_ask, long=True)
        # Adjust portfolios N, N-1, ... for short positions
        for p in np.arange(n_portfolios, n_portfolios-n_to_adjust,-1):
            portfolio = portfolios["portfolio"+str(p)]
            adj_portfolios["portfolio"+str(p)] =\
                            spot_bas_adjustment(portfolio, spot_mid,
                                                spot_bid, spot_ask, long=False)
    # Add returns of equally-weighted portfolios
    for p in np.arange(1, n_portfolios+1):
        adj_portfolios["p" + str(p)] =\
                                adj_portfolios["portfolio"+str(p)].mean(axis=1)
        adj_portfolios["p" + str(p)].name = "p" + str(p)

    return adj_portfolios


def rank_weighted_strat(returns, signals):
    """
    Computes returns to a long-minus-shrort portfolio where at each time point
    portfolio return is a cross-sectional weighted average of individual asset
    returns, with weights determined by the rank of the corresponding signal.

    Rank based weighted scheme: w(i)  = Rank(i) - Average XS Rank, whith ranks
    in ascending order

    Parameters
    ----------
    returns: DataFrame
        of asset returns
    signals: DataFrame
        of signals accroding to which returns are sorted, this
             dataframe has same shape as returns

    Returns
    -------
    strategy_returns: DataFrame
        of cross-sectional means of weigthed returns

    """

    # Get signal ranks
    signal_ranks = signals.rank(axis = 1,           # get ranks for each row
                                numeric_only=True,  # ignore NaNs
                                method="dense")     # rank changes by 1 for
                                                    # each unique value in a xs

    # Estimate rank-based weights
    rank_weights = signal_ranks.subtract(signal_ranks.mean(axis=1), axis=0)

    # Get weighted returns and returns of the long-short portfolio
    weighted_returns = returns * rank_weights
    strategy_returns  = weighted_returns.mean(axis=1).to_frame()

    return strategy_returns


def z_score(data, axis=0):
    """Utility function applying z-score transformation to a dataframe along
    specified axis.

    Parameters
    ----------
    data: pandas.DataFrame
        of input data
    axis: int
        specifying axis along which z-scores are to be computed, 0 for column-
        wise, and 1 for rowwise. Default is 0

    Returns
    -------
    z_scores: pandas.DataFrame
        with z-scores of input data

    """
    if axis == 1:
        z_scores = data.subtract(data.mean(axis=1), axis=0).\
                        divide(data.std(axis=1), axis=0)
    else:
        z_scores = data.subtract(data.mean(axis=0), axis=1).\
                        divide(data.std(axis=0), axis=1)

    return z_scores


def risk_parity(returns, volas):
    """Estimates returns to risk-parity portfolio, weighing returns by inverse
    of their volatilities, i.e. assuming zero off-diagonal elements of the
    variance-covariance matrix

    Parameters
    ----------
    returns: pandas.DataFrame
        of asset returns
    volas: pandas.DataFrame
        of the same shape, containing corresponding volatilites to determine
        weights in each period

    Returns
    -------
    risk_parity: pandas.DataFrame
        of returns to risk parity portfolio

    """
    # Compute weights for each period
    weights = volas.pow(-1).div(volas.pow(-1).sum(axis=1), axis="index")
    # Compute returns on risk-parity portfolio
    risk_parity = (returns * weights).sum(axis=1).to_frame()
    risk_parity.columns = ["risk_parity"]

    return risk_parity


def get_hml(returns, signals, n_portf=3):
    """
    """
    pf = rank_sort(returns, signals, n_portf)
    hml = get_factor_portfolios(pf, hml=True).loc[:,"hml"]

    hml = hml.reindex(index=signals.index)

    return hml


def multiple_timing(returns, signals, xs_avg=True):
    """Wrapper around the 'timed_strat()' function allowing application to
    multiple assets

    Parameters
    ----------
    returns: pandas.DataFrame
        of returns to individual assets to which timing is to be applied
    signals: pandas.DataFrame
        of the corresponding signals. The dataframe should consist of asset-
        specific signals, concatenated over the time-series index with outer
        join
    xs_avg: bool
        specifying whether fuction should return returns of individual assets
        if False or cross-sectional average of these returns if True. Default
        is True

    Returns
    -------
    timed_strats: pandas.DataFrame
        with individually timed returns if xs-avg is False, or cross-sectional
        average of these returns merged by index if True

    """
    # If input is series, simply invoke behavior of 'timed_strat'
    if isinstance(returns, pd.Series) and isinstance(signals, pd.Series):
        timed_strats = timed_strat(returns, signals)

    elif isinstance(returns, pd.DataFrame) and isinstance(signals,
                                                          pd.DataFrame):
        # ipdb.set_trace()
        # Apply the 'timed_strat()' function to each column
        tmp_list = list()
        for col in returns.columns:
            tmp_list.append(timed_strat(returns[col], signals[col]))
        # Concatendate the timed returns using outer join
        timed_strats = pd.concat(tmp_list, join="outer", axis=1)

        # Average the output if required
        if xs_avg:
            timed_strats = timed_strats.mean(axis=1).to_frame()
            timed_strats.columns = ["avg"]
    else:
        raise TypeError("Both returns and signals should be either pd.Series"
                        "or pd.DataFrame objects")

    return timed_strats


def timed_strat(returns, signals):
    """Times the returns given signals, such that long position is taken for
    positive signals' values and short position is taken for the negative ones.

    Parameters
    ----------
    returns: pandas.Series or pandas.DataFrame
        of returns to an asset or portfolio to be timed according to signals
    signals: pandas.Series or pandas.DataFrame
        of signals to time returns. Signals' index should be a subset of that
        of returns

    Returns
    -------
    timed_returns: pandas.Series
        of returns timed according to signals

    """
    # Transform dataframe inputs into series
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    if isinstance(signals, pd.DataFrame):
        signals = signals.iloc[:, 0]

    # Get the returns corresponding to signals's dates
    timed_returns = returns.reindex(signals.index)

    # Time the returns according to signals' signs
    # Drop zeros from signals inference purposses -- zeros should be omitted
    # from sample statistics -- no position is established
    timed_returns = timed_returns * (np.sign(signals).replace(0, np.nan))

    return timed_returns


def many_monthly_rx(s_d, f_d):
    """ Get 28 overlapping monthly excess returns.
    Parameters
    ----------
    fdisc_d : pd.DataFrame
        forward discounts (30-day forwards) from perspective of US investor
    """
    s_d, f_d = s_d.align(f_d, axis=0)
    idx = pd.date_range(s_d.index[0], s_d.index[-1], freq='D')

    f_d = f_d.reindex(idx).ffill()
    s_d = s_d.reindex(idx).fillna(value=0)

    res = pd.DataFrame(index=s_d.index, columns=s_d.columns)*np.nan

    for t in s_d.index:
        # t = s_d.index[100]
        prev_t = t - DateOffset(months=1)
        if prev_t < (f_d.index[0]):
            continue
        # subsample y and x
        r = s_d.loc[(prev_t+DateOffset(days=1)):t,:].sum().add(
            f_d.iloc[f_d.index.get_loc(prev_t, "ffill")])

        res.loc[t,:] = r

    all_m = dict()
    for p in range(28):
        all_m[p] = res.ix[(res.index.day == p+1),:]

    return all_m

    # s = s_d.rolling(22).sum()
    # f = fdisc_d.shift(22)
    #
    # all_m = dict()
    # for p in range(22):
    #     all_m[p] = s_d.iloc[p::22,:] + f.iloc[p::22,:]
    #
    # return all_m


def many_monthly_s(s_d):
    """ Get 22 overlapping monthly spot returns.
    Parameters
    ----------
    """

    s = s_d.rolling(22).sum()

    all_m = dict()
    for p in range(22):
        all_m[p] = s_d.iloc[p::22,:]

    return all_m


def normalize_weights(wght):
    """

    Parameters
    ----------
    wght : pandas.DataFrame or pandas.Series

    Returns
    -------

    """
    # helper to discrimimnate between series and df
    def normalize_weights_series(x):
        """Rescale weights watching out for long-short holdings.
        """
        if (x == 0).all():
            res = x

        elif (x < 0).any() & (x > 0).any():
            short_leg = x.where(x < 0)
            long_leg = x.where(x >= 0)

            short_leg = short_leg / np.abs(short_leg).sum()
            long_leg = long_leg / np.abs(long_leg).sum()

            res = short_leg.fillna(long_leg)

        else:
            res = x / np.abs(x).sum()

        return res

    # apply helper
    if isinstance(wght, pd.DataFrame):
        res = wght.apply(normalize_weights_series, axis=1)
    else:
        res = normalize_weights_series(wght)

    return res


def weighted_return(ret, weights):
    """
    """
    if isinstance(ret, pd.Series):
        return ret
    if isinstance(weights, pd.Series):
        weights = weights.to_frame()

    # mask = (ret + w).notnull()
    # w_rescaled = rescale_weights(w.where(mask))

    # this is just weighted product, needed because .sum() gives 0 when the
    #   row is full of nans
    res = (ret*weights).mean(axis=1)*(ret*weights).count(axis=1)

    return res
