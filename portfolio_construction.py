import pandas as pd
import numpy as np
from foolbox.data_mgmt import set_credentials as setc
import pickle
# import ipdb

def rank_sort_adv(returns, signals, n_portfolios, holding_period=None,
                  rebalance_dates=None):
    """Experimental version of the 'rank_sort' funciton. Sorts a dataframe of
    returns into portfolios according to dataframe of signals, holding period,
    and rebalancing dates. If in any given cross-section the number of assets
    is not a multiple of number of portfolios, additional assets are allocated
    according to the rule specified in 'custom_bins' function.

    Parameters
    ----------
    returns: pandas.DataFrame
        of returns on assets to be sorted into portfolios
    signals: pandas.DataFrame
        of signals according to which returns are sorted
    n_portfolios: int
        specifying the number of portfolios to sort returns into
    holding_period: int
        number of periods after which the portfolio is liquidated. If None, the
        portoflio is either held for a single period (if rebalance_dates is
        also None), or until the next rebalancing date. Default is None
    rebalance_dates: pandas.DatetimeIndex
        specifying a sequence of rebalance dates. If None, then portfolio is
        either held for a single period (if holding period is also None), or
        dates are derived from the holding period. Default is None

    Returns
    -------
    portfolios: dict
        of dataframes. Keys p1, p2, ..., pN, where N = n_portfolios contain
        returns on equally weighted portfolios from low to high. Keys
        portfolio1, portfolio2, ..., portfolioN contain returns of individual
        asssets in the corresponding portfolio

    """
    # Assign output structure, ensure inputs integrity
    portfolios = {}

    # Create a list of dataframes with number of elements equal n_portfolios
    # the first element of the list is the 'low' portfolio, the last is 'high'
    portf_list = [pd.DataFrame(columns=returns.columns)
                  for portfolio in range(n_portfolios)]

    # Consider observationas where both signals and returns are available
    #returns = returns[pd.notnull(signals)]
    #signals = signals[pd.notnull(returns)]

    # Align two frames to ensure the index is the same
    # returns, signals = returns.align(signals, axis=0)

    # Get signal ranks row by row
    signal_ranks = signals.rank(axis=1,             # get ranks for each row
                                numeric_only=True,  # ignore NaNs
                                pct=True,           # map ranks to [0, 1]
                                method="average")   # equal values' ranks are
                                                    # averaged

    # Deal with rebalancing dates and holding periods

    # First valid date, and the corresponding integer value
    first_date = returns.first_valid_index()
    first_date_int = returns.index.get_loc(first_date)

    # CASE 1: neither rebalance dates nor holding periods are supplied,
    # rebalance each period
    if holding_period is None and rebalance_dates is None:
        # Set rebalance dates to index values of the returns
        rebalance_dates = returns.ix[first_date:].index

    # CASE 2: no rebalance dates, portfolio rebalanced at the end of the
    # holding period
    if holding_period is not None and rebalance_dates is None:
        # Set rebalance dates, every holding_periods steps from the first obs
        rebalance_dates = returns.iloc[first_date_int::holding_period, :].\
            index

    # Create a list of holdings masks, i.e. list of dataframes corresponding to
    # the number of portfolios, where values will be set to True if an asset is
    # in the portfolio at some date. The first element of the list corresponds
    # to the 'low' portfolio, and the last element to the 'high' portfolio
    holdings_masks = [pd.DataFrame(columns=returns.columns) for portfolio in
                      range(n_portfolios)]

    # Start iteration over rebalancing dates
    for k, t in enumerate(rebalance_dates.tolist()):
        # Get number of assets available in the row xs
        n_assets = pd.notnull(signal_ranks.ix[t]).sum()
        # Generate quantile bins, applying rule specified in 'custom_bins'
        bins = custom_bins(n_assets, n_portfolios)
        # Get positions by cutting xs into bins
        rank_cuts = rank_cut_old(pd.Series(True, index=signal_ranks.columns,
                                       name=t), signal_ranks.ix[t], bins)

        # Append masks with True/False dataframes reflecting holdings in the
        # particular portfolio

        for portf_num, mask in enumerate(holdings_masks):

            if holding_period is not None and rebalance_dates is not None:
                # The case where positions are liquidated before next rebalance
                tmp_idx = returns.iloc[returns.index.get_loc(t):\
                    returns.index.get_loc(t)+holding_period].index
            else:
                # The case where holding period is effectively spans time
                # between rebalancings. Enforce right-open time interval

                # Check for the last period
                if k+1 == len(rebalance_dates):
                    tmp_idx = returns[(returns.index >= t)].index
                else:
                    tmp_idx = returns[(returns.index >= t) &
                                (returns.index < rebalance_dates[k+1])].index

            tmp_mask = pd.DataFrame(False, columns=returns.columns,
                                    index=tmp_idx)
            tmp_mask[rank_cuts[portf_num].index] = True
            holdings_masks[portf_num] =\
                holdings_masks[portf_num].append(tmp_mask)

    # Apply masks to returns, getting portfolios with constituents
    for p in np.arange(1, n_portfolios+1):
        # Write each portfolios'constituent assets
        portfolios["portfolio"+str(p)] = returns.where(holdings_masks[p-1])
        # Get the equally-weighted return on each portfolio
        portfolios["p"+str(p)] = portfolios["portfolio"+str(p)].mean(axis=1)
        portfolios["p"+str(p)].name = "p" + str(p)

    return portfolios


def rank_cut_old(returns, signal_ranks, bins):
    """
    """
    rank_cuts = []  # output is a list
    # For each tuple in bins, select assets whose signals'percentile ranks lie
    # within bins, then append the output
    for p in range(len(bins)):
        if p == 0:  # first bin is closed interval
            rank_cuts.append(returns[
                (signal_ranks >= bins[p][0]) & (signal_ranks <= bins[p][1])])
        else:
            rank_cuts.append(returns[
                (signal_ranks > bins[p][0]) & (signal_ranks <= bins[p][1])])

    return rank_cuts


def rank_sort_adv_2(ret, sig, n_portf, hold_per=None, reb_dt=None,
    hold_between=None):
    """
    """
    no_hold_per = False
    if hold_per is None:
        no_hold_per = True
        hold_per = 0
    if reb_dt is None:
        # rebalance dates are all time periods
        reb_dt = ret.index

    # check datatype of rebalance_dates
    if isinstance(reb_dt[0], str):
        reb_dt = pd.to_datetime(reb_dt)

    # construct index -------------------------------------------------------
    # from [jan 11] to [jan 11, jan 12, jan 13...]
    hold_pattern = [
        np.arange(hold_per+1) + ret.index.get_loc(p,"bfill") for p in reb_dt]

    # hold_pattern is a list of arrays -> collapse; unique is needed because
    #   of overlaps in hold_pattern
    hold_pattern = np.unique(np.concatenate(hold_pattern))

    # now hold_pattern is an array of integers -> use it to select dates and
    #   convert to pandas.Index (otherwise complains on .reindex)
    hold_index = pd.Index(ret.index[hold_pattern])

    # reindex with forward fill: overlaps are taken care of
    sig = sig.reindex(index=hold_index, method="ffill")

    if no_hold_per:
        sig = sig.reindex(index=ret.index, method="ffill")

    # align
    al_sig, al_ret = sig.align(ret, join='left')

    # sort
    pf = rank_sort(returns=al_ret, signals=al_sig, n_portfolios=n_portf)

    if hold_between is not None:
        # hold_between = pd.DataFrame.from_dict(
        #     {c: hold_between.values for c in pf.keys if "portfolio" not in c)
        for key in pf.keys():
            if "portfolio" in key:
                pf[key].fillna(hold_between, inplace=True)

    return pf


def rank_sort(returns, signals, n_portfolios=3):
    """
    """
    returns = returns.dropna(how="all")
    signals = signals.dropna(how="all")

    # align two frames to ensure the index is the same
    returns, signals = returns.align(signals, axis=0, join="inner")

    # Get signal ranks row by row
    signal_ranks = signals.rank(
        axis=1,
        numeric_only=True,
        pct=True,
        method="average")

    # -----------------------------------------------------------------------
    # init space for bins
    bins = signal_ranks*np.nan

    # start hashing!
    hash_bins = {}

    # loop over rows; for each calculate # of assets, construct bins, hash
    #   them; result is a DataFrame of bin numbers
    for idx, row in signal_ranks.iterrows():
        # drop nans: needed for digitize later
        this_row = row.dropna()
        # Get number of assets available in the row xs
        n_assets = this_row.count()

        # hash bins
        if n_assets not in hash_bins:
            # Generate quantile bins, applying rule specified in 'custom_bins'
            hash_bins[n_assets] = custom_bins_2(n_assets, n_portfolios)

        # cut into bins
        bins.loc[idx,this_row.index] = np.digitize(
            this_row, hash_bins[n_assets])

    # -----------------------------------------------------------------------
    # write the list's contents into output dictionary
    portfolios = dict()

    # allocate
    for p in range(n_portfolios):
        # where bins are equal to 1,2,...
        this_portf = returns.where(bins == p)
        # write each portfolio's constituent assets
        portfolios["portfolio" + str(p+1)] = this_portf
        # get the equally-weighted return on each portfolio
        portfolios["p" + str(p+1)] = this_portf.mean(axis=1)
        portfolios["p" + str(p+1)].name = "p" + str(p+1)

    return portfolios

def custom_bins_2(n_assets, n_portf, epsilon=1e-05):
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
    custom_bins_2(3, 4) -> [1/3+e, 1/3+2e, 2/3+3e, 3/3+4e]
    """
    # number of assets in each portfolio
    nass = assets_in_each(n_assets, n_portf)

    # bin ranges
    bins = (np.array(nass)/n_assets+epsilon).cumsum()

    return bins

def assets_in_each(N, K):
    """ Calculate the number of assets in each portfolio with Dmitry's rule.

    Example with 7 assets, 5 portfolios: having assigned 2 assets to the 5th
    portfolio, the problem is to assign 5 assets to 4 portfolios, but in
    reversed order (2 to the first). The rest is careful handling of quotients.

    Parameters
    ----------
    N : int
        number of assets
    K : int
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
    quot = N // K
    rem = N % K

    # if N is a multiple of K, just assign them equally
    if rem == 0:
        return [quot,]*K

    # else start with the last portfolio
    init = [quot+1]

    # continue with the assigning fewer assets into fewer portfolios
    return assets_in_each(N-quot-1, K-1)[::-1] + init

def custom_bins(n_assets, n_portfolios):
    """
    Estimates quantiles for portfolio bins, using following allocation rule:
    if number of assets is not a multiple of number of portfolios, first
    additional asset goes to the last portfolio, second - to the first, third
    to the last-1, fourth to the first+1 and so on.

    Parameters
    ----------
    n_assets: int
        number of assets
    n_portfolios: int
        number of portfolios

    Returns
    -------
    bins: list
        of size n_portfolios where each element is a tuple, containing lower
        and upper quantiles for the corresponding portfolio

    """

    # CASE 1: #Assets / #Portfolios is an integer
    if n_assets % n_portfolios == 0:
        bins = [((p - 1) / n_portfolios, p / n_portfolios)
                for p in np.arange(1, n_portfolios+1)]
                            # +1 because the right interval in arange is open

    # CASE 2: #Assets / #Portfolios is not an integer
    else:
        # Get indices of portfolios where additionl assets to be placed
        add_asset_idx = [-1]  # the first
        for k in np.arange(2, n_assets % n_portfolios+1):
            add_asset_idx.append(add_asset_idx[-1] - (k-1) * (-1) ** (k-1))

        # Get number of assets in each portfolio
        assets_in_p = np.ones(n_portfolios) * (n_assets // n_portfolios)
        assets_in_p[add_asset_idx] = assets_in_p[add_asset_idx] + 1

        # Finally, construct custom bins
        bins = list(zip((assets_in_p.cumsum() - assets_in_p) / n_assets,
                        assets_in_p.cumsum() / n_assets))

    return bins


def get_holdings(portfolio):
    """ Utility function, returning DataFrame of holdings corresponding to the
    'portfolio' key of the 'rank_sort()' family of functions.

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


def rowCut(row, n_groups):
    """
    Assigns values in *row* to quantile groups. Groups are given names "p1",
    "p2",...,"pK", where K = n_groups.

    Parameters
    ----------
    row: array-like
        values to split according to quantiles
    n_groups: int
        number of groups to create

    Returns
    -------
    res: Categorical

    """
    res = pd.qcut(x      = row,
                  q      = n_groups,
                  labels = ['p'+str(n) for n in range(1,n_groups+1)])  # names
    return(res)


def constructPortfolios(data, signals, n_groups):
    """
    Based on the *signals* (of the same shape as *data*), constructs equally-
    weighted portfolios taking labels from *signals*.

    Parameters
    ----------
    data: pd.DataFrame
        assets to assign to portfolios
    signals: pd.DataFrame
        values to use for assigning to groups
    n_groups: int
        number of groups to create

    Returns
    -------
    res: pd.DataFrame
        constructed portfolios; df of dimension len(*data*.index) x n_groups

    """
    # allocate space for portfolios
    res = pd.DataFrame(index   = data.index,
                       columns = ['p'+str(n) for n in range(1,n_groups+1)],
                       dtype   = np.float)

    # for each row from the dataset
    for row in signals.iterrows():
        # cut quantiles into groups
        cut_sort_by = rowCut(row[1], n_groups)

        # sort data into these groups
        res.ix[row[0]] = data.ix[row[0]].groupby(cut_sort_by).mean()

    return(res)


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

def get_carry(pickle_name, key_name="spot_ret", transform=None, n_portf=3):
    """ Construct carry using sorts on forward discounts.
    """
    if transform is None:
        transform = lambda x: x

    # fetch data
    data_path = setc.gdrive_path("research_data/fx_and_events/")
    with open(data_path + pickle_name + ".p", mode='rb') as fname:
        data = pickle.load(fname)

    s = data[key_name]
    f = data["fwd_disc"]

    pfs = rank_sort(s, transform(f).shift(1), n_portfolios=n_portf)

    return get_factor_portfolios(pfs, hml=True)

def get_hml(returns, signals, n_portf=3):
    """
    """
    pf = rank_sort(returns, signals, n_portf)
    hml = get_factor_portfolios(pf, hml=True).loc[:,"hml"]

    return hml

def buy_before_events(s_d, fdisc_d, evts, fwd_maturity='W', burnin=1):
    """
    Parameters
    ----------
    s_d : pandas.DataFrame
        of spot returns; >0 means appreciation of foreign cur
    fdisc_d : pandas.DataFrame
        of forward discounts; >0 means higher US rf than foreign rate
    evts : pandas.DataFrame
        of events with diff in rates for values
    fwd_maturity : str
        'W','M'
    burnin : int
        how many periods to skip before/after events
    """
    # align -----------------------------------------------------------------
    common_cols = s_d.columns.intersection(
        fdisc_d.columns.intersection(evts.columns))
    start_dt = fwd.index[0]

    s_d = s_d.loc[start_dt:,common_cols]
    fdisc_d = fdisc_d.loc[start_dt:,common_cols]
    evts = evts.loc[start_dt:,common_cols]

    # maturity --------------------------------------------------------------
    if fwd_maturity == 'W':
        tau = 5
    elif fwd_maturity == 'M':
        tau = 22
    else:
        ValueError("maturity not implemented!")

    # hikes and cuts --------------------------------------------------------
    hikes = evts.where(evts > 0)
    cuts = evts.where(evts < 0)

    # init space
    res = dict()

    # before ----------------------------------------------------------------
    # hikes ---------------------------------------------------------------
    # reindex with events' dates
    fwd_reixed = fdisc_d.shift(burnin+tau).reindex(
        index=hikes.index, method="bfill")
    # sum of spot return over 5 days
    s_reixed = s_d.rolling(tau).sum().shift(burnin).reindex(
        index=hikes.index, method="bfill")
    # store
    res["before_hikes"] = \
        (-fwd_reixed+s_reixed).where(hikes.notnull())

    # cuts ----------------------------------------------------------------
    # reindex with events' dates
    fwd_reixed = fdisc_d.shift(burnin+tau).reindex(
        index=cuts.index, method="bfill")
    # sum of spot return over 5 days
    s_reixed = s_d.rolling(tau).sum().shift(burnin).reindex(
        index=cuts.index, method="bfill")
    # store
    res["before_cuts"] = \
        (-fwd_reixed+s_reixed).where(cuts.notnull())

    # after -----------------------------------------------------------------
    # hikes ---------------------------------------------------------------
    # reindex with events' dates
    fwd_reixed = fdisc_d.shift(-burnin).reindex(
        index=hikes.index, method="bfill")
    # sum of spot return over 5 days
    s_reixed = s_d.rolling(tau).sum().shift(-burnin-tau).reindex(
        index=hikes.index, method="bfill")
    # store
    res["after_hikes"] = \
        (-fwd_reixed+s_reixed).where(hikes.notnull())

    # cuts ----------------------------------------------------------------
    # reindex with events' dates
    fwd_reixed = fdisc_d.shift(-burnin).reindex(
        index=cuts.index, method="bfill")
    # sum of spot return over 5 days
    s_reixed = s_d.rolling(tau).sum().shift(-burnin-tau).reindex(
        index=cuts.index, method="bfill")
    # store
    res["after_cuts"] = \
        (-fwd_reixed+s_reixed).where(cuts.notnull())

    # merge
    res = pd.Panel.from_dict(res, orient="minor")
    merged_before = res.loc[:,:,"before_hikes"].fillna(
        -1*res.loc[:,:,"before_cuts"])
    merged_after = res.loc[:,:,"after_hikes"].fillna(
        -1*res.loc[:,:,"after_cuts"])

    # plot
    merged_before.mean(axis=1).dropna().cumsum().plot(color='g', label="bef")
    merged_after.mean(axis=1).dropna().cumsum().plot(color='r', label="aft")
    plt.gca().legend(loc="upper left")

    return merged_before, merged_after


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
    # from sample statistics -- no position is etablished
    timed_returns = timed_returns * (np.sign(signals).replace(0, np.nan))

    return timed_returns

def many_monthly_rx(s_d, fdisc_d):
    """ Get 22 overlapping monthly excess returns.
    Parameters
    ----------
    fdisc_d : pd.DataFrame
        forward discounts (30-day forwards) from perspective of US investor
    """
    s_d, fdisc_d = s_d.align(fdisc_d, axis=0)

    s = s_d.rolling(22).sum()
    f = fdisc_d.shift(22)

    all_m = dict()
    for p in range(22):
        all_m[p] = s_d.iloc[p::22,:] + f.iloc[p::22,:]

    return all_m

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

# ---------------------------------------------------------------------------
# limbo
# ---------------------------------------------------------------------------

# Dmitry's original rank sort, with hash

# def rank_sort(returns, signals, n_portfolios=3):
#     """Sorts a dataframe of returns into portfolios according to dataframe of
#     signals. If in any given cross-section the number of assets is not a
#     multiple of number of portfolios, additional assets are allocated according
#     to the rule specified in 'custom_bins' function.
#
#     Parameters
#     ----------
#     returns: DataFrame
#         of asset returns to be sorted in portfolios
#     signals: DataFrame
#         of signals accroding to which returns are sorted, this dataframe has
#         same shape as returns
#     n_portfolios: int
#         number if portfolios to sort returns into
#
#     Returns
#     -------
#     protfolios: dictionary
#         of dataframes. Keys p1, p2, ..., pN, where N = n_portfolios contain
#         returns on equally weighted portfolios from low to high. Keys
#         portfolio1, portfolio2, ..., portfolioN contain returns of individual
#         asssets in the corresponding portfolio
#
#     """
#     # output is a dictionary
#     portfolios = {}
#
#     # Consider observationas where both signals and returns are available
#     # returns = returns.where(returns * signals).dropna(how="all")
#     # signals = signals.where(returns * signals).dropna(how="all")
#     returns = returns.dropna(how="all")
#     signals = signals.dropna(how="all")
#
#     # align two frames to ensure the index is the same
#     returns, signals = returns.align(signals, axis=0)
#
#     # Get signal ranks row by row
#     signal_ranks = signals.rank(
#         axis = 1,           # get ranks for each row
#         numeric_only=True,  # ignore NaNs
#         pct=True,           # map ranks to [0, 1]
#         method="average")   # equal values' ranks are
#                             #   averaged
#
#     # create panel with number of elements equal n_portfolios;
#     portfolio_pan = pd.Panel(
#         items=returns.columns,
#         major_axis=returns.index,
#         minor_axis=range(n_portfolios))
#
#     # hash bins
#     hash_bins = {}
#
#     # iterate over signals' rows
#     for idx, row in signal_ranks.iterrows():
#
#         # Get number of assets available in the row xs
#         n_assets = row.count()
#
#         # hash bins
#         if n_assets not in hash_bins:
#             # Generate quantile bins, applying rule specified in 'custom_bins'
#             hash_bins[n_assets] = custom_bins(n_assets, n_portfolios)
#
#         #  cut into bins
#         portfolio_pan.loc[:,idx,:] = rank_cut(row, hash_bins[n_assets])
#
#     # write the list's contents into output dictionary
#     for p in range(n_portfolios):
#         this_portf = returns.where(portfolio_pan.loc[:,:,p])
#         # write each portfolios'constituent assets
#         portfolios["portfolio" + str(p+1)] = this_portf
#         # get the equally-weighted return on each portfolio
#         portfolios["p" + str(p+1)] = this_portf.mean(axis=1)
#         portfolios["p" + str(p+1)].name = "p" + str(p+1)
#
#     return portfolios

# Dmitry's rank_cut function, slightly changed

# def rank_cut(signal_ranks, bins):
#     """
#     Cuts a dataframe of returns into dataframes of returns on rank-sorted
#     portfolios, accepting percentile signal ranks and custom bins to cut upon,
#     and returning list of dataframes where each element contains returns on
#     assets whose rank is within corresponding bin.
#
#     Parameters
#     ----------
#     returns: DataFrame
#         (or series) containing a cross-section of returns for given time point
#     signal_ranks: DataFrame
#         (or series) containing ranks on which returns are to be sorted
#     bins: list
#         where each element is a tuple, containing lower and upper quantiles for
#         the corresponding portfolio, number of portfolios equals the number of
#         elements in the list
#
#     Returns
#     -------
#     rank_cuts: list
#         of dataframes with each dataframe containing returns whose pecentile
#         signal rank lies within quantiles in the corresponding element of bins
#
#     """
#     M, N = len(bins), len(signal_ranks)
#     rank_cuts = pd.DataFrame(
#         columns=signal_ranks.index,
#         index=range(M))
#
#     rank_cuts.loc[0,:] = \
#         (signal_ranks >= bins[0][0]) & (signal_ranks <= bins[0][1])
#
#     for p in range(1,M):
#         rank_cuts.loc[p,:] = \
#             (signal_ranks > bins[p][0]) & (signal_ranks <= bins[p][1])
#
#     return rank_cuts
#
