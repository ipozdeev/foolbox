import pandas as pd
import numpy as np

def rank_sort(returns, signals, n_portfolios):
    """
    Sorts a dataframe of returns into portfolios according to dataframe of
    signals. If in any given cross-section the number of assets is not a
    multiple of number of portfolios, additional assets are allocated according
    to the rule specified in 'custom_bins' function.

    Parameters
    ----------
    returns: DataFrame
        of asset returns to be sorted in portfolios
    signals: DataFrame
        of signals accroding to which returns are sorted, this dataframe has
        same shape as returns
    n_portfolios: int
        number if portfolios to sort returns into

    Returns
    -------
    protfolios: dictionary
        of dataframes. Keys p1, p2, ..., pN, where N = n_portfolios contain
        returns on equally weighted portfolios from low to high. Keys
        portfolio1, portfolio2, ..., portfolioN contain returns of individual
        asssets in the corresponding portfolio

    """

    portfolios = {}  # output is a dictionary

    # Consider observationas where both signals and returns are available
    returns = returns[pd.notnull(signals)]
    signals = signals[pd.notnull(returns)]

    # align two frames to ensure the index is the same
    returns, signals = returns.align(signals, axis = 0)

    # Get signal ranks row by row
    signal_ranks = signals.rank(axis = 1,           # get ranks for each row
                                numeric_only=True,  # ignore NaNs
                                pct=True,           # map ranks to [0, 1]
                                method="average")   # equal values' ranks are
                                                    # averaged

    # Create a list of dataframes with number of elements equal n_portfolios
    # the first element of the list is the 'low' portfolio, the last is 'high'
    portf_list = [pd.DataFrame(columns=returns.columns)
                    for portfolio in range(n_portfolios)]

    # Start iteration through signals' rows

    for row in signal_ranks.iterrows():
        # Get number of assets available in the row xs
        n_assets = pd.notnull(row[1]).sum()
        # Generate quantile bins, applying rule specified in 'custom_bins'
        bins = custom_bins(n_assets, n_portfolios)
        # Get portfolios by cutting xs into bins
        rank_cuts = rank_cut(returns.ix[row[0]], row[1], bins)
        # Finally, append the dataframes in portf_list with new rows
        for p in np.arange(1, n_portfolios+1):
            portf_list[p-1] = portf_list[p-1].append(rank_cuts[p-1])

    # Write the list's contents into output dictionary
    for p in np.arange(1, n_portfolios+1):
        # Write each portfolios'constituent assets
        portfolios["portfolio" + str(p)] = portf_list[p-1]
        # Get the equally-weighted return on each portfolio
        portfolios["p" + str(p)] = portf_list[p-1].mean(axis=1)
        portfolios["p" + str(p)].name = "p" + str(p)

    return portfolios


def rank_cut(returns, signal_ranks, bins):
    """
    Cuts a dataframe of returns into dataframes of returns on rank-sorted
    portfolios, accepting percentile signal ranks and custom bins to cut upon,
    and returning list of dataframes where each element contains returns on
    assets whose rank is within corresponding bin.

    Parameters
    ----------
    returns: DataFrame
        (or series) containing a cross-section of returns for given time point
    signal_ranks: DataFrame
        (or series) containing ranks on which returns are to be sorted
    bins: list
        where each element is a tuple, containing lower and upper quantiles for
        the corresponding portfolio, number of portfolios equals the number of
        elements in the list

    Returns
    -------
    rank_cuts: list
        of dataframes with each dataframe containing returns whose pecentile
        signal rank lies within quantiles in the corresponding element of bins

    """
    rank_cuts = []  # output is a list
    # For each tuple in bins, select assets whose signals'percentile ranks lie
    # within bins, then append the output
    for p in range(len(bins)):
        if p == 0:  # first bin is closed interval
            rank_cuts.append(returns[(signal_ranks >= bins[p][0]) &
                          (signal_ranks <= bins[p][1])])
        else:
            rank_cuts.append(returns[(signal_ranks > bins[p][0]) &
                          (signal_ranks <= bins[p][1])])

    return rank_cuts


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
