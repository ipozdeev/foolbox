import pandas as pd
from pandas.tseries.offsets import DateOffset
import numpy as np
from foolbox.data_mgmt import set_credentials as setc
import pickle


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


def get_carry(pickle_name, key_name="spot_ret", transform=None,
    x_curs=[], n_portf=3):
    """ Construct carry using sorts on forward discounts.
    """
    if transform is None:
        transform = lambda x: x

    # fetch data
    data_path = setc.gdrive_path("research_data/fx_and_events/")
    with open(data_path + pickle_name + ".p", mode='rb') as fname:
        data = pickle.load(fname)

    s = data[key_name].drop(x_curs, axis=1, errors="ignore")
    f = data["fwd_disc"].drop(x_curs, axis=1, errors="ignore")

    pfs = rank_sort(s, transform(f).shift(1), n_portfolios=n_portf)

    return get_factor_portfolios(pfs, hml=True)


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


def wght_grid(pf):
    """
    """
    pf_keys = [p for p in pf.keys() if "portfolio" in p]
    n_pfs = len(pf_keys)

    wghts = dict()
    for p in pf_keys:
        # p = "portfolio2"
        this_pf = pf[p].mask(pf[p].notnull(), 1.0)
        wghts[p] = this_pf.divide(this_pf.mean(axis=1)*this_pf.count(axis=1),
            axis=0)

    return wghts


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
