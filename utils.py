import pandas as pd
from pandas.tseries.offsets import *
from pandas.tseries.holiday import *

import numpy as np
import pickle


def remove_outliers(data, stds):
    """
    """
    res = data.where(np.abs(data) < data.std()*stds)

    return res


def fetch_the_data(path_to_data, drop_curs=[], align=False, add_usd=False):
    """
    """
    # Import the FX data
    with open(path_to_data+"fx_by_tz_aligned_d.p", mode="rb") as fname:
        data_merged_tz = pickle.load(fname)

    # Import the all fixing times for the dollar index
    with open(path_to_data+"fx_by_tz_sp_fixed.p", mode="rb") as fname:
        data_all_tz = pickle.load(fname)

    # Get the individual currencies, spot rates:
    spot_bid = data_merged_tz["spot_bid"]
    spot_ask = data_merged_tz["spot_ask"]
    swap_ask = data_merged_tz["tnswap_ask"]
    swap_bid = data_merged_tz["tnswap_bid"]

    # outliers in swap points
    swap_ask = remove_outliers(swap_ask, 50)
    swap_bid = remove_outliers(swap_bid, 50)

    # Align and ffill the data, first for tz-aligned countries
    (spot_bid, spot_ask, swap_bid, swap_ask) = align_and_fillna(
        (spot_bid, spot_ask, swap_bid, swap_ask),
        "B", method="ffill")

    if add_usd:

        spot_ask_us = data_all_tz["spot_ask"].loc[:,:,"NYC"]
        spot_bid_us = data_all_tz["spot_bid"].loc[:,:,"NYC"]
        swap_ask_us = data_all_tz["tnswap_ask"].loc[:,:,"NYC"]
        swap_bid_us = data_all_tz["tnswap_bid"].loc[:,:,"NYC"]

        swap_ask_us = \
            swap_ask_us.where(np.abs(swap_ask_us) < swap_ask_us.std()*25)\
            .dropna(how="all")
        swap_bid_us = \
            swap_bid_us.where(np.abs(swap_bid_us) < swap_bid_us.std()*25)\
            .dropna(how="all")

        # Now for the dollar index
        (spot_bid_us, spot_ask_us, swap_bid_us, swap_ask_us) =\
            align_and_fillna((spot_bid_us, spot_ask_us,
                              swap_bid_us, swap_ask_us),
                             "B", method="ffill")

        spot_bid.loc[:,"usd"] = spot_bid_us.drop(drop_curs,axis=1).mean(axis=1)
        spot_ask.loc[:,"usd"] = spot_ask_us.drop(drop_curs,axis=1).mean(axis=1)
        swap_bid.loc[:,"usd"] = swap_bid_us.drop(drop_curs,axis=1).mean(axis=1)
        swap_ask.loc[:,"usd"] = swap_ask_us.drop(drop_curs,axis=1).mean(axis=1)

    prices = pd.Panel.from_dict(
        {"bid": spot_bid, "ask": spot_ask},
        orient="items").drop(drop_curs, axis="minor_axis")
    swap_points = pd.Panel.from_dict(
        {"bid": swap_bid, "ask": swap_ask},
        orient="items").drop(drop_curs, axis="minor_axis")

    return prices, swap_points


def align_and_fillna(data, reindex_freq=None, common_index="outer", **kwargs):
    """
    Parameters
    ----------
    data : list-like
        of pandas.DataFrames or pandas.Series
    reindex_freq : str
        pandas frequency string, e.g. 'B' for business day
    common_index: str
        argument of 'pd.concat()' inner or outer index concatenation. Default
        is outer
    kwargs : dict
        arguments to .fillna()
    """
    common_idx = pd.concat(data, axis=1, join="outer").index

    if reindex_freq is not None:
        common_idx = pd.date_range(common_idx[0], common_idx[-1],
            freq=reindex_freq)

    new_data = None

    if isinstance(data, dict):
        new_data = {}
        for k,v in data.items():
            new_data.update({k: v.reindex(index=common_idx).fillna(**kwargs)})
    elif isinstance(data, tuple):
        new_data = tuple(
            [p.reindex(index=common_idx).fillna(**kwargs) for p in data])
    elif isinstance(data, list):
        new_data = [p.reindex(index=common_idx).fillna(**kwargs) for p in data]

    return new_data


def add_fake_signal(ret, sig):
    """ Add fake series to signal: the median in each row.
    """
    r, s = ret.copy(), sig.copy()

    # calculate median across rows
    fake_sig = sig.apply(np.nanmedian, axis=1)

    # reinstall
    s.loc[:,"fake"] = fake_sig
    r.loc[:,"fake"] = np.nan

    return r, s


def interevent_quantiles(events, max_fill=None, df=None):
    """ Split inter-event intervals in two parts.
    Parameters
    ----------
    events : pandas.Series or pandas.DataFrame
        of events; any non-na value counts as separate event
    df : (optional) pandas.DataFrame
        optional dataframe to concatenate the indicator column to

    Returns
    -------
    res : (if df is None) pandas.DataFrame with new columns: event number and
        quantiles for each column of `events` (if more than one); (if df is
        not None) df.copy with new columns added

    Example
    -------
    pd.set_option("display.max_rows", 20)
    data = pd.DataFrame(14.9, index=range(20), columns=["data",])
    events = pd.DataFrame(index=range(20), columns=["woof","meow"])
    events.iloc[[2,10,14], 0] = 1
    events.iloc[[5,10,15], 1] = 1
    events = events.iloc[:,[0]]
    %timeit interevent_quantiles(events, max_fill=None, df=data)
    %timeit -n 100 interevent_qcut(data, events.dropna(), 2)
    """
    # ensure no 1-columns dataframes enter
    events = events.squeeze()

    # recursion, if `events` have more than 1 column
    if isinstance(events, pd.DataFrame):
        res = interevent_quantiles(
            events=events.iloc[:,1:].squeeze(),
            max_fill=max_fill,
            df=interevent_quantiles(events.iloc[:,0], max_fill, df))
        return res

    # no limit on the amount filled by default
    if max_fill is None:
        max_fill = 1e05

    # name
    evt_name = "evt" if events.name is None else events.name

    # leave only 0.0 and nan's
    evts_q = events.notnull().where(events.notnull())*0

    # remove trailing and leading nans: no info where these periods start(end)
    evts_q = evts_q.loc[evts_q.first_valid_index():evts_q.last_valid_index()]

    # index events
    # evts_idx = evts_q.dropna()
    # evts_idx = evts_idx.add(np.arange(1,len(evts_idx)+1))
    # evts_idx = evts_idx.reindex(index=evts_q.index, method="ffill")
    evts_idx = evts_q.expanding().count()

    # helper: this will be modified each iteration
    evts_help = evts_q.copy().astype(float)

    # while there are nan's, forward-fill then backward-fill one cell
    cnt = 1
    while evts_q.isnull().sum() > 0:

        temp_evts = evts_help.replace(0.0, 1.0)
        evts_q.fillna(temp_evts.fillna(method="ffill", limit=1), inplace=True)
        temp_evts = evts_help.replace(0.0, 2.0)
        evts_q.fillna(temp_evts.fillna(method="bfill", limit=1), inplace=True)

        # make sure helper changes
        evts_help = evts_q.copy()

        cnt += 1

        # break if more than a certain amount
        if cnt > max_fill:
            break

    # concatenate
    res = pd.concat((evts_idx, evts_q), axis=1)
    res.columns = ["_evt_" + p + evt_name for p in ["num_", "q_"]]

    # keep original index
    res = res.reindex(index=events.index)

    # patch with two additional columns
    if df is not None:
        # df = df.reindex(index=evts_q.index)
        df = pd.concat((df, res), axis=1)

    return (res if df is None else df)


def interevent_qcut(data_to_slice, events, n_quantiles):
    """Given a dataframe of data assigns each data point to an interevent
    quantile.


    Parameters
    ----------
    data_to_slice: pd.DataFrame
        with data to classify by interevent quantile
    events: pd.DataFrame
        of events according to whuch the data is to be classified
    n_quantiles: int
        number of quantiles within each interevent period

    Returns
    -------
    out: pd.DataFrame

    Example
    -------
    pd.set_option("display.max_rows", 20)
    data_to_slice = pd.DataFrame(14.9, index=range(20), columns=["woof"])
    events = pd.DataFrame(index=range(20), columns=["one",])
    events.iloc[[2,10,14], 0] = 1
    events.iloc[[5,10,15], 1] = 1
    interevent_qcut(data_to_slice,events.dropna(),n_quantiles=3)
    """
    # assert isinstance(events, pd.DataFrame)
    #
    # # recursion if multiple events columns
    # if events.shape[1] > 1:
    #     res = interevent_qcut(
    #         data_to_slice=interevent_qcut(
    #             data_to_slice, events.iloc[:,[0]], n_quantiles),
    #         events=events.iloc[:,1:],
    #         n_quantiles=n_quantiles)
    #     return res

    # store name
    col_name = "evt_q"

    data = data_to_slice.copy()

    # Make sure events do not contain NaNs
    if events.isnull().any()[0]:
        raise ValueError("A NaN found in events, ain't going no further")
    # if events.isnull().sum()[0] > 0:
    #     warnings.warn("There are missing values in data! Will drop.")
    #     events = events.dropna()

    # Locate first and last event dates
    evt_first = events.index[0]
    evt_last = events.index[-1]

    # Sample the data between the first and last events
    data = data.loc[evt_first:evt_last, :]

    # Span events by data, get event number for each day
    events_spanned = events.reindex(data.index)
    event_number = events_spanned.expanding().count()

    # Add event number and classification columns for further groupby
    data["evt_num"] = event_number
    data[col_name] = np.nan

    # Take a subsample between first and last events only contingent on both
    # data and events
    data = data.loc[data.evt_num > 0]
    events = events.loc[data.index[0]: data.index[-1]]

    # Output has the same layout as data
    out = pd.DataFrame(index=data.index, columns=data.columns)

    # Take the event days as a separate case and drop them from the data
    announcement_days = data.loc[events.index]
    announcement_days[col_name] = "event"

    # Drop the announcement days
    data = data.drop(announcement_days.index, axis=0)

    # Make quantile labels
    quantile_labels = ["q" + str(q+1) for q in range(n_quantiles)]

    for evt_num, df in data.groupby(["evt_num"]):
        df["day_count"] = df["evt_num"].expanding().count()
        df[col_name] = pd.qcut(df["day_count"], n_quantiles, quantile_labels)
        out.loc[df.index, :] = df.loc[:, data.columns]

    # Plug in the announcement days
    out.loc[announcement_days.index, data.columns] = announcement_days

    return out.drop(["evt_num"], axis=1)


def parse_bloomberg_excel(filename, colnames_sheet, data_sheets):
    """Parse .xlsx frile constructed with Bloomberg Excel API.

    Returns
    -------
    this_data : pandas.DataFrame
        of data
    """
    if isinstance(data_sheets, str):
        data_sheets = [data_sheets, ]
        flag_single_input = True
    else:
        flag_single_input = False

    # converter for date
    def converter(x):
        try:
            res = pd.to_datetime(x)
        except:
            res = pd.NaT
        return res

    # if data_sheets is None, read in all sheets
    if data_sheets is None:
        data_dict_full = pd.read_excel(filename,
            sheet_name=None,
            header=0)
    else:
        data_dict_full = pd.read_excel(filename,
            sheet_name=data_sheets+[colnames_sheet],
            header=0)

    # fetch column names from respective sheet
    colnames = data_dict_full.pop(colnames_sheet).columns

    # loop over sheetnames
    all_data = dict()

    for s, data_df in data_dict_full.items():
        # loop over triplets, map dates, extract
        new_data_df = []
        for p in range((data_df.shape[1]+1)//3):
            # this triplet
            this_piece = data_df.iloc[1:, p*3:(p+1)*3-1]

            # map date
            this_piece.iloc[:, 0] = this_piece.iloc[:, 0].map(converter)

            # drop nans (with dates)
            this_piece = this_piece.dropna()

            # extract date as index
            this_piece = this_piece.set_index(this_piece.columns[0])

            # rename
            this_piece.columns = [colnames[p]]
            this_piece.index.name = "date"

            # store
            new_data_df += [this_piece, ]

        # concat
        all_data[s] = pd.concat(new_data_df, axis=1, join="outer")

    # if only one sheet was asked for
    if flag_single_input:
        all_data = list(all_data.values())[0]

    return all_data


def compute_floating_leg_return(trade_dates, returns, maturity, settings):
    """

    Parameters
    ----------
    trade_dates: pd.tseries.index.DatetimeIndex
        with trade dates of OIS contracts (dates when the fixed leg is quoted)
    returns: pd.Series
        of returns on the underlying overnight (or tom/next) rate
    maturity: pd.tseries.offsets.DateOffset
        specifying maturity of the contract. For example DateOffset(months=1)
        corresponds to the maturity of one month
    settings: dict
        specifying contract's conventions

    The dictionary should look like:

        {"start_date": 2,
         "fixing_lag": 1,
         "day_count_float_dnm": 360,
         "date_rolling": "modified following"}

    The items in the dictionary are:
    start_date: int
        number of business days until the first accrual of floating rate
    fixing_lag: int
        number of periods to shift returns _*forward*_ that is
        if the underlyig rate is reported with one period lag, today's rate on
        the floating leg is determined tomorrow. The value is negative if
        today's rate is determined yesterday
    day_count_float_dnm: int
        360 or 365 corresponding to Act/360 and Act/365 day count conventions
    date_rolling: str
        specifying date rolling convention if the end date of the contract is
        not a business day. Typical conventions are: "previous", "following",
        and "modified following". Currently only the latter two are implemented


    Returns
    -------
    out: dict
        with two dataframe and series as items. Key 'dates' contains dataframe
        of start and end dates for each realized return for a swap traded on
        date in trade dates. Key 'ret' contains the realized return

    """
    # Prepare the output structures
    out_dates = pd.DataFrame(index=trade_dates, columns=["start", "end"],
                             dtype="datetime64[ns]")
    out_returns = pd.Series(index=trade_dates, name=returns.name)

    # Prepare the data and get a shrothand notation for settings
    returns = returns.shift(-settings["fixing_lag"])  # publication or t/n lag
    day_count = settings["day_count_float_dnm"]

    # Start the main lOOp
    for date in trade_dates:
        # Set the start date - the first date of accrued floating interest
        # TODO: there's pandas.tseries.holiday, with holiday calendars...
        start_date = date + BDay(settings["start_date"])

        # End date
        end_date = start_date + maturity

        # Handle the date rolling if end_date is not a business day
        # TODO: relegate date rolling to a separate function to avoid clutter?
        if settings["date_rolling"] == "previous":
            # Roll to the previous business day
            end_date -= BDay(0)

        elif settings["date_rolling"] == "following":
            # Roll to the next business day
            end_date += BDay(0)

        elif settings["date_rolling"] == "modified following":
            # Try to roll forward
            tmp_end_date = end_date + BDay(0)

            # If the date is in the following month roll backwards instead
            if tmp_end_date.month == end_date.month:
                end_date += BDay(0)
            else:
                end_date -= BDay(0)

        else:
            raise NotImplementedError("{} date rolling is not supported"
                                      .format(settings["date_rolling"]))

        # Append the output dates
        out_dates.loc[date, "start"] = start_date
        out_dates.loc[date, "end"] = end_date

        # Append returns, if the end date has already happened
        if end_date <= returns.index[-1]:
            # Reindex to calendar day, carrying rates forward over non b-days
            tmp_ret = returns.reindex(
                pd.DatetimeIndex(start=start_date, end=end_date, freq="D"),
                method="ffill")

            # Compute the cumulative floating leg return over the period
            out_returns.loc[date] = \
                100 * (day_count / tmp_ret.size) * \
                ((1 + tmp_ret / day_count / 100).prod() - 1)

    # Populate the output
    out = {"dates": out_dates, "ret": out_returns}

    return out


def next_days_same_rate(dt, b_day=None):
    """
    dt : str/np.datetime64
    """
    if b_day is None:
        b_day = BDay()

    dt = pd.to_datetime(dt)

    next_b_day = dt + b_day

    res = (next_b_day - dt).days

    return res


def to_better_latex(df_coef, df_tstat, fmt_coef="{}", fmt_tstat="{}",
    **kwargs):
    """
    df_coef = pd.DataFrame(np.eye(2)*3, index=["on","tw"], columns=["A","B"])
    df_coef.iloc[0, 0] = None
    df_tstat = pd.DataFrame(np.eye(2), index=["on","tw"], columns=["A","B"])
    fmt_coef = '{:3.2f}'
    fmt_tstat = fmt_coef

    """
    weird_fmt = lambda x: "\multicolumn{1}{c}{" + str(x) + "}"

    # def signif_fmt(x):
    #     if x <= 0.01:
    #         star = "***"
    #     elif x <= 0.05:

    mask_coef = df_coef.applymap(np.isfinite)
    mask_tstat = df_tstat.applymap(np.isfinite)

    df_coef_fmt = df_coef.applymap(fmt_coef.format)
    df_tstat_fmt = df_tstat.applymap(('('+fmt_tstat+')').format)

    new_df = []
    for p, q in zip(*[list(r.values) for r in (df_coef_fmt, df_tstat_fmt)]):
        new_df += [p]
        new_df += [q]

    new_mask = []
    for p, q in zip(*[list(r.values) for r in (mask_coef, mask_coef)]):
        new_mask += [p]
        new_mask += [q]

    new_idx = []
    for p, q in zip(list(df_coef.index), [' ',]*len(df_coef.index)):
        new_idx += [p]
        new_idx += [q]

    # new_col = [weird_fmt(p) for p in df_coef.columns]
    new_col = df_coef.columns

    new_df = pd.DataFrame(new_df, index=new_idx, columns=new_col)
    new_mask = pd.DataFrame(new_mask, index=new_idx, columns=new_col)
    new_df = new_df.applymap(weird_fmt).where(~new_mask).fillna(new_df)

    return new_df.to_latex(**kwargs)


def resample_between_events(data, events, fun, mask=None):
    """
    """
    mask = mask.fillna(False)
    mask = ~mask

    idx = pd.Series(events.index, index=events.index).reindex(
        index=data.index, method="ffill")
    idx.name = "index"

    # mask; this way is better than .loc[mask_idx] because of non-empty set
    #   difference of the two
    if mask is not None:
        idx = idx.where(mask)

    # concat `data` and `idx` to be able to group by column
    to_group = pd.concat((data, idx), axis=1)

    # groupby!
    res = to_group.groupby("index").agg(fun)

    return res


def apply_between_events(data, events, func, lag=None, **kwargs):
    """

    Parameters
    ----------
    data
    events
    func : callable
    lag : int/DateOffset or dict/pandas.Series thereof
        if dict, {key: value} for key in data.columns
    kwargs : dict
        arguments to `func`

    Returns
    -------
    res

    """
    # works on pandas.Series
    if isinstance(data, pd.Series):
        # lag: defaults
        if lag is None:
            lag = Day(1)

        if isinstance(lag, (int, np.int)):
            add_lag = Day(1)*lag
        elif isinstance(lag, DateOffset):
            add_lag = lag*1

        # watch out for nan in `events`
        data_a, _ = data.align(events.dropna(), axis=0, join="outer")

        # space for data
        res = data_a * np.nan

        # fill forward-like
        for t in events.dropna().index[::-1]:
            res.loc[(t + add_lag):] = res.loc[(t + add_lag):].fillna(
                func(data_a.loc[(t + add_lag):])
            )

        # add the very start
        res = res.fillna(func(data_a))

    # else recursion
    elif isinstance(data, pd.DataFrame):
        assert data.columns.equals(events.columns)
        assert isinstance(lag, (dict, pd.Series))
        res = {
            apply_between_events(data.loc[:, c], events.loc[:, c],
                                 func, lag[c])
            for c in data.columns
        }

    else:
        res = None

    return res


if __name__ == "__main__":
    pass
