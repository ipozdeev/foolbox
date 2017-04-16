import pandas as pd
import numpy as np
from scipy.stats import norm
import random
from pandas.tseries.offsets import DateOffset
import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import ipdb
# %matplotlib

# matplotlib settings
plt.rc("font", family="serif", size=12)
gr_1 = "#8c8c8c"

class EventStudy():
    """
    Parameters
    ----------
    data : pandas.DataFrame/pandas.Series
        of assets
    events : list/pandas.DataFrame/pandas.Series
        of events indexed with dates; if DataFrame, must contain one column
        only
    window : tuple of int
        [a,b,c,d] where each element is relative (in periods) date of
        a - start of cumulating returns in `data` before event (negative int);
        b - stop of cumulating returns before event (negative int),
        c - start of cumulating returns in `data` after event,
        d - stop of cumulating returns before event.
        For example, [-5,-1,1,5] means that one starts to look at returns 5
        periods before each event, ends 1 day before, starts again 1 day after
        and stops 5 days after.
    """
    def __init__(self, data, events, window):
        """ Create class, pivot data and center on events.
        """
        # unpack window into individual items
        ta, tb, tc, td = window

        # if `events` is not a pandas object:
        if not issubclass(events.__class__, pd.core.generic.NDFrame):
            events = pd.Series(data=np.arange(len(events)),index=events)

        # if not a DataFrame already -> convert to it because iterrows later
        if isinstance(events, pd.DataFrame):
            events = events.iloc[:,0]

        # # check that events are uniquely labelled (e.g. 0,1,2...)
        # if events.duplicated().sum() > 0:
        #     events = \
        #         pd.Series(index=events.index,data=np.arange(len(events)))

        events = \
            pd.Series(index=events.index,data=np.arange(len(events)))

        # shape
        T = len(data)
        #
        # # keep record of dimensions (needed for later)
        # self.is_1d = N < 2

        # if last event is closer to the end than td - delete last event
        # TODO: write logger for that
        good_evt_idx = events.index.map(
            lambda x: (get_idx(data, x) < (T-td)) & \
                (get_idx(data, x) > -ta-1))
        events = events.ix[good_evt_idx]

        # pivot
        before, after, grp_var, tot_mu, tot_var = self.pivot_for_event_study(
            data, events, window)

        self.data = data
        self.events = events
        self.window = window
        self.before = before
        self.after = after
        self.before_idx = before.index
        self.after_idx = after.index
        self.stacked_idx = np.hstack((before.index, after.index))
        self.grp_var = grp_var
        self.tot_mu = tot_mu
        self.tot_var = tot_var

    @staticmethod
    def pivot_for_event_study(data, events, window):
        """ Pivot `data` to center on events.
        """
        assert isinstance(data, pd.Series)

        if data.name is None:
            data.name = "response"

        # pdb.set_trace()
        ta, tb, tc, td = window

        # find dates belonging to two event windows simultaneously
        belongs_idx = pd.Series(data=0,index=data.index)
        for t in events.index:
            # fetch position index of this event
            this_t = get_idx(data,t)
            # record span of its event window
            belongs_idx.ix[(this_t+ta):(this_t+td+1)] += 1

        # set values in rows belonging to multiple events to nan
        data_to_pivot = data.copy().to_frame()
        data_to_pivot.loc[belongs_idx > 1] *= np.nan

        # introduce columns for pivoting
        data_to_pivot.loc[:,"event_window"] = np.nan
        # introduce columns for pivoting
        data_to_pivot.loc[:,"event_no"] = np.nan
        # introduce columns for standard deviations
        data_to_pivot.loc[:,"which_evt"] = 0

        # loop over events, save snapshot of returns around each
        for t, evt in events.iteritems():
            # fetch index of this event
            this_t = get_idx(data_to_pivot,t)
            # index [ta:tb]+[tc:td]
            window_idx = np.concatenate((
                np.arange((this_t+ta),(this_t+tb+1)),
                np.arange((this_t+tc),(this_t+td+1))))
            # put event identifier to be used for pivoting later
            data_to_pivot.ix[window_idx,"event_no"] = evt
            # index with [ta:tb]+[tc:td]
            data_to_pivot.ix[window_idx,"event_window"] = \
                    np.concatenate((np.arange(ta, tb+1),
                    np.arange(tc, td+1)))
            # for standard deviations
            data_to_pivot.ix[(this_t+ta):,"which_evt"] += 1

        # drop rows except those selected around event
        pivoted = data_to_pivot.dropna(subset=["event_no"])

        # column to integers: need events and periods to be integers
        pivoted.loc[:,"event_no"] = pivoted.loc[:,"event_no"].map(int)
        pivoted.loc[:,"event_window"] = pivoted.loc[:,"event_window"].map(int)

        pivoted = pivoted.pivot(
            index="event_window",
            columns="event_no",
            values=data.name)
        # for std
        for_var = data_to_pivot.where(data_to_pivot["event_no"].isnull())

        # calculate variances
        grouped_var = for_var[data.name].groupby(for_var["which_evt"]).var()
        grouped_var.index = grouped_var.index.map(int)

        # variance without grouping by
        total_var = for_var[data.name].var()

        # calculate means
        grouped_mu = for_var[data.name].groupby(for_var["which_evt"]).mean()\
            .mean()

        # before
        # ipdb.set_trace()
        before = pivoted.loc[:tb,:]
        # after
        after = pivoted.loc[tc:,:]

        return before, after, grouped_var, grouped_mu, total_var

    @staticmethod
    def get_ts_cumsum(before, after):
        """ Calculate cumulative sum over time.
        """
        # the idea is to buy stuff at relative date a, where T is the event
        #   date and keep it until and including relative date b, so need to
        #   reverse this Panel for further cumulative summation
        ts_mu_before = before.ix[::-1,:].cumsum().ix[::-1,:]
        # after events is ok as is
        ts_mu_after = after.cumsum()

        # concat
        ts_mu = pd.concat((ts_mu_before, ts_mu_after), axis=0)

        return ts_mu

    @staticmethod
    def get_cs_mean(before, after):
        """ Calculate mean across events
        """
        cs_mu_before = before.mean(axis=1)
        cs_mu_after = after.mean(axis=1)

        # concat
        cs_mu = pd.concat((cs_mu_before, cs_mu_after), axis=0)

        return cs_mu

    def get_cs_ts(self, before, after):
        """ Calculate time-series cumulative sum and its average across events.
        """
        ts_mu = self.get_ts_cumsum(before, after)
        cs_ts_mu = ts_mu.mean(axis=1)

        return cs_ts_mu

    def get_ci(self, ps, method="simple", **kwargs):
        """ Calculate confidence bands.
        Parameters
        ----------
        p : float / tuple of floats
            interval width or tuple of (lower bound, upper bound)
        Returns
        -------
        ci : pd.Panel

        """
        # if `ps` was provided as single float
        if isinstance(ps, float):
            ps = ((1-ps)/2, ps/2+1/2)

        # calculate confidence band
        if method == "simple":
            ci = self.simple_ci(ps=ps)
        elif method == "boot":
            ci = self.boot_ci(ps, **kwargs)
        else:
            raise NotImplementedError("ci you asked for is not implemented")

        self.ci = ci

        return ci

    def simple_ci(self, ps):
        """
        """
        ta, tb, tc, td = self.window

        sigma = np.sqrt(self.grp_var.sum())/self.grp_var.count()
        self.sigma = sigma

        # mu = boot_from.mean()

        # # sd of mean across events is mean outside of events divided
        # #   through sqrt(number of events)
        # sd_across = boot_from.std()/np.sqrt(len(self.events))
        # # sd of cumulative sum of mean across events is sd times sqrt(# of
        # #   cumulants)
        # # pdb.set_trace()

        q = np.sqrt(np.hstack(
            (np.arange(-ta+tb+1,0,-1), np.arange(1,td-tc+2))))
        # q = q[:,np.newaxis]

        # # multiply with broadcast, add mean
        # ci_lo = norm.ppf(ps[0])*q*sd_across.values[np.newaxis,:] + \
        #     boot_from.mean().values[np.newaxis,:]*q**2
        # ci_hi = norm.ppf(ps[1])*q*sd_across.values[np.newaxis,:] + \
        #     boot_from.mean().values[np.newaxis,:]*q**2
        # multiply with broadcast, add mean
        ci_lo = norm.ppf(ps[0])*q*sigma + self.tot_mu*(q**2)
        ci_hi = norm.ppf(ps[1])*q*sigma + self.tot_mu*(q**2)

        # concatenate: items keeps columns of Y
        ci = pd.DataFrame(
            index=self.stacked_idx,
            columns=ps)

        ci.loc[:,ps[0]] = ci_lo
        ci.loc[:,ps[1]] = ci_hi

        return ci

    def boot_ci(self, ps, M=500):
        """
        Returns
        -------
        ci : pandas.DataFrame
            with columns for confidence interval bands
        """
        ta, tb, tc, td = self.window

        # drop dates around events, G times window length
        boot_from = self.data.copy()
        for t in self.events.index:
            this_t = get_idx(boot_from, t)
            boot_from.drop(
                boot_from.index[(this_t+ta):(this_t+td)],
                axis="index",
                inplace=True)

        # space for df's of pivoted tables
        booted = pd.DataFrame(
            columns=range(M),
            index=self.stacked_idx)

        # possible times should exclude values that are too early or too late
        possible_dates = boot_from.index.tolist()[
            (-self.window[0]):-(self.window[-1]+1)]

        for p in range(M):
            # pdb.set_trace()
            # draw sequence of events, sort them
            events_batch = sorted(
                random.sample(possible_dates, len(self.events)))
            events_batch = pd.Series(
                data=np.arange(len(self.events)), index=events_batch)

            # pivot this batch
            bef, aft, _, _ = self.pivot_for_event_study(
                boot_from, events_batch, self.window)

            # calculate cumsum + cross-sectional mean
            this_cs_ts = self.get_cs_ts(bef, aft)

            # store
            booted.iloc[:,p] = this_cs_ts

        # quantiles
        # lower, upper
        ci = booted.quantile(ps, axis=1).T

        return ci

    def plot(self, **kwargs):
        """
        Parameters
        ----------
        kwargs : dict
            of arguments to EventStudy.get_ci() method.

        Returns
        -------
        fig :
        """
        ta, tb, tc, td = self.window

        cs_mu = self.get_cs_mean(self.before, self.after)
        ts_mu = self.get_ts_cumsum(self.before, self.after)
        cs_ts_mu = self.get_cs_ts(self.before, self.after)

        if not hasattr(self, "ci"):
            ci = self.get_ci(**kwargs)

        fig, ax = plt.subplots(2, figsize=(8.4,11.7/2), sharex=True)

        # # 1st plot: before and after for each event in light gray
        # self.before.plot(ax=ax[0], color=gr_1)
        # self.after.plot(ax=ax[0], color=gr_1)
        # # add points at initiation
        # self.before.iloc[[-1],:].plot(ax=ax[0], color="k",
        #     linestyle="none", marker=".", markerfacecolor="k")
        # self.after.iloc[[0],:].plot(ax=ax[0], color="k",
        #     linestyle="none", marker=".", markerfacecolor="k")
        # # plot mean in black =)
        # cs_mu.plot(ax=ax[0], color='k', linewidth=1.5)

        # 2nd plot: cumulative sums
        ts_mu.loc[:tb,:].plot(ax=ax[0], color=gr_1)
        ts_mu.loc[tc:,:].plot(ax=ax[0], color=gr_1)
        # add points at initiation
        self.before.iloc[[-1],:].plot(ax=ax[0], color="k",
            linestyle="none", marker=".", markerfacecolor="k")
        self.after.iloc[[0],:].plot(ax=ax[0], color="k",
            linestyle="none", marker=".", markerfacecolor="k")
        # mean in black
        cs_ts_mu.plot(ax=ax[0], color='k', linewidth=1.5)
        ax[0].set_title("cumulative, individual")

        # 3rd plot: ci around avg cumsum
        cs_ts_mu.loc[:tb].plot(ax=ax[1], color='k', linewidth=1.5)
        cs_ts_mu.loc[tc:].plot(ax=ax[1], color='k', linewidth=1.5)
        ax[1].fill_between(self.stacked_idx,
            self.ci.iloc[:,0].values,
            self.ci.iloc[:,1].values,
            color=gr_1, alpha=0.5, label="conf. interval")
        ax[1].set_title("cumulative average")

        # some parameters common for all ax
        for x in range(len(ax)):
            # ax[x].legend_.remove()
            ax[x].xaxis.set_ticks(self.stacked_idx)
            ax[x].axhline(y=0, color='r', linestyle='--', linewidth=1.0)
            ax[x].grid(axis="both", alpha=0.33, linestyle=":")
            legend = ax[x].legend()
            legend.remove()

        ax[x].set_xlabel("periods after event")

        # super title
        fig.suptitle(self.data.name, fontsize=14)

        return fig

def event_study_wrapper(data, events, reix_w_bday=False,
    direction="all", crisis="both", window=None, ps=0.9, ci_method="simple",
    plot=False):
    """ Convenience fun to run studies of many series on the same `events`.

    Parameters
    ----------
    data : pandas.DataFrame
        of data, with columns for something summable (e.g. log-returns)
    events : pandas.Series/DataFrame
        of events, indexed by event dates; values can be e.g. interest rates
    reix_w_bday : boolean
        if `data` should be reindexed with business days
    direction : str
        'ups' for considering positive changes in `events`.values only;
        'downs' for negative changes;
        'changes' for positive and negative changes;
        'constants' for no changes whatsoever;
        everything else for considering all events
    crisis : str
        'pre' for considering pre-crisis subsample only;
        'post' for post-crisis one;
        everything else - for the whole sample.
        Crisis date is taken to be 2008-06-30.
    window : list
        of 4 elements as in EventStudy
    ps : float/tuple
        confidence interval/bands thereof

    Returns:
    es : EventStudy
        isntance of EventStudy with all attributes (cs, ts etc.) calculated

    """
    data = data.loc[events.index[0]:]
    events = events.loc[data.index[0]:]

    # window default
    if window is None:
        window = [-5,-1,0,5]

    this_data = data.copy()

    # reindex if needed
    if reix_w_bday:
        bday_idx = pd.date_range(data.index[0], data.index[-1], freq='B')
        this_data = this_data.reindex(index=bday_idx)

    # subsample events: ups, downs, constants, ups and down or all
    if direction == "ups":
        events = events.where(events.diff() > 0).dropna()
    elif direction == "downs":
        events = events.where(events.diff() < 0).dropna()
    elif direction == "changes":
        events = events.where(events.diff() != 0).dropna()
    elif direction == "constants":
        events = events.where(events.diff() == 0).dropna()
    elif direction == "all":
        events = events
    else:
        raise ValueError("direction not implemented")

    # index to start data at: needed to discard stuff way too old
    start_at = min(events.index) - DateOffset(months=2)

    # pre- or post-crisis
    if crisis == "pre":
        this_data = this_data.loc[start_at:"2008-06-30"]
        events = events.loc[:"2008-06-30"]
    elif crisis == "post":
        this_data = this_data.loc["2008-06-30":]
        events = events.loc["2008-06-30":]
    else:
        this_data = this_data.loc[start_at:]

    # init EventStudy
    es = EventStudy(data=this_data, events=events, window=window)
    # plot
    if plot:
        es.plot(ps=ps, method=ci_method)

    return es

def get_idx(data, t):
    """ Fetch integer index given time index.
    If index `t` is not present in `data`.index, this function finds the
    first present later.
    """
    return data.index.get_loc(t, method="ffill")

def signal_from_events(data, events, window, func=None):
    """ Create signal based on events.

    Returns a DataFrame where at each index from `events` there will be a
    statistic of `data` calculated over `window` near each event in `events`.

    Parameters
    ----------
    data : (T,N) pandas.DataFrame
        of data
    events : (K,) pandas.Series
        of events
    window : tuple of int
        (start, end)
    func : callable
        function to apply to window of data

    Returns
    -------
    pivoted : (K,N) pd.DataFrame
        of statistics of `data` over `window` near each event

    Example
    -------
    with open(data_path+"data_dev_d.p", mode='rb') as fname:
        data = pickle.load(fname)
    with open(data_path+"events.p", mode='rb') as fname:
        events = pickle.load(fname)
    s_d = data["spot_ret"]
    fomc = events["fomc"].squeeze()
    # maximal rally
    res = signal_from_events(s_d, fomc, (-10, -2), lambda x: max(x.cumsum()))
    """
    assert isinstance(data, pd.DataFrame)
    assert isinstance(events, pd.Series)

    if func is None:
        func = np.nansum

    # pdb.set_trace()
    # unpack window ---------------------------------------------------------
    t_start, t_end = window

    # find dates belonging to two event windows simultaneously --------------
    belongs_idx = pd.Series(data=0,index=data.index)
    for t in events.index:
        # fetch position index of this event
        this_t = get_idx(data,t)
        # record span of its event window
        belongs_idx.ix[(this_t+t_start):(this_t+t_end+1)] += 1

    # set values in rows belonging to multiple events to nan
    data_to_pivot = data.copy()
    data_to_pivot.loc[belongs_idx > 1] *= np.nan

    # space for result ------------------------------------------------------
    pivoted = pd.DataFrame(
        columns=data.columns,
        index=events.index)

    # loop over events, save snapshot of returns around each ----------------
    for t, evt in events.iteritems():
        # fetch index of this event
        this_t = get_idx(data_to_pivot, t)
        # index as [t_start:t_end]
        window_idx = np.arange(this_t+t_start, this_t+t_end+1)
        # put event identifier to be used for pivoting later
        pivoted.loc[t,:] = data_to_pivot.ix[window_idx,:].apply(func)


    return pivoted.astype(np.float)

# if __name__ == "__main__":
#     # parameters of distribution
#     sigma = 1.5
#     mu = 1.0
#     # number of observations
#     T = 2000
#     # number of events
#     K = 10
#
#     # time index
#     idx = pd.date_range("1990-01-01", periods=T, frequency='D')
#
#     # simulate data
#     data_1d = pd.Series(
#         data=np.random.normal(size=(T,))*sigma+mu,
#         index=idx)
#
#     # simulate events
#     events = sorted(random.sample(idx.tolist()[T//3:T//2], K))
#     events = pd.Series(index=events, data=np.arange(len(events)))
#
#     evt_study = EventStudy(
#         data=data_1d,
#         events=events,
#         window=[-5,-1,0,5])
#
#     # fig = evt_study.plot(ps=0.9)


# ---------------------------------------------------------------------------
# alternative spec, once needed for Mirkov, Pozdeev, Soederlind (2016)
# ---------------------------------------------------------------------------
# fig, ax = plt.subplots(3, figsize=(12,12/1.25), sharex=True)
# cnt = 0
# for c in self.data.columns:
#     # 3rd plot: ci around avg cumsum
#     self.cs_ts_mu.loc[tc:,c].plot(ax=ax[cnt], color='k', linewidth=1.5)
#     self.cs_ts_mu.loc[tc:,c].plot(ax=ax[cnt], color='k',
#         linestyle="none", marker="o", markerfacecolor="k",
#         markersize=6)
#     ax[cnt].fill_between(self.after_idx,
#         np.zeros(6),
#         self.ci.loc[c,tc:,:].iloc[:,0].values,
#         color=gr_1, alpha=0.5, label="conf. interval")
#     # ax[cnt].axhline(y=0, color='r', linestyle='--', linewidth=1.0)
#     legend = ax[cnt].legend()
#     legend.remove()
#     ax[cnt].grid(axis="both", alpha=0.33, linestyle=":")
#     ax[cnt].set_ylim([-90,10])
#     cnt += 1
#
# ax[2].set_xlim([-0.5, 5.5])
