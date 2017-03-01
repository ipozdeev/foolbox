import pandas as pd
import numpy as np
from scipy.stats import norm
import random
from pandas.tseries.offsets import DateOffset
import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import pdb
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
            events = pd.DataFrame(data=np.arange(len(events)),index=events)

        # if not a DataFrame already -> convert to it because iterrows later
        if isinstance(events, pd.Series):
            events = events.to_frame()

        # check that events are uniquely labelled (e.g. 0,1,2...)
        if events.duplicated().sum() > 0:
            events = \
                pd.DataFrame(index=events.index,data=np.arange(len(events)))

        # shape
        T,N = data.shape

        # keep record of dimensions (needed for later)
        self.is_1d = N < 2

        # if last event is closer to the end than td - delete last event
        # TODO: write logger for that
        good_evt_idx = events.index.map(
            lambda x: (get_idx(data, x) < (T-td)) & \
                (get_idx(data, x) > -ta-1))
        events = events.ix[good_evt_idx,:]

        # pivot
        before, after = self.pivot_for_event_study(data, events, window)
        # pdb.set_trace()

        self.data = data
        self.events = events
        self.window = window
        self.before = before
        self.after = after
        self.before_idx = before.major_axis
        self.after_idx = after.major_axis
        self.stacked_idx = np.hstack((before.major_axis, after.major_axis))

    # @staticmethod
    # def pivot_for_event_study(data, events, window):
    #     """ Pivot `data` to center on events.
    #     """
    #     # pdb.set_trace()
    #     ta, tb, tc, td = window
    #
    #     # find dates belonging to two event windows simultaneously
    #     belongs_idx = pd.Series(data=0,index=data.index)
    #     for t, evt in events.iterrows():
    #         # fetch position index of this event
    #         this_t = get_idx(data,t)
    #         # record span of its event window
    #         belongs_idx.ix[(this_t+ta):(this_t+td+1)] += 1
    #
    #     # set values in rows belonging to multiple events to nan
    #     data_to_pivot = data.copy()
    #     data_to_pivot.loc[belongs_idx > 1] *= np.nan
    #
    #     # loop over events, save snapshot of returns around each
    #     before_dict = {}
    #     after_dict = {}
    #     for (t, evt) in events.iterrows():
    #         # fetch index of this event
    #         this_t = get_idx(data_to_pivot,t)
    #         # from a to b
    #         this_df = data_to_pivot.ix[(this_t+ta):(this_t+tb+1),:]
    #         # index with [a,...,b]
    #         this_df.index = np.arange(ta, tb+1)
    #         # store
    #         before_dict[evt.values[0]] = this_df
    #
    #         # from c to d, same steps
    #         this_df = data_to_pivot.ix[(this_t+tc):(this_t+td+1),:]
    #         this_df.index = np.arange(tc, td+1)
    #         after_dict[evt.values[0]] = this_df
    #
    #     # create panel of event-centered dataframes, where minor_axis keeps
    #     #   individual events, items keeps columns of Y
    #     before = pd.Panel.from_dict(before_dict, orient="minor")
    #
    #     # after
    #     after = pd.Panel.from_dict(after_dict, orient="minor")
    #
    #     return before, after

    @staticmethod
    def pivot_for_event_study(data, events, window):
        """ Pivot `data` to center on events.
        """
        # pdb.set_trace()
        ta, tb, tc, td = window

        # find dates belonging to two event windows simultaneously
        belongs_idx = pd.Series(data=0, index=data.index)
        for t in events.index:
            # fetch position index of this event (ffill if missing in `data`)
            this_t = get_idx(data, t)
            # record span of its event window
            belongs_idx.ix[(this_t+ta):(this_t+td+1)] += 1

        # set values in rows belonging to multiple events to nan
        data_to_pivot = data.copy()
        data_to_pivot.loc[belongs_idx > 1] *= np.nan

        # loop over events, save snapshot of returns around each
        for t, evt in events.iteritems():
            # fetch index of this event
            this_t = get_idx(data_to_pivot,t)
            # from a to b
            this_df = data_to_pivot.ix[(this_t+ta):(this_t+tb+1),:]
            # index with [a,...,b]
            this_df.index = np.arange(ta, tb+1)
            # store
            before_dict[evt.values[0]] = this_df

            # from c to d, same steps
            this_df = data_to_pivot.ix[(this_t+tc):(this_t+td+1),:]
            this_df.index = np.arange(tc, td+1)
            after_dict[evt.values[0]] = this_df

        # create panel of event-centered dataframes, where minor_axis keeps
        #   individual events, items keeps columns of Y
        before = pd.Panel.from_dict(before_dict, orient="minor")

        # after
        after = pd.Panel.from_dict(after_dict, orient="minor")

        return before, after

    def get_ts_cumsum(self):
        """ Calculate cumulative sum over time.
        """
        # the idea is to buy stuff at relative date a, where T is the event
        #   date and keep it until and including relative date b, so need to
        #   reverse this Panel for further cumulative summation
        ts_mu_before = self.before.ix[:,::-1,:].cumsum().ix[:,::-1,:]
        # after events is ok as is
        ts_mu_after = self.after.cumsum()

        # concat
        ts_mu = pd.concat((ts_mu_before, ts_mu_after), axis=1)

        self.ts_mu = ts_mu

        return ts_mu

    def get_cs_mean(self):
        """ Calculate mean across events
        """
        cs_mu_before = self.before.mean(axis="minor_axis")
        cs_mu_after = self.after.mean(axis="minor_axis")

        # concat
        cs_mu = pd.concat((cs_mu_before, cs_mu_after), axis=0)

        self.cs_mu = cs_mu

        return cs_mu

    def get_cs_ts(self):
        """ Calculate time-series cumulative sum and its average across events.
        """
        ts_mu = self.get_ts_cumsum()
        cs_ts_mu = ts_mu.mean(axis="minor_axis")

        self.cs_ts_mu = cs_ts_mu

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
        ta, tb, tc, td = self.window

        # if `ps` was provided as single float
        if isinstance(ps, float):
            ps = ((1-ps)/2, ps/2+1/2)

        # drop dates around events, G times window length
        # TODO: control for overlaps
        boot_from = self.data.copy()
        # pdb.set_trace()
        for (t, number) in self.events.iterrows():
            this_t = get_idx(boot_from, t)
            boot_from.drop(
                boot_from.index[(this_t+ta):(this_t+td)],
                axis="index",
                inplace=True)

        # calculate confidence band
        if method == "simple":
            ci = self.simple_ci(boot_from=boot_from, ps=ps)

        elif method == "boot":
            ci = self.boot_ci(boot_from=boot_from, ps=ps, **kwargs)

        self.ci = ci

        return ci

    def simple_ci(self, boot_from, ps):
        """
        """
        ta, tb, tc, td = self.window

        # mu = boot_from.mean()

        # sd of mean across events is mean outside of events divided
        #   through sqrt(number of events)
        sd_across = boot_from.std()/np.sqrt(len(self.events))
        # sd of cumulative sum of mean across events is sd times sqrt(# of
        #   cumulants)
        # pdb.set_trace()
        q = np.sqrt(np.hstack(
            (np.arange(-ta+tb+1,0,-1), np.arange(1,td-tc+2))))
        q = q[:,np.newaxis]

        # multiply with broadcast, add mean
        ci_lo = norm.ppf(ps[0])*q*sd_across.values[np.newaxis,:] + \
            boot_from.mean().values[np.newaxis,:]*q**2
        ci_hi = norm.ppf(ps[1])*q*sd_across.values[np.newaxis,:] + \
            boot_from.mean().values[np.newaxis,:]*q**2

        # concatenate: items keeps columns of Y
        ci = pd.Panel(
            major_axis=self.stacked_idx,
            items=self.data.columns,
            minor_axis=ps)
        ci.loc[:,:,ps[0]] = ci_lo.T
        ci.loc[:,:,ps[1]] = ci_hi.T

        return ci

    def boot_ci(self, boot_from, ps, M=500):
        """
        """
        # space for df's of pivoted tables
        booted = pd.Panel(
            items=self.data.columns,
            major_axis=self.stacked_idx,
            minor_axis=range(M))

        # possible times should exclude values that are too early or too late
        possible_dates = boot_from.index.tolist()[
            (-self.window[0]):-(self.window[-1]+1)]

        for p in range(M):
            # pdb.set_trace()
            # draw sequence of events, sort them
            events_batch = sorted(
                random.sample(possible_dates, len(self.events)))
            events_batch = pd.DataFrame(
                data=np.arange(len(self.events)), index=events_batch)

            # pivot this batch
            bef, aft = self.pivot_for_event_study(
                boot_from, events_batch, self.window)

            # cumsum
            bef = bef.ix[:,::-1,:].cumsum().ix[:,::-1,:]
            # after events is ok as is
            aft = aft.cumsum()
            # concat (automatically along major_axis)
            tot = pd.concat((bef, aft), axis=1)

            # mean across events
            tot = tot.mean(axis="minor_axis")

            # store
            booted.iloc[:,:,p] = tot

        # quantiles
        # lower
        ci_lo = booted.apply(lambda x: x.quantile(ps[0]), axis="minor")
        # upper
        ci_hi = booted.apply(lambda x: x.quantile(ps[1]), axis="minor")

        # concatenate
        ci = pd.Panel(
            major_axis=self.stacked_idx,
            items=self.data.columns,
            minor_axis=ps)
        ci.loc[:,:,ps[0]] = ci_lo
        ci.loc[:,:,ps[1]] = ci_hi

        return ci

    def plot(self, **kwargs):
        """
        Parameters
        ----------
        kwargs : dict
            of arguments to EventStudy.get_ci() method.

        Returns
        -------
        figs : list
             of figures
        """
        ta, tb, tc, td = self.window

        # check if cross-sectional avg of cumsum exists
        if not hasattr(self, "cs_mu"):
            cs_mu = self.get_cs_mean()
        if not hasattr(self, "ts_mu"):
            ts_mu = self.get_cs_mean()
        if not hasattr(self, "cs_ts_mu"):
            cs_ts_mu = self.get_cs_ts()
        if not hasattr(self, "ci"):
            ci = self.get_ci(**kwargs)

        # for each column in `data` create one plot with three subplots
        figs = {}
        for c in self.data.columns:
            fig, ax = plt.subplots(3, figsize=(12,12*0.9), sharex=True)

            # 1st plot: before and after for each event in light gray
            self.before.loc[c,:,:].plot(ax=ax[0], color=gr_1)
            self.after.loc[c,:,:].plot(ax=ax[0], color=gr_1)
            # add points at initiation
            self.before.loc[c,[-1],:].plot(ax=ax[0], color="k",
                linestyle="none", marker=".", markerfacecolor="k")
            self.after.loc[c,[0],:].plot(ax=ax[0], color="k",
                linestyle="none", marker=".", markerfacecolor="k")
            # plot mean in black =)
            self.cs_mu.loc[:,c].plot(ax=ax[0], color='k', linewidth=1.5)
            ax[0].set_title("at respective period")

            # 2nd plot: cumulative sums
            self.ts_mu.loc[c,:tb,:].plot(ax=ax[1], color=gr_1)
            self.ts_mu.loc[c,tc:,:].plot(ax=ax[1], color=gr_1)
            # add points at initiation
            self.before.loc[c,[-1],:].plot(ax=ax[1], color="k",
                linestyle="none", marker=".", markerfacecolor="k")
            self.after.loc[c,[0],:].plot(ax=ax[1], color="k",
                linestyle="none", marker=".", markerfacecolor="k")
            # mean in black
            self.cs_ts_mu.loc[:,c].plot(ax=ax[1], color='k', linewidth=1.5)
            ax[1].set_title("cumulative")

            # 3rd plot: ci around avg cumsum
            self.cs_ts_mu.loc[:tb,c].plot(ax=ax[2], color='k', linewidth=1.5)
            self.cs_ts_mu.loc[tc:,c].plot(ax=ax[2], color='k', linewidth=1.5)
            ax[2].fill_between(self.stacked_idx,
                self.ci.loc[c,:,:].iloc[:,0].values,
                self.ci.loc[c,:,:].iloc[:,1].values,
                color=gr_1, alpha=0.5, label="conf. interval")
            ax[2].set_title("average cumulative")

            # some parameters common for all ax
            for x in range(len(ax)):
                # ax[x].legend_.remove()
                ax[x].xaxis.set_ticks(self.stacked_idx)
                ax[x].axhline(y=0, color='r', linestyle='--', linewidth=1.0)
                ax[x].grid(axis="both", alpha=0.33, linestyle=":")
                legend = ax[x].legend()
                legend.remove()

            ax[x].set_xlabel("periods after event")
            ax[x].set_ylabel("return, in % per period")

            # super title
            fig.suptitle(c, fontsize=14)

            figs[c] = fig

        return figs

def event_study_wrapper(data, events, exclude_cols=[], direction="all",
    crisis="both", window=None, ps=0.9, ci_method="simple"):
    """ Convenience fun to run studies of many series on the same `events`.

    Parameters
    ----------
    data : pandas.DataFrame
        of data, with columns for something summable (e.g. log-returns)
    events : pandas.Series/DataFrame
        of events, indexed by event dates; values can be e.g. interest rates
    exclude_cols : list-like
        of columns to drop from data when doing the magic
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
    # window default
    if window is None:
        window = [-5,-1,0,5]
    # drop certain currencies
    this_data = data.drop(exclude_cols, axis=1)

    # subsample events: ups, downs, constants, ups and down or all
    if direction == "ups":
        events = events.where(events.diff() > 0).dropna()
    elif direction == "downs":
        events = events.where(events.diff() < 0).dropna()
    elif direction == "changes":
        events = events.where(events.diff() != 0).dropna()
    elif direction == "constants":
        events = events.where(events.diff() == 0).dropna()

    # index to start data at: needed to discard stuff way too old
    start_at = min(events.index) - DateOffset(months=2)

    # pre- or post-crisis
    if crisis == "pre":
        this_data = this_data.loc[start_at:"2008-06-30",:]
        events = events.loc[:"2008-06-30",:]
    elif crisis == "post":
        this_data = this_data.loc["2008-06-30":,:]
        events = events.loc["2008-06-30":,:]
    else:
        this_data = this_data.loc[start_at:,:]

    # init EventStudy
    es = EventStudy.EventStudy(data=this_data, events=events, window=window)
    # plot
    es.plot(ps=ps, method=ci_method)

    return es

def get_idx(data, t):
    """ Fetch integer index given time index.
    If index `t` is not present in `data`.index, this function finds the
    first present later.
    """
    return data.index.get_loc(t, method="ffill")


if __name__ == "__main__":
    pass


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
