import pandas as pd
import numpy as np
from scipy.stats import norm
import random
from pandas.tseries.offsets import DateOffset
import matplotlib.pyplot as plt
import warnings
# import ipdb

class EventStudy():
    """
    """
    def __init__(self, data, events, window, normal_data=0, x_overlaps=False):
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
            a - start of cumulating returns in `data` before event (neg. int);
            b - stop of cumulating returns before event (neg. int),
            c - start of cumulating returns in `data` after event,
            d - stop of cumulating returns before event.
            For example, [-5,-1,1,5] means that one starts to look at returns 5
            periods before each event, ends 1 day before, starts again 1 day
            after and stops 5 days after.
        normal_data : pandas.DataFrame/Series
            of 'normal' returns
        """
        self._raw = {
            "data": data,
            "events": events}

        self._x_overlaps = x_overlaps
        self._window = window
        self._normal_data = normal_data

        # break down window
        ta, tb, tc, td = self._window
        self.event_index = \
            np.array(list(range(ta, tb+1)) + list(range(tc, td+1)))

        # prepare data
        abnormal_data, events = self.prepare_data()

        self.abnormal_data = abnormal_data
        self.events = events

    def prepare_data(self):
        """
        """
        # calculate abnormal data
        data = self._raw["data"] - self._normal_data

        # take events as is
        events = self._raw["events"].copy()

        # type manipulations to ensure that everthing is conformable DataFrames
        if isinstance(data, pd.Series):
            assert isinstance(events, pd.Series)
            data = data.to_frame(
                "data" if data.name is None else data.name)
            events = events.to_frame(data.columns[0])
        elif isinstance(data, pd.DataFrame):
            if isinstance(events, pd.Series):
                events = pd.concat(
                    [events.rename(p) for p in data.columns], axis=1)
            elif isinstance(events, pd.DataFrame):
                assert all(data.columns == events.columns)

        # check data and events for severe overlaps (not the pre vs. post kind)
        if self._x_overlaps:
            new_evts = dict()

            # ipdb.set_trace()

            for c in data.columns:
                this_evt, this_evt_diff = self.exclude_overlaps(
                    events[c].dropna(), data[c], self._window)

                if len(this_evt_diff) > 0:
                    warnings.warn("The following events for {} will be " +
                        " excluded because of overlaps: {}".format(c,
                            list(this_evt_diff)))

                new_evts[c] = this_evt

            events = pd.DataFrame.from_dict(new_evts)

        return data, events

    @staticmethod
    def exclude_overlaps(events, data, window):
        """Exclude overlapping data.

        Parameters
        ----------
        events : pandas.Series
        window : tuple
        """
        ta, tb, tc, td = window

        overlaps = pd.Series(0, index=data.index)

        for t in events.index:
            # t = events.index[0]
            tmp_series = pd.Series(index=data.index)
            tmp_series.loc[t] = 1

            tmp_series = tmp_series.fillna(method="bfill", limit=-ta).fillna(0)

            overlaps += tmp_series

        # set values in rows corresponding to overlaps to nan
        overlaps.ix[overlaps > 1] *= np.nan

        # catch deleted events
        evt_diff = events.index.difference(overlaps.dropna().index)

        # remove events where overlaps occur
        events = events.where(overlaps < 2).dropna()

        return events, evt_diff

    @staticmethod
    def mark_timeline(data, events, window, x_overlaps):
        """Mark the timeline by group such as within-event, pre-event etc.

        Parameters
        ----------
        data : pandas.Series
            of data
        events : pandas.Series
            of events (collapsed)
        window : tuple
            ta, tb, tc, td as before

        Returns
        -------
        timeline : pandas.DataFrame
            with columns
                evt_no - number of event within event window,
                evt_wind_pre - indexes from ta to tb,
                evt_wind_post - indexes from tc to td
                inter_evt - 1 for periods between events
                next_evt_no - number of the next event
        """
        ta, tb, tc, td = window
        wind_idx_pre = np.arange(ta, tb+1)
        wind_idx_post = np.arange(tc, td+1)

        # helper to find indexes
        def get_idx(t, data):
            return data.index.get_loc(t)

        # count events
        # evt_count = events.expanding().count() - 1
        evt_count = pd.Series(data=events.index.date, index=events.index)
        evt_count = evt_count.reindex(index=data.index)

        # allocate space for marked timeline: need pre and post to capture
        #   cases when deleteing overlapping events is not desired
        cols = [
            "evt_no",
            "evt_wind_pre",
            "evt_wind_post",
            "inter_evt",
            "next_evt_no"]

        timeline = pd.DataFrame(index=data.index, columns=cols)

        # next event number is the number of event immediately following date
        timeline.loc[:, "next_evt_no"] = evt_count.fillna(method="bfill")

        # window around each event referring to this event
        timeline.loc[:, "evt_no"] = evt_count.\
            fillna(method="bfill", limit=ta*-1).\
            fillna(method="ffill", limit=td)

        # loop over events to write down -10, -9, -8, ... 4, 5 and the like
        for t in events.index:
            # t = events.index[-2]
            # fetch index of this event
            this_t = get_idx(t, timeline)

            # index with window indexes
            timeline.ix[this_t + wind_idx_pre, "evt_wind_pre"] = wind_idx_pre
            timeline.ix[this_t + wind_idx_post, "evt_wind_post"] = \
                wind_idx_post

        # fill nans
        # inter_evt_idx = timeline.loc[:, ["evt_wind_pre", "evt_wind_post"]].\
        #     dropna(how="all").index
        timeline.ix[timeline.loc[:, "evt_no"].isnull(), "inter_evt"] = 1

        # ensure overlaps do not enter
        if x_overlaps:
            idx = timeline.loc[:, ["evt_wind_pre", "evt_wind_post"]].\
                notnull().all(axis=1)
            timeline.ix[idx, ["evt_wind_pre", "evt_wind_post"]] = np.nan

        return timeline

    @staticmethod
    def pivot_with_timeline(data, timeline):
        """Pivot `data` to center it on events, using a timeline.

        Parameters
        ----------
        data : pandas.Series
        timeline : pandas.DataFrame
            output of mark_timeline

        Returns
        -------
        data_pivoted : pandas.DataFrame
            indexed by window, columned by event dates
        """
        assert isinstance(data, pd.Series)

        if data.name is None:
            data.name = "data"

        # concat
        both = pd.concat((data, timeline), axis=1).dropna(subset=["evt_no"])

        # # event window
        # data_to_pivot_pre = timeline.loc[:, ["evt_wind_pre", "evt_no"]]
        # data_to_pivot_post = timeline.loc[:, ["evt_wind_post", "evt_no"]]

        data_pivoted_pre = both.dropna(subset=["evt_wind_pre"]).pivot(
            index="evt_wind_pre",
            columns="evt_no",
            values=data.name)
        data_pivoted_post = both.dropna(subset=["evt_wind_post"]).pivot(
            index="evt_wind_post",
            columns="evt_no",
            values=data.name)

        # main data
        data_pivoted = pd.concat((data_pivoted_pre, data_pivoted_post), axis=0)

        return data_pivoted

    def collect_responses(self):
        """Pivot for every column in `self.abnormal_data`

        Returns
        -------
        res : pandas.Panel
            of data, where major_axis keep the window, minor_axis keeps assets
        """
        res = dict()

        # loop over columns in `abnormal_data`
        for c in self.abnormal_data.columns:
            print(c)
            # if c == "sek":
            #     ipdb.set_trace()
            this_evt = self.events[c].dropna()

            # timeline
            this_timeline = self.mark_timeline(
                data=self.abnormal_data[c],
                events=this_evt,
                window=self._window,
                x_overlaps=self._x_overlaps)
            # pivot
            res[c] = self.pivot_with_timeline(
                data=self.abnormal_data[c],
                timeline=this_timeline)

        res = pd.Panel.from_dict(res, orient="minor")

        return res

    @staticmethod
    def get_ts_cumsum(ndframe, window):
        """Calculate cumulative sum over time."""
        ta, tb, tc, td = window

        if len(ndframe.shape) > 2:
            ts_cumsum_before = ndframe.loc[:, :tb, :].iloc[:, ::-1, :].\
                cumsum().iloc[:, ::-1, :]
            ts_cumsum_after = ndframe.loc[:, tc:, :].cumsum()

            # concat
            ts_cumsum = pd.concat((ts_cumsum_before, ts_cumsum_after),
                axis="major")
        else:
            ts_cumsum_before = ndframe.loc[:tb, :].iloc[::-1, :].\
                cumsum().iloc[::-1, :]
            ts_cumsum_after = ndframe.loc[tc:, :].cumsum()

            # concat
            ts_cumsum = pd.concat((ts_cumsum_before, ts_cumsum_after),
                axis=0)

        return ts_cumsum

    @staticmethod
    def get_cs_mean(ndframe, window):
        """Calculate the average across events."""
        cs_mu = ndframe.mean(axis=("items" if len(ndframe.shape) > 2 else 1))

        return cs_mu

    # def get_ci(self, ps, method="simple", **kwargs):
    #     """ Calculate confidence bands.
    #     Parameters
    #     ----------
    #     p : float / tuple of floats
    #         interval width or tuple of (lower bound, upper bound)
    #     Returns
    #     -------
    #     ci : pd.Panel
    #
    #     """
    #     # if `ps` was provided as single float
    #     if isinstance(ps, float):
    #         ps = ((1-ps)/2, ps/2+1/2)
    #
    #     # calculate confidence band
    #     if method == "simple":
    #         ci = self.simple_ci(ps=ps)
    #     elif method == "boot":
    #         ci = self.boot_ci(ps, **kwargs)
    #     else:
    #         raise NotImplementedError("ci you asked for is not implemented")
    #
    #     self.ci = ci
    #
    #     return ci

    # def simple_ci(self, ps):
    #     """
    #     """
    #     ta, tb, tc, td = self._window
    #
    #     sigma = np.sqrt(self.grp_var.sum())/self.grp_var.count()
    #     self.sigma = sigma
    #
    #     # mu = boot_from.mean()
    #
    #     # # sd of mean across events is mean outside of events divided
    #     # #   through sqrt(number of events)
    #     # sd_across = boot_from.std()/np.sqrt(len(self.events))
    #     # # sd of cumulative sum of mean across events is sd times sqrt(# of
    #     # #   cumulants)
    #     # # pdb.set_trace()
    #
    #     q = np.sqrt(np.hstack(
    #         (np.arange(-ta+tb+1,0,-1), np.arange(1,td-tc+2))))
    #     # q = q[:,np.newaxis]
    #
    #     # # multiply with broadcast, add mean
    #     # ci_lo = norm.ppf(ps[0])*q*sd_across.values[np.newaxis,:] + \
    #     #     boot_from.mean().values[np.newaxis,:]*q**2
    #     # ci_hi = norm.ppf(ps[1])*q*sd_across.values[np.newaxis,:] + \
    #     #     boot_from.mean().values[np.newaxis,:]*q**2
    #     # multiply with broadcast, add mean
    #     ci_lo = norm.ppf(ps[0])*q*sigma + self.tot_mu*(q**2)
    #     ci_hi = norm.ppf(ps[1])*q*sigma + self.tot_mu*(q**2)
    #
    #     # concatenate: items keeps columns of Y
    #     ci = pd.DataFrame(
    #         index=self.stacked_idx,
    #         columns=ps)
    #
    #     ci.loc[:,ps[0]] = ci_lo
    #     ci.loc[:,ps[1]] = ci_hi
    #
    #     return ci

    def boot_ci(self, ps, M=500):
        """
        Returns
        -------
        ci : pandas.DataFrame
            with columns for confidence interval bands
        """
        ta, tb, tc, td = self._window

        ci = pd.Panel(
            items=ps,
            major_axis=self.event_index,
            minor_axis=self.abnormal_data.columns)

        # loop over columns in `abnormal_data`
        for c in self.abnormal_data.columns:

            this_evt = self.events[c].dropna()
            K = len(this_evt)

            # timeline
            this_timeline = self.mark_timeline(
                data=self.abnormal_data[c],
                events=this_evt,
                window=self._window,
                x_overlaps=self._x_overlaps)

            # boot from `abnormal_data` without intervals around events
            this_interevt_idx = this_timeline.loc[:, "inter_evt"].astype(bool)
            boot_from = self.abnormal_data[c].copy().ix[this_interevt_idx]

            # possible dates exclude values that are too early or too late
            possible_dt = boot_from.index.tolist()[(-ta):-(td+1)]

            # space for df's of pivoted tables
            booted = pd.DataFrame(columns=range(M), index=self.event_index)

            for p in range(M):
                # print(p)
                # draw sequence of events, sort them
                tmp_events = pd.Series(
                    data=list(range(K)),
                    index=sorted(random.sample(possible_dt, K)))
                # ipdb.set_trace()
                tmp_events, _ = self.exclude_overlaps(tmp_events, boot_from,
                    self._window)

                tmp_timeline = self.mark_timeline(
                    data=boot_from,
                    events=tmp_events,
                    window=self._window,
                    x_overlaps=self._x_overlaps)

                # pivot this batch
                this_pivoted_data = self.pivot_with_timeline(
                    data=boot_from,
                    timeline=tmp_timeline)

                # calculate cumsum + cross-sectional mean
                this_cs_ts = self.get_cs_mean(
                    self.get_ts_cumsum(this_pivoted_data, window=self._window),
                    window=self._window)

                # store
                booted.iloc[:, p] = this_cs_ts

            # extract percentiles
            # ipdb.set_trace()
            this_ci = booted.quantile(ps, axis=1)

            # store
            ci.loc[:, :, c] = this_ci.T

        return ci

    # def plot(self, **kwargs):
    #     """
    #     Parameters
    #     ----------
    #     kwargs : dict
    #         of arguments to EventStudy.get_ci() method.
    #
    #     Returns
    #     -------
    #     fig :
    #     """
    #     ta, tb, tc, td = self._window
    #
    #     cs_mu = self.get_cs_mean(self.before, self.after)
    #     ts_mu = self.get_ts_cumsum(self.before, self.after)
    #     cs_ts_mu = self.get_cs_ts(self.before, self.after)
    #
    #     if not hasattr(self, "ci"):
    #         ci = self.get_ci(**kwargs)
    #
    #     fig, ax = plt.subplots(2, figsize=(8.4,11.7/2), sharex=True)
    #
    #     # # 1st plot: before and after for each event in light gray
    #     # self.before.plot(ax=ax[0], color=gr_1)
    #     # self.after.plot(ax=ax[0], color=gr_1)
    #     # # add points at initiation
    #     # self.before.iloc[[-1],:].plot(ax=ax[0], color="k",
    #     #     linestyle="none", marker=".", markerfacecolor="k")
    #     # self.after.iloc[[0],:].plot(ax=ax[0], color="k",
    #     #     linestyle="none", marker=".", markerfacecolor="k")
    #     # # plot mean in black =)
    #     # cs_mu.plot(ax=ax[0], color='k', linewidth=1.5)
    #
    #     # 2nd plot: cumulative sums
    #     ts_mu.loc[:tb,:].plot(ax=ax[0], color=gr_1)
    #     ts_mu.loc[tc:,:].plot(ax=ax[0], color=gr_1)
    #     # add points at initiation
    #     self.before.iloc[[-1],:].plot(ax=ax[0], color="k",
    #         linestyle="none", marker=".", markerfacecolor="k")
    #     self.after.iloc[[0],:].plot(ax=ax[0], color="k",
    #         linestyle="none", marker=".", markerfacecolor="k")
    #     # mean in black
    #     cs_ts_mu.plot(ax=ax[0], color='k', linewidth=1.5)
    #     ax[0].set_title("cumulative, individual")
    #
    #     # 3rd plot: ci around avg cumsum
    #     cs_ts_mu.loc[:tb].plot(ax=ax[1], color='k', linewidth=1.5)
    #     cs_ts_mu.loc[tc:].plot(ax=ax[1], color='k', linewidth=1.5)
    #     ax[1].fill_between(self.stacked_idx,
    #         self.ci.iloc[:,0].values,
    #         self.ci.iloc[:,1].values,
    #         color=gr_1, alpha=0.5, label="conf. interval")
    #     ax[1].set_title("cumulative average")
    #
    #     # some parameters common for all ax
    #     for x in range(len(ax)):
    #         # ax[x].legend_.remove()
    #         ax[x].xaxis.set_ticks(self.stacked_idx)
    #         ax[x].axhline(y=0, color='r', linestyle='--', linewidth=1.0)
    #         ax[x].grid(axis="both", alpha=0.33, linestyle=":")
    #         legend = ax[x].legend()
    #         legend.remove()
    #
    #     ax[x].set_xlabel("periods after event")
    #
    #     # super title
    #     fig.suptitle(self.data.name, fontsize=14)
    #
    #     return fig

# def event_study_wrapper(data, events, reix_w_bday=False,
#     direction="all", crisis="both", window=None, ps=0.9, ci_method="simple",
#     plot=False, impose_mu=None):
#     """ Convenience fun to run studies of many series on the same `events`.
#
#     Parameters
#     ----------
#     data : pandas.DataFrame
#         of data, with columns for something summable (e.g. log-returns)
#     events : pandas.Series/DataFrame
#         of events, indexed by event dates; values can be e.g. interest rates
#     reix_w_bday : boolean
#         if `data` should be reindexed with business days
#     direction : str
#         'ups' for considering positive changes in `events`.values only;
#         'downs' for negative changes;
#         'changes' for positive and negative changes;
#         'constants' for no changes whatsoever;
#         everything else for considering all events
#     crisis : str
#         'pre' for considering pre-crisis subsample only;
#         'post' for post-crisis one;
#         everything else - for the whole sample.
#         Crisis date is taken to be 2008-06-30.
#     window : list
#         of 4 elements as in EventStudy
#     ps : float/tuple
#         confidence interval/bands thereof
#
#     Returns:
#     es : EventStudy
#         isntance of EventStudy with all attributes (cs, ts etc.) calculated
#
#     """
#     data = data.loc[events.index[0]:]
#     events = events.loc[data.index[0]:]
#
#     # window default
#     if window is None:
#         window = [-5,-1,0,5]
#
#     this_data = data.copy()
#
#     # reindex if needed
#     if reix_w_bday:
#         bday_idx = pd.date_range(data.index[0], data.index[-1], freq='B')
#         this_data = this_data.reindex(index=bday_idx)
#
#     # subsample events: ups, downs, constants, ups and down or all
#     if direction == "ups":
#         events = events.where(events > 0).dropna()
#     elif direction == "downs":
#         events = events.where(events < 0).dropna()
#     elif direction == "changes":
#         events = events.where(events != 0).dropna()
#     elif direction == "constants":
#         events = events.where(events == 0).dropna()
#     elif direction == "all":
#         events = events
#     else:
#         raise ValueError("direction not implemented")
#
#     # index to start data at: needed to discard stuff way too old
#     start_at = min(events.index) - DateOffset(months=2)
#
#     # pre- or post-crisis
#     if crisis == "pre":
#         this_data = this_data.loc[start_at:"2008-06-30"]
#         events = events.loc[:"2008-06-30"]
#     elif crisis == "post":
#         this_data = this_data.loc["2008-06-30":]
#         events = events.loc["2008-06-30":]
#     else:
#         this_data = this_data.loc[start_at:]
#
#     # init EventStudy
#     es = EventStudy(data=this_data, events=events, window=window)
#     # plot
#     if plot:
#         es.plot(ps=ps, method=ci_method)
#
#     return es

# def get_idx(data, t):
#     """ Fetch integer index given time index.
#     If index `t` is not present in `data`.index, this function finds the
#     first present later.
#     """
#     return data.index.get_loc(t, method="ffill")

if __name__ == "__main__":

    import ipdb
    from foolbox.api import *

    # currency to drop
    drop_curs = ["usd","jpy","dkk"]

    # window
    wa,wb,wc,wd = -10,-1,1,5
    window = (wa,wb,wc,wd)

    s_dt = settings["sample_start"]
    e_dt = settings["sample_end"]

    # data ------------------------------------------------------------------
    data_path = set_credentials.gdrive_path("research_data/fx_and_events/")
    out_path = set_credentials.gdrive_path("opec_meetings/tex/figs/")

    # spot returns + drop currencies ----------------------------------------
    with open(data_path + settings["fx_data"], mode='rb') as fname:
        fx = pickle.load(fname)
    ret = np.log(fx["spot_mid"].drop(drop_curs,axis=1,errors="ignore")).diff()

    # events + drop currencies ----------------------------------------------
    with open(data_path + settings["events_data"], mode='rb') as fname:
        events = pickle.load(fname)

    events_perf = events["joint_cbs"].drop(drop_curs, axis=1, errors="ignore")
    events_perf = events_perf.loc[s_dt:e_dt]

    data = ret["nzd"]
    data = ret.copy()
    events = events_perf["nzd"].dropna()
    events = events_perf.copy()
    events = events.where(events < 0)

    es = EventStudy(data, events, window, normal_data=0.0, x_overlaps=True)

    ipdb.set_trace()
    res = es.collect_responses()

    res.squeeze().iloc[:10, 0].sum()

    res.squeeze()

    mu = es.get_ts_cumsum(res, window).mean(axis="items")
    number_evts = es.get_ts_cumsum(res, window).count(axis="items")

    r = (mu * number_evts.divide(number_evts.sum(axis=1), axis=0)).sum(axis=1)

    r.plot()


    ax = es.get_cs_mean(es.get_ts_cumsum(res, window), window).plot()
    es.get_ts_cumsum(res, window).loc[:,:,"sek"].dropna(how="all", axis=1)
    res

    es.events

    ci = es.boot_ci(ps=(0.05, 0.95), M=100)

    ci.squeeze().plot(ax=ax, color='gray')
