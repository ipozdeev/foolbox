import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import warnings
from foolbox.data_mgmt import set_credentials
from foolbox.utils import resample_between_events
from foolbox.wp_tabs_figs.wp_settings import *
import pickle

# import ipdb

path_out = set_credentials.set_path("research_data/fx_and_events/",
    which="gdrive")

class EventStudy():
    """
    """
    def __init__(self, data, events, wind, mean_type="simple",
        normal_data=0.0, x_overlaps=False):
        """
        Parameters
        ----------
        data : pandas.DataFrame/pandas.Series
            of returns (need to be summable)
        events : list/pandas.DataFrame/pandas.Series
            of events (if a pandas object, must be indexed with dates)
        wind : tuple of int
            [a,b,c,d] where each element is relative (in periods) date of
            a - start of cumulating `data` before event (usually an int < 0);
            b - stop of cumulating `data` before event (usually an int < 0),
            c - start of cumulating `data` after event,
            d - stop of cumulating `data` before event;
            for example, [-5,-1,1,5] means that one starts to look at returns 5
            periods before each event, ends 1 day before, starts again 1 day
            after and stops 5 days after
        mean_type : str
            method for calculating cross-sectional average; "simple" or
            "count_weighted" are supported
        normal_data : float/pandas.DataFrame/pandas.Series
            of 'normal' data; must have the same shape, index and column names
            as `events`.
        x_overlaps : bool
            True for excluding events and days in case of overlaps
        """
        # conversions of stuff to ensure that everything appears as DataFrames
        if isinstance(normal_data, (float, np.float, int, np.int)):
            normal_data = events.notnull()*normal_data

        # indexes better be all of the same type
        events.index = events.index.map(pd.to_datetime)
        data.index = data.index.map(pd.to_datetime)
        normal_data.index = normal_data.index.map(pd.to_datetime)

        if isinstance(data, pd.Series):
            # ensure `data` has a name
            data_name = "data" if data.name is None else data.name

            # if `data` is a Series, all else better be too
            assert isinstance(events, pd.Series)
            assert isinstance(normal_data, pd.Series)
            # assert normal_data.index.equals(events.index)

            # everything to frames
            data = data.to_frame(data_name)
            events = events.to_frame(data_name)
            normal_data = normal_data.to_frame(data_name)

        elif isinstance(data, pd.DataFrame):
            if isinstance(events, pd.Series):
                events = pd.concat(
                    [events.rename(p) for p in data.columns], axis=1)
            elif isinstance(events, pd.DataFrame):
                assert all(data.columns == events.columns)

        # assert normal_data.index.equals(events.index)
        # assert normal_data.columns.equals(events.columns)

        self._raw = {
            "data": data,
            "events": events}

        self._x_overlaps = x_overlaps
        self._wind = wind
        self._normal_data = normal_data
        self._mean_type = mean_type

        # break down window, construct index of holding days
        ta, tb, tc, td = self._wind
        self.event_index = \
            np.array(list(range(ta, tb+1)) + list(range(tc, td+1)))

        # prepare data
        abnormal_data, events = self.prepare_data()

        self.abnormal_data = abnormal_data
        self.events = events

        self.assets = self.abnormal_data.columns

        # property-based
        self._timeline = None
        self._evt_avg_ts_sum = None
        self._the_mean = None
        self._booted = None

    @classmethod
    def with_normal_data(cls, data, events, wind, mean_type="simple",
        x_overlaps=False, norm_data_method="rolling", **kwargs):
        """Construct an  EventStudy with normal data.

        'between_events' is understood to result in the average between event
        windows, that is, from (previous event + wd) to (this_event + wa)

        Parameters
        ----------
        norm_data_method : str
            'rolling' and 'between_events' are supported
        kwargs : dict
            keyword arguments to method (if 'rolling' or 'between_events')

        Returns
        -------
        instance of cls (EventStudy)
        """
        # break down window
        ta, tb, tc, td = wind

        # make sure all time indexes in `events` are also in `data`
        data, _ = data.align(events, axis=0, join="outer")

        # rolling mean method
        if norm_data_method == "rolling":
            # rolling mean
            normal_data = data.rolling(**kwargs).mean()
            normal_data = normal_data.shift(-ta+1).where(events.notnull())
            normal_data = normal_data.loc[events.index]

        elif norm_data_method == "between_events":
            # between event windows; the idea is to use a temp EventStudy
            #   instance with normal_data = 0 and its .timeline property
            tmp_es = cls(data, events, wind, mean_type,
                normal_data=0.0, x_overlaps=x_overlaps)

            normal_data = list()

            # loop over columns, calculate between-events mean
            for c in tmp_es.abnormal_data.columns:
                this_data = tmp_es.abnormal_data.loc[:, c]
                this_timeline = tmp_es.timeline[c]\
                    .loc[:, ["inter_evt", "next_evt_no"]]

                both = pd.concat((this_data, this_timeline), axis=1,
                    join="inner").dropna()

                normal_data.append(both.drop("inter_evt", axis=1)\
                    .groupby(["next_evt_no"]).mean())

            normal_data = pd.concat(normal_data, axis=1, join="outer")
            normal_data = normal_data.loc[events.index, :]

        res = cls(data, events, wind, mean_type, normal_data=normal_data,
            x_overlaps=x_overlaps)

        return res

    @property
    def data_between_events(self):
        data = list()
        for k, v in self.timeline.items():
            this_data = self.abnormal_data.loc[:, k]
            this_tml = v.loc[:, "inter_evt"].notnull()

            data.append(this_data.where(this_tml))

        res = pd.concat(data, axis=1)

        res = res.loc[:, sorted(res.columns)]

        return res

    @property
    def timeline(self):
        if self._timeline is None:
            timelines = dict()
            # loop over columns in `abnormal_data`
            for c in self.abnormal_data.columns:
                this_evt = self.events[c].dropna()
                this_timeline = self.mark_timeline(
                    data=self.abnormal_data[c],
                    events=this_evt,
                    wind=self._wind,
                    x_overlaps=self._x_overlaps)
                timelines[c] = this_timeline
            self._timeline = timelines

        return self._timeline

    @property
    def evt_avg_ts_sum(self):
        if self._evt_avg_ts_sum is None:
            responses = self.collect_responses()
            self._evt_avg_ts_sum = self.get_ts_cumsum(responses, self._wind)

        return self._evt_avg_ts_sum

    @property
    def the_mean(self):
        if self._the_mean is None:
            fun = self.mean_fun(method=self._mean_type)
            self._the_mean = fun(self.evt_avg_ts_sum)

        return self._the_mean

    @property
    def booted(self):
        return self._booted

    @booted.setter
    def booted(self, value):
        self._booted = value

        # also, pickle
        with open(path_out + "evt_study_booted_mean.p", mode="wb") as han:
            pickle.dump(value, han)

    @booted.getter
    def booted(self):
        if self._booted is None:
            with open(path_out + "evt_study_booted_mean.p", mode="rb") as han:
                res = pickle.load(han)
            return res

        return self._booted

    def prepare_data(self):
        """Align, exclude overlapping events, calculate abnormal data.
        """
        # events
        events = self._raw["events"].copy()

        # data: makes sure all data points from `events` are in data too
        data, _ = self._raw["data"].align(events, axis=0, join="outer")

        # check data and events for severe overlaps (not the pre vs. post kind)
        if self._x_overlaps:
            new_evts = dict()

            for c in data.columns:
                this_evt, this_evt_diff = self.exclude_overlaps(
                    events[c].dropna(), data[c], self._wind)

                if len(this_evt_diff) > 0:
                    warnings.warn("The following events for {} will be " +
                        " excluded because of overlaps: {}".format(c,
                            list(this_evt_diff)))

                new_evts[c] = this_evt

            events = pd.DataFrame.from_dict(new_evts)

        # get normal data to the same dimensionality as `data`
        normal_data = self._normal_data.reindex(index=data.index)
        # normal_data = normal_data.shift(self._wind[0]).ffill()

        # abnormal data
        abn_data = data - normal_data

        return abn_data, events

    @staticmethod
    def exclude_overlaps(events, data, wind):
        """Exclude overlapping data.

        Parameters
        ----------
        events : pandas.Series
        wind : tuple
        """
        ta, tb, tc, td = wind

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
    def mark_timeline(data, events, wind, x_overlaps):
        """Mark the timeline by group such as within-event, pre-event etc.

        Parameters
        ----------
        data : pandas.Series
            of data
        events : pandas.Series
            of events (collapsed)
        wind : tuple
            ta, tb, tc, td as before
        x_overlaps : bool
            see above

        Returns
        -------
        timeline : pandas.DataFrame
            with columns
                evt_no - number of event within event wind,
                evt_wind_pre - indexes from ta to tb,
                evt_wind_post - indexes from tc to td
                inter_evt - 1 for periods between events
                next_evt_no - number of the next event
        """
        ta, tb, tc, td = wind
        wind_idx_pre = np.arange(ta, tb+1)
        wind_idx_post = np.arange(tc, td+1)

        # helper to find indexes
        def get_idx(t, data):
            return data.index.get_loc(t)

        # count events
        # evt_count = events.expanding().count() - 1
        evt_count = pd.Series(data=events.index, index=events.index)
        evt_count = evt_count.reindex(index=data.index)

        # allocate space for marked timeline: need pre and post to capture
        #   cases when deleting overlapping events is not desired
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
        # ipdb.set_trace()
        timeline.ix[timeline.loc[:, "evt_no"].isnull(), "inter_evt"] = 1

        # ensure overlaps do not enter
        if x_overlaps:
            idx = timeline.loc[:, ["evt_wind_pre", "evt_wind_post"]].\
                notnull().all(axis=1)
            timeline.ix[idx, ["evt_wind_pre", "evt_wind_post"]] = np.nan
            # timeline.dropna(subset=["evt_wind_pre", "evt_wind_post"],
            #     how="all", inplace=True)

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
        both = both.dropna(subset=["evt_wind_pre", "evt_wind_post"],
            how="all")

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
            # pivot
            res[c] = self.pivot_with_timeline(
                data=self.abnormal_data[c],
                timeline=self.timeline[c])

        res = pd.Panel.from_dict(res, orient="minor")

        return res

    @staticmethod
    def get_ts_cumsum(ndframe, wind):
        """Calculate cumulative sum over time."""
        ta, tb, tc, td = wind

        if len(ndframe.shape) > 2:
            ts_cumsum_before = ndframe.loc[:, :tb, :].iloc[:, ::-1, :].\
                cumsum().iloc[:, ::-1, :]
            ts_cumsum_after = ndframe.loc[:, tc:, :].cumsum()

            # concat
            ts_cumsum = pd.concat((ts_cumsum_before, ts_cumsum_after),
                axis=1)
        else:
            ts_cumsum_before = ndframe.loc[:tb, :].iloc[::-1, :].\
                cumsum().iloc[::-1, :]
            ts_cumsum_after = ndframe.loc[tc:, :].cumsum()

            # concat
            ts_cumsum = pd.concat((ts_cumsum_before, ts_cumsum_after),
                axis=0)

        return ts_cumsum

    def get_ci(self, ps, method="simple", **kwargs):
        """Calculate confidence bands for self.`the_mean`; wrapper.

        Parameters
        ----------
        ps : float / tuple of floats
            interval width or tuple of (lower bound, upper bound)
        method : str
            "simple" or "boot"
        **kwargs : dict
            additional arguments to methods 'simple' or 'boot'
        Returns
        -------
        ci : pandas.DataFrame
            column'ed by quantiles

        sets attribute `ci` needed for plots later
        """
        # if `ps` was provided as single float
        if isinstance(ps, float):
            ps = ((1-ps)/2, ps/2+1/2)

        # calculate confidence band
        if method == "simple":
            ci = self.simple_ci(ps=ps)
        elif method == "boot":
            booted = self.boot_the_mean(ps, **kwargs)
            self.booted = booted
            ci = booted.quantile(ps, axis=1).T
        else:
            raise NotImplementedError("ci you asked for is not implemented")

        self.ci = ci

        return ci

    def simple_ci(self, ps, variances=None):
        """
        """
        ta, tb, tc, td = self._wind

        # get the huge panel of everything stacked together
        the_panel = self.collect_responses()

        var_sums = pd.DataFrame(
            index=self.events.columns,
            columns=["var", "count"])

        for c in self.events.columns:
            # c = "nzd"
            # evts_used = self.timeline[c].loc[:, "evt_no"].dropna().unique()
            evts_used = the_panel.loc[:,:,c].dropna(axis=1, how="all").columns

            mask = self.timeline[c].loc[:, "inter_evt"].isnull()

            # calculate variance between events
            # TODO: think about var here
            var_btw = resample_between_events(self.abnormal_data[c],
                events=self.events[c].dropna(),
                fun=lambda x: np.nanvar(x),
                mask=mask)

            var_sums.loc[c, "count"] = var_btw.loc[evts_used].count().values[0]

            if variances is None:
                var_sums.loc[c, "var"] = var_btw.loc[evts_used].sum().values[0]
            elif isinstance(variances, pd.Series):
                var_sums.loc[c, "var"] = var_sums.loc[c, "count"]*variances[c]
            else:
                raise ValueError("Unknown type for variances!")

        # daily variances
        daily_var = var_sums.loc[:, "var"] / var_sums.loc[:, "count"]**2

        # weighted
        if self._mean_type == "simple":
            wght = pd.Series(1/len(self.assets), index=self.assets)
        elif self._mean_type == "count_weighted":
            wght = var_sums.loc[:, "count"] / var_sums.loc[:, "count"].sum()

        avg_daily_var = (daily_var * wght**2).sum()

        # time index
        time_idx = pd.Series(
            data=np.hstack((np.arange(-ta+tb+1,0,-1), np.arange(1,td-tc+2))),
            index=self.event_index)

        cumul_avg_daily_var = avg_daily_var * time_idx

        ci = pd.concat((
            np.sqrt(cumul_avg_daily_var).rename(ps[0]) * norm.ppf(ps[0]),
            np.sqrt(cumul_avg_daily_var).rename(ps[1]) * norm.ppf(ps[1])),
            axis=1)

        return ci

    def boot_the_mean(self, ps, M=500, n_blocks=None):
        """Block bootstrap.
        Returns
        -------
        ci : pandas.DataFrame
            with columns for confidence interval bands
        ps : float/tuple
            see above
        M : int
            number of iterations
        n_blocks : int
            size of blocks to use for resampling
        """
        ta, tb, tc, td = self._wind

        if n_blocks is None:
            n_blocks = td - ta + 1

        ci = pd.Panel(
            items=ps,
            major_axis=self.event_index,
            minor_axis=self.abnormal_data.columns)

        # boot from `abnormal_data` without intervals around events
        this_interevt_idx = {
            c: self.timeline[c].loc[:, "inter_evt"].notnull() \
                for c in self.assets}
        boot_from = {
            c: self.abnormal_data[c].copy().ix[this_interevt_idx[c]] \
                for c in self.assets}
        Ks = {c: self.events[c].count() for c in self.assets}

        boot_from = pd.DataFrame.from_dict(boot_from)

        # space for df's of pivoted tables
        booted = pd.DataFrame(columns=range(M), index=self.event_index)

        # loop over simulations
        for p in range(M):

            print(p)
            # ipdb.set_trace()

            # shuffle data
            shuffled_data = self.shuffle_data(boot_from, n_blocks=n_blocks)

            # shuffle events
            # make sure resampled events do not lie outside normal range
            possible_dt = shuffled_data.index.tolist()[(-ta+1):-(td+1)]
            shuffled_evts = pd.DataFrame.from_dict({
                c: pd.Series(1,
                    index=np.random.choice(possible_dt, Ks[c], replace=False))\
                for c in self.assets})
            shuffled_evts = shuffled_evts.sort_index(axis=0)

            this_evt_study = EventStudy(
                data=shuffled_data,
                events=shuffled_evts,
                wind=self._wind,
                mean_type=self._mean_type,
                normal_data=0.0,
                x_overlaps=self._x_overlaps)

            booted.iloc[:, p] = this_evt_study.the_mean

        return booted

    @staticmethod
    def shuffle_data(data, n_blocks):
        """
        """
        T, N = data.shape

        arr_no_na = np.array([
            np.random.choice(
                data[p].dropna().values, T) for p in data.columns]).T

        old_df_no_na = pd.DataFrame(
            data=arr_no_na,
            index=data.index,
            columns=data.columns)

        new_df_no_na = data.fillna(old_df_no_na).values

        new_df_no_na = np.concatenate((
            new_df_no_na,
            new_df_no_na[np.random.choice(range(T), n_blocks - T % n_blocks)]))

        M = new_df_no_na.shape[0] // n_blocks
        res = new_df_no_na.reshape(M, -1, N)
        res = res[np.random.permutation(M)].reshape(-1, N)

        res = pd.DataFrame(data=res, columns=data.columns)

        return res

    def plot(self):
        """
        Parameters
        ----------
        kwargs : dict
            of arguments to EventStudy.get_ci() method.

        Returns
        -------
        fig :
        """
        ta, tb, tc, td = self._wind

        ts_mu = self.evt_avg_ts_sum
        cs_ts_mu = self.the_mean

        fig, ax = plt.subplots(2, figsize=(8.4,11.7/2), sharex=True)

        # 2nd plot: cumulative sums
        which_axis = "items" if ts_mu.shape[2] > 1 else "minor_axis"
        ts_mu.loc[:,:tb,:].mean(axis=which_axis).plot(ax=ax[0], color=new_gray)
        ts_mu.loc[:,tc:,:].mean(axis=which_axis).plot(ax=ax[0], color=new_gray)
        # add points at initiation
        ax[0].plot([tb]*len(self.assets), ts_mu.loc[:,tb,:].mean(axis=1),
            color="k", linestyle="none", marker=".", markerfacecolor="k")
        ax[0].plot([tc]*len(self.assets), ts_mu.loc[:,tc,:].mean(axis=1),
            color="k", linestyle="none", marker=".", markerfacecolor="k")
        # mean in black
        cs_ts_mu.plot(ax=ax[0], color='k', linewidth=1.5)
        title = "all assets" if ts_mu.shape[2] > 1 else "all events"
        ax[0].set_title(title)

        # 3rd plot: ci around avg cumsum
        cs_ts_mu.loc[:tb].plot(ax=ax[1], color='k', linewidth=1.5)
        cs_ts_mu.loc[tc:].plot(ax=ax[1], color='k', linewidth=1.5)
        ax[1].fill_between(self.event_index,
            self.ci.iloc[:,0].values,
            self.ci.iloc[:,1].values,
            color=new_gray, alpha=0.5, label="conf. interval")
        ax[1].set_title("cumulative average")

        # some parameters common for all ax
        for x in range(len(ax)):
            # ax[x].legend_.remove()
            ax[x].xaxis.set_ticks(self.event_index)
            ax[x].axhline(y=0, color='r', linestyle='--', linewidth=1.0)
            ax[x].grid(axis="both", alpha=0.33, linestyle=":")
            legend = ax[x].legend()
            legend.remove()

        ax[x].set_xlabel("periods after event")

        # # super title
        # fig.suptitle(self.data.name, fontsize=14)

        return fig, ax

    @staticmethod
    def mean_fun(method="simple"):
        """
        method : str
            'simple' or 'count_weighted'
        """
        def simple_avg(x):
            assert isinstance(x, pd.Panel)
            res = x.mean(axis=("items")).mean(axis=1)
            return res

        def count_weighted_avg(x):
            assert isinstance(x, pd.Panel)
            wght = x.count(axis="items").divide(
                x.count(axis="items").sum(axis=1),
                axis=0)
            res = (x.mean(axis=("items"))*wght).sum(axis=1)
            return res

        if method == "simple":
            return simple_avg
        elif method == "count_weighted":
            return count_weighted_avg
        else:
            raise ValueError("type '{}' not implemented".format(method))


if __name__ == "__main__":

    from foolbox.api import *
    from pandas.tseries.offsets import BDay

    # currency to drop
    drop_curs = ["jpy","dkk"]

    # window
    wind = (-10, -1, 1, 5)
    wa, wb, wc, wd = wind

    # start, end dates
    s_dt = pd.to_datetime(settings["sample_start"])
    e_dt = pd.to_datetime(settings["sample_end"])

    # data ------------------------------------------------------------------
    data_path = set_credentials.gdrive_path("research_data/fx_and_events/")
    out_path = set_credentials.gdrive_path("opec_meetings/tex/figs/")

    # spot returns + drop currencies ----------------------------------------
    with open(data_path + settings["fx_data"], mode='rb') as fname:
        fx = pickle.load(fname)
    ret = np.log(fx["spot_mid"].drop(drop_curs,axis=1,errors="ignore")).diff()
    ret = ret.loc[(s_dt - BDay(22)):, :]

    # events + drop currencies ----------------------------------------------
    with open(data_path + settings["events_data"], mode='rb') as fname:
        events_data = pickle.load(fname)

    events = events_data["joint_cbs"].drop(drop_curs + ["usd"],
        axis=1, errors="ignore")
    events = events.loc[s_dt:e_dt]

    # reindex with business day ---------------------------------------------
    data = ret.reindex(
        index=pd.date_range(ret.index[0], ret.index[-1], freq='B'))

    # normal data, all events sample ----------------------------------------
    es = EventStudy(data=data*100,
        events=events,
        mean_type="count_weighted",
        wind=wind,
        x_overlaps=True,
        normal_data=0.0)

    data_between_events = es.data_between_events
    norm_data_all_evt = data_between_events.ewm(alpha=0.95).mean().shift(-wa+1)

    # hike? cut? status quo? ------------------------------------------------
    evt = events.where(events < 0).dropna(how="all")

    # normal data
    normal_data = norm_data_all_evt.where(evt.notnull()).loc[evt.index, :]

    # event study!
    es = EventStudy(data=data*100,
        events=evt,
        mean_type="count_weighted",
        wind=wind,
        x_overlaps=True,
        normal_data=normal_data)

    ci = es.get_ci(ps=0.95, method="boot", M=222)
    fig, ax = es.plot()



    # recipe #2: constructor ------------------------------------------------
    es = EventStudy.with_normal_data(data=data*100,
        events=evt,
        mean_type="count_weighted",
        wind=wind,
        x_overlaps=True,
        norm_data_method="between_events")

    fig, ax = es.plot()

    # recipe #3: zero mean --------------------------------------------------
    es = EventStudy(data=data*100,
        events=evt,
        mean_type="count_weighted",
        wind=wind,
        x_overlaps=True,
        normal_data=0)

    ci = es.get_ci(ps=0.95, method="simple")
    es.the_mean
    fig, ax = es.plot()
