import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import warnings
from foolbox.data_mgmt import set_credentials
from foolbox.utils import resample_between_events
from foolbox.wp_tabs_figs.wp_settings import *
import pickle
from foolbox.linear_models import PureOls

path_out = set_credentials.set_path("research_data/fx_and_events/",
    which="gdrive")


class EventStudy:
    """
    """
    def __init__(self, data, events, window, mean_type="simple"):
        """
        Parameters
        ----------
        data : pandas.DataFrame/pandas.Series
            of returns (need to be summable)
        events : list/pandas.DataFrame/pandas.Series
            of events (if a pandas object, must be indexed with dates)
        window : tuple of int
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
        """
        # indexes better be all of the same type
        events.index = events.index.map(pd.to_datetime)
        data.index = data.index.map(pd.to_datetime)

        self.data = data
        self.events = events

        # parameters --------------------------------------------------------
        self.assets = self.data.columns
        self._window = window
        self._mean_type = mean_type

        # break down window, construct index of holding days
        ta, tb, tc, td = self._window
        self.event_index = np.concatenate((np.arange(ta, tb+1),
                                           np.arange(tc, td+1)))

        # property-based ----------------------------------------------------
        self._timeline = None
        self._ar = None
        self._car = None
        self._the_mean = None
        self._booted = None

    @property
    def mask_between_events(self):
        res = self.timeline.xs("inter_evt", axis=1, level=1, drop_level=True)

        return res

    @property
    def timeline(self):
        if self._timeline is None:
            self._timeline = self.mark_timeline(self.data, self.events,
                                                self._window)

        return self._timeline

    @property
    def ar(self):
        if self._ar is None:
            self._ar = self.get_ar()
        return self._ar

    @property
    def car(self):
        if self._car is None:
            self._car = self.get_car(self.get_ar(), self._window)

        return self._car

    @property
    def the_mean(self):
        if self._the_mean is None:
            fun = self.mean_fun(method=self._mean_type)
            self._the_mean = fun(self.car)

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

    def reindex(self, freq, inplace=False, **kwargs):
        """Reindex by frequency.

        Worth doing if one suspects the datais not evenly indexed.

        Parameters
        ----------
        freq : str
            e.g. 'B'
        inplace : bool
            False to return a new EventStudy instance
        kwargs : dict
            arguments to pandas.DataFrame.reindex()

        Returns
        -------

        """
        new_idx = pd.date_range(self.data.index[0], self.data.index[-1],
                                freq=freq)
        new_data = self.data.reindex(index=new_idx, **kwargs)

        if inplace:
            self.data = new_data
        else:
            return EventStudy(new_data, self.events, self._window,
                              self._mean_type)

    def exclude_overlapping_events(self, inplace=False):
        """

        Parameters
        ----------
        inplace : bool
            False to return a new EvetnStudy instance

        Returns
        -------
        if inplace is True, nothing (sets a new value on self.events),
        else a new EventStudy instance

        """
        new_evt = dict()
        evt_diff = dict()

        # loop over columns of data
        for c_name, c_val in self.data.iteritems():
            # this will reindex events with c_val.index and exclude overlapping
            this_new_evt, this_x_evt = self.exclude_overlaps(
                self.events[c_name].dropna(), c_val, self._window)

            # save
            new_evt[c_name] = this_new_evt
            evt_diff[c_name] = pd.Series('x', index=this_x_evt)

        # dicts to DataFrames
        new_evt = pd.DataFrame.from_dict(new_evt)
        evt_diff = pd.DataFrame.from_dict(evt_diff)

        # return, based on inplace
        if inplace:
            self.events = new_evt
            return evt_diff
        else:
            print(evt_diff)
            return EventStudy(self.data, new_evt, self._window,
                              self._mean_type)

    def align_data_with_events(self, inplace=False):
        """Align data and events thus ensuring that all events are represented.

        Makes sure all data points from `events` are in data too.

        Returns
        -------

        """
        new_data, _ = self.data.align(self.events, axis=0, join="outer")

        if inplace:
            self.data = new_data
        else:
            return EventStudy(new_data, self.events, self._window,
                              self._mean_type)

    def default_data_manipulation(self, inplace=False):
        """Align, exclude overlapping events.
        """
        # align
        if inplace:
            self.align_data_with_events(inplace=True)
            self.reindex(freq='B', inplace=True)
            # self.exclude_overlapping_events(inplace=True)
        else:
            new_evt_study = self.align_data_with_events(inplace=False)
            new_evt_study.reindex(freq='B', inplace=False)
            # new_evt_study.exclude_overlapping_events(inplace=False)

            return new_evt_study

    @staticmethod
    def exclude_overlaps(events, data, window):
        """Exclude overlapping data.

        This is probably not needed, as overlaps are taken care of in
        mark_timeline.

        Parameters
        ----------
        events : pandas.Series
        data : pandas.DataFrame or pandas.Series
            only need to take the index and reindex `events` accordingly
        window : tuple

        Returns
        -------
        new_evt : pandas.Series
            of events without overlaps
        evt_diff : pandas.Index
            index of dropped events

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
        new_evt = events.where(overlaps < 2).dropna()

        return new_evt, evt_diff

    @staticmethod
    def mark_timeline(data, events, window):
        """

        Parameters
        ----------
        data
        events
        window

        Returns
        -------

        """
        # defenestrate
        ta, tb, tc, td = window
        window_pre = np.arange(ta, 0)
        window_post = np.arange(1, td + 1)

        # pre- and post-event indices, by window ----------------------------
        # reindex events, create df with 1 where an event took place
        evt_notnull = events.reindex_like(data).notnull()
        evt = evt_notnull.where(evt_notnull)

        # replace 1 with 0 to have day-zero events
        evt_pre = evt.copy().replace(1, 0)
        evt_post = evt.copy().replace(1, 0)

        # pre-event
        for p in window_pre[::-1]:
            evt_pre.fillna((evt * p).bfill(limit=np.abs(p)), inplace=True)

        # kill day-0 events
        evt_pre = evt_pre.where(evt.isnull())

        # post-event
        for p in window_post:
            evt_post.fillna((evt * p).ffill(limit=np.abs(p)), inplace=True)

        # exclude values not in (ta, tb) and (tc, td)
        idx_correct_window = (evt_pre <= tb) | (evt_post >= tc)
        idx_between_evt = ~((evt_pre >= ta) | (evt_post <= td))

        # exclude overlaps
        idx_overlap = (evt_pre <= tb) & (evt_post >= tc)
        idx_correct_window = idx_correct_window & (~idx_overlap)

        evt_pre = evt_pre.where(idx_correct_window)
        evt_post = evt_post.where(idx_correct_window)

        # event id (date), gets repeated over the whole window --------------
        # repeat series of dates as many times as there are assets
        dt_series = pd.Series(data=data.index, index=data.index)
        dt_df = pd.concat([dt_series.rename(c) for c in data.columns], axis=1)

        # leave only cells where there are events
        dt_df = dt_df.where(evt_notnull)

        # next event number is the number of event immediately following date
        next_evt_no = dt_df.fillna(method="bfill")

        # fill with dates
        dt_df.bfill(limit=-ta, inplace=True)
        dt_df.ffill(limit=td, inplace=True)
        dt_df = dt_df.where(idx_correct_window)

        # concat to a df with MultiIndex; swaplevel needed to have level [0]
        #  to keep asset names
        timeline = pd.concat({
            "evt_no": dt_df,
            "evt_wind_pre": evt_pre,
            "evt_wind_post": evt_post,
            "inter_evt": idx_between_evt,
            "next_evt_no": next_evt_no}, axis=1).swaplevel(axis=1)

        return timeline

    def get_ar(self):
        """Wrapper for pivot_with_timeline for every column in self.`data`.

        Returns
        -------
        res : pandas.DataFrame
            of abnormal retuns, indexed by window, column'ed by
            a MultiIndex of [0] asset names, [1] event ids (dates)
        """
        res = dict()

        # loop over columns in `data`
        for c_name, c_val in self.data.iteritems():
            # pivot
            res[c_name] = self.pivot_with_timeline(
                data=c_val, timeline=self.timeline[c_name])

        # concat all assets to a df with MultiIndex
        res = pd.concat(res, axis=1)

        # name the MultiIndex
        res.columns.names = ["assets", "events"]

        return res

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
            indexed by window, columned by event ids (dates)
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

    @staticmethod
    def get_car(data, window):
        """Calculate cumulative abnormal return from abnormal returns.

        As the indexing is from very negative (up) to very positive (down),
        say, from period -10 to period +2, the cumulative return is taken
        'upwards' from -1 to -10 and 'downwards' from +1 to +2.

        Parameters
        ----------
        data : pandas.DataFrame
        window : tuple

        Returns
        -------
        car : pandas.DataFrame
            of cumulative abnormal returns, indexed by window, column'ed by
            a MultiIndex of [0] asset names, [1] event ids (dates)

        """
        ta, tb, tc, td = window

        # 'upwards'
        car_before = data.loc[:tb, :].iloc[::-1, :].cumsum().iloc[::-1, :]
        # 'downwards'
        car_after = data.loc[tc:, :].cumsum()

        # concat both
        car = pd.concat((car_before, car_after), axis=0)

        return car

    def boot_ci(self, ps, M=10):
        """

        Parameters
        ----------
        ps
        M

        Returns
        -------

        """
        pass

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

    def simple_ci(self, ps, variances):
        """
        Parameters
        ----------
        ps : float or tuple
        variances : pandas.DataFrame
            of same shape and form as self.`events`
        """
        # defenestrate
        wa, wb, wc, wd = self._window

        # confidence interval levels
        if isinstance(ps, float):
            ps = ((1-ps)/2, ps/2+1/2)

        # construct time index to scale variance
        time_idx = np.hstack((np.arange(-wa + wb + 1, 0, -1),
                              np.arange(1, wd - wc + 2))).reshape(-1, 1)

        # convert variances to the same shape as ar
        var_ar = variances.unstack().dropna().to_frame().T
        var_ar.columns.names = self.ar.columns.names

        # broadcast + multiply with wa:wd; this is a column (scale factors)
        # times row (variances for each event) multiplication bound to
        # result in a matrix
        var_car = pd.DataFrame(data=time_idx * var_ar.values,
                               index=self.event_index,
                               columns=var_ar.columns)

        # mask var_car to have na where there are na in car
        var_car = var_car.where(self.car.notnull())

        # var of average CAR across events and assets
        var_fun = self.var_fun(method=self._mean_type)
        var_car_x_evt_x_ast = var_fun(var_car)
        
        # confidence interval
        ci = pd.concat((
            np.sqrt(var_car_x_evt_x_ast).rename(ps[0]) * norm.ppf(ps[0]),
            np.sqrt(var_car_x_evt_x_ast).rename(ps[1]) * norm.ppf(ps[1])),
            axis=1)

        # var_sums = pd.DataFrame(
        #     index=self.events.columns,
        #     columns=["var", "count"])
        #
        # for c in self.assets:
        #     # c = "nzd"
        #     # evts_used = self.timeline[c].loc[:, "evt_no"].dropna().unique()
        #     evts_used = self.ar.loc[:,:,c].dropna(axis=1, how="all").columns
        #
        #     mask = self.timeline[c].loc[:, "inter_evt"].isnull()
        #
        #     # calculate variance between events
        #     # TODO: think about var here
        #     var_btw = resample_between_events(self.data[c],
        #                                       events=self.events[c].dropna(),
        #                                       fun=lambda x: np.nanvar(x),
        #                                       mask=mask)
        #
        #     var_sums.loc[c, "count"] = var_btw.loc[evts_used].count().values[0]
        #
        #     if variances is None:
        #         var_sums.loc[c, "var"] = var_btw.loc[evts_used].sum().values[0]
        #     elif isinstance(variances, pd.Series):
        #         var_sums.loc[c, "var"] = var_sums.loc[c, "count"]*variances[c]
        #     else:
        #         raise ValueError("Unknown type for variances!")
        #
        # # daily variances
        # daily_var = var_sums.loc[:, "var"] / var_sums.loc[:, "count"]**2
        #
        # # weighted
        # if self._mean_type == "simple":
        #     wght = pd.Series(1/len(self.assets), index=self.assets)
        # elif self._mean_type == "count_weighted":
        #     wght = var_sums.loc[:, "count"] / var_sums.loc[:, "count"].sum()
        #
        # avg_daily_var = (daily_var * wght**2).sum()
        #
        # # time index
        # time_idx = pd.Series(
        #     data=np.hstack((np.arange(-wa+wb+1,0,-1), np.arange(1,wd-wc+2))),
        #     index=self.event_index)
        #
        # cumul_avg_daily_var = avg_daily_var * time_idx
        #
        # ci = pd.concat((
        #     np.sqrt(cumul_avg_daily_var).rename(ps[0]) * norm.ppf(ps[0]),
        #     np.sqrt(cumul_avg_daily_var).rename(ps[1]) * norm.ppf(ps[1])),
        #     axis=1)

        return ci

    def boot_the_mean(self, what="car", n_iter=500, n_blocks=None,
                      fillna=False):
        """Block bootstrap.

        Returns
        -------
        what : str
            what to take the mean of: 'ar' or 'car'
        n_iter : int
            number of iterations
        n_blocks : int
            size of blocks to use for resampling
        fillna : bool
            fill na in the data when bootstrapping

        Returns
        -------
        booted : pandas.DataFrame
        """
        # defenestrate
        ta, tb, tc, td = self._window

        # default for the no of blocks
        if n_blocks is None:
            n_blocks = td - ta + 1

        # boot from `data` without windows around events --------------------
        # boot_from = dict()
        # for k, v in self.timeline.groupby(axis=1, level=0):
        #     idx = v[k].loc[:, "inter_evt"]
        #     boot_from[k] = self.data.loc[idx, k]
        #
        # boot_from = pd.DataFrame.from_dict(boot_from)
        boot_from = self.data.where(self.mask_between_events)

        # no of events for each asset ---------------------------------------
        n_evts = {c: self.events[c].count() for c in self.assets}

        # loop over simulations ---------------------------------------------
        # space for df's of pivoted tables
        booted = pd.DataFrame(columns=range(n_iter), index=self.event_index)

        for p in range(n_iter):
            print(p)

            # shuffle data
            shuff_data = self.shuffle_data(boot_from, n_blocks, fillna)

            # shuffle events
            # make sure resampled events do not lie outside normal range
            possible_dt = shuff_data.index.tolist()[(-ta+1):-(td+1)]

            shuff_evts = dict()
            for c in self.assets:
                smpl = np.random.choice(possible_dt, n_evts[c], replace=False)
                shuff_evts[c] = pd.Series(1, index=smpl)

            shuff_evts = pd.DataFrame.from_dict(shuff_evts).sort_index(axis=0)

            # event study on simulated data
            this_evt_study = EventStudy(
                data=shuff_data,
                events=shuff_evts,
                window=self._window,
                mean_type=self._mean_type)

            # calculate the mean based on the reshuffled dataframe
            if what == "car":
                this_what = this_evt_study.car
            else:
                this_what = this_evt_study.ar

            func = this_evt_study.mean_fun(self._mean_type)
            booted.iloc[:, p] = func(this_what)

        return booted

    @staticmethod
    def shuffle_data(data, n_blocks, fillna=False):
        """Shuffle data in blocks.

        First, fill the na in `data` using randomly sampled observations in
        `data` itself. Then, the data is shuffled in blocks.

        Parameters
        ----------
        data : pandas.DataFrame
        n_blocks : int
        fillna : bool
            fill na in the data when bootstrapping
        Returns
        -------
        res : pandas.DataFrame
            of shuffled data
        """
        n_obs, n_cols = data.shape

        # fill na in `data` by botstrapping it ------------------------------
        if fillna:
            # bootstrap `data`
            arr_no_na = np.array([
                np.random.choice(
                    data[p].dropna().values, n_obs) for p in data.columns]).T

            # to frame
            old_df_no_na = pd.DataFrame(
                data=arr_no_na,
                index=data.index,
                columns=data.columns)

            # fill na
            new_df_no_na = data.fillna(old_df_no_na).values

        else:
            new_df_no_na = data.values

        # make the number of rows be a multiple of the number of blocks (for
        #   easier resampling)
        new_df_no_na = np.concatenate((
            new_df_no_na,
            new_df_no_na[np.random.choice(range(n_obs),
                                          n_blocks - n_obs % n_blocks)]))

        # resample
        M = new_df_no_na.shape[0] // n_blocks
        res = new_df_no_na.reshape(M, -1, n_cols)
        res = res[np.random.permutation(M)].reshape(-1, n_cols)

        # to frame again
        res = pd.DataFrame(data=res, columns=data.columns)

        return res

    def plot(self, xs_level="assets", plot_ci=False):
        """
        Parameters
        ----------
        xs_level : str
            'assets' or 'events'
        plot_ci : bool

        Returns
        -------

        """
        # defenestrate
        ta, tb, tc, td = self._window

        # handles
        car = self.car
        cs_car = self.the_mean

        # figure
        fig, ax = plt.subplots(2, figsize=(8.4, 11.7/2), sharex=True)

        # plot cumulative sums ----------------------------------------------
        to_plot = car.mean(axis=1, level=xs_level)

        # lower part
        to_plot.loc[:tb, :].plot(ax=ax[0], color=new_gray)

        # upper part
        to_plot.loc[tc:, :].plot(ax=ax[0], color=new_gray)

        # add points at initiation
        ax[0].plot(x=[tb]*to_plot.shape[1],
                   y=to_plot.loc[tb, :],
                   color='k', linestyle="none", marker=".",
                   markerfacecolor='k')

        ax[0].plot(x=[tc]*to_plot.shape[1],
                   y=to_plot.loc[tc, :],
                   color='k', linestyle="none", marker=".",
                   markerfacecolor='k')

        # mean in black
        cs_car.plot(ax=ax[0], color='k', linewidth=1.5)
        ax[0].set_title(xs_level)

        # plot ci around avg cumsum -----------------------------------------
        cs_car.loc[:tb].plot(ax=ax[1], color='k', linewidth=1.5)
        cs_car.loc[tc:].plot(ax=ax[1], color='k', linewidth=1.5)

        # furbish plot ------------------------------------------------------
        for x in range(len(ax)):
            # lost tickos
            ax[x].xaxis.set_ticks(self.event_index)
            # la linea de zero
            ax[x].axhline(y=0, color='k', linestyle='--', linewidth=1.0)
            # la grida
            ax[x].grid(axis="both", alpha=0.33, linestyle=":")
            # la legenda
            legend = ax[x].legend()
            legend.remove()

        # los lables
        ax[1].set_xlabel("periods after event")

        if not plot_ci:
            return fig, ax

        ax[1].fill_between(self.event_index,
                           self.ci.iloc[:, 0].values,
                           self.ci.iloc[:, 1].values,
                           color=new_gray, alpha=0.33, label="conf. interval")
        ax[1].set_title("cumulative average")

        # # super title
        # fig.suptitle(self.data.name, fontsize=14)

        return fig, ax

    @staticmethod
    def var_fun(method="simple"):
        """
        method : str
            'simple' or 'count_weighted'
        """

        def simple_var(x):
            # variance acrosss events (equally-weighted), for each asset
            var_x_evt = x.sum(axis=1, level="assets").div(
                x.count(axis=1, level="assets") ** 2)

            # var of average average CAR accross assets
            var_x_evt_x_ast = var_x_evt.sum(axis=1).div(
                var_x_evt.count(axis=1) ** 2)

            return var_x_evt_x_ast

        def count_weighted_var(x):
            # in the total mean, each asset's mean across events is weighted
            #  by the no of events for that asset
            evt_count = x.count(axis=1, level="assets")
            wght = evt_count.divide(evt_count.sum(axis=1), axis=0)

            # variance acrosss events (equally-weighted), for each asset
            var_x_evt = x.sum(axis=1, level="assets").div(
                x.count(axis=1, level="assets") ** 2)

            # var of average average CAR accross assets, weighted
            var_x_evt_x_ast = (var_x_evt * wght**2).sum(axis=1)

            return var_x_evt_x_ast

        if method == "simple":
            return simple_var
        elif method == "count_weighted":
            return count_weighted_var
        else:
            raise ValueError("type '{}' not implemented".format(method))

    @staticmethod
    def mean_fun(method="simple"):
        """
        method : str
            'simple' or 'count_weighted'
        """
        def simple_avg(x):
            res = x.mean(axis=1)
            return res

        def count_weighted_avg(x):
            # in the total mean, each asset's mean across events is weighted
            #  by the no of events for that asset
            evt_count = x.count(axis=1, level="assets")
            wght = evt_count.divide(evt_count.sum(axis=1), axis=0)
            res = (x.mean(axis=1, level="assets")*wght).sum(axis=1)
            return res

        if method == "simple":
            return simple_avg
        elif method == "count_weighted":
            return count_weighted_avg
        else:
            raise ValueError("type '{}' not implemented".format(method))


class EventStudyFactory:
    """
    """
    def __init__(self):
        """
        """
        pass

    @staticmethod
    def broadcast(*args, columns):
        """Broadcast columns if asset returns share events or other series.

        Series are concatenated into DataFrames, DataFrames are aggregated
        into dicts.

        Parameters
        ----------
        args : pandas.Series or pandas.DataFrame
        columns : list-like

        Returns
        -------

        """
        def broadcast_fun(arg):
            if isinstance(arg, pd.Series):
                # concat series into a df
                res = pd.concat([arg.rename(c) for c in columns],
                                axis=1, join="outer")
            elif isinstance(arg, pd.DataFrame):
                # dfs go into dicts
                res = {c: arg for c in columns}
            else:
                raise ValueError("Use Series or DataFrames only!")

            return res

        return (broadcast_fun(arg) for arg in args)

    @staticmethod
    def mark_windows(data, events, window, outside=False):
        """

        Parameters
        ----------
        data
        events
        window
        outside : bool
            True to mark outside windows

        Returns
        -------

        """
        wa, wb, wc, wd = window

        # find where any event takes place
        any_event = events.notnull()

        # reindex to match data's index
        any_event = any_event.notnull().reindex(index=data.index)

        # fill na values to mark window start and end
        all_windows = any_event.bfill(limit=(-1*wa)).ffill(limit=wd)

        # if need outside widows, invert
        if outside:
            res = all_windows.isnull()
        else:
            res = all_windows.notnull()

        return res

    def get_normal_data_exog(self, data, events, exog, window):
        """

        Parameters
        ----------
        data
        events
        exog
        window

        Returns
        -------

        """
        # defenestrate
        wa, wb, wc, wd = window

        # get mask with True outside event windows
        outside_windows = self.mark_windows(data, events, window, outside=True)

        # space for the normal data and residual variance
        norm_data = data * np.nan
        resid_var = events * np.nan

        # loop over assets
        for col, this_y in data.iteritems():
            # columns of events
            this_evt = events.loc[:, col].dropna()

            # put na in the df of regressors where event widnows are
            this_x_est = exog[col].where(outside_windows[col])
            this_x_fcast = exog[col].where(~outside_windows[col])

            # loop over events, use expanding windows TODO: what about other?
            for t in this_evt.index:
                # it is ensured by outside_windows that .loc[:t, ] also
                # excludes the window around t
                mod = PureOls(this_y, this_x_est.loc[:t, ], add_constant=True)

                # get x around t to make a forecast; +1 needed seince day 0
                # is the event day
                x_fcast_right = this_x_fcast.loc[t:].head(wd+1)
                x_fcast_left = this_x_fcast.loc[:t].tail(-wa+1)

                # avoid duplicates
                t_x_fcast = pd.concat((x_fcast_left, x_fcast_right.iloc[1:]),
                                      axis=0)

                # forecast, fill na in `norm_data`
                norm_data.fillna(
                    (this_y-mod.get_yhat(newdata=t_x_fcast)).to_frame(col),
                    inplace=True)

                # residual variance
                resid_var.loc[t, col] = mod.get_residuals().var()

        return norm_data, resid_var


if __name__ == "__main__":

    from foolbox.api import *
    from pandas.tseries.offsets import BDay

    # currency to drop
    drop_curs = ["jpy","dkk", "nok", ]

    # window
    wind = (-10, -1, 0, 5)
    wa, wb, wc, wd = wind

    # start, end dates
    s_dt = pd.to_datetime(settings["sample_start"])
    e_dt = pd.to_datetime(settings["sample_end"])

    # data ------------------------------------------------------------------
    data_path = set_credentials.gdrive_path("research_data/fx_and_events/")
    out_path = set_credentials.gdrive_path("opec_meetings/tex/figs/")

    # spot returns + drop currencies ----------------------------------------
    # with open(data_path + settings["fx_data"], mode='rb') as fname:
    #     fx = pickle.load(fname)
    fx = pd.read_pickle(data_path + settings["fx_data"])

    fx = pd.read_pickle(data_path + "bond_index_data.p")
    mat = "10+y"
    data = list()
    currs = list()
    for key, df in fx.items():
        data.append(np.log(df[mat]).diff() - np.log(df["5-7y"]).diff())
        currs.append(key)
    data = pd.concat(data, axis=1)
    data.columns = currs

    data = data.drop(["jpy", "usd"], axis=1)
    ret = data

    # ret = np.log(data).diff()
    ret = ret.loc[(s_dt - BDay(22)):, :]


    # ret = np.log(fx["spot_mid"].drop(drop_curs,axis=1,errors="ignore")).diff()
    # ret = ret.loc[(s_dt - BDay(22)):, :]

    # events + drop currencies ----------------------------------------------
    # with open(data_path + settings["events_data"], mode='rb') as fname:
    #     events_data = pickle.load(fname)
    events_data = pd.read_pickle(data_path + settings["events_data"])

    events = events_data["joint_cbs"].drop(drop_curs + ["usd"],
        axis=1, errors="ignore")
    events = events.loc[s_dt:e_dt]

    # reindex with business day ---------------------------------------------
    data = ret.reindex(
        index=pd.date_range(ret.index[0], ret.index[-1], freq='B'))

    # data_frequency = "H1"
    #
    # out_counter_usd_name = "fxcm_counter_usd_" + data_frequency + ".p"
    #
    # data = pd.read_pickle(data_path + out_counter_usd_name)
    # data = data["ask_close"].loc[s_dt:e_dt, events.columns].dropna(how="all")
    # data = data.pct_change()
    # data = data.loc[(s_dt - BDay(22)):, :]
    #
    # events.index = [ix.tz_localize("UTC") for ix in events.index]
    # events = events.loc[data.index[0]:data.index[-1], :]
    # events = events.dropna(how="all")


    # window
    # wind = (-240, -5, 0, 100)
    wa, wb, wc, wd = wind

    # exog = pd.DataFrame(1, index=data.index, columns=currs)
    #
    # exog_normal, exog_res = \
    #     EventStudyFactory().get_normal_data_exog(data, events, exog, wind)

    # normal data, all events sample ----------------------------------------
    es = EventStudy(data=(data)*100,
                    events=events,
                    mean_type="count_weighted",
                    window=wind)

    esh = EventStudy(data=(data)*100,
                     events=events.where(events > 0).dropna(how="all"),
                     mean_type="count_weighted",
                     window=wind)
    esl = EventStudy(data=(data)*100,
                     events=events.where(events < 0).dropna(how="all"),
                     mean_type="count_weighted",
                     window=wind)
    esn = EventStudy(data=(data)*100,
                     events=events.where(events == 0).dropna(how="all"),
                     mean_type="count_weighted",
                     window=wind)
    es.plot()
    esh.plot()
    esl.plot()
    esn.plot()

    # event study!
    es = EventStudy(data=data*100,
                    events=evt,
                    mean_type="count_weighted",
                    window=wind,
                    x_overlaps=True,
                    normal_data=normal_data)

    ci = es.get_ci(ps=0.95, method="boot", M=222)
    fig, ax = es.plot()



    # recipe #2: constructor ------------------------------------------------
    es = EventStudy.with_normal_data(data=data*100,
                                     events=evt,
                                     mean_type="count_weighted",
                                     window=wind,
                                     x_overlaps=True,
                                     norm_data_method="between_events")

    fig, ax = es.plot()

    # recipe #3: zero mean --------------------------------------------------
    es = EventStudy(data=data*100,
                    events=evt,
                    mean_type="count_weighted",
                    window=wind,
                    x_overlaps=True,
                    normal_data=0)

    ci = es.get_ci(ps=0.95, method="simple")
    es.the_mean
    fig, ax = es.plot()
