import pandas as pd
import numpy as np
from scipy.stats import norm
from random import choice
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from foolbox.econometrics.linear_models import OLS
from foolbox.visuals import *


class EventStudy:
    """Event study machinery.

    Event studies with N assets and M(n) asset-specific events for each
    (trivially generalized to one asset or the same events for each asset).

    Window (-5,-1,1,5) means that one starts to look at returns 5
    periods before each event, ends 1 day before, starts again 1 day
    after and stops 5 days after.

    """

    def __init__(self, data, events, window, cross_asset_weights="equal"):
        """
        Parameters
        ----------
        data : pandas.DataFrame or pandas.Series
            of returns (need to be summable)
        events : list or pandas.DataFrame or pandas.Series
            of events (if a pandas object, must be indexed with dates)
        window : tuple
            of int (a,b,c,d) or (a,d) where each element is a relative (in
            periods) date to explore patterns in `data` either (for the case
            of a 4-tuple) from a to b and c to d or (for the case of a
            2-tuple) from a to d.
        cross_asset_weights : str
            method for calculating the average across assets; "equal" or
            "by_event" are supported
        """
        # window is either a 4-tuple covering zero or a 2-tuple
        assert ((len(window) == 4) & (window[0] < 0 < window[-1])) | \
               (len(window) == 2)

        # property-based stuff
        self._timeline = None

        # always a 4-tuple will be returned here, even if a 2-tuple is supplied
        self._window = window
        self._the_mean = None

        # indexes better be all of the same type (PeriodIndexes are fine and
        #   need not be converted)
        data_idx = isinstance(data.index, (pd.DatetimeIndex, pd.PeriodIndex))
        evt_idx = isinstance(data.index, (pd.DatetimeIndex, pd.PeriodIndex))

        if data_idx | evt_idx:
            data.index = data.index.map(pd.to_datetime)
            events.index = events.index.map(pd.to_datetime)

        # data types better be float
        data = data.astype(float)

        # rename, align
        data, events = EventStudyFactory.align_for_event_study(data, events)

        self.data = data
        self.events = events

        # parameters
        self.assets = self.data.columns
        self.cross_asset_weights = cross_asset_weights

        # construct index of holding days, e.g. [-3, -2, -1, 1, 2] for
        #   window = (-3, 2)
        ta, tb, tc, td = self.window
        self.event_index = np.unique(
            np.concatenate((np.arange(ta, tb + 1), np.arange(tc, td + 1)))
        )

    @property
    def window(self):
        """Return a 4-tuple, broadcast if necessary."""
        if len(self._window) == 4:
            return self._window
        else:
            return self._window * 2

    @property
    def mask_between_events(self):
        res = self.timeline.xs("inter_evt", axis=1, level=1, drop_level=True)

        return res

    @property
    def timeline(self):
        if self._timeline is None:
            self._timeline = EventStudyFactory.mark_timeline(
                self.data, self.events, self.window)

        return self._timeline

    @property
    def the_mean(self):
        if self._the_mean is None:
            self._the_mean = self.cs_mean(self.get_car())

        return self._the_mean

    def cs_mean(self, df):
        """Calculate cross-sectional mean.

        Weighting is different based on `self.cross_asset_weights`.

        Parameters
        ----------
        df : pandas.DataFrame
            with (asset, event) for columns

        Returns
        -------
        res : pandas.Series

        """
        func = self.mean_fun(method=self.cross_asset_weights)

        res = func(df)

        return res

    def reindex(self, freq, inplace=False, **kwargs):
        """Reindex by frequency.

        Worth doing if one suspects the data is not evenly indexed.

        Parameters
        ----------
        freq : str
            valid pandas frequency e.g. 'B'
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
                              self.cross_asset_weights)

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

    def get_ar(self):
        """Calculate abnormal returns.

        Returns
        -------
        res : pandas.DataFrame

        """
        res = self._get_ar(self.data, self.timeline)

        return res

    def get_car(self):
        """Calculate cumulative abnormal returns.

        Returns
        -------
        res : pandas.DataFrame

        """
        # get abnormal returns first
        ar = self.get_ar()

        # cumulate
        res = self._get_car(ar, self.window)

        return res

    @staticmethod
    def _get_ar(data, timeline):
        """Pivot data based on timeline to arrive at abnormal returns.

        Returns
        -------
        ar : pandas.DataFrame
            of abnormal returns, indexed by window, column'ed by
            a MultiIndex of [0] asset names, [1] event ids (dates)
        """
        ar = dict()

        # loop over columns in `data`, pivot for each
        for c_name, c_val in data.iteritems():
            ar[c_name] = EventStudyFactory.pivot_with_timeline(
                data=c_val, timeline=timeline[c_name])

        # concat all assets to a df with MultiIndex
        ar = pd.concat(ar, axis=1)

        # name the MultiIndex
        ar.columns.names = ["assets", "events"]

        return ar

    @staticmethod
    def _get_car(ar, window):
        """Calculate cumulative abnormal return from abnormal returns.

        `window` can be of the form (-10, -1, 1, 5) - standard case - or of
        the form (-10, -1, -10, -1) - transformed version of window (-10, -1)

        Parameters
        ----------
        ar : pandas.DataFrame
        window : tuple
            of 4 elements

        Returns
        -------
        car : pandas.DataFrame
            of cumulative abnormal returns, indexed by window, column'ed by
            a MultiIndex of [0] asset names, [1] event ids (dates)

        """
        ta, tb, tc, td = window

        car_before = None
        car_after = None

        # 'upwards'
        if ta < 0:
            car_before = ar.loc[:tb, :].iloc[::-1, :].cumsum().iloc[::-1, :]
        if td > 0:
            # 'downwards'
            car_after = ar.loc[tc:, :].cumsum()

        # concat both
        car = pd.concat((car_before, car_after), axis=0)

        return car

    def get_ci(self, ps, method="simple", **kwargs):
        """Calculate confidence bands for the average CAR.

        Parameters
        ----------
        ps : float or tuple
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
            ps = ((1 - ps) / 2, ps / 2 + 1 / 2)

        # calculate confidence band
        if method == "simple":
            if "variances" not in kwargs.keys():
                raise ValueError("Specify variances for the confidence "
                                 "interval!")
            ci = self.simple_ci(ps=ps, **kwargs)

        elif method == "boot":
            booted = self.bootstrap_sample(**kwargs)
            ci = booted.quantile(ps, axis=1).T

        else:
            raise NotImplementedError("ci you asked for is not implemented")

        self.ci = ci

        return ci

    def boot_ci(self, boot_from=None, prob=0.9, n_sim=101, blocksize=None,
                fill_missing=False, use_ar=False):
        """Construct a bootstrapped confidence interval for the average CAR.

        Parameters
        ----------
        boot_from : pandas.DataFrame
            to use for bootstrapping; by default, `self.data` is used,
            with event windows taken out
        prob : float or tuple
            size of the confidence interval or the bands thereof
        n_sim : int
            number of simulations
        blocksize : int
            size of blocks to resample by
        fill_missing : bool
            True to fill NA in data
        use_ar : bool
            True to construct the interval for abnormal returns rather than
            for cumulative abnormal returns

        Returns
        -------
        ci : pandas.DataFrame
            columned by interval bands, indexed by `event_index`

        """
        if boot_from is None:
            boot_from = self.data.where(self.mask_between_events)

        if isinstance(prob, float):
            ps = ((1 - prob) / 2, prob / 2 + 1 / 2)

        # bootstrap
        booted = self.bootstrap_sample(
            smpl=boot_from, what=("ar" if use_ar else "car"),
            n_iter=n_sim, blocksize=blocksize, fillna=fill_missing
        )

        ci = booted.quantile(prob, axis=1).T

        return ci

    def simple_ci(self, ps, variances):
        """
        Parameters
        ----------
        ps : float or tuple
        variances : pandas.DataFrame
            of same shape and form as self.`events`
        """
        raise NotImplementedError
        # defenestrate
        wa, wb, wc, wd = self.window

        # confidence interval levels
        if isinstance(ps, float):
            ps = ((1 - ps) / 2, ps / 2 + 1 / 2)

        # construct time index to scale variance
        time_idx = np.hstack((np.arange(-wa + wb + 1, 0, -1),
                              np.arange(1, wd - wc + 2))).reshape(-1, 1)

        # convert variances to the same shape as ar
        var_ar = variances.unstack().dropna().to_frame().T
        var_ar.columns.names = self.get_ar().columns.names

        # broadcast + multiply with wa:wd; this is a column (scale factors)
        # times row (variances for each event) multiplication bound to
        # result in a matrix
        var_car = pd.DataFrame(data=time_idx * var_ar.values,
                               index=self.event_index,
                               columns=var_ar.columns)

        # mask var_car to have na where there are na in car
        # TODO: do we need .notnull()? maybe not...
        # var_car = var_car.where(self.car.notnull())

        # var of average CAR across events and assets
        var_fun = self.var_fun(method=self.mean_type)
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
        # if self.cross_asset_weights == "simple":
        #     wght = pd.Series(1/len(self.assets), index=self.assets)
        # elif self.cross_asset_weights == "count_weighted":
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

    def bootstrap_sample(self, smpl, what="car", n_iter=101,
                         blocksize=None, fillna=False):
        """Block bootstrap.

        Returns
        -------
        smpl : pandas.DataFrame
            to bootstrap from
        what : str
            what to take the mean of: 'ar' or 'car'
        n_iter : int
            number of iterations
        blocksize : int
            size of blocks to use for resampling
        fillna : bool
            True to fill na in `smpl`

        Returns
        -------
        res : pandas.DataFrame
        """
        if blocksize is None:
            blocksize = self.window[-1] - self.window[0] + 1

        # loop over simulations ---------------------------------------------
        # space for df's of pivoted tables
        res = dict()

        for p in range(n_iter):
            print(p)

            # shuffle data
            shuff_data = self.shuffle_data(smpl, blocksize, fillna)

            # event study on simulated data
            this_evt_study = EventStudy(
                data=shuff_data,
                events=self.events,
                window=self._window,
                cross_asset_weights=self.cross_asset_weights)

            # calculate the mean based on the reshuffled dataframe
            if what == "car":
                this_what = this_evt_study.get_car()
            else:
                this_what = this_evt_study.get_ar()

            func = this_evt_study.mean_fun(self.cross_asset_weights)
            res[p] = func(this_what)

        res = pd.DataFrame.from_dict(res, orient="columns")

        return res

    def shuffle_data(self, data, blocksize, fillna=False):
        """Shuffle data in blocks.

        First, fill the na in `data` using randomly sampled observations in
        `data` itself. Then, the data is shuffled in blocks.

        Parameters
        ----------
        data : pandas.DataFrame
        blocksize : int
        fillna : bool
            fill na in the data when bootstrapping
        Returns
        -------
        res : pandas.DataFrame
            of shuffled data
        """
        n_obs, n_cols = data.shape
        n_blocks = n_obs // blocksize + 1

        if n_cols is None:
            res = self.shuffle_data(data[:, np.newaxis], blocksize, fillna) \
                .squeeze()
            return res

        def shuffle_blocks(x):
            """Shuffle `x` in blocks ensuring results has length `n_obs`"""
            if x.ndim < 2:
                return shuffle_blocks(x[:, np.newaxis]).squeeze()

            # define random location of split in two subsamples
            rand_split_loc = np.random.choice(np.arange(len(x)))

            # split
            x_split = np.vstack((x[rand_split_loc:], x[:rand_split_loc]))

            # construct a sequence of blocks
            shuffle_from = [
                x[p:(p + blocksize)] for p in range(len(x_split) - blocksize)
            ]

            # boot
            res = np.vstack([choice(shuffle_from) for _ in range(n_blocks)])

            # trim excess
            res = res[:n_obs]

            return res

        if fillna:
            data_filled_na = pd.DataFrame.from_dict(
                {c_name: shuffle_blocks(col.dropna().values)
                 for c_name, col in data.iteritems()},
                orient="columns")
            data_filled_na.index = data.index
            data_to_shuffle = data.fillna(data_filled_na)

        else:
            data_to_shuffle = data

        # bootstrap
        bootstrapped = shuffle_blocks(data_to_shuffle.values)

        # to frame again
        res = pd.DataFrame(data=bootstrapped, columns=data.columns,
                           index=data.index)

        return res

    def plot(self, xs_level="assets", plot_ci=False, annotate="none", ax=None):
        """Plot the mean and possibly confidence interval around it.

        Parameters
        ----------
        xs_level : str
            'assets' or 'events'
        plot_ci : bool
        annotate : str
            'left', 'right', 'both' or 'none'
        ax : matplotlib.Axes

        Returns
        -------
        fig : matplotlib.Figure
        ax : matplotlib.Axes

        """
        # defenestrate
        ta, tb, tc, td = self.window

        # plot elements
        car = self.get_car()
        cs_car = self.the_mean

        # figure
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize_single)
        else:
            fig = ax.figure

        # plot cumulative sums ----------------------------------------------
        to_plot = car.mean(axis=1, level=xs_level)

        # lower part
        to_plot.loc[:tb, :].plot(
            ax=ax, color=color_blue, linewidth=1.0, alpha=0.8,
            # marker='o', markerfacecolor='w', markersize=4
        )

        # upper part
        to_plot.loc[tc:, :].plot(
            ax=ax, color=color_blue, linewidth=1.0, alpha=0.8,
            # marker='o', markerfacecolor='w', markersize=4
        )

        # add points at initiation
        pd.Series(index=[tb] * to_plot.shape[1],
                  data=to_plot.loc[[tb], :].iloc[0].values) \
            .plot(ax=ax, linestyle="none", marker='o', markersize=4,
                  markerfacecolor=color_blue, markeredgecolor=color_blue)
        cs_car.loc[[tb]].iloc[[0]] \
            .plot(ax=ax, linestyle="none", marker='o', markersize=4,
                  markerfacecolor=color_red, markeredgecolor=color_red)

        if tc != ta:
            pd.Series(index=[tc] * to_plot.shape[1],
                      data=to_plot.loc[[tc], :].iloc[0].values) \
                .plot(ax=ax, linestyle="none", marker='o', markersize=4,
                      markerfacecolor=color_blue, markeredgecolor=color_blue)
            cs_car.loc[[tc]].iloc[[0]] \
                .plot(ax=ax, linestyle="none", marker='o', markersize=4,
                      markerfacecolor=color_red, markeredgecolor=color_red)

        # mean in black
        cs_car.loc[:tb].plot(ax=ax, color=color_red, linewidth=2)
        cs_car.loc[tc:].plot(ax=ax, color=color_red, linewidth=2)
        # pd.Series(index=[tb, tc], data=cs_car.loc[[tb, tc]].values) \
        #     .plot(ax=ax[0], linestyle="none",
        #           marker='o', markerfacecolor='k', markeredgecolor='k',
        #           markersize=3
        #           )
        ax.set_title(xs_level)

        # furbish plot ------------------------------------------------------
        ax.set_xlim(ta - 1.5, td + 1.5)

        # lost tickos
        ax.xaxis.set_ticks(self.event_index)
        # la linea de zero
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1.0)
        # la grida
        ax.grid(axis="both", alpha=0.33, linestyle=":")
        # la legenda
        ax.legend_.remove()

        if plot_ci:
            ax.fill_between(self.ci.index,
                            self.ci.iloc[:, 0].values,
                            self.ci.iloc[:, 1].values,
                            color=color_gray, alpha=0.5,
                            label="conf. interval")

        # ax.set_title("cumulative average", fontsize=10)

        # annotate?
        def annotater(labels, x_acnhor, position):
            cnt = 0
            x_margin = (0.1 + 0.6) * (-1 if position == "left" else 1)
            horizontalalignment = ("right" if position == "left" else "left")

            for c, p in labels.iteritems():
                this_x_pos = x_acnhor + \
                    0.1 * (-1 if position == "left" else 1) +\
                    x_margin * (cnt % 2)
                ax.annotate(
                    c, xy=(this_x_pos, p),
                    horizontalalignment=horizontalalignment,
                    verticalalignment='center',
                    fontsize=10)

                cnt += 1

        labels_l = to_plot.iloc[0, :].sort_values(ascending=False)
        labels_r = to_plot.iloc[-1, :].sort_values(ascending=False)

        if annotate == "left":
            # sort in descending order
            annotater(labels_l, ta, annotate)

        elif annotate == "right":
            annotater(labels_r, ta, annotate)

        elif annotate == "both":
            annotater(labels_l, ta, "left")
            annotater(labels_r, ta, "right")

        # legend
        lines = [
            mlines.Line2D([], [], linewidth=1.5, color=color_blue,
                          label="individual CAR"),
            mlines.Line2D([], [], linewidth=2, color=color_red,
                          label="cross-currency CAR"),
            mlines.Line2D([], [], linestyle="none", color="k", marker='o',
                          markersize=4, markerfacecolor="k",
                          markeredgecolor="k",
                          label="1-day AR")
        ]
        if plot_ci:
            ci_patch = mpatches.Patch(color=color_gray, label='conf. int.')
            lines += [ci_patch, ]

        labels = [line.get_label() for line in lines]
        ax.legend(lines, labels, loc="upper right")

        # los lables
        ax.set_xlabel("periods after event")

        self.legend_lines = ax.get_legend().legendHandles

        # ax.get_legend().legendHandles

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
            var_x_evt_x_ast = (var_x_evt * wght ** 2).sum(axis=1)

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
            res = (x.mean(axis=1, level="assets") * wght).sum(axis=1)
            return res

        if method == "equal":
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
        both = both.dropna(subset=["evt_wind"])

        # # event window
        # data_to_pivot_pre = timeline.loc[:, ["evt_wind_pre", "evt_no"]]
        # data_to_pivot_post = timeline.loc[:, ["evt_wind_post", "evt_no"]]

        # data_pivoted_pre = both.dropna(subset=["evt_wind_pre"]).pivot(
        #     index="evt_wind_pre",
        #     columns="evt_no",
        #     values=data.name)
        # data_pivoted_post = both.dropna(subset=["evt_wind_post"]).pivot(
        #     index="evt_wind_post",
        #     columns="evt_no",
        #     values=data.name)

        # # main data
        # data_pivoted = pd.concat((data_pivoted_pre, data_pivoted_post), axis=0)

        data_pivoted = both.pivot(index="evt_wind", columns="evt_no",
                                  values=data.name)

        return data_pivoted

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
        # ta, tb, tc, td = window
        ta, td = window[0], window[-1]
        if len(window) == 4:
            tb, tc = window[1:3]
            event_index = np.unique(
                np.concatenate((np.arange(ta, tb + 1),np.arange(tc, td + 1)))
            )
        else:
            tb, tc = (0, 0)
            event_index = np.arange(ta, td + 1)

        # pre- and post-event indices, by window ----------------------------
        # reindex events, create df with 1 where an event took place
        evt = events.reindex_like(data).mul(0.0)
        evt_c = evt.copy()

        window_neg_part = event_index[event_index < 0][::-1]
        window_pos_part = event_index[event_index >= 0]

        both_parts = pd.concat({
            "neg": pd.Series(window_neg_part, index=np.abs(window_neg_part)),
            "pos": pd.Series(window_pos_part, index=np.abs(window_pos_part))
        }, axis=1)

        for t, row in both_parts.iterrows():
            try:
                evt_c.fillna((evt + row["neg"]).bfill(limit=-int(row["neg"])),
                             inplace=True)
            except ValueError:
                pass
            try:
                evt_c.fillna((evt + row["pos"]).ffill(limit=int(row["pos"])),
                             inplace=True)
            except ValueError:
                pass

        idx_between_evt = evt_c.isnull() & evt.isnull()

        # exclude overlaps
        evt_c = evt_c.where((evt_c <= tb) | (evt_c >= tc))
        evt_c = evt_c.where(evt_c.isin(event_index))

        # event id (date), gets repeated over the whole window --------------
        # repeat series of dates as many times as there are assets
        dt_series = pd.Series(data=data.index, index=data.index)
        dt_df = pd.concat([dt_series.rename(c) for c in data.columns], axis=1)

        # leave only cells where there are events
        dt_df = dt_df.where(evt.notnull())

        # next event number is the number of event immediately following date
        next_evt_no = dt_df.fillna(method="bfill")

        # fill with dates
        for t, row in both_parts.iterrows():
            try:
                dt_df.bfill(limit=-int(row["neg"]), inplace=True)
            except ValueError:
                pass
            try:
                dt_df.ffill(limit=int(row["pos"]), inplace=True)
            except ValueError:
                pass

        # if ta < 0:
        #     dt_df.bfill(limit=-ta, inplace=True)
        # if td > 0:
        #     dt_df.ffill(limit=td, inplace=True)

        dt_df = dt_df.where(evt_c.notnull())

        # concat to a df with MultiIndex; swaplevel needed to have level [0]
        #  to keep asset names
        timeline = pd.concat({
            "evt_no": dt_df,
            # "evt_wind_pre": evt_pre,
            # "evt_wind_post": evt_post,
            "evt_wind": evt_c,
            "inter_evt": idx_between_evt,
            "next_evt_no": next_evt_no}, axis=1).swaplevel(axis=1)

        timeline.columns.names = ["data", "element"]

        return timeline

    @staticmethod
    def fill_in_turns(sparse_df, bfill_values, ffill_values):
        """Fill a sparse dataframe forwards and backwards in turns.

        Avoids nasty one-sided overlaps.

        Parameters
        ----------
        sparse_df : pandas.DataFrame
        bfill_values : list-like
            of negative values
        ffill_values : list-like
            of positive values

        Returns
        -------

        """
        res = sparse_df.copy()

        both_parts = pd.concat({
            "neg": pd.Series(np.sort(bfill_values)[::-1],
                             index=np.sort(np.abs(bfill_values))),
            "pos": pd.Series(np.sort(ffill_values),
                             index=np.abs(ffill_values))
        }, axis=1)

        for t, row in both_parts.iterrows():
            try:
                res.fillna((sparse_df + row["neg"]).bfill(limit=t),
                                 inplace=True)
            except ValueError:
                pass
            try:
                res.fillna((sparse_df + row["pos"]).ffill(limit=t),
                                 inplace=True)
            except ValueError:
                pass

        return res

    @staticmethod
    def align_for_event_study(data, events):
        """Rename, align and transform `data` and `events` to DataFrames.

        Returns two DataFrames of equal column names; broadcasts `events` if
        `events` is a Series but `data` is not.

        Parameters
        ----------
        data : pandas.Series or pandas.DataFrame
        events : pandas.Series or pandas.DataFrame

        Returns
        -------
        data : pandas.DataFrame
        events : pandas.DataFrame

        """
        if isinstance(data, pd.Series):
            # rename if needed
            n = getattr(data, "name", "data")

            # to a DataFrame
            data = data.to_frame(n)

            if isinstance(events, pd.Series):
                # events must be an equally-named frame
                events = events.to_frame(n)
            else:
                events.columns = [n, ]
        else:
            n = data.columns
            if isinstance(events, pd.Series):
                # the same events for all data columns
                events = pd.DataFrame.from_dict({k: events for k in n})
            else:
                if data.shape[1] > events.shape[1]:
                    warnings.warn("Some data columns do not have "
                                  "corresponding columns in `events` and "
                                  "will be removed.")
                events, data = events.align(data, axis=1, join="inner")

        return data, events

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

        return tuple([broadcast_fun(arg) for arg in args])

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
        bfilled = any_event.fillna(method=("bfill" if wa < 0 else "ffill"),
                                   limit=abs(wa))
        ffilled = any_event.fillna(method=("bfill" if wd < 0 else "ffill"),
                                   limit=abs(wd))
        all_windows = bfilled.fillna(ffilled)

        # <deprecated>
        # all_windows = any_event.bfill(limit=(-1*wa)).ffill(limit=wd)
        # </deprecated>

        # if need outside widows, invert
        if outside:
            res = all_windows.isnull()
        else:
            res = all_windows.notnull()

        return res

    def get_normal_data_exog(self, data, events, exog, window, add_constant):
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
        # data, events = self.align_for_event_study(data, events)
        # exog, = self.broadcast(exog, columns=data.columns)

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
            # this_x_fcast = exog[col].where(~outside_windows[col])
            this_x_fcast = exog[col]

            # loop over events, use expanding windows TODO: what about other?
            for t in this_evt.index:
                # it is ensured by outside_windows that .loc[:t, ] also
                # excludes the window around t
                mod = OLS(y=this_y, x=this_x_est.loc[:t],
                          add_constant=add_constant)

                # get x around t to make a forecast; +1 needed since day 0
                # is the event day
                x_fcast_right = this_x_fcast.loc[t:].head(wd + 1)
                # x_fcast_left = this_x_fcast.loc[:t].tail(-wa+1)
                x_fcast_left = this_x_fcast.loc[:t]

                # avoid duplicates
                # TODO: watch out for Series vs. DataFrame
                t_x_fcast = pd.concat((x_fcast_left, x_fcast_right.iloc[1:]),
                                      axis=0)

                # forecast, fill na in `norm_data`
                norm_data.fillna(
                    (this_y - mod.get_yhat(newdata=t_x_fcast)).to_frame(col),
                    inplace=True)

                # residual variance
                resid_var.loc[t, col] = mod.get_residuals().var()

        return norm_data, resid_var

    def get_normal_data_ewm(self, data, events, event_window, **kwargs):
        """

        Parameters
        ----------
        data : pandas.DataFrame
        events : pandas.DataFrame
        event_window : tuple
        kwargs : dict

        Returns
        -------
        normal_data : pandas.DataFrame

        """
        if "min_periods" not in kwargs.keys():
            kwargs["min_periods"] = 1

        msk = self.mark_timeline(data=data, events=events, window=event_window)
        msk = msk.xs("next_evt_no", axis=1, level=1).where(
            msk.xs("inter_evt", axis=1, level=1))

        # normal_data = data.where(msk).ewm(**kwargs).mean()\
        #     .where(~msk).replace(np.nan, 0.0)

        normal_data = data.where(msk.notnull()).ewm(**kwargs).mean().shift(1)

        # variances
        eps_s2 = dict()

        for c, c_col in data.iteritems():
            eps_s2[c] = c_col.where(msk[c].notnull()).groupby(msk[c]).var()

        eps_s2 = pd.concat(eps_s2, axis=1)

        return normal_data, eps_s2

    def get_normal_data_between_events(self, data, events, event_window):
        """

        Parameters
        ----------
        data
        events
        event_window

        Returns
        -------

        """
        # mark timeline to be able to do the cool stuff
        timeline = self.mark_timeline(data, events, event_window)

        res_means = dict()
        res_vars = dict()

        for c, c_col in data.iteritems():
            this_inter_evt = timeline[c].loc[:, "inter_evt"].rename(c)
            this_grouper = timeline[c].loc[:, "next_evt_no"].rename(c)
            this_m = c_col.where(this_inter_evt).groupby(this_grouper).mean()
            this_v = c_col.where(this_inter_evt).groupby(this_grouper).var()

            res_means[c] = this_m.ffill()
            res_vars[c] = this_v.ffill()

        filler = \
            timeline.loc[:, (slice(None), ["evt_wind_pre", "evt_wind_post"])]
        filler = filler.notnull().any(axis=1, level=0)

        # means
        res_means = pd.concat(res_means, axis=1)
        res_means = res_means.reindex(index=data.index)
        res_means = res_means.bfill(limit=max(-1 * event_window[0], 0)) \
            .ffill(limit=max(event_window[-1], 0))
        res_means = res_means.where(filler)

        # variances
        res_vars = pd.concat(res_vars, axis=1)
        res_vars = res_vars.reindex(index=data.index)
        res_vars = res_vars.bfill(limit=max(-1 * event_window[0], 0)) \
            .ffill(limit=max(event_window[-1], 0))
        res_vars = res_vars.where(filler).where(events.notnull())

        return res_means, res_vars


if __name__ == "__main__":
    pass
