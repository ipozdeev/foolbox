import pandas as pd
import numpy as np
import itertools as itools
from sklearn.metrics import confusion_matrix
from pandas.tseries.offsets import DateOffset, MonthBegin, MonthEnd, \
    relativedelta, BDay
import pickle

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D

from foolbox.utils import *
from foolbox.bankcalendars import *
from foolbox.portfolio_construction import multiple_timing
from foolbox.api import *
from foolbox.trading import EventTradingStrategy

# import ipdb

my_red = "#ce3300"
my_blue = "#2f649e"
my_gray = "#8c8c8c"

plt.rc("font", family="serif", size=12)


class PolicyExpectation():
    """
    """
    def __init__(self, meetings=None, policy_rate=None, proxy_rate=None,
                 expected_proxy_rate=None):
        """
        """
        self.meetings = meetings
        self.policy_rate = policy_rate
        self.proxy_rate = proxy_rate
        self.expected_proxy_rate = expected_proxy_rate

    @classmethod
    def from_forward_rates(cls, meetings, rate_object, rate_long,
        rate_on_const, **kwargs):
        """Construct PolicyExpectation from forward rates.

        Parameters
        ----------
        meetings : pandas.Series
            of events, without nans
        rate_object : FixedIncome instance
            e.g. LIBOR
        rate_long : pandas.Series
            e.g. 1-month libor rate
        rate_on_const : pandas.Series
            rate to prevail from value date to meeting date
        kwargs : dict
            arguments to PolicyExpectation.__init__ such as
            policy_rate and proxy_rate

        Returns
        -------
        pe : PolicyExpectation instance

        """
        # allocate space
        expected_proxy_rate = rate_long.copy() * np.nan

        # loop over time
        for t in expected_proxy_rate.index:
            # t = "2015-01-23"
            # set quote date to this t
            rate_object.quote_dt = t

            # break if out of range
            if rate_object.value_dt >= meetings.last_valid_index():
                break

            # meetings ------------------------------------------------------
            # next closest to the quote date
            very_nx_meet = meetings.index[
                meetings.index.get_loc(rate_object.quote_dt, method="bfill")]

            # next closest to the value date
            nx_meet = meetings.index[
                meetings.index.get_loc(rate_object.value_dt, method="bfill")]

            # sometimes there is a meeting between quote and value dates
            if very_nx_meet < nx_meet:
                # if quote_dt < meeting < end_dt < another meeting, perfect
                if rate_object.end_dt < nx_meet:
                    expected_proxy_rate.loc[t] = rate_long.loc[t]
                else:
                    continue
            else:
                # end_dt < next meeting, cannot tell anything
                if rate_object.end_dt < nx_meet:
                    continue

            # extract implied rate ------------------------------------------
            expected_proxy_rate.loc[t] = rate_object.get_forward_rate(
                rate=rate_long.loc[t],
                forward_dt=nx_meet,
                rate_until=rate_on_const.loc[t])

        # constructor
        pe = PolicyExpectation(
            meetings=meetings,
            expected_proxy_rate=expected_proxy_rate,
            **kwargs)

        return pe

    @classmethod
    def from_ois(cls, meetings, ois_object, ois_rate, rate_on_const, **kwargs):
        """Construct PolicyExpectation from ois rates.

        Parameters
        ----------
        meetings : pandas.Series
            of events, without nans
        ois_object : OIS instance
        ois_rate : pandas.Series
            e.g. 1-month ois rates
        rate_on_const : pandas.Series
            rate to prevail from value date to meeting date
        kwargs : dict
            arguments to PolicyExpectation.__init__ such as
            policy_rate and proxy_rate

        Returns
        -------
        pe : PolicyExpectation instance

        """
        # allocate space
        expected_proxy_rate = ois_rate.copy() * np.nan

        # loop over time
        for t in expected_proxy_rate.index:
            # if t > pd.Timestamp("2009-02-15"):
            #     print("ololo")
            # else:
            #     continue

            # set quote date to this t
            ois_object.quote_dt = t

            # break if out of range
            if ois_object.value_dt >= meetings.last_valid_index():
                break

            # meetings ------------------------------------------------------
            # next closest to the quote date
            very_nx_meet = meetings.index[
                meetings.index.get_loc(ois_object.quote_dt, method="bfill")]

            # next closest to the value date
            nx_meet = meetings.index[
                meetings.index.get_loc(ois_object.value_dt, method="bfill")]

            # # sometimes there is a meeting between quote and value dates
            if very_nx_meet < nx_meet:
                if ois_object.end_dt < nx_meet:
                    expected_proxy_rate.loc[t] = ois_rate.loc[t]
                else:
                    continue
            else:
                # end_dt < next meeting, cannot tell anything
                if ois_object.end_dt < nx_meet:
                    continue

            # extract implied rate
            expected_proxy_rate.loc[t] = ois_object.get_implied_rate(
                event_dt=nx_meet,
                swap_rate=ois_rate.loc[t],
                rate_until=rate_on_const.loc[t])

        pe = PolicyExpectation(
            meetings=meetings,
            expected_proxy_rate=expected_proxy_rate,
            **kwargs)

        return pe

    @classmethod
    def from_funds_futures(cls, meetings, funds_futures):
        """ Extract policy expectations from Fed funds futures.

        For each date when a Fed funds futures is available, it is possible to
        extract the expectation of the rate to be set at the next policy
        meeting. Assumptions: the rate stays constant at the level of the
        previously settled futures until the next meeting when it changes;
        expectation hypothesis holds.

        Both rate levels and changes are needed in `meetings` since some
        events might be skipped as non-scheduled: in this case, if there was a
        rate change in an unscheduled event, the actual change on the closest
        scheduled one will be biased.

        Only monthly futures are supported.

        Parameters
        ----------
        meetings : pd.DataFrame
            indexed with meeting dates, containing respective policy rates
            (column "rate_level") and rate changes (column "rate_change"),
            in percentage points

        funds_futures : pd.DataFrame
            with dates on the index and expiry dates on the columns axis,
            priced as usually: 100-(rate in percentage points)

        Returns
        -------
        pe : PolicyExpectation instance
        """
        # fill na forward (today's na equals yesterday's value)
        funds_futures = funds_futures.fillna(method="ffill", limit=30)

        # space for stuff
        expected_proxy_rate = pd.Series(index=funds_futures.index)

        for t in expected_proxy_rate.index:
            # t = rate_exp.index[0]
            # next meeting
            try:
                nx_fomc = meetings.index[meetings.index.get_loc(
                    t+relativedelta(minutes=1), method="bfill")]
            except:
                continue

            # meeting after that: will crash for the last fomc date
            try:
                nx_nx_fomc = \
                    meetings.index[meetings.index.get_loc(nx_fomc)+1]
            except:
                continue

            # first day of next fomc month
            nx_fomc_mstart = MonthBegin().rollback(nx_fomc)
            # last day of next fomc month
            nx_fomc_mend = MonthEnd().rollforward(nx_fomc)
            # days in fomc month
            days_in_m = nx_fomc.daysinmonth

            # next fomc month
            if nx_nx_fomc.year+nx_nx_fomc.month/100 > \
                nx_fomc.year+(nx_fomc.month+1)/100:
                # Scenario 1
                # implied rate = 100-ff_fut()
                nx_nx_fomc_mend = MonthEnd().rollforward(nx_nx_fomc)

                # rate decision
                expected_proxy_rate.loc[t] = \
                    100-funds_futures.loc[t, nx_nx_fomc_mend]
            else:
                # Scenario 2
                # previos month
                nx_fomc_mbefore = nx_fomc_mstart - relativedelta(days=1)
                M = nx_fomc.day

                # implied rate
                impl_rate = 100 - funds_futures.loc[t, nx_fomc_mend]
                # rate at start of fomc month
                start_rate = 100 - funds_futures.loc[t, nx_fomc_mbefore]

                # rate decision
                expected_proxy_rate.loc[t] = days_in_m/(days_in_m-M)*(
                    impl_rate - (M/days_in_m)*start_rate)

        # create a new PolicyExpectation instance to return
        pe = PolicyExpectation(
            meetings=meetings,
            expected_proxy_rate=expected_proxy_rate)

        return pe

    @classmethod
    def from_pickles(cls, data_path, currency, start_dt="1990",
                     proxy_rate_pickle=None, e_proxy_rate_pickle=None,
                     meetings_pickle=None, ffill=False):
        """Construct PolicyExpectation from existing pickled rates.

        Parameters
        ----------
        data_path : str
        currency : str
            e.g. 'usd'
        start_dt : str
        proxy_rate_pickle : str
        e_proxy_rate_pickle : str
        meetings_pickle : str
        ffill : boolean

        Returns
        -------
        PolicyExpectation

        """
        if proxy_rate_pickle is None:
            proxy_rate_pickle = "overnight_rates.p"
        if e_proxy_rate_pickle is None:
            e_proxy_rate_pickle = "implied_rates_from_1m_ois.p"
        if meetings_pickle is None:
            meetings_pickle = "events.p"

        # meetings
        meetings_data = pd.read_pickle(data_path + meetings_pickle)
        meetings = meetings_data["joint_cbs"].loc[start_dt:, currency]
        policy_rate = meetings_data["joint_cbs_lvl"].loc[start_dt:, currency]

        # drop nans from policy rates
        meetings = meetings.dropna()
        policy_rate = policy_rate.dropna()

        # proxy rates
        proxy_rate_data = pd.read_pickle(data_path + proxy_rate_pickle)
        # proxy_rate = proxy_rate_data[currency].loc[start_dt:, "ON"].rename(
        #     currency)
        proxy_rate = proxy_rate_data.loc[start_dt:, currency]

        # implied rates
        e_proxy_rate_data = pd.read_pickle(data_path + e_proxy_rate_pickle)
        e_proxy_rate = e_proxy_rate_data.loc[start_dt:, currency]

        if ffill:
            e_proxy_rate = e_proxy_rate.reindex(index=pd.date_range(
                e_proxy_rate.index[0], e_proxy_rate.index[-1], freq='B'),
                method="ffill")

        # init class, manually insert policy expectation
        pe = cls(meetings=meetings,
                 proxy_rate=proxy_rate,
                 policy_rate=policy_rate,
                 expected_proxy_rate=e_proxy_rate)

        return pe

    # @classmethod
    # def from_ois(cls, meetings, instrument, tau, reference_rate=None):
    #     """ Extract policy expectations from money market instruments.
    #
    #     For each date when an instrument observation is available, it is
    #     possible to extract the expectation of the rate to be set at the next
    #     policy meeting. Assumptions: the regulator targets the rate which
    #     `instrument` is derivative of; the rate stays constant at the
    #     previosuly set policy level (`reference_rate` is None) or at the most
    #     recent level of `reference_rate` (`reference_rate` is not None) until the next
    #     meeting; at the expiry of the `instrument` the buyer receives the
    #     accumulated short rate; expectation hypothesis holds.
    #
    #     Parameters
    #     ----------
    #     meetings : pd.DataFrame
    #         indexed with meeting dates, containing respective policy rates
    #         (column "rate_level") and rate changes (column "rate_change"),
    #         in percentage points
    #     instrument : pd.Series
    #         of instrument (e.g. OIS) values, in frac of 1
    #     tau : int
    #         maturity of `instrument`, in months
    #     reference_rate : pd.Series, optional
    #         of the rate which the `instrument` is derivative of (if differs
    #         from the rate in `meetings`), in frac of 1
    #
    #     Returns
    #     -------
    #     pe : PolicyExpectation instance
    #         with attribute `rate_expectation` for expectation of the next set rate
    #
    #     """
    #     assert isinstance(meetings, pd.DataFrame)
    #     assert isinstance(instrument, pd.Series)
    #
    #     if instrument.name is None:
    #         raise AttributeError("instrument must be called with ISO of "+
    #             "respective currency")
    #
    #     # day count convention
    #     day_count = {
    #         "gbp": 365,
    #         "cad": 365,
    #         "usd": 360,
    #         "eur": 360,
    #         "chf": 360,
    #         "jpy": 365,
    #         "nzd": 365,
    #         "aud": 365,
    #         "sek": 360}
    #
    #     # days in year for this currency:
    #     days_in_year = day_count[instrument.name]
    #
    #     # # do not let instrument be too full of nans
    #     # min_start = max((meetings.index[0], instrument.index[0]))
    #     # instrument = instrument.loc[min_start:]
    #     # meetings = meetings.loc[min_start:]
    #
    #     # if no rate provided, meetings must contain a rate; in this case
    #     #   reference_rate be the rate set at meetings projected onto the
    #     #   dates of
    #     #   instrument.
    #     if reference_rate is None:
    #         reference_rate = meetings.loc[:,"rate_level"].reindex(
    #             index=instrument.index, method="ffill")
    #     else:
    #         # project onto the dates too
    #         reference_rate = reference_rate.reindex(
    #             index=instrument.index, method="ffill")
    #
    #     # make sure meeting dates have corresponding dates in instrument
    #     reference_rate = reference_rate.reindex(
    #         index=reference_rate.index.union(meetings.index),
    #         method="ffill")
    #
    #     # from percentage points to frac of 1
    #     meetings = meetings.copy()/100
    #     reference_rate = reference_rate.copy()/100
    #     instrument = instrument.copy()/100
    #
    #     # main loop ---------------------------------------------------------
    #     # allocate space for forward rates
    #     rate_expectation = instrument.copy()*np.nan
    #
    #     # loop over dates in instrument
    #     for t in rate_expectation.index:
    #         # t = rate_expectation.index[3000]
    #         # t = pd.to_datetime("2016-02-08")
    #
    #         # break if out of range
    #         if t > meetings.last_valid_index():
    #             break
    #
    #         # continue if too early
    #         if t < meetings.first_valid_index():
    #             continue
    #
    #         # find two meeting dates: the previous, whose rate will serve as
    #         #   reference rate, and the next, which will possibly set a new
    #         #   rate
    #         # previous
    #         prev_meet = \
    #             meetings.index[meetings.index.get_loc(t, method="ffill")]
    #
    #         # next closest
    #         nx_meet = \
    #             meetings.index[meetings.index.get_loc(t, method="bfill")]
    #
    #         # maturity date of instrument: contract starts tomorrow; final
    #         #   date is then today tau month forward
    #         setl_date = t + DateOffset(months=tau)
    #
    #         # if next meeting is earlier than maturity, skip
    #         if setl_date <= nx_meet:
    #             continue
    #
    #         # number of days between them
    #         ndays = (setl_date - t).days
    #
    #         # number of days until next meeting
    #         ndays_until = (nx_meet - t).days
    #
    #         # previously set rate, to be effective until next meeting
    #         # TODO: not prev_meet, but prev_meet + 1
    #         idx = reference_rate.index[reference_rate.index.get_loc(
    #             prev_meet + DateOffset(days=1), "bfill")]
    #         bench_reixed = reference_rate.loc[idx:t].reindex(
    #             index=pd.date_range(idx,t,freq='D'),
    #             method="ffill")
    #         prev_rate = (
    #             (1+bench_reixed/days_in_year).product()**\
    #             (1/bench_reixed.count())-1)*days_in_year
    #
    #         # implied rate
    #         r_instr = instrument.loc[t]
    #
    #         impl_rate = (
    #             (
    #                 (1+r_instr/days_in_year*ndays)/
    #                 (1+prev_rate/days_in_year)**(ndays_until)
    #             )**(1/(ndays-ndays_until))-1)*days_in_year
    #
    #         # store
    #         rate_expectation.loc[t] = impl_rate
    #
    #     # create a new PolicyExpectation instance to return
    #     pe = PolicyExpectation(
    #         meetings=meetings*100,
    #         instrument=instrument*100,
    #         reference_rate=reference_rate*100)
    #
    #     # supply it with policy expectation
    #     pe.rate_expectation = rate_expectation*100
    #
    #     return pe

    def ts_plot(self, lag=None, ax=None):
        """ Plot implied rate before vs. realized after meetings.

        Implied rate at time -`lag` before each event
        """
        if not hasattr(self, "rate_expectation"):
            raise ValueError("Estimate first!")

        # defaults
        if lag is None:
            lag = 1
        if ax is None:
            f, ax = plt.subplots(figsize=(8.3,8.3/2))
        else:
            f = plt.gcf()

        # smooth + align ----------------------------------------------------
        actual_rate = \
            self.reference_rate.rolling(lag, min_periods=1).mean()\
                .shift(-lag).reindex(
                    index=self.meetings.index, method="bfill")

        # policy expectation to plot
        rate_expectation = self.rate_expectation.shift(lag).reindex(
            index=actual_rate.index, method="ffill")

        # rename a bit
        rate_expectation.name = "rate_expectation"
        actual_rate.name = "actual_rate"

        # plot --------------------------------------------------------------
        # plot expectation
        (rate_expectation).plot(
            ax=ax,
            linestyle='none',
            marker='o',
            color=my_blue,
            mec="none",
            label="implied rate")

        # plot the actually set rate
        (actual_rate).plot(
            ax=ax,
            marker='.',
            color=my_red,
            label="actual rate")

        # set artist properties
        ax.set_xlim(
            max([actual_rate.first_valid_index(),
                self.rate_expectation.first_valid_index()])-\
            DateOffset(months=6), ax.get_xlim()[1])
        ax.legend(fontsize=12)

        return f, ax

    def error_plot(self, avg_over=None, ax=None):
        """ Plot predicted vs. realized and the error plot.
        """
        if not hasattr(self, "rate_expectation"):
            raise ValueError("Estimate first!")

        # defaults
        if avg_over is None:
            avg_over = 1
        if ax is None:
            f, ax = plt.subplots(figsize=(8.3,8.3/2))
        else:
            f = plt.gcf()

        # smooth + align ----------------------------------------------------
        actual_rate = \
            self.reference_rate.rolling(avg_over, min_periods=1).mean()\
                .shift(-avg_over).reindex(
                    index=self.meetings.index, method="bfill")

        # policy expectation to plot
        rate_expectation = self.rate_expectation.shift(avg_over).reindex(
            index=actual_rate.index, method="ffill")

        # rename a bit
        rate_expectation.name = "rate_expectation"
        actual_rate.name = "actual_rate"
        # ax.set_xlim(
        #     max([meetings_c.first_valid_index(),
        #         self.rate_expectation.first_valid_index()])-\
        #     DateOffset(months=6), ax.get_xlim()[1])
        # ax.legend(fontsize=12)

        # plot --------------------------------------------------------------
        # predictive power
        both = pd.concat((rate_expectation, actual_rate), axis=1).dropna()
        both.plot.scatter(
            ax=ax,
            x="rate_expectation",
            y="actual_rate",
            alpha=0.66,
            s=33,
            color=my_blue,
            edgecolor='none')

        # limits to produce a nice square picture
        max_x = both.max().max()+1.0
        min_x = both.min().min()-1.0

        lim_x = np.array([min_x, max_x])

        ax.plot(lim_x, lim_x, color='r', linestyle='--')

        # reset limits
        ax.set_xlim(lim_x)
        ax.set_ylim(lim_x)

        # text: absolute error
        abs_err = np.abs(rate_expectation-actual_rate).mean()
        ax.annotate(r"$|err|={:3.2f}$".format(abs_err*100),
            xy=(0.5, 0.05), xycoords='axes fraction')

        return f, ax

    @staticmethod
    def classify_direction(dr, h_low=None, h_high=None):
        """
        """
        if (h_low is None) and (h_high is None):
            raise ValueError("One of h_low and h_high must be set.")

        if h_low is None:
            h_low = h_high*-1

        if h_high is None:
            h_high = h_low*-1

        # allocate space
        res = pd.Series(index=dr.index, dtype=float)

        # classify!
        res.ix[dr < h_low] = -1
        res.ix[dr > h_high] = 1
        res.ix[(dr <= h_high) & (dr >= h_low)] = 0

        return res

    def forecast_policy_direction(self, lag, h_low=None, h_high=None,
        map_proxy_rate=None, map_expected_rate=None):
        """

        Parameters
        ----------
        lag : int
        h_low : float
        h_high : float
        map_proxy_rate : callable
            a good idea is to at least lag the rate, since e.g. in the US it is
            published with a 1-day lag
        map_expected_rate : callable

        Returns
        -------

        """
        # default maps are mickey mouse
        if map_proxy_rate is None:
            map_proxy_rate = lambda x: x
        if map_expected_rate is None:
            map_expected_rate = lambda x: x

        # make sure all meeting dates are there, even if rates are not
        e_proxy_rate, _ = self.expected_proxy_rate.align(
            self.meetings, join="outer")

        # align two rates
        proxy_rate, e_proxy_rate = self.proxy_rate.align(
            e_proxy_rate, join="outer")

        # apply transforms
        proxy_rate = map_proxy_rate(proxy_rate)
        e_proxy_rate = map_expected_rate(e_proxy_rate)

        # difference, reindex with meeting dates
        dr = (e_proxy_rate - proxy_rate).shift(lag).loc[self.meetings.index]

        # call to direction classifier
        res = self.classify_direction(dr, h_low=h_low, h_high=h_high)

        return res

    # def forecast_policy_change(self,
    #     lag=1,
    #     threshold=0.125,
    #     avg_impl_over=1,
    #     avg_refrce_over=None,
    #     bday_reindex=False):
    #     """ Predict +1, -1 or 0 for every meeting date.
    #
    #     Some periods before each event (`lag`) we take the implied rate
    #     smoothed over the previous `avg_impl_over` periods and compare it to
    #     the reference rate smoothed over the previous `avg_refrce_over`
    #     periods. Should the difference between the two exceed the `threshold`,
    #     we predict the next set rate be higher and so on.
    #
    #     Parameters
    #     ----------
    #     lag : int
    #         such that the forecast is made at t-`lag` with t being the meeting
    #         date
    #     threshold : float
    #         the maximum absolute difference (in %) between the implied rate
    #         and the reference rate such that no policy change is predicted;
    #         values above that would signal a hike, below - cut
    #     avg_impl_over : int
    #         such that the implied rate that is compared to the reference rate
    #         is first smoothed over this number of periods
    #     avg_refrce_over : int
    #         the number of periods to average the reference rate over before
    #         comparing it to the implied rate
    #     bday_reindex : boolean
    #         True if the rates are reindexed with business days ('B' in pandas)
    #
    #     Returns
    #     -------
    #     policy_fcast : pd.Series
    #         of forecasts: -1 for cut, +1 for hike, 0 for no change; indexed
    #         with the index of self.meetings
    #
    #     Example
    #     -------
    #     data_path = set_credentials.gdrive_path("research_data/fx_and_events/")
    #     pe = PolicyExpectation.from_pickles(data_path, "gbp")
    #     fcast = pe.forecast_policy_change(10, 0.1250, 2)
    #
    #     """
    #     # take implied rate `lag` periods before
    #     impl_rate = self.rate_expectation.copy()
    #     refrce_rate = self.reference_rate.copy()
    #
    #     # reindex if asked to
    #     if bday_reindex:
    #         first_dt = min((impl_rate.index[0], refrce_rate.index[0]))
    #         last_dt = max((impl_rate.index[-1], refrce_rate.index[-1]))
    #         bday_dt = pd.date_range(first_dt, last_dt, freq='B')
    #
    #         impl_rate = impl_rate.reindex(index=bday_dt, method="ffill")
    #         refrce_rate = refrce_rate.reindex(index=bday_dt, method="ffill")
    #
    #     # will need to compare it to either some rolling average of the
    #     #   reference rate or the previously set policy rate (avg_refrce_over
    #     #   is None)
    #     # 1) asked to smooth over several periods
    #     if isinstance(avg_refrce_over, int):
    #         # fill some NAs
    #         avg_refrce = refrce_rate.fillna(method="ffill", limit=2)
    #         # smooth
    #         avg_refrce = avg_refrce.rolling(
    #             avg_refrce_over, min_periods=1).mean()
    #         # shift to alighn with the event dates
    #         avg_refrce = avg_refrce.shift(lag)\
    #             .reindex(index=self.meetings.index, method="ffill")
    #
    #     elif isinstance(avg_refrce_over, str):
    #         # collect everything between events
    #         refrce_aligned, meets_aligned = refrce_rate.align(
    #             self.meetings, join="outer", axis=0)
    #         refrce_aligned = refrce_aligned.to_frame()
    #         refrce_aligned.loc[:,"dates"] = meets_aligned.index
    #         refrce_aligned.loc[:,"dates"].fillna(method="bfill")
    #
    #         for dt in range(1,len(self.meetings.index)):
    #             this_dt = self.meetings.index[dt]
    #             prev_dt = self.meetings.index[dt-1] + DateOffset(days=2)
    #             this_piece = refrce_aligned.ix[
    #                 refrce_aligned.loc[:,"dates"]==this_dt,0]
    #             refrce_aligned.ix[
    #                 refrce_aligned.loc[:,"dates"]==this_dt,0].loc[prev_dt:] = \
    #                     this_piece.loc[prev_dt:].expanding().mean()
    #
    #             # shift to alighn with the event dates
    #             avg_refrce = refrce_aligned.shift(lag).\
    #                 reindex(index=self.meetings.index, method="ffill")
    #
    #     else:
    #         # else, if None, take the last meetings decision
    #         avg_refrce = self.meetings.loc[:,"rate_level"].shift(1)
    #         # rename!
    #         avg_refrce.name = refrce_rate.name
    #
    #     # smooth implied rate
    #     avg_impl = impl_rate.rolling(avg_impl_over, min_periods=1).mean()\
    #         .shift(lag).reindex(index=self.meetings.index, method="ffill")
    #
    #     # difference between rate implied some periods earlier and the
    #     #   reference_rate rate
    #     impl_less_bench = (avg_impl-avg_refrce).dropna()
    #
    #     # forecast is the sign of the difference if it is large enough
    #     #   (as measured by `threshold`)
    #     policy_fcast = np.sign(impl_less_bench).where(
    #         abs(impl_less_bench) > threshold).fillna(0.0)
    #
    #     # add name
    #     policy_fcast.name = impl_rate.name
    #
    #     return policy_fcast

    def assess_forecast_quality(self, policy_dir_fcast):
        """
        Parameters
        ----------
        policy_dir_fcast : pd.Series
            e.g. from self.`forecast_policy_direction`

        """
        policy_dir_fcast.name = "fcast"

        # policy change: should already be expressed as difference
        policy_dir_actual = np.sign(self.meetings)
        policy_dir_actual.name = "actual"

        # concat to be able to drop NAs
        both = pd.concat((policy_dir_actual, policy_dir_fcast), axis=1)
        both = both.dropna(how="any")

        # percentage of correct guesses
        # rho = (both.loc[:,"fcast"] == both.loc[:,"actual"]).mean()

        # confusion matrix
        cmx = confusion_matrix(both.loc[:, "fcast"], both.loc[:, "actual"])

        # all possible outcomes
        idx = sorted(list(set(
            list(pd.unique(both.loc[:, "fcast"])) +\
            list(pd.unique(both.loc[:, "actual"])))))

        # confusion matrix to dataframe
        cmx = pd.DataFrame(cmx, index=idx, columns=idx)

        # # TODO: why?
        # cmx = cmx.reindex(index=[-1, 0, 1], columns=[-1, 0, 1]).fillna(0)
        cmx = cmx.astype(np.int8)

        return cmx

    @staticmethod
    def calculate_vus(measurements, classes, order=None):
        """

        Parameters
        ----------
        measurements : pd.Series
        classes : pd.Series
        order : list

        Returns
        -------

        """
        # measurements = (ir_us - r_until).shift(5).loc[evt_us.index]
        # measurements = pd.Series(np.random.random((186,)),index=evt_us.index)
        # classes = np.sign(evt_us)
        grp = pd.concat((measurements, classes), axis=1).dropna()
        grp.columns = ["m", "c"]

        d = {c: m.loc[:, "m"].values for c, m in grp.groupby("c")}

        if order is None:
            order = sorted(d.keys())

        p = np.array(list(itools.product(*[d[k] for k in order])))

        f = lambda x: (x[1] > x[0]) & (x[2] > x[1])

        res = np.apply_along_axis(func1d=f, axis=1, arr=p).mean()

        return res

    def get_vus(self, lag, map_proxy_rate=None, map_expected_rate=None):
        """

        Parameters
        ----------
        lag : int
        map_proxy_rate : callable
        map_expected_rate : callable

        Returns
        -------

        """
        if map_proxy_rate is None:
            map_proxy_rate = lambda x: x
        if map_expected_rate is None:
            map_expected_rate = lambda x: x

        # define classes
        classes = np.sign(self.meetings.dropna())

        # ensure all meeting dates are in the expected rate, even if nan
        e_proxy_rate, _ = self.expected_proxy_rate.align(classes, join="outer")

        # align two rates
        proxy_rate, e_proxy_rate = self.proxy_rate.align(
            e_proxy_rate, join="outer")

        proxy_rate = map_proxy_rate(proxy_rate)
        e_proxy_rate = map_expected_rate(e_proxy_rate)

        # difference
        dr = (e_proxy_rate - proxy_rate).shift(lag).loc[classes.index]
        print("{:d} non-na markers".format(len(dr.dropna())))

        # watch out for nans!
        if dr.isnull().any():
            nans_df = dr.isnull().where(dr.isnull()).dropna()
            nans_dates = [str(p.date()) for p in nans_df.index]
            n_nans = len(nans_df)
            warnings.warn(("there are nans on some meeting dates: "+\
                          "{}, "*(n_nans-1) + "{}.").format(*nans_dates))
            print(classes.loc[nans_df.index])

        vus = self.calculate_vus(dr, classes, order=[-1, 0, 1])

        return vus

    def plot_roc_surface(self, lag, ref_rate):
        """
        Parameters
        ----------
        kwargs : dict
            keyword args to self.forecast_policy_direction
        """
        # different thresholds
        thresholds = np.linspace(-1.50, 1.50, 201)

        mix = pd.MultiIndex.from_product((thresholds, thresholds))
        tprs = pd.DataFrame(index=mix, columns=[-1, 0, 1], dtype=float)
        cmxs = dict()

        # loop over thresholds; for each, calculate true pos probabilities for
        #   each outcome

        for h_low in thresholds:
            for h_high in thresholds:

                if h_high <= h_low:
                    continue

                this_m_idx = (h_low, h_high)

                # p = 0.085
                fc = self.forecast_policy_direction(lag=lag, ref_rate=ref_rate,
                    h_low=h_low, h_high=h_high)
                cmx = self.assess_forecast_quality(fc)

                # # save cmx
                # cmxs[th] = cmx

                tprs.loc[this_m_idx, :] = np.diag(cmx / cmx.sum())

        # sort values to plot a nice surface
        tprs = tprs.sort_values(
            by=[-1, 1], ascending=[True, True])

        # # cmxs to Panel
        # cmxs = pd.Panel.from_dict(cmxs, orient="minor")

        # # plot
        # fig = plt.figure()
        # ax = Axes3D(fig)
        #
        # ax.plot_trisurf(
        #     tprs.loc[:, -1],
        #     tprs.loc[:, 1],
        #     tprs.loc[:, 0], linewidth=0.2, cmap=cm.coolwarm)
        #
        # ax.set_xlim((0.0, 1.0))
        # ax.set_ylim((0.0, 1.0))
        # ax.set_zlim((0.0, 1.0))
        #
        # ax.set_xlabel('-1')
        # ax.set_ylabel('1')
        # ax.set_zlabel('0')

        return tprs, cmxs

    def roc_curve(self, lag=1, out_path=None, **kwargs):
        """ Construct ROC curve.

        Parameters
        ----------
        kwargs : dict
            args to `.forecast_policy_change`
        """
        thresholds = np.linspace(-0.50, 0.50, 101)

        # plot
        fig, ax = plt.subplots(figsize=(8.4,11.7/2))

        # allocate space
        fcast_accy = pd.Panel(
            major_axis=thresholds,
            minor_axis=["true_pos", "false_pos"],
            items=["hike", "cut"])

        # loop over thresholds
        for p in thresholds:
            # p = 0.085
            fc = self.forecast_policy_direction(lag=lag,
                ref_rate=self.reference_rate, h_low=-p, h_high=p)
            cmx = self.assess_forecast_quality(fc)
            # ipdb.set_trace()

            fcast_accy.loc["hike",p,"true_pos"] = \
                cmx.loc[1,1]/cmx.loc[:,1].sum()
            fcast_accy.loc["hike",p,"false_pos"] = \
                cmx.loc[1,-1:0].sum()/cmx.loc[:,-1:0].sum().sum()
            fcast_accy.loc["cut",p,"true_pos"] = \
                cmx.loc[-1,-1]/cmx.loc[:,-1].sum()
            fcast_accy.loc["cut",p,"false_pos"] = \
                cmx.loc[-1,0:1].sum()/cmx.loc[:,0:1].sum().sum()

            # add back extreme values
            fcast_accy.loc["hike",1,:] = [1.0, 1]
            fcast_accy.loc["hike",-1,:] = [0.0, 0]
            fcast_accy.loc["cut",1,:] = [0.0, 0]
            fcast_accy.loc["cut",-1,:] = [1.0, 1]

        for h in range(2):
            # h = 0
            this_ax = plt.subplot(121+h)
            # ipdb.set_trace()
            self.plot_roc(fcast_accy.iloc[h,:,:], ax=this_ax,
                linewidth=1.5)
            this_ax.set_title(fcast_accy.items[h]+'s')


        this_ax.set_ylabel('', visible=False)
        # this_ax.legend(loc="lower right", prop={'size':12},
        #     bbox_to_anchor=((1+0.01)/1.1, (0.05+0.01)/1.1))
        # this_ax.legend_.remove()

        fig.suptitle("roc curves", fontsize=12)

        if out_path is not None:
            fig.savefig(out_path+"roc_lag_"+str(lag)+\
                ".png", dpi=300, bbox_inches="tight")

        return ax

    @staticmethod
    def plot_roc(data, ax=None, **kwargs):
        """
        Parameters
        ----------
        kwargs : dict
            args to `matplotlib.pyplot.plot`
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8.4,8.4))

        data_sorted = data.sort_values(["false_pos","true_pos"])
        data_sorted.plot(
            ax=ax,
            x="false_pos",
            y="true_pos",
            drawstyle="steps",
            alpha=1.0,
            # marker='o',
            # markersize=3,
            # color='k',
            # markerfacecolor='k',
            **kwargs)

        ax.set_xlim((-0.05, 1.05))
        ax.set_ylim((-0.05, 1.05))
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.grid(which="major", alpha=0.85, linestyle="--")
        ax.grid(which="minor", alpha=0.33, linestyle=":")
        ax.set_xlabel("false positive")
        ax.set_ylabel("true positive")
        ax.legend_.remove()

        # area under the curve
        auc = np.trapz(
            data_sorted["true_pos"].values,
            data_sorted["false_pos"].values)

        ax.annotate(r"$AUC={:5.4f}$".format(auc),
            xy=(0.5, 0.1), xycoords='axes fraction')

        return


def into_currency(data, new_cur, counter_cur="usd"):
    """
    Parameters
    ----------
    data :
        expressed in USD per unit
    """
    if new_cur == counter_cur:
        return data

    # re-express everything in units of new_cur
    new_data = data.drop([new_cur], axis=1).subtract(
        data[new_cur], axis="index")

    # reinstall
    new_data["usd"] = -1*data[new_cur]

    return new_data

def pe_backtest(returns, holding_range, threshold_range,
                data_path, avg_impl_over=2, avg_refrce_over=2, sum=False):
    """

    Parameters
    ----------
    returns: pd.DataFrame
        of returns to assets, for which implied rates are available via the
        'PolicyExpectation' class data api
    holding_range: np.arange
        specifying the range of holding periods
    threshold_range: np.arange
        specifying threshold levels in basis points
    data_path: str
        to the data for the 'PolicyExpectation.from_pickles()'
    avg_impl_over : int
        such that the implied rate that is compared to the reference rate
        is first smoothed over this number of periods
    avg_refrce_over : int
        the number of periods to average the reference rate over before
        comparing it to the implied rate
    sum: bool
        controlling whether to sum or average aggregated returns over the cross
        section. Default is False, i.e. mean is taken

    Returns
    -------
    results: dict
        with key 'aggr' containing a MultiIndex dataframe with the first level
        corresponding to holding period and second level corresponding to the
        threshold levels, and columns containing average return on the expected
        policy rate strategy across assets in returns. The second key 'disaggr'
        is a dict of dicts with first level corresponding to holding strategy
        and second level corresponding to the threshold levels with dataframes
        of returns to individual currencies as items. For example

        results = {
            "aggr": MultiIndex of average returns,
            "disaggr": {
                "10":
                   {"5": DataFrame,
                    "10": DataFrame}
                "11":
                    {"5": DataFrame,
                     "10": DataFrame}
                    },
            }

    """
    # Get he pandas slicer for convenient MultiIndex reference
    ix = pd.IndexSlice

    # Set up the output structure
    results = dict()
    results["disaggr"] = dict()

    # The aggregated outpu is a multiindex
    combos = list(itools.product(holding_range, threshold_range))
    cols = pd.MultiIndex.from_tuples(combos, names=["holding", "threshold"])
    aggr = pd.DataFrame(index=returns.index, columns=cols)

    # Start backtesting looping over holding periods and thresholds
    for holding_period in holding_range:

        # Transform holding period into lag_expect argument for the
        # 'forecast_policy_change()' method of the 'PolicyExpectation' class
        lag_expect = holding_period + 2  # forecast rate before trading FX

        # Create an entry for the disaggregated output
        results["disaggr"][str(holding_period)] = dict()

        # A soup of several loops
        for threshold in threshold_range:

            # For the predominant case of multiple currencies pool the signals
            pooled_signals = list()

            # Get the policy forecast for each currency
            for curr in returns.columns:
                tmp_pe = PolicyExpectation.from_pickles(data_path, curr)
                tmp_fcast =\
                    tmp_pe.forecast_policy_change(lag=lag_expect,
                                                  threshold=threshold/100,
                                                  avg_impl_over=avg_impl_over,
                                                  avg_refrce_over=avg_refrce_over,
                                                  bday_reindex=True)
                # Append the signals
                pooled_signals.append(tmp_fcast)

            # Aggregate the signals, construct strategies, append the output
            pooled_signals = pd.concat(pooled_signals, join="outer", axis=1)

            # Replace 0 with nan to consider only expected hikes and cuts
            strat = multiple_timing(returns.rolling(holding_period).sum(),
                                    pooled_signals.replace(0, np.nan),
                                    xs_avg=False)

            # Append the disaggregated and aggregated outputs
            # TODO: Review mean vs sum/count
            if sum:
                aggr.loc[:, ix[holding_period, threshold]] =\
                    strat.mean(axis=1) * strat.count(axis=1)
            else:
                aggr.loc[:, ix[holding_period, threshold]] = strat.mean(axis=1)
            results["disaggr"][str(holding_period)][str(threshold)] = strat

            print("Policy expectation backtest\n",
                  "Holding period:", holding_period,
                  "Threshold level:", threshold, "\n")

    results["aggr"] = aggr

    return results

def pe_perfect_foresight_strat(returns, holding_range, data_path,
                               forecast_consistent=False,
                               smooth_burn=5, sum=False):
    """Generate a backetst of perfect foresight strategies, with optional
    forecast availability consistency.

    Parameters
    ----------
    returns: pd.DataFrame
        of returns to assets, for which implied rates are available via the
        'PolicyExpectation' class data api
    holding_range: np.arange
        specifying the range of holding periods
    data_path: str
        to the data for the 'PolicyExpectation.from_pickles()'
    forecast_consistent: bool
        controlling whether perfect foresight strategy should be consistent
        with implied rates in terms of data availability. If True the output is
        contngent on the forecast availability. If False the whole sample of
        events is taken. Default is False.
    smooth_burn: int
        additional number of days to burn in order to account for forecast
        smoothing, as in the real backtest. Corresponds to  avg_XXX_over of
        policy_forecast. Default is five
    sum: bool
        controlling whether to sum or average aggregated returns over the cross
        section. Default is False, i.e. mean is taken

    Returns
    -------
    results: dict
        with key 'aggr' containing a dataframe with columns indexed by holding
        periods, with each column containing returns of aggregate strategy, and
        key 'disaggr' containing a dictionary with keys corresponding to
        holding periods and items containing dataframes with returns asset-by-
        asset for the corresponding holding period. For example:

        results = {
            "aggr": pd.DataFrame of average returns,
            "disaggr": {
                "10": pd.DataFrame,
                "11": pd.DataFrame
                }

    """
    # Set up the output
    results = dict()
    results["disaggr"] = dict()

    aggr = pd.DataFrame(index=returns.index, columns=holding_range)

    # Start backtesting looping over holding periods and thresholds
    for holding_period in holding_range:
        # Transform holding period into lag_expect argument for the
        # 'forecast_policy_change()' method of the 'PolicyExpectation' class
        lag_expect = holding_period + 2  # forecast rate before trading FX

        # For the (predominant) case of multiple currenices pool the signals
        pooled_signals = list()

        # Get the policy forecast for each currency
        for curr in returns.columns:
            tmp_pe = PolicyExpectation.from_pickles(data_path, curr)
            # For forecast availability consistent perfect foresight strats
            # align the meetings accordingly
            # igor ---
            # signal + rename
            this_signal = tmp_pe.meetings.loc[:,"rate_change"]
            this_signal.name = curr
            # --------
            if forecast_consistent:
                # Get the first forecast date available, leave enough data
                # to make a forecast, control for averaging
                first_date = tmp_pe.rate_expectation.dropna()\
                    .iloc[[lag_expect+smooth_burn-1]].index[0]
                # igor ---
                pooled_signals.append(this_signal.loc[first_date:])
                # --------
            else:
                pooled_signals.append(this_signal)

        # Aggregate the signals, construct strategies, append the output
        pooled_signals = pd.concat(pooled_signals, join="outer", axis=1)

        # Replace 0 with nan to consider only realized hikes and cuts
        strat = multiple_timing(returns.rolling(holding_period).sum(),
                                pooled_signals.replace(0, np.nan),
                                xs_avg=False)

        # Append the disaggregated and aggregated outputs
        if sum:
            aggr.loc[:, holding_period] =\
                strat.mean(axis=1) * strat.count(axis=1)
        else:
            aggr.loc[:, holding_period] = strat.mean(axis=1)
        results["disaggr"][str(holding_period)] = strat

        print("Perfect foresight backtest\n",
              "Holding period:", holding_period, "\n")

    results["aggr"] = aggr

    return results

def event_backtest_wrapper(fx_data, fx_data_us, holding_range, threshold_range,
                           data_path, **kwargs):
    """Wrapper around 'event_trading_backtest' combining fomc dollar index and
    other currencies into single return

    Parameters
    ----------
    fx_data: dictionary
        containing dataframes with mid, ask, bid spot quotes and bid, ask t/n
        swap points. The FX data should adhere to the following convention:
        all exchange rates are in units of base currency per unit of quote
        currency, swap points should be added to get the excess returns. The
        keys are as follows: "spot_mid", "spot_bid", "spot_ask", "tnswap_bid",
        "tnswap_ask". Furthermore the data are assumed to be 'prepared' in
        terms of nan handling and index alignment
    fx_data: dictionary
        structured similarly to the one above, containing fx data for the
        dollar index trading strategy around the fomc meetings
    holding_range: np.arange
        specifying the range of holding periods
    threshold_range: np.arange
        specifying threshold levels in basis points
    data_path: str
        to the 'events.p' file with data for policy forecasts
    kwargs: dict
        of arguments of PolicyExpectation().forecast_policy_change() method,
        namely: avg_impl_over, avg_refrce_over, bday_reindex

    Returns
    -------
    strat_ret: Multiindex DataFrame
        with the first level corresponding to holding periods and the second
        level corresponding to the threshold levels, and columns containing sum
        of return on the expected policy rate strategy across assets

    """
    # Get returns of the currencies around local events, except for fomc
    ret_x_us = event_trading_backtest(fx_data, holding_range, threshold_range,
                                      data_path, fomc=False, **kwargs)["aggr"]

    # Get returns of the dollar portfolio around the FOMC meetings
    ret_us = event_trading_backtest(fx_data_us, holding_range, threshold_range,
                                    data_path, fomc=True, **kwargs)["aggr"]

    # Add one to the other
    strat_ret = ret_x_us.add(ret_us, axis=1).\
        fillna(value=ret_x_us).fillna(value=ret_us)

    return strat_ret

def event_trading_backtest(fx_data, holding_range, threshold_range,
                           data_path, fomc=False, **kwargs):
    """Backtest using policy forecast and EventTrading strategy, to compute
    bid-ask spread and t/n-swap -adjusted returns around forecast events

    Parameters
    ----------
    fx_data: dictionary
        containing dataframes with mid, ask, bid spot quotes and bid, ask t/n
        swap points. The FX data should adhere to the following convention:
        all exchange rates are in units of base currency per unit of quote
        currency, swap points should be added to get the excess returns. The
        keys are as follows: "spot_mid", "spot_bid", "spot_ask", "tnswap_bid",
        "tnswap_ask". Furthermore the data are assumed to be 'prepared' in
        terms of nan handling and index alignment
    holding_range: np.arange
        specifying the range of holding periods
    threshold_range: np.arange
        specifying threshold levels in basis points
    data_path: str
        to the 'events.p' file with data for policy forecasts
    fomc: bool
        If false the forecasts are obtained for each currency in the currencies
        list. If false the inverse fomc events are returned for each currency
        in the list. Default is False
    kwargs: dict
        of arguments of PolicyExpectation().forecast_policy_change() method,
        namely: avg_impl_over, avg_refrce_over, bday_reindex

    Returns
    -------
    results: dict
        with key 'aggr' containing a MultiIndex dataframe with the first level
        corresponding to holding period and second level corresponding to the
        threshold levels, and columns containing sum of return on the expected
        policy rate strategy across assets (mean for the case of dollar index
        i.e. if fomc=True). The second key 'disaggr' is a dict of dicts with
        first level corresponding to holding strategy and second level
        corresponding to the threshold levels with dataframes of returns to
        individual currencies as items. For example

        results = {
            "aggr": MultiIndex of summed (or averaged for fomc) returns,
            "disaggr": {
                "10":
                   {"5": DataFrame,
                    "10": DataFrame}
                "11":
                    {"5": DataFrame,
                     "10": DataFrame}
                    },
            }

    """
    # Get he pandas slicer for convenient MultiIndex reference
    ix = pd.IndexSlice

    # Initialize trading strategy settings
    trade_strat_settings = {"horizon_a": None,
                            "horizon_b": -1,
                            "bday_reindex": kwargs["bday_reindex"]}

    # Get the shorthand notation of the data for all countries
    spot_mid = fx_data["spot_mid"]
    spot_bid = fx_data["spot_bid"]
    spot_ask = fx_data["spot_ask"]
    swap_bid = fx_data["tnswap_bid"]
    swap_ask = fx_data["tnswap_ask"]

    # Set up the output structure
    results = dict()
    results["disaggr"] = dict()

    # The aggregated output is a multiindex
    combos = list(itools.product(holding_range, threshold_range))
    cols = pd.MultiIndex.from_tuples(combos, names=["holding", "threshold"])
    aggr = pd.DataFrame(index=spot_mid.index, columns=cols)

    # Start backtesting looping over holding periods and thresholds
    for holding_period in holding_range:
        # Transform holding period into lag_expect argument for the
        # 'forecast_policy_change()' method of the 'PolicyExpectation' class
        lag_expect = holding_period + 2  # forecast rate before trading FX

        # Adjust the trading strategy settings
        trade_strat_settings["horizon_a"] = -holding_period

        # Create an entry for the disaggregated output
        results["disaggr"][str(holding_period)] = dict()

        # A soup of several loops
        for threshold in threshold_range:
            # Get the signals
            signals = get_pe_signals(currencies=spot_mid.columns,
                                     lag=lag_expect, threshold=threshold,
                                     data_path=data_path, fomc=fomc, **kwargs)

            # Get the trading strategy
            strat = EventTradingStrategy(
                signals=signals,
                prices={"mid": spot_mid, "bid": spot_bid, "ask": spot_ask},
                settings=trade_strat_settings)

            # Adjust for transaction costs and swap points, get returns
            strat = strat.bas_adjusted()\
                .roll_adjusted({"bid": swap_bid, "ask": swap_ask})._returns

            # Append the output
            results["disaggr"][str(holding_period)][str(threshold)] = strat

            # For FOMC, report aggregated mean, else sum over the x-section
            if fomc:
                aggr.loc[:, ix[holding_period, threshold]] = strat.mean(axis=1)
            else:
                aggr.loc[:, ix[holding_period, threshold]] =\
                    strat.mean(axis=1) * strat.count(axis=1)

            print("Policy expectation backtest\n",
                  "Holding period:", holding_period,
                  "Threshold level:", threshold, "\n")

    results["aggr"] = aggr

    return results

def get_pe_signals(currencies, lag, threshold, data_path, fomc=False,
                   **kwargs):
    """Fetches a dataframe of signals from the 'PolicyExpectation' class, for
    currencies in the list for a given holding period and threshold. If fomc
    is True, returns a dataframe of inverse fomc signal for each currency in
    currencies

    Parameters
    ----------
    currencies: list
        of strings with currency codes, e.g ["aud", "gbp", "jpy"]
    lag: int
        forecast is made 'T-lag' days before a meeting, where T is announcement
        day
    threshold: float
        threshold for policy expectation forecast in basis points
    data_path: str
        to the 'events.p' file with data for policy forecasts
    fomc: bool
        If false the forecasts are obtained for each currency in the currencies
        list. If false the inverse fomc events are returned for each currency
        in the list. Default is False
    kwargs: dict
        of arguments of PolicyExpectation().forecast_policy_change() method,
        namely: avg_impl_over, avg_refrce_over, bday_reindex

    Returns
    -------
    signals. pd.DataFrame
        of predicted policy changes (+1 - hike, 0 - no change, -1 - cut)

    """
    # Transformation applied to reference and implied rates
    map_expected_rate = lambda x: x.rolling(kwargs["avg_impl_over"],
                                            min_periods=1).mean()
    map_proxy_rate = lambda x: x.rolling(kwargs["avg_refrce_over"],
                                         min_periods=1).mean()

    # For the US get the fomc announcements and use them for every currency
    if fomc:
        # Construct signals for the dollar index
        us_pe = PolicyExpectation.from_pickles(data_path, "usd", ffill=True)
        us_fcast = \
            us_pe.forecast_policy_direction(
                lag=lag, h_high=threshold/100,
                map_proxy_rate=map_proxy_rate,
                map_expected_rate=map_expected_rate)

        # Create inverse signals for every currency around FOMC announcements
        signals = pd.concat([-1*us_fcast]*len(currencies), axis=1)
        signals.columns = currencies

    # Otherwise get signals for each currency in the list
    else:
        policy_fcasts = list()
        # Get signals for each currency in the list
        for curr in currencies:
            tmp_pe = PolicyExpectation.from_pickles(data_path, currency=curr,
                                                    ffill=True)
            # policy_fcasts.append(
            #     tmp_pe.forecast_policy_change(lag=lag,threshold=threshold/100,
            #                                   **kwargs))
            policy_fcasts.append(
                tmp_pe.forecast_policy_direction(
                    lag=lag, h_high=threshold/100,
                    map_proxy_rate=map_proxy_rate,
                    map_expected_rate=map_expected_rate))

        # Pool signals into a single dataframe
        signals = pd.concat(policy_fcasts, join="outer", axis=1)
        signals.columns = currencies

    return signals


if __name__  == "__main__":

    pe = PolicyExpectation.from_pickles(data_path, currency="usd",
        meetings_pickle="ois_project_events.p", start_dt="2001-12",
        e_proxy_rate_pickle="implied_rates_1m_ois.p")

    pe.get_vus(10, lambda x: x.rolling(5, min_periods=1).mean().shift(1))
