import pandas as pd
import numpy as np
import itertools as itools
from pandas.tseries.offsets import DateOffset, MonthBegin, MonthEnd, \
    relativedelta
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import foolbox.data_mgmt.set_credentials as set_credentials
import pickle
from foolbox.portfolio_construction import multiple_timing
# import ipdb

my_red = "#ce3300"
my_blue = "#2f649e"
my_gray = "#8c8c8c"

plt.rc("font", family="serif", size=12)


class PolicyExpectation():
    """
    """
    def __init__(self, meetings, instrument=None, benchmark=None):
        """
        """
        self.meetings = meetings
        self.instrument = instrument
        self.benchmark = benchmark


    @classmethod
    def from_funds_futures(cls, meetings, funds_futures):
        """ Extract policy expectations from Fed funds futures.

        For each date when a Fed funds futures is available, it is possible to
        extract the expectation of the rate to be set at the next policy
        meeting. Assumptions: the rate stays constant at the level of the
        previously settled futures until the next meeting when it changes;
        expectation hypothesis holds.

        Parameters
        ----------
        meetings : pd.Series
            of meeting dates and respective policy rate decisions
        funds_futures : pd.DataFrame
            with dates on the index and expiry dates on the columns axis

        Returns
        -------
        pe : PolicyExpectation instance
            with attribute `policy_exp` for expectation of the next set rate

        """
        # fill na forward (today's na equals yesterday's value)
        funds_futures = funds_futures.fillna(method="ffill", limit=30)

        # space for stuff
        rate_exp = pd.Series(index=funds_futures.index)

        for t in rate_exp.index:
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
                rate_exp.loc[t] = 100-funds_futures.loc[t,nx_nx_fomc_mend]
            else:
                # Scenario 2
                # previos month
                nx_fomc_mbefore = nx_fomc_mstart - relativedelta(days=1)
                M = nx_fomc.day

                # implied rate
                impl_rate = 100 - funds_futures.loc[t,nx_fomc_mend]
                # rate at start of fomc month
                start_rate = 100 - funds_futures.loc[t,nx_fomc_mbefore]

                # rate decision
                rate_exp.loc[t] = days_in_m/(days_in_m-M)*(
                    impl_rate - (M/days_in_m)*start_rate)

        # create a new PolicyExpectation instance to return
        pe = PolicyExpectation(
            meetings=meetings,
            benchmark=meetings)

        # supply it with policy expectation
        pe.policy_exp = rate_exp

        return pe


    @classmethod
    def from_money_market(cls, meetings, instrument, tau, benchmark=None):
        """ Extract policy expectations from money market instruments.

        For each date when an instrument observation is available, it is
        possible to extract the expectation of the rate to be set at the next
        policy meeting. Assumptions: the regulator targets the rate which
        `instrument` is derivative of; the rate stays constant at the
        previosuly set policy level (`benchmark` is None) or at the most
        recent level of `benchmark` (`benchmark` is not None) until the next
        meeting; at the expiry of the `instrument` the buyer receives the
        accumulated short rate; expectation hypothesis holds.

        Parameters
        ----------
        meetings : pd.Series
            of meeting dates and respective policy rate decisions, in frac of 1
        instrument : pd.Series
            of instrument (e.g. OIS) values, in frac of 1
        tau : int
            maturity of `instrument`, in months
        benchmark : pd.Series, optional
            of the rate which the `instrument` is derivative of (if differs
            from the rate in `meetings`), in frac of 1

        Returns
        -------
        pe : PolicyExpectation instance
            with attribute `policy_exp` for expectation of the next set rate

        """
        assert isinstance(meetings, pd.Series)
        assert isinstance(instrument, pd.Series)

        if instrument.name is None:
            raise AttributeError("instrument must be called with ISO of "+
                "respective currency")

        # day count convention
        day_count = {
            "gbp": 365,
            "cad": 365,
            "usd": 360,
            "eur": 360,
            "chf": 360,
            "jpy": 365,
            "nzd": 365,
            "aud": 365,
            "sek": 360}

        # days in year for this currency:
        days_in_year = day_count[instrument.name]

        # # do not let instrument be too full of nans
        # min_start = max((meetings.index[0], instrument.index[0]))
        # instrument = instrument.loc[min_start:]
        # meetings = meetings.loc[min_start:]

        # if no rate provided, meetings must contain a rate; in this case
        #   benchmark be the rate set at meetings projected onto the dates of
        #   instrument.
        if benchmark is None:
            benchmark = meetings.reindex(
                index=instrument.index, method="ffill")
        else:
            # project onto the dates too
            benchmark = benchmark.reindex(
                index=instrument.index, method="ffill")

        # make sure meeting dates have corresponding dates in instrument
        benchmark = benchmark.reindex(
            index=benchmark.index.union(meetings.index),
            method="ffill")

        # main loop ---------------------------------------------------------
        # allocate space for forward rates
        rate_exp = instrument.copy()*np.nan

        # loop over dates in instrument
        for t in rate_exp.index:
            # t = rate_exp.index[3000]
            # t = pd.to_datetime("2016-02-08")

            # break if out of range
            if t > meetings.last_valid_index():
                break

            # continue if too early
            if t < meetings.first_valid_index():
                continue

            # find two meeting dates: the previous, whose rate will serve as
            #   reference rate, and the next, which will possibly set a new
            #   rate
            # previous
            prev_meet = \
                meetings.index[meetings.index.get_loc(t, method="ffill")]

            # next closest
            nx_meet = \
                meetings.index[meetings.index.get_loc(t, method="bfill")]

            # maturity date of instrument: contract starts tomorrow; final
            #   date is then today tau month forward
            setl_date = t + DateOffset(months=tau)

            # if next meeting is earlier than maturity, skip
            if setl_date <= nx_meet:
                continue

            # number of days between them
            ndays = (setl_date - t).days

            # number of days until next meeting
            ndays_until = (nx_meet - t).days

            # previously set rate, to be effective until next meeting
            # TODO: not prev_meet, but prev_meet + 1
            idx = benchmark.index[benchmark.index.get_loc(
                prev_meet + DateOffset(days=1), "bfill")]
            bench_reixed = benchmark.loc[idx:t].reindex(
                index=pd.date_range(idx,t,freq='D'),
                method="ffill")
            prev_rate = (
                (1+bench_reixed/days_in_year).product()**\
                (1/bench_reixed.count())-1)*days_in_year

            # implied rate
            r_instr = instrument.loc[t]

            impl_rate = (
                (
                    (1+r_instr/days_in_year*ndays)/
                    (1+prev_rate/days_in_year)**(ndays_until)
                )**(1/(ndays-ndays_until))-1)*days_in_year

            # store
            rate_exp.loc[t] = impl_rate

        # create a new PolicyExpectation instance to return
        pe = PolicyExpectation(
            meetings=meetings,
            instrument=instrument,
            benchmark=benchmark)

        # supply it with policy expectation
        pe.policy_exp = rate_exp

        return pe


    def plot(self, lag):
        """ Plot predicted vs. realized and the error plot.
        """
        if not hasattr(self, "policy_exp"):
            raise ValueError("Estimate first!")

        # meetings_c = self.meetings.copy()
        meetings_c = self.benchmark.rolling(lag).mean().shift(-lag).reindex(
            index=self.meetings.index, method="bfill")

        # policy expectation to plot
        to_plot = self.policy_exp.shift(lag).reindex(
            index=meetings_c.index, method="ffill")

        # rename a bit
        to_plot.name = "policy_exp"
        meetings_c.name = "policy_rate"

        f, ax = plt.subplots(2, figsize=(11.7,8.3))

        # plot forward rate
        (to_plot*100).plot(
            ax=ax[0],
            linestyle='none',
            marker='o',
            color=my_blue,
            mec="none",
            label="implied rate")

        # plot meetings-set rate
        (meetings_c*100).plot(
            ax=ax[0],
            marker='.',
            color=my_red,
            label="policy rate")

        # set artist properties
        ax[0].set_xlim(
            max([meetings_c.first_valid_index(),
                self.policy_exp.first_valid_index()])-\
            DateOffset(months=6), ax[0].get_xlim()[1])
        ax[0].legend(fontsize=12)

        # predictive power
        pd.concat((to_plot*100, meetings_c*100), axis=1).\
            plot.scatter(
                ax=ax[1],
                x="policy_exp",
                y="policy_rate",
                alpha=0.66,
                s=33,
                color=my_blue,
                edgecolor='none')

        lim_x = ax[1].get_xlim()
        ax[1].plot(lim_x, lim_x, color='r', linestyle='--')

        return f, ax

    def forecast_policy_change(self, lag, threshold, avg_impl_over=1,
        avg_refrce_over=None):
        """ Predict +1, -1 or 0 for every meeting date.

        Some periods before each event (`lag`) we take the implied rate
        smoothed over the previous `avg_impl_over` periods and compare it to
        the reference rate smoothed over the previous `avg_refrce_over`
        periods. Should the difference between the two exceed the `threshold`,
        we predict the next set rate be higher and so on.

        Parameters
        ----------
        lag : int
            such that the forecast is made at t-`lag` with t being the meeting
            date
        threshold : float
            the maximum absolute difference between the implied rate and the
            reference rate such that no policy change is predicted; values
            above that would signal a hike, below - cut
        avg_impl_over : int
            such that the implied rate that is compared to the reference rate
            is first smoothed over this number of periods
        avg_refrce_over : int
            the number of periods to average the reference rate over before
            comparing it to the implied rate

        Returns
        -------
        policy_fcast : pd.Series
            of forecasts: -1 for cut, +1 for hike, 0 for no change; indexed
            with the index of self.meetings

        Example
        -------
        data_path = set_credentials.gdrive_path("research_data/fx_and_events/")
        pe = PolicyExpectation.from_pickles(data_path, "gbp")
        fcast = pe.forecast_policy_change(10, 0.1250, 10)

        """
        # take implied rate `lag` periods before
        impl_rate = self.policy_exp

        # will need to compare it to either some rolling average of the
        #   reference rate or the previously set policy rate (avg_refrce_over
        #   is None)
        if isinstance(avg_refrce_over, int):
            # fill some NAs
            avg_refrce = self.benchmark.fillna(method="ffill", limit=2)
            # +1 is needed because e.g. in the USA the rate is published on
            #   the next day
            avg_refrce = avg_refrce.rolling(avg_refrce_over).mean().\
                shift(lag+1).\
                reindex(index=self.meetings.index, method="ffill")

        elif isinstance(avg_refrce_over, str):
            # collect everything between events
            refrce_aligned, meets_aligned = self.benchmark.align(
                self.meetings, join="outer", axis=0)
            refrce_aligned = refrce_aligned.to_frame()
            refrce_aligned.loc[:,"dates"] = meets_aligned.index
            refrce_aligned.loc[:,"dates"].fillna(method="bfill")

            for dt in range(1,len(self.meetings.index)):
                this_dt = self.meetings.index[dt]
                prev_dt = self.meetings.index[dt-1] + DateOffset(days=2)
                this_piece = refrce_aligned.ix[
                    refrce_aligned.loc[:,"dates"]==this_dt,0]
                refrce_aligned.ix[
                    refrce_aligned.loc[:,"dates"]==this_dt,0].loc[prev_dt:] = \
                        this_piece.loc[prev_dt:].expanding().mean()

                avg_refrce = refrce_aligned.shift(lag+1).\
                    reindex(index=self.meetings.index, method="ffill")

        else:
            # else, if None, take the last meetings decision
            avg_refrce = self.meetings.shift(1)

        # smooth implied rate
        avg_impl = impl_rate.rolling(avg_impl_over).mean().shift(lag).\
            reindex(index=self.meetings.index, method="ffill")

        # difference between rate implied some periods earlier and the
        #   benchmark rate
        impl_less_bench = (avg_impl-avg_refrce).dropna()

        # forecast is the sign of the difference if it is large enough
        #   (as measured by `threshold`)
        policy_fcast = np.sign(impl_less_bench).where(
            abs(impl_less_bench) > threshold).fillna(0.0)

        return policy_fcast

    def assess_forecast_quality(self, lag, threshold, avg_impl_over=1,
        avg_refrce_over=None):
        """
        """
        # ipdb.set_trace()
        # forecast `lag` periods before
        policy_fcast = self.forecast_policy_change(
            lag, threshold, avg_impl_over, avg_refrce_over)
        policy_fcast.name = "fcast"

        # policy change: should already be expressed as difference
        #   (see __init__)
        policy_diff = self.meetings
        policy_actual = np.sign(policy_diff)
        policy_actual.name = "actual"

        # concat to be able to drop NAs
        both = pd.concat((policy_actual, policy_fcast), axis=1).\
            dropna(how="any")

        # percentage of correct guesses
        rho = (both.loc[:,"fcast"] == both.loc[:,"actual"]).mean()
        cmx = confusion_matrix(both.loc[:,"fcast"], both.loc[:,"actual"])

        idx = sorted(list(set(
            list(pd.unique(both.loc[:,"fcast"])) + \
            list(pd.unique(both.loc[:,"actual"])))))

        cmx = pd.DataFrame(cmx,
            index=idx,
            columns=idx)

        cmx = cmx.reindex(index=[-1, 0, 1], columns=[-1, 0, 1]).fillna(0)
        cmx = cmx.astype(np.int16)

        return rho, cmx

    @classmethod
    def from_pickles(cls, data_path, currency, s_dt="1990"):
        """
        """
        # events
        with open(data_path + "events_new.p", mode='rb') as hangar:
            events = pickle.load(hangar)
        evts = events["joint_cbs"].loc[:,currency].dropna()

        # reference rates
        with open(data_path + "overnight_rates.p", mode='rb') as hangar:
            overnight_rates = pickle.load(hangar)

        # implied rates
        with open(data_path + "implied_rates.p", mode='rb') as hangar:
            implied_rates = pickle.load(hangar)

        # init class, manually insert policy expectation
        pe = PolicyExpectation(
            meetings=evts.loc[s_dt:],
            benchmark=overnight_rates[currency])
        pe.policy_exp = implied_rates[currency]

        return pe

def into_currency(data, new_cur):
    """
    Parameters
    ----------
    data :
        expressed in USD per unit
    """
    # re-express everything in units of new_cur
    new_data = data.drop([new_cur], axis=1).subtract(
        data[new_cur], axis="index")

    # reinstall
    new_data["usd"] = -1*data[new_cur]

    return new_data


def pe_backtest(returns, holding_range, threshold_range,
                data_path, avg_impl_over=2, avg_refrce_over=2):
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
        lag_expect = holding_period + 1  # forecast rate before trading FX

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
                                                  avg_refrce_over=avg_refrce_over)
                # Append the signals
                pooled_signals.append(tmp_fcast)

            # Aggregate the signals, construct strategies, append the output
            pooled_signals = pd.concat(pooled_signals, join="outer", axis=1)

            # Replace 0 with nan to consider only expected hikes and cuts
            strat = multiple_timing(returns.rolling(holding_period).sum(),
                                    pooled_signals.replace(0, np.nan),
                                    xs_avg=False)

            # Append the disaggregated and aggregated outputs
            aggr.loc[:, ix[holding_period, threshold]] = strat.mean(axis=1)
            results["disaggr"][str(holding_period)][str(threshold)] = strat

            print("Policy expectation backtest\n",
                  "Holding period:", holding_period,
                  "Threshold level:", threshold, "\n")

    results["aggr"] = aggr

    return results


def pe_perfect_foresight_strat(returns, holding_range, data_path,
                               forecast_consistent=False):
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
        lag_expect = holding_period + 1  # forecast rate before trading FX

        # For the (predominant) case of multiple currenices pool the signals
        pooled_signals = list()

        # Get the policy forecast for each currency
        for curr in returns.columns:
            tmp_pe = PolicyExpectation.from_pickles(data_path, curr)
            # For forecast availability consistent perfect foresight strats
            # align the meetings accordingly
            if forecast_consistent:
                # Get the first forecast date available
                first_date = tmp_pe.policy_exp.dropna().iloc[[0]].index[0]
                pooled_signals.append(tmp_pe.meetings[first_date:])
            else:
                pooled_signals.append(tmp_pe.meetings)

        # Aggregate the signals, construct strategies, append the output
        pooled_signals = pd.concat(pooled_signals, join="outer", axis=1)

        # Replace 0 with nan to consider only realized hikes and cuts
        strat = multiple_timing(returns.rolling(holding_period).sum(),
                                pooled_signals.replace(0, np.nan),
                                xs_avg=False)

        # Append the disaggregated and aggregated outputs
        aggr.loc[:, holding_period] = strat.mean(axis=1)
        results["disaggr"][str(holding_period)] = strat

        print("Perfect foresight backtest\n",
              "Holding period:", holding_period, "\n")

    results["aggr"] = aggr

    return results


def broomstick_plot(data, ci=(0.1, 0.9)):
    """Given an input array of data, produces a broomstick plot, with all
    series in gray, the mean of the series in black (along with confidence
    interval). The data are intended to be cumulative returns on a set of
    trading strategies

    Parameters
    ----------
    data: pd.DataFrame
        of the cumulative returns to plot
    ci: tuple
        of floats specifying lower and upper quantiles for empirical confidence
        interval. Default is (0.1, 0.9)

    Returns
    -------
    figure: matplotlib.pyplot.plot
        with plotted data


    """
    # Check the input data
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Nope, the input should be a DataFrame")

    # Drop absent observations if any
    data = data.dropna()

    # Get the mean and confidence bounds
    cb_u = data.quantile(ci[1], axis=1)
    cb_l = data.quantile(ci[0], axis=1)
    avg = data.mean(axis=1)
    stats = pd.concat([avg, cb_u, cb_l], axis=1)
    stats.columns = ["mean", "cb_u", "cb_l"]

    # Start plotting
    fig, ax = plt.subplots(figsize=(8.4, 11.7/3))
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    # Grid
    ax.grid(which="both", alpha=0.33, linestyle=":")
    ax.plot(data, color=my_gray, lw=0.75, alpha=0.25)

    ax.plot(stats["mean"], color="k", lw=2)
    ax.plot(stats[["cb_l", "cb_u"]], color="k", lw=2, linestyle="--")

    # Construct lines for the custom legend
    solid_line = mlines.Line2D([], [], color='black', linestyle="-",
                               lw=2, label="Mean")

    ci_label = "{}th and {}th percentiles"\
        .format(int(ci[0]*100), int(ci[1]*100))
    dashed_line = mlines.Line2D([], [], color='black', linestyle="--",
                                lw=2, label=ci_label)
    ax.legend(handles=[solid_line, dashed_line], loc="upper left")

    return fig


if __name__  == "__main__":
    pass
    # from foolbox.data_mgmt import set_credentials
    #
    # # data ------------------------------------------------------------------
    # path = "c:/Users/Igor/Google Drive/Personal/opec_meetings/calc/"
    # data_path = set_credentials.gdrive_path("research_data/fx_and_events/")
    #
    # pe = PolicyExpectation.from_pickles(data_path, "eur")
    # ipdb.set_trace()
    # pe.forecast_policy_change(5, 0.1250, 5, 5)


    #
    # with open(data_path + "ois.p", mode='rb') as fname:
    #     oidataata = pickle.load(fname)
    # with open(data_path + "overnight_rates.p", mode='rb') as fname:
    #     overnight_rates = pickle.load(fname)
    # with open(data_path + "events.p", mode='rb') as fname:
    #     events = pickle.load(fname)
    # many_meetings = events["joint_cbs_lvl"]
    #
    # # loop over providers (icap, tr etc.) -----------------------------------
    # # init storage
    # policy_expectation = pd.Panel.from_dict(
    #     data={k: v*np.nan for k,v in oidataata.items()},
    #     orient="minor")
    #
    # for provider, many_instr in oidataata.items():
    #     # provider, many_instr = list(oidataata.items())[1]
    #     tau = int(provider[-2:-1])
    #
    #     # loop over columns = currencies ------------------------------------
    #     for cur in many_instr.columns:
    #         if cur not in overnight_rates.columns:
    #             continue
    #         # cur = many_instr.columns[0]
    #         instrument = many_instr[cur]
    #         benchmark = overnight_rates.loc[:,cur]
    #         meetings = many_meetings[cur].dropna()
    #
    #         pe = PolicyExpectation.from_money_market(
    #             meetings=meetings/100,
    #             instrument=instrument/100,
    #             benchmark=benchmark/100,
    #             tau=tau)
    #
    #         policy_expectation.loc[cur,:,provider] = pe.policy_exp*100

    # # %matplotlib
    # lol, wut = pe.plot(5)
    # policy_expectation.loc[cur,:,provider] = pe.policy_exp*100
    # events["snb"].loc[:,["lower","upper"]].plot(ax=wut[0], color='k')
    # benchmark.rolling(5).mean().shift(5).dropna().plot(
    #     ax=wut[0], color='g', linewidth=1.5)
    #
    # one = pe.policy_exp.shift(5).reindex(
    #     index=meetings.index, method="ffill")
    # two = benchmark.rolling(5).mean().shift(5).reindex(
    #     index=meetings.index, method="ffill")
    # both = (one*100-two).dropna()
    # policy_diff = meetings.diff()
    #
    # f, ax = plt.subplots()
    # policy_diff.plot(ax=ax, color='r', linestyle="none", marker='.')
    # ax.set_ylim([-2.5, 1.0])
    # both.plot(ax=ax, color='k', linestyle="none", marker='o',
    #     alpha=0.33)
    #
    #
    # meetings.loc["2016-07":]
    # pe.policy_exp.loc["2016-07":]
    # instrument.loc["2016-06-15":]
    # benchmark.loc["2016-07":]
    #
    # with open(data_path + "implied_rates.p", "wb") as fname:
    #     pickle.dump(policy_expectation, fname)

    # %matplotlib
    # lol,wut = pe.plot(5)
    #
    # with open(data_path + "implied_rates.p", "rb") as fname:
    #     policy_expectation = pickle.load(fname)
    #
    # gbp_impl = policy_expectation.loc["gbp",:,"icap_1m"]
    # gbp_meet = many_meetings["gbp"].dropna()
    # gbp_rate = overnight_rates.loc[:,"gbp"]
    #
    # # %matplotlib
    # gbp_rate.rolling(5).mean().plot(color='r')
    #
    # one = gbp_impl.shift(5).reindex(
    #     index=gbp_meet.index, method="ffill")
    # two = gbp_rate.fillna(method="ffill", limit=2).\
    #     rolling(5).mean().shift(5).reindex(
    #         index=gbp_meet.index, method="ffill")
    # both = (one*100-two).dropna()
    # policy_diff = gbp_meet.diff()
    #
    # gbp_rate.loc["2016-05"]
    #
    # policy_fcast = np.sign(both).where(abs(both) > 0.24).fillna(0.0)
    # policy_actual = np.sign(policy_diff)
    #
    # pd.concat((policy_fcast, policy_actual), axis=1).corr()
