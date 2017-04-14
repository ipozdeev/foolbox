import pandas as pd
import numpy as np
from pandas.tseries.offsets import DateOffset, MonthBegin, MonthEnd, \
    relativedelta
import matplotlib.pyplot as plt

# import ipdb

my_red = "#ce3300"
my_blue = "#2f649e"

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
                M = nx_fomc.day-1

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
            of meeting dates and respective policy rate decisions
        instrument : pd.Series
            of instrument (e.g. OIS) values
        tau : int
            maturity of `instrument`, in months
        benchmark : pd.Series, optional
            of the rate which the `instrument` is derivative of (if differs
            from the rate in `meetings`)

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
                    (r_instr/days_in_year + 1)**ndays/
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

        meetings_c = self.meetings.copy()

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


    def assess_forecast_quality(self, lag, threshold, benchmark_lag=None):
        """
        """
        # ipdb.set_trace()
        # forecast `lag` periods before
        fcast = self.policy_exp.shift(lag).reindex(
            index=self.meetings.index, method="bfill")

        # average benchmark value `lag` periods before
        if benchmark_lag is not None:
            avg_bench = self.benchmark.fillna(method="ffill", limit=2).\
                rolling(benchmark_lag).mean().shift(lag).reindex(
                    index=self.meetings.index, method="bfill")
        else:
            avg_bench = self.meetings.shift(1)

        # difference thereof
        fcast_less_instr = (fcast-avg_bench).dropna()

        # policy change
        policy_diff = self.meetings.diff()

        # signs
        policy_fcast = np.sign(fcast_less_instr).where(
            abs(fcast_less_instr) > threshold).fillna(0.0)
        policy_actual = np.sign(policy_diff)

        both = pd.concat((policy_fcast, policy_actual), axis=1)

        # correlation
        res_1 = both.corr().iloc[0,1]
        res_2 = both.dropna(how="any").index[0]

        return res_1, res_2


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


if __name__  == "__main__":

    from foolbox.api import *

    # data ------------------------------------------------------------------
    data_path = set_credentials.gdrive_path("research_data/fx_and_events/")

    with open(data_path + "ois.p", mode='rb') as fname:
        oidataata = pickle.load(fname)
    with open(data_path + "overnight_rates.p", mode='rb') as fname:
        overnight_rates = pickle.load(fname)
    with open(data_path + "events.p", mode='rb') as fname:
        events = pickle.load(fname)
    many_meetings = events["joint_cbs_lvl"]

    # loop over providers (icap, tr etc.) -----------------------------------
    # init storage
    policy_expectation = pd.Panel.from_dict(
        data={k: v*np.nan for k,v in oidataata.items()},
        orient="minor")

    for provider, many_instr in oidataata.items():
        # provider, many_instr = list(oidataata.items())[1]
        tau = int(provider[-2:-1])

        # loop over columns = currencies ------------------------------------
        for cur in many_instr.columns:
            if cur not in overnight_rates.columns:
                continue
            # cur = many_instr.columns[0]
            instrument = many_instr[cur]
            benchmark = overnight_rates.loc[:,cur]
            meetings = many_meetings[cur].dropna()

            pe = PolicyExpectation.from_money_market(
                meetings=meetings/100,
                instrument=instrument/100,
                benchmark=benchmark/100,
                tau=tau)

            policy_expectation.loc[cur,:,provider] = pe.policy_exp*100

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
