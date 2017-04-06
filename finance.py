import pandas as pd
import numpy as np
from pandas.tseries.offsets import DateOffset, MonthBegin, MonthEnd, \
    relativedelta
import matplotlib.pyplot as plt

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
        """
        Parameters
        ----------
        meetings : pd.Series
            of meeting dates and respective policy rate decisions
        funds_futures : pd.DataFrame
            with dates on the index and expiry dates on the columns axis
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
        policy_exp_instance = PolicyExpectation(
            meetings=meetings,
            benchmark=meetings)

        # supply it with policy expectation
        policy_exp_instance.policy_exp = rate_exp

        return policy_exp_instance

    @classmethod
    def from_money_market(cls, meetings, instrument, tau, benchmark=None):
        """
        """
        assert isinstance(instrument, pd.Series)
        assert isinstance(meetings, pd.Series)

        # do not let
        instrument = instrument.loc[meetings.index[0]:]

        # if no rate provided, meetings must contain a rate
        if benchmark is None:
            benchmark = meetings.reindex(
                index=instrument.index, method="ffill")
        else:
            benchmark = benchmark.reindex(
                index=instrument.index, method="ffill")

        # allocate space for forward rates
        rate_exp = instrument.copy()*np.nan

        # loop over dates in instrument
        for t in rate_exp.index:
            # t = fwd_rate.index[2822]
            # t = pd.to_datetime("2016-02-08")

            # break if overshoot
            if t > meetings.last_valid_index():
                break

            # find two meeting dates: the previous, whose rate will serve as
            #   reference rate, and the next, which will possibly set new rate
            # previous
            prev_meet = \
                meetings.index[meetings.index.get_loc(t, method="ffill")]

            # next closest meeting date
            nx_meet = \
                meetings.index[meetings.index.get_loc(t, method="bfill")]

            # maturity date of ois: t + 1 month (actual/360 convention)
            setl_date = t + DateOffset(months=tau)

            # if next meeting is earlier than maturity, skip
            if setl_date <=  nx_meet:
                continue

            # number of days between them
            ndays = (setl_date - t).days

            # number of days until next meeting
            ndays_until = (nx_meet - t).days

            # previously set rate, to be effective until next meeting
            prev_rate = benchmark.loc[prev_meet]

            # implied rate
            r_isntr = instrument.loc[t]
            # impl_rate = ((1+/360)**ndays /
            #     (1+prev_rate/360)**ndays_until - 1)*(360/(ndays-ndays_until))
            impl_rate = (
                (
                    (r_isntr/360 + 1)**ndays/
                    (1+prev_rate/360)**(ndays_until)
                )**(1/(ndays-ndays_until))-1)*360

            # store
            rate_exp.loc[t] = impl_rate

        # create a new PolicyExpectation instance to return
        policy_exp_instance = PolicyExpectation(
            meetings=meetings,
            instrument=instrument,
            benchmark=benchmark)

        # supply it with policy expectation
        policy_exp_instance.policy_exp = rate_exp

        return policy_exp_instance

    def plot(self, lag):
        """
        """
        if not hasattr(self, "policy_exp"):
            raise ValueError("Estimate first!")

        # policy expectation to plot
        to_plot = self.policy_exp.shift(lag).loc[self.meetings.index]

        # rename a bit
        to_plot.name = "fwd_rate"
        meetings.name = "policy_rate"

        f, ax = plt.subplots(2, figsize=(11,8))

        # plot forward rate
        (to_plot*100).plot(
            ax=ax[0],
            linestyle='none',
            marker='o',
            color=my_blue,
            mec="none",
            label="implied rate")

        # plot meetings-set rate
        (meetings*100).plot(
            ax=ax[0],
            marker='.',
            color=my_red,
            label="policy rate")

        # set artist properties
        ax[0].set_xlim(
            max([meetings.first_valid_index(),
                instrument.first_valid_index()])-\
            DateOffset(months=6), ax[0].get_xlim()[1])
        ax[0].legend(fontsize=12)

        # predictive power
        pd.concat((to_plot*100, meetings*100), axis=1).\
            plot.scatter(
                ax=ax[1],
                x="fwd_rate",
                y="policy_rate",
                alpha=0.66,
                s=33,
                color=my_blue,
                edgecolor='none')
        lim_x = ax[1].get_xlim()
        ax[1].plot(lim_x, lim_x, color='r', linestyle='--')

        f.show()

#
# def expect_policy(instrument, meetings, tau, rate=None, plot=None):
#     """ Calculate expectation of the next policy meeting.
#
#     Uses forward rates implied in `instrument` as proxy for the rate set at
#     the next policy meeting. Looping over index of `instrument`, assumes that
#     before the next meeting the rate, which the instrument is derivative of,
#     stays at the level of the previous meeting (`rate`=None) or at the level
#     specified in `rate`.
#
#     Parameters
#     ----------
#     tau : int
#         maturity, in months
#     """
#     ax = None
#
#     assert isinstance(instrument, pd.Series)
#     assert isinstance(meetings, pd.Series)
#
#     # do not let
#     instrument = instrument.loc[meetings.index[0]:]
#
#     # if no rate provided, meetings must contain a rate
#     if rate is None:
#         rate = meetings.reindex(index=instrument.index, method="ffill")
#     else:
#         rate = rate.reindex(index=instrument.index, method="ffill")
#
#     # allocate space for forward rates
#     fwd_rate = instrument.copy()*np.nan
#
#     # loop over dates in instrument
#     for t in fwd_rate.index:
#         # t = fwd_rate.index[2822]
#         # t = pd.to_datetime("2016-02-08")
#
#         # break if overshoot
#         if t > meetings.last_valid_index():
#             break
#
#         # find two meeting dates: the previous, whose rate will serve as
#         #   reference rate, and the next, which will possibly set a new rate
#         # previous
#         prev_meet = meetings.index[meetings.index.get_loc(t, method="ffill")]
#
#         # next closest meeting date
#         nx_meet = meetings.index[meetings.index.get_loc(t, method="bfill")]
#
#         # maturity date of ois: t + 1 month (actual/360 convention)
#         setl_date = t + DateOffset(months=tau)
#
#         # if next meeting is earlier than maturity, skip
#         if setl_date <=  nx_meet:
#             continue
#
#         # number of days between them
#         ndays = (setl_date - t).days
#
#         # number of days until next meeting
#         ndays_until = (nx_meet - t).days
#
#         # previously set rate, to be effective until next meeting
#         prev_rate = meetings.loc[prev_meet]
#         # prev_rate = sonia.loc[t]
#
#         # implied rate
#         r_ois = instrument.loc[t]
#         # impl_rate = ((1+/360)**ndays /
#         #     (1+prev_rate/360)**ndays_until - 1)*(360/(ndays-ndays_until))
#         impl_rate = (
#             (
#                 (r_ois/360 + 1)**ndays/
#                 (1+prev_rate/360)**(ndays_until)
#             )**(1/(ndays-ndays_until))-1)*360
#
#         # store
#         fwd_rate.loc[t] = impl_rate
#
#     # plot --------------------------------------------------------------
#     if plot is not None:
#         assert isinstance(plot, int)
#
#         to_plot = fwd_rate.shift(plot).loc[meetings.index]
#
#         # rename a bit
#         to_plot.name = "fwd_rate"
#         meetings.name = "policy_rate"
#
#         f, ax = plt.subplots(2, figsize=(11,8))
#
#         # plot forward rate
#         (to_plot*100).plot(
#             ax=ax[0],
#             linestyle='none',
#             marker='o',
#             color=my_blue,
#             mec="none",
#             label="implied rate")
#
#         # plot meetings-set rate
#         (meetings*100).plot(
#             ax=ax[0],
#             marker='.',
#             color=my_red,
#             label="policy rate")
#
#         # set artist properties
#         ax[0].set_xlim(
#             max([meetings.first_valid_index(),
#                 instrument.first_valid_index()])-\
#             DateOffset(months=6), ax[0].get_xlim()[1])
#         ax[0].legend(fontsize=12)
#
#         # predictive power
#         pd.concat((to_plot*100, meetings*100), axis=1).\
#             plot.scatter(
#                 ax=ax[1],
#                 x="fwd_rate",
#                 y="policy_rate",
#                 alpha=0.66,
#                 s=33,
#                 color=my_blue,
#                 edgecolor='none')
#         lim_x = ax[1].get_xlim()
#         ax[1].plot(lim_x, lim_x, color='r', linestyle='--')
#
#         f.show()
#
#     return fwd_rate, ax
#
# if __name__  == "__main__":
#
#     with open(data_path + "ois.p", mode='rb') as fname:
#         ois_data = pickle.load(fname)
#     instrument = ois_data["icap_1m"]
#     instrument = instrument["cad"]
#     meetings = events["cad"].dropna()
#
#     pe = PolicyExpectation.from_funds_futures(
#         meetings=fomc, funds_futures=ff_fut_data)
#
#     pe.policy_exp.shift(5).loc[meetings.index].plot(
#         linestyle='none',marker='o')
#     pe.meetings.plot(color='r', marker='.')
#
#     pe = PolicyExpectation.from_money_market(
#         meetings=meetings, instrument=instrument, tau=1)
