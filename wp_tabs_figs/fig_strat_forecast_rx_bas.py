from foolbox.api import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
import matplotlib.lines as mlines
from foolbox.wp_tabs_figs.wp_settings import settings
from foolbox.trading import EventTradingStrategy
from foolbox.utils import *

"""Plots returns to strategy based on monetary policy action forecasts along
with prediction-data-availability-consistent prefect foresight. Adjust returns
for rollover yields and bid-ask spreads
"""
# Set the output path, input data and sample
out_path = data_path + settings["fig_folder"]
input_dataset = settings["fx_data"]
start_date = settings["sample_start"]

#start_date = "2011-10-03"

end_date = settings["sample_end"]

no_good_curs = ["dkk", "jpy", "nok", "gbp", "cad", "chf", "eur", "nzd", "sek"]

# Set up the parameters of trading strategies
lag = 10              # actually the holding period,
lag_expect = lag + 2  # forecast rate one day in advance of trading
threshold = 10        # threshold in bps

# Forecast, and forecast consistency parameters
avg_impl_over = 5    # smooth implied rate
avg_refrce_over = 5  # smooth reference rate
smooth_burn = 5      # discard number of periods corresponding to smoothing for
                     # the forecast-consistent perfect foresight
e_proxy_rate_pickle_name = "implied_rates_bloomberg_1m.p"
e_proxy_rate_pickle_name = "implied_rates_from_1m_ois.p"
#e_proxy_rate_pickle_name = "implied_rates_from_1m_ois_roll_5d.p"
# EventTradingStrategy() settings
trad_strat_settings = {
    "horizon_a": -lag,
    "horizon_b": -1,
    "bday_reindex": True
    }

# matplotlib settings -------------------------------------------------------
# font, colors
plt.rc("font", family="serif", size=12)
# locators
minor_locator = mdates.YearLocator()
major_locator = mdates.YearLocator(2)

# Import the FX data
# with open(data_path+input_dataset, mode="rb") as fname:
#     data = pickle.load(fname)
data = pd.read_pickle(data_path+input_dataset)

# Get the individual currenices, spot rates:
spot_mid = data["spot_mid"][start_date:end_date].drop(no_good_curs,
                                                      axis=1)
spot_bid = data["spot_bid"][start_date:end_date].drop(no_good_curs,
                                                      axis=1)
spot_ask = data["spot_ask"][start_date:end_date].drop(no_good_curs,
                                                      axis=1)

# And swap points
swap_ask = data["tnswap_ask"][start_date:end_date].drop(no_good_curs,
                                                        axis=1)
swap_ask = remove_outliers(swap_ask, 50)
swap_bid = data["tnswap_bid"][start_date:end_date].drop(no_good_curs,
                                                        axis=1)
swap_bid = remove_outliers(swap_bid, 50)

# Import the all fixing times for the dollar index
# with open(data_path+"fx_by_tz_sp_fixed.p", mode="rb") as fname:
#     data_usd = pickle.load(fname)
data_usd = pd.read_pickle(data_path+"fx_by_tz_sp_fixed.p")

# Construct a pre-set fixing time dollar index
us_spot_mid = data_usd["spot_mid"].loc[:, :, settings["usd_fixing_time"]]\
    .drop(["dkk"], axis=1)[start_date:end_date]
us_spot_bid = data_usd["spot_bid"].loc[:, :, settings["usd_fixing_time"]]\
    .drop(["dkk"], axis=1)[start_date:end_date]
us_spot_ask = data_usd["spot_ask"].loc[:, :, settings["usd_fixing_time"]]\
    .drop(["dkk"], axis=1)[start_date:end_date]
us_swap_ask = data_usd["tnswap_ask"].loc[:, :, settings["usd_fixing_time"]]\
    .drop(["dkk"], axis=1)[start_date:end_date]
us_swap_ask = remove_outliers(us_swap_ask, 50)
us_swap_bid = data_usd["tnswap_bid"].loc[:, :, settings["usd_fixing_time"]]\
    .drop(["dkk"], axis=1)[start_date:end_date]
us_swap_bid = remove_outliers(us_swap_bid, 50)

# Align and ffill the data, first for tz-aligned countries
(spot_mid, spot_bid, spot_ask, swap_bid, swap_ask) =\
    align_and_fillna((spot_mid, spot_bid, spot_ask, swap_bid, swap_ask),
                     "B", method="ffill")
# Now for the dollar index
(us_spot_mid, us_spot_bid, us_spot_ask, us_swap_bid, us_swap_ask) =\
    align_and_fillna((us_spot_mid, us_spot_bid, us_spot_ask,
                      us_swap_bid, us_swap_ask),
                     "B", method="ffill")

# TODO: check the spot-based strategy for consistency, otherwise uncomment
# TODO: the stuff below for the same result by setting costs and swaps to zero
# spot_bid = spot_mid
# spot_ask = spot_mid
# swap_bid *= 0
# swap_ask *= 0
#
# us_spot_bid = us_spot_mid
# us_spot_ask = us_spot_mid
# us_swap_bid *= 0
# us_swap_ask *= 0

# Transformation applied to reference and implied rates
map_expected_rate = lambda x: x.rolling(avg_impl_over,
                                        min_periods=1).mean()
map_proxy_rate = lambda x: x.rolling(avg_refrce_over,
                                     min_periods=1).mean()

# Get signals for all countries except for the US
policy_fcasts = list()
for curr in spot_mid.columns:
    # Get the predicted change in policy rate
    tmp_pe = PolicyExpectation.from_pickles(
        data_path, curr, ffill=True,
        e_proxy_rate_pickle=e_proxy_rate_pickle_name)
    # policy_fcasts.append(
    #     tmp_pe.forecast_policy_change(lag=lag_expect,
    #                                   threshold=threshold/100,
    #                                   avg_impl_over=avg_impl_over,
    #                                   avg_refrce_over=avg_refrce_over,
    #                                   bday_reindex=True))
    policy_fcasts.append(
        tmp_pe.forecast_policy_direction(
                    lag=lag_expect, h_high=threshold/100,
                    map_proxy_rate=map_proxy_rate,
                    map_expected_rate=map_expected_rate))

# Put individual predictions into a single dataframe
signals = pd.concat(policy_fcasts, join="outer", axis=1)[start_date:end_date]
signals.columns = spot_mid.columns

# Get the trading strategy
strat = EventTradingStrategy(
    signals=signals,
    prices={"mid": spot_mid, "bid": spot_bid, "ask": spot_ask},
    settings=trad_strat_settings)

strat_bas_adj = strat.bas_adjusted().roll_adjusted(
    {"bid": swap_bid, "ask": swap_ask})

strat_ret = strat_bas_adj._returns.dropna(how="all")

# Construct signals for the dollar index
us_pe = PolicyExpectation.from_pickles(
    data_path, "usd", ffill=True,
    e_proxy_rate_pickle=e_proxy_rate_pickle_name)
# us_fcast = us_pe.forecast_policy_change(lag=lag_expect,
#                                         threshold=threshold/100,
#                                         avg_impl_over=avg_impl_over,
#                                         avg_refrce_over=avg_refrce_over,
#                                         bday_reindex=True)

us_fcast = us_pe.forecast_policy_direction(
                    lag=lag_expect, h_high=threshold/100,
                    map_proxy_rate=map_proxy_rate,
                    map_expected_rate=map_expected_rate)

# Create signals for every currency around FOMC announcements, mind the minus
us_signal = pd.concat([-us_fcast]*len(us_spot_mid.columns), axis=1)\
    [start_date:end_date]
us_signal.columns = us_spot_mid.columns

# Get the returns of each currency around the FOMC meetings
us_strat = EventTradingStrategy(
    signals=us_signal,
    prices={"mid": us_spot_mid, "bid": us_spot_bid, "ask": us_spot_ask},
    settings=trad_strat_settings,
    weights=pd.DataFrame(1/len(us_spot_mid.columns),
                         index=us_spot_mid.index,
                         columns=us_spot_mid.columns))

# Adjust for spreadsa
us_strat_bas_adj = us_strat.bas_adjusted().roll_adjusted(
    {"bid": us_swap_bid, "ask": us_swap_ask})

# Get the return of dollar index
usd_ret = us_strat_bas_adj._returns.dropna(how="all").mean(axis=1).to_frame()
usd_ret.columns = ["usd"]

# Add it to other returns
fcst_strat = pd.concat([strat_ret, usd_ret], axis=1).sum(axis=1).to_frame()
fcst_strat.columns = ["fcst"]

# Get the descriptives for the forecast strategy, use mean instead of sum
descr_fcst = taf.descriptives(
    pd.concat([strat_ret, usd_ret], axis=1).mean(axis=1).to_frame(), 1)

# Drop the period from June 2008 to June 2009, repeat descriptives
tmp_df_descr = pd.concat([strat_ret, usd_ret], axis=1).mean(axis=1).to_frame()
tmp_df_descr = pd.concat([tmp_df_descr[:"2008-06"],
                          tmp_df_descr["2009-06":]], axis=0)
tmp_df_descr = taf.descriptives(tmp_df_descr, 1)



# Construct the perfect foresight, forecastablility consistent signals
# Start with all countries
pfct_signals = list()
for curr in spot_mid.columns:
    # Get the predicted change in policy rate
    tmp_pe = PolicyExpectation.from_pickles(
        data_path, curr, ffill=True,
        e_proxy_rate_pickle=e_proxy_rate_pickle_name)

    # Get the signals
    this_signal = tmp_pe.meetings#.loc[:,"rate_change"]
    this_signal.name = curr

    # Get the first forecast date available, leave enough data
    # to make a forecast, control for averaging
    first_date = tmp_pe.expected_proxy_rate.dropna() \
         .iloc[[lag_expect + smooth_burn - 1]].index[0]
    pfct_signals.append(this_signal.loc[first_date:])

pfct_signals = pd.concat(pfct_signals, axis=1)[start_date:end_date]
pfct_signals = np.sign(pfct_signals)

# Get the trading strategy
pfct_strat = EventTradingStrategy(
    signals=pfct_signals,
    prices={"mid": spot_mid, "bid": spot_bid, "ask": spot_ask},
    settings=trad_strat_settings)

pfct_strat_bas_adj = pfct_strat.bas_adjusted().roll_adjusted(
    {"bid": swap_bid, "ask": swap_ask})

pfct_strat_ret = pfct_strat_bas_adj._returns.dropna(how="all")


# Similarly for the US
us_pe = PolicyExpectation.from_pickles(data_path, "usd")
us_pfct_signal = us_pe.meetings.loc[start_date:end_date]
us_pfct_signal = np.sign(us_pfct_signal)

# Create signals for every currency around FOMC announcements, mind the minus
us_pfct_signal = pd.concat([-us_pfct_signal] * len(us_spot_mid.columns),
                           axis=1)[start_date:end_date]
us_pfct_signal.columns = us_spot_mid.columns

# Get the returns of each currency around the FOMC meetings
us_pfct_strat = EventTradingStrategy(
    signals=us_pfct_signal,
    prices={"mid": us_spot_mid, "bid": us_spot_bid, "ask": us_spot_ask},
    settings=trad_strat_settings,
    weights=pd.DataFrame(1/len(us_spot_mid.columns),
                         index=us_spot_mid.index,
                         columns=us_spot_mid.columns))

# Adjust for spreads
us_pfct_strat_bas_adj = us_pfct_strat.bas_adjusted().roll_adjusted(
    {"bid": us_swap_bid, "ask": us_swap_ask})

# Get the return of dollar index
usd_pfct_ret = us_pfct_strat_bas_adj._returns.dropna(how="all").mean(axis=1)\
    .to_frame()
usd_pfct_ret.columns = ["usd"]

# Combine with all countries
perfect_consistent = pd.concat([pfct_strat_ret, usd_pfct_ret], axis=1)\
    .sum(axis=1).to_frame()
perfect_consistent.columns = ["pfct"]


# Figure 1: plot OIS-availability consistent perfect foresight and real strat
fig1, ax = plt.subplots()
# Concatenate the data first
to_plot = pd.concat([perfect_consistent.dropna().cumsum()*100,
                    fcst_strat.dropna().cumsum()*100], axis=1).ffill().fillna(0)
to_plot.columns = ["pfct", "fcst"]

# Plot it
to_plot[["pfct"]].plot(ax=ax, color='k', linewidth=1.5, linestyle="--")
to_plot[["fcst"]].plot(ax=ax, color='k', linewidth=1.5, linestyle="-")

# Rotate the axis labels
plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

# Use time-indexed scale use the locators from settings
ax.xaxis.set_major_locator(major_locator)
ax.xaxis.set_minor_locator(minor_locator)

# Polish the layout
ax.grid(which="both", alpha=0.33, linestyle=":")
ax.set_xlabel("date", visible=True)
ax.set_ylabel("cumulative return in percent", visible=True)
ax.legend_.remove()

# Add some descriptives
descr = descr_fcst
ax.annotate(r"$\mu={:3.2f}%$".format(descr.loc["mean"].values[0]*10000),
    xy=(0.9, 0.30), xycoords='axes fraction',
    horizontalalignment='right', fontsize=12)
ax.annotate(r"$se={:3.2f}%$".format(descr.loc["se_mean"].values[0]*10000),
    xy=(0.9, 0.20), xycoords='axes fraction',
    horizontalalignment='right', fontsize=12)
ax.annotate(r"$SR={:3.2f}%$".format(descr.loc["sharpe"].values[0]),
    xy=(0.9, 0.10), xycoords='axes fraction',
    horizontalalignment='right', fontsize=12)

# Add legend for lines
solid_line = mlines.Line2D([], [], color='black', linestyle="-",
                           lw=2, label="forecast-based")
dashed_line = mlines.Line2D([], [], color='black', linestyle="--",
                            lw=2, label="perfect foresight")
ax.legend(handles=[solid_line, dashed_line], loc="upper left", fontsize=10)

fig1.tight_layout()
fig1.savefig(out_path +
             "ueber_fcast_lag{:d}_thresh{:d}_ai{:d}_ar{:d}".\
             format(lag, int(threshold*100), avg_impl_over, avg_refrce_over) +
             "_rx_bas" +
             "_time" +
             ".pdf")


# Figure 2: forecast on the event axis
fig2, ax = plt.subplots()
pd.DataFrame(100*fcst_strat.dropna().values).cumsum().\
    plot(ax=ax, color='k', linewidth=1.5, linestyle="-")
# Rotate the axis labels
plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

# Use time-indexed scale use the locators from settings
ax.xaxis.set_major_locator(MultipleLocator(50))
ax.xaxis.set_minor_locator(MultipleLocator(25))

# Polish the layout
ax.grid(which="both", alpha=0.33, linestyle=":")
ax.set_xlabel("event number", visible=True)
ax.set_ylabel("cumulative return in percent", visible=True)
ax.legend_.remove()

fig2.tight_layout()
fig2.savefig(out_path +
             "ueber_fcast_lag{:d}_thresh{:d}_ai{:d}_ar{:d}".\
             format(lag, int(threshold*100), avg_impl_over, avg_refrce_over) +
             "_rx_bas" +
             "_count" +
             ".pdf")


def main():
    pass

if __name__ == "__main__":
    main()
