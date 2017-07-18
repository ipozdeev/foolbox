from foolbox.api import *
from foolbox.fxtrading import FXTrading
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
import matplotlib.lines as mlines
from wp_tabs_figs.wp_settings import settings
from foolbox.utils import *

"""Plots returns to strategy based on monetary policy action forecasts along
with prediction-data-availability-consistent prefect foresight. Adjust returns
for rollover yields and bid-ask spreads, set leverage to zero
"""

# Set the output path, input data and sample
out_path = data_path + settings["fig_folder"]
input_dataset = settings["fx_data"]
start_date = settings["sample_start"]
end_date = settings["sample_end"]

# Choose fixing time
fixing_time = "LON"

currencies = ["aud", "cad", "chf", "eur", "gbp", "nzd", "sek"]

# Policy expectations keyword arguments
pol_exp_args = {"avg_impl_over": settings["avg_impl_over"],
                "avg_refrce_over": settings["avg_refrce_over"],
                "bday_reindex": settings["bday_reindex"]}

# Set up the parameters of trading strategies
holding_period = settings["base_holding_h"]
threshold = settings["base_threshold"] * 100
lag_expect = holding_period + 2  # forecast rate one day in advance of trading
smooth_burn = 5  # discard number of periods corresponding to smoothing for
                 # the forecast-consistent perfect foresight

# matplotlib settings -------------------------------------------------------
# font, colors
plt.rc("font", family="serif", size=12)
# locators
minor_locator = mdates.YearLocator(1)
major_locator = mdates.YearLocator(2)

# Fetch the data
with open(data_path + "fx_by_tz_sp_fixed.p", mode="rb") as fname:
    data_all_tz = pickle.load(fname)

# Get the data for the fixing time, drop DKK
spot_mid = data_all_tz["spot_mid"].loc[:, :, fixing_time]\
    .drop(["dkk"], axis=1)[start_date:end_date]
spot_bid = data_all_tz["spot_bid"].loc[:, :, fixing_time]\
    .drop(["dkk"], axis=1)[start_date:end_date]
spot_ask = data_all_tz["spot_ask"].loc[:, :, fixing_time]\
    .drop(["dkk"], axis=1)[start_date:end_date]
swap_ask = data_all_tz["tnswap_ask"].loc[:, :, fixing_time]\
    .drop(["dkk"], axis=1)[start_date:end_date]
swap_ask = remove_outliers(swap_ask, 50)
swap_bid = data_all_tz["tnswap_bid"].loc[:, :, fixing_time]\
    .drop(["dkk"], axis=1)[start_date:end_date]
swap_bid = remove_outliers(swap_bid, 50)

# Align and ffill the data, first for tz-aligned countries
(spot_mid, spot_bid, spot_ask, swap_bid, swap_ask) =\
    align_and_fillna((spot_mid, spot_bid, spot_ask, swap_bid, swap_ask),
                     "B", method="ffill")

# Then for the new one. Organize data first
prices = pd.Panel.from_dict({"bid": spot_bid,
                             "ask": spot_ask},
                            orient="items")
swap_points = pd.Panel.from_dict({"bid": swap_bid,
                                  "ask": swap_ask},
                                 orient="items")

# Get signals for the individual currencies
signals = get_pe_signals(currencies, lag_expect,
                         threshold, data_path, fomc=False,
                         **pol_exp_args)[start_date:end_date]

# Add NOK and JPY to signals as an empty columns
signals["nok"] = np.nan
signals["jpy"] = np.nan

signals_us = get_pe_signals(currencies + ["jpy", "nok"], lag_expect,
                            threshold, data_path, fomc=True,
                            **pol_exp_args)[start_date:end_date]

fx_trading = FXTrading(prices=prices, swap_points=swap_points,
                       signals=signals,
                       settings={"holding_period": holding_period,
                                 "blackout": 1})
# Add the US signals, keeping the local ones dominant
fx_trading.add_junior_signals(signals_us)


nav_forecast = fx_trading.backtest().to_frame()
nav_forecast.columns = ["fcst"]
print("Forecast:", nav_forecast.ffill().iloc[-1])


# Get the perfect foresight signals
pfct_signals = list()
for curr in currencies + ["usd"]:
    # Get the predicted change in policy rate
    tmp_pe = PolicyExpectation.from_pickles(data_path, curr)

    # Get the signals
    this_signal = tmp_pe.meetings.loc[:, "rate_change"]
    this_signal.name = curr

    # Get the first forecast date available, leave enough data
    # to make a forecast, control for averaging
    first_date = tmp_pe.rate_expectation.dropna() \
        .iloc[[lag_expect + smooth_burn - 1]].index[0]
    pfct_signals.append(this_signal.loc[first_date:])

pfct_signals = pd.concat(pfct_signals, axis=1)[start_date:end_date]
pfct_signals = np.sign(pfct_signals)

# Construct the fomc perfect signals
pfct_signals_us = pd.concat(
    [-1 * pfct_signals["usd"]]*len(currencies+["nok", "jpy"]), axis=1
    ).dropna()
pfct_signals_us.columns = currencies+["nok", "jpy"]

# Drop the fomc from indiviual currencies' signals
pfct_signals = pfct_signals.drop(["usd"], axis=1)

# Do the traiding with perfect signals
fx_trading_pfct = FXTrading(prices=prices, swap_points=swap_points,
                            signals=pfct_signals,
                            settings={"holding_period": holding_period,
                                      "blackout": 1})
# Add the US signals, keeping the local ones dominant
fx_trading.add_junior_signals(pfct_signals_us)


nav_pfct = fx_trading_pfct.backtest().to_frame()
nav_pfct.columns = ["pfct"]
print("Forecast:", nav_pfct.ffill().iloc[-1])


# Get the descriptives for the forecast strategy
descr_fcst = taf.descriptives(np.log(nav_forecast).diff(), 10)

# Plot the results
# Figure 1: plot OIS-availability consistent perfect foresight and real strat
fig1, ax = plt.subplots()
# Concatenate the data first
to_plot = pd.concat([nav_pfct, nav_forecast], axis=1).ffill().fillna(1) - 1
to_plot.columns = ["pfct", "fcst"]

# Plot it
to_plot[["pfct"]].plot(ax=ax, color='k', linewidth=1.5, linestyle="--")
to_plot[["fcst"]].plot(ax=ax, color='k', linewidth=1.5, linestyle="-")

# Rotate the axis labels
plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

# Use time-indexed scale use the locators from settings
#ax.xaxis.set_major_locator(major_locator)
#ax.xaxis.set_minor_locator(minor_locator)

# Polish the layout
ax.grid(which="both", alpha=0.33, linestyle=":")
ax.set_xlabel("date", visible=True)
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
             format(holding_period, int(threshold*100),
                    pol_exp_args["avg_impl_over"],
                    pol_exp_args["avg_refrce_over"]) +
             "_rx_bas_leverage_control" +
             "_time" +
             ".pdf")

# Figure 2: forecasted on the event axis
fig2, ax = plt.subplots()
count_idx = signals.replace(0, np.nan).dropna(how="all").index
data_to_plot = (nav_forecast.ffill().fillna(1) - 1).loc[count_idx]
data_to_plot.index = range(len(data_to_plot.index))
data_to_plot.plot(ax=ax, color='k', linewidth=1.5, linestyle="-")
# Rotate the axis labels
plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

# Use time-indexed scale use the locators from settings
ax.xaxis.set_major_locator(MultipleLocator(50))
ax.xaxis.set_minor_locator(MultipleLocator(25))

# Polish the layout
ax.grid(which="both", alpha=0.33, linestyle=":")
ax.set_xlabel("event number", visible=True)
ax.legend_.remove()

fig2.tight_layout()
fig2.savefig(out_path +
             "ueber_fcast_lag{:d}_thresh{:d}_ai{:d}_ar{:d}".\
             format(holding_period, int(threshold*100),
                    pol_exp_args["avg_impl_over"],
                    pol_exp_args["avg_refrce_over"]) +
             "_rx_bas_leverage_control" +
             "_count" +
             ".pdf")


if __name__ == "__main__":
    pass
