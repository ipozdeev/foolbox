from foolbox.finance import PolicyExpectation
from foolbox.api import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
import matplotlib.lines as mlines
from foolbox.wp_tabs_figs.wp_settings import settings
from foolbox.utils import *

"""Plots returns to strategy based on monetary ploicy action forecasts along
with prediction-data-availability-consistent prefect foresight.
"""
# Set the output path, input data and sample
out_path = data_path + settings["fig_folder"]
input_dataset = settings["fx_data"]
start_date = settings["sample_start"]
end_date = settings["sample_end"]

# Set up the parameters of trading strategies
lag = 10        # actually the holding period,
threshold = 10  # threshold in bps

# Forecast, and forecast consistency parameters
avg_impl_over = 5    # smooth implied rate
avg_refrce_over = 5  # smooth reference rate
smooth_burn = 5      # discard number of periods corresponding to smoothing for
                     # the forecast-consistent perfect foresight

# matplotlib settings -------------------------------------------------------
# font, colors
plt.rc("font", family="serif", size=12)
# locators
minor_locator = mdates.YearLocator()
major_locator = mdates.YearLocator(2)

# Import the FX data
with open(data_path+input_dataset, mode="rb") as fname:
    data = pickle.load(fname)

# Get the individual currenices
spot_mid = data["spot_mid"][start_date:end_date].drop(["dkk", "jpy", "nok"],
                                                      axis=1)
(spot_mid, ) = align_and_fillna((spot_mid, ), "B", method="ffill")
rx = np.log(spot_mid/spot_mid.shift(1))

# Import the all fixing times for the dollar index
with open(data_path+"fx_by_tz_d.p", mode="rb") as fname:
    data_all_fix = pickle.load(fname)

# Construct a pre-set fixing time dollar index
data_usd = data_all_fix["spot_mid"].loc[:, :, settings["usd_fixing_time"]]\
    .drop(["dkk"], axis=1)
(data_usd, ) = align_and_fillna((data_usd, ), "B", method="ffill")
rx_usd = -np.log(data_usd/data_usd.shift(1))[start_date:end_date].mean(axis=1)

# Add it to the data
rx["usd"] = rx_usd

# Lag to ensure trading before the announcements
rx = rx.rolling(lag).sum().shift(1)#.drop(["usd"], axis=1)

# Reformat lag and threshold to be consistent with backtest functions
holding_range = np.arange(lag, lag+1, 1)
threshold_range = np.arange(threshold, threshold+1, 1)


events = []
for curr in rx.columns:
    tmp_pe = PolicyExpectation.from_pickles(data_path, curr)
    this_signal = tmp_pe.meetings.loc[:, "rate_change"]
    this_signal.name = curr
    events.append(this_signal)

events = pd.concat(events, axis=1)


# For the predominant case of multiple currencies pool the signals
pooled_signals = list()
lag_expect = lag + 2
# Get the policy forecast for each currency
for curr in rx.columns:
    tmp_pe = PolicyExpectation.from_pickles(data_path, curr)
    tmp_fcast =\
        tmp_pe.forecast_policy_change(lag=lag_expect,
                                      threshold=threshold/100,
                                      avg_impl_over=avg_impl_over,
                                      avg_refrce_over=avg_refrce_over,
                                      bday_reindex=True)
    # Append the signals
    pooled_signals.append(tmp_fcast[0])
pooled_signals = pd.concat(pooled_signals, axis=1)


pooled_signals = pooled_signals.where(events != 0)
#pooled_signals = pooled_signals.where((pooled_signals > 0) & (events < 0))

rx, pooled_signals = rx.align(pooled_signals, axis=0)

ix = pd.IndexSlice
df = pd.concat([rx.stack()*100, pooled_signals.stack()],
               axis=1).dropna().loc[ix[:, "eur"], :]
df.columns = ["ret", "rate_diff"]

taf.ts_ap_tests(df[["ret"]], df[["rate_diff"]], 1)

def rsp(df, window):
    return np.sign(df.rolling(window).mean()) * \
           (df.rolling(window).std())*np.abs(df.rolling(window).cov(df.mean(axis=1)))
w = 22
w2 = 22
sig = rsp(rx, w) #- rsp(rx, w2).shift(w)

# go = multiple_timing(rx, sig.shift(1))
go = poco.rank_sort_adv(rx, sig.shift(1), 2, 1)
go = poco.get_factor_portfolios(go, True)
go.cumsum().plot()
taf.descriptives(go*100, 261)


import pandas as pd
import numpy as np
import matplotlib
from matplotlib import dates as mdates, ticker
from wp_ois.wp_settings import *
from foolbox.api import data_path

import pickle
ois_rx = data_path + "olsen_d.p"
lol = pd.read_pickle(ois_rx)

with open(ois_rx, mode="rb") as hangar:
    ois_rx = pickle.load(hangar)

ix = ois_rx["usd"]["1m"]["2014":"2016"].index
T = len(ix)

spot = pd.DataFrame(np.random.standard_normal((T, 1)).cumsum(),
                    columns=["spot"], index=ix)
stock_chf = pd.DataFrame(np.random.standard_normal((T, 1)).cumsum(),
                    columns=["stock_chf"], index=ix) + 1000
stock_usd = stock_chf.sub(spot["spot"], axis=0)
stock_usd.columns = ["stock_usd"]

this_data = pd.concat([spot, stock_chf, stock_usd], axis=1)

stock_data = this_data[["stock_usd", "stock_chf"]]
stock_data = stock_data.div(stock_data.iloc[0, :], axis=1)

this_title = "MSCI Switzerland in CHF and USD"
label_spot = "USDCHF"
label_stock_chf = "MSCI SWI in CHF"
label_stock_usd = "MSCI SWI in USD"

figure, ax = plt.subplots(2, sharex=True, figsize=(8.27,10.0))

ax_stocks = ax[0]
ax_spot = ax[1]

this_data["spot"].plot(ax=ax_spot, x_compat=True, color=new_blue,
                       linewidth=1.5, label=label_spot)
stock_data["stock_chf"].plot(ax=ax_stocks, x_compat=True, color=new_red,
                           linewidth=1.5, label=label_stock_chf)
stock_data["stock_usd"].plot(ax=ax_stocks, x_compat=True, color="k",
                           linewidth=1.5, label=label_stock_usd)

# Set the same x-axis for all subplots. Sharex is fucked.
#this_ax.set_xlim(x_limits[0], x_limits[1])

# Maximum y-limit is the maximum value in the data plus 10%
#y_limit = max(this_data.iloc[-1, :]) * 1.1
#this_ax.set_ylim(0, y_limit)

# Make the plot look nicer
# this_ax.xaxis.set_major_locator(mdates.YearLocator(1))
# this_ax.xaxis.set_minor_locator(mdates.MonthLocator((1,2)))
# this_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
# this_ax.xaxis.set_minor_formatter(mdates.DateFormatter('%m'))
plt.setp(this_ax.get_xticklabels(), rotation=0, ha="center")

# this_ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
# this_ax.grid(axis='y', which="major", alpha=0.66)
# this_ax.grid(axis='x', which="major", alpha=0.66)
# this_ax.grid(axis='x', which="minor", alpha=0.33)

# Set the plot title
this_ax.set_title(this_title)

this_ax.legend(loc="lower left", fontsize=12, frameon=False)


