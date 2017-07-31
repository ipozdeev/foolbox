from foolbox.api import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
import matplotlib.lines as mlines
from wp_tabs_figs.wp_settings import settings
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
rx = rx.shift(1)

# Reformat lag and threshold to be consistent with backtest functions
holding_range = np.arange(lag, lag+1, 1)
threshold_range = np.arange(threshold, threshold+1, 1)

perfect_consistent =\
    pe_perfect_foresight_strat(rx, holding_range, data_path, True,
                               smooth_burn, True)["aggr"]
forecasted =\
    pe_backtest(rx, holding_range, threshold_range, data_path, avg_impl_over,
                avg_refrce_over, True)["aggr"]

# Get the corresponding descriptives, mind averaging instead of taking a sum
descr_fcst = taf.descriptives(
    pe_backtest(rx, holding_range, threshold_range, data_path, avg_impl_over,
                avg_refrce_over, False)["aggr"], 1)

# Figure 3: plot OIS-availability consistent perfect foresight and real strat
fig3, ax = plt.subplots(figsize=(8.4, 11.7/3))
# Concatenate the data first
to_plot = pd.concat([perfect_consistent.dropna().cumsum(),
                    forecasted.dropna().cumsum()], axis=1).ffill().fillna(0)
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
ax.set_ylabel("cumulative return", visible=True)
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

fig3.tight_layout()
fig3.savefig(out_path +
             "ueber_fcast_lag{:d}_thresh{:d}_ai{:d}_ar{:d}".\
             format(lag, int(threshold*100), avg_impl_over, avg_refrce_over) +
             "_spot" +
             "_time" +
             ".pdf")


# Figure 4: forecasted on the event axis
fig4, ax = plt.subplots(figsize=(8.4, 11.7/3))
pd.DataFrame(forecasted.dropna().values).cumsum().\
    plot(ax=ax, color='k', linewidth=1.5, linestyle="-")
# Rotate the axis labels
plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

# Use time-indexed scale use the locators from settings
ax.xaxis.set_major_locator(MultipleLocator(50))
ax.xaxis.set_minor_locator(MultipleLocator(25))

# Polish the layout
ax.grid(which="both", alpha=0.33, linestyle=":")
ax.set_xlabel("event number", visible=True)
ax.set_ylabel("cumulative return", visible=True)
ax.legend_.remove()

fig4.tight_layout()
fig4.savefig(out_path +
             "ueber_fcast_lag{:d}_thresh{:d}_ai{:d}_ar{:d}".\
             format(lag, int(threshold*100), avg_impl_over, avg_refrce_over) +
             "_spot" +
             "_count" +
             ".pdf")
