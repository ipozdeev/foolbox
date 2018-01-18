from foolbox.api import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator

"""Produces figures with perfect foresight strategy, and predicted strategy
along with prediction-data-consistent prefect foresight. With spot rates
"""
out_path = data_path + "ir_events/"
# Set up the parameters of trading strategies
# Set up the backtest parameters
lag = 10        # actually the holding period,
threshold = 10  # threshold in bps

start_date = "1990-01-01"

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

# Reformat lag and threshold to be consistent with backtest functions
holding_range = np.arange(lag, lag+1, 1)    # for both perfect foresight and irl
threshold_range = np.arange(threshold, threshold+1, 1)  # for real strat only


# Import the FX data, prepare dollar index
with open(data_path+"data_wmr_dev_d.p", mode="rb") as fname:
    data = pickle.load(fname)

# Lag returns to ensure pre-announcements
rx = data["spot_ret"][start_date:].drop(["dkk"], axis=1).shift(1)
rx["usd"] = -1 * rx.mean(axis=1).shift(-1)  # not the case for the dollar tho
rx = rx.drop(["jpy", "nok"], axis=1)        # no ois data for these gentlemen

# Generate retunrs on the strategies, perfect foresight and all events,
# perfect foresight and forecastable evenets (in terms of data availability)
# forecasted events
perfect_all =\
    pe_perfect_foresight_strat(rx, holding_range, data_path, False)["aggr"]
perfect_consistent =\
    pe_perfect_foresight_strat(rx, holding_range, data_path, True,
                               smooth_burn)["aggr"]
forecasted =\
    pe_backtest(rx, holding_range, threshold_range, data_path, avg_impl_over,
                avg_refrce_over)["aggr"]

# Get the corresponding descriptives
descr_pfct_all = taf.descriptives(perfect_all, 1)
descr_pfct_consistent = taf.descriptives(perfect_consistent, 1)
descr_fcst = taf.descriptives(forecasted, 1)

# Plot the results
# Perfect foresight all events, time axis
fig1, ax = plt.subplots(figsize=(8.4, 11.7/3))
perfect_all.dropna().cumsum().plot(ax=ax, color='k', linewidth=1.5,
                                   linestyle="-")
# Rotate the axis labels
plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

# Use time-indexed scale use the locators from settings
ax.xaxis.set_major_locator(major_locator)
ax.xaxis.set_minor_locator(minor_locator)

# Polish the layout
ax.grid(which="both", alpha=0.33, linestyle=":")
ax.set_xlabel("date", visible=True)
ax.legend_.remove()

# Add some descriptives
descr = descr_pfct_all
ax.annotate(r"$\mu={:3.2f}%$".format(descr.loc["mean"].values[0]*10000),
    xy=(0.9, 0.30), xycoords='axes fraction',
    horizontalalignment='right', fontsize=12)
ax.annotate(r"$se={:3.2f}%$".format(descr.loc["se_mean"].values[0]*10000),
    xy=(0.9, 0.20), xycoords='axes fraction',
    horizontalalignment='right', fontsize=12)
ax.annotate(r"$SR={:3.2f}%$".format(descr.loc["sharpe"].values[0]),
    xy=(0.9, 0.10), xycoords='axes fraction',
    horizontalalignment='right', fontsize=12)

fig1.tight_layout()
fig1.savefig(out_path +
             "ueber_perfect_lag{:d}".\
             format(lag) +
             "_spot" +
             "_time" +
             ".pdf")


# Figure 2: same stuff on the event axis
fig2, ax = plt.subplots(figsize=(8.4, 11.7/3))
pd.DataFrame(perfect_all.dropna().values).cumsum().\
    plot(ax=ax, color='k', linewidth=1.5, linestyle="-")
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
             "ueber_perfect_lag{:d}".\
             format(lag) +
             "_spot" +
             "_count" +
             ".pdf")


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
             format(lag, threshold, avg_impl_over, avg_refrce_over) +
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
ax.legend_.remove()

fig4.tight_layout()
fig4.savefig(out_path +
             "ueber_fcast_lag{:d}_thresh{:d}_ai{:d}_ar{:d}".\
             format(lag, threshold, avg_impl_over, avg_refrce_over) +
             "_spot" +
             "_count" +
             ".pdf")
