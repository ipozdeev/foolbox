from foolbox.api import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
from wp_tabs_figs.wp_settings import settings

with open(data_path+"overnight_rates.p", mode="rb") as fname:
    data = pickle.load(fname)

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
rx = data["spot_ret"][start_date:end_date].drop(["dkk", "jpy", "nok"],
                                                axis=1)

# Import the all fixing times for the dollar index
with open(data_path+"fx_by_tz_all_fixings_d.p", mode="rb") as fname:
    data_all_fix = pickle.load(fname)

# Construct a pre-set fixing time dollar index
data_usd = data_all_fix["spot"].loc[:, :, settings["usd_fixing_time"]]\
    .drop(["dkk"], axis=1)
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
                               smooth_burn)["aggr"]
forecasted =\
    pe_backtest(rx, holding_range, threshold_range, data_path, avg_impl_over,
                avg_refrce_over)

disaggr = forecasted["disaggr"]


# Get the approximate 10-day interest rate
fd_file = "data_wmr_dev_d.p"
with open (data_path+fd_file, mode="rb") as fname:
    fd_data = pickle.load(fname)

fwd_5 = fd_data["fwd_mid_w"]
sp = fd_data["spot_mid"]
f_disc = np.log(fwd_5/sp)/5


# Get aud spot and the corresponding swap points
sp = data_all_fix["spot"].loc["aud", :, "NYC"].to_frame()
sp.columns = ["aud"]
sp = sp["2000-11-01":"2017-03-31"]

swap_points = pd.read_excel(data_path + "fx_tn_diff_tz_1994_2017_d.xlsx",
                            sheetname="NYC", index_col=0,
                            parse_cols="A:B", skiprows=1).dropna()
swap_points = swap_points["2000-11-01":"2017-03-31"]
swap_points = swap_points.reindex(sp.index).ffill()/10000
swap_points.columns = ["aud"]

# Construct 10-days rolling rx
accrued_interest = swap_points.rolling(9).sum()
eff_fwd = sp

rx_10 = np.log(sp/(sp.shift(10) + swap_points.rolling(9).sum().shift(1)))
sp_10 = np.log(sp/sp.shift(10))


# Set up the parameters of trading strategies
lag = 10        # actually the holding period,
holding_period = lag
lag_expect = lag + 2
curr = "aud"

tmp_pe = PolicyExpectation.from_pickles(data_path, curr)
tmp_fcast =\
    tmp_pe.forecast_policy_change(lag=lag_expect,
                                  threshold=threshold/100,
                                  avg_impl_over=avg_impl_over,
                                  avg_refrce_over=avg_refrce_over)

pooled_signals = tmp_fcast.to_frame()

returns = np.log(sp/sp.shift(1))
strat3 = multiple_timing(rx_10,
                        pooled_signals.replace(0, np.nan),
                        xs_avg=False)





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
rx = data["spot_ret"][start_date:end_date].drop(["dkk", "jpy", "nok"],
                                                axis=1)

# Import the all fixing times for the dollar index
with open(data_path+"fx_by_tz_all_fixings_d.p", mode="rb") as fname:
    data_all_fix = pickle.load(fname)

# Construct a pre-set fixing time dollar index
data_usd = data_all_fix["spot"].loc[:, :, settings["usd_fixing_time"]]\
    .drop(["dkk"], axis=1)
rx_usd = -np.log(data_usd/data_usd.shift(1))[start_date:end_date].mean(axis=1)

# Add it to the data
rx["usd"] = rx_usd

# Lag to ensure trading before the announcements
rx = rx.shift(1)

# Reformat lag and threshold to be consistent with backtest functions
holding_range = np.arange(lag, lag+1, 1)

perfect_all =\
    pe_perfect_foresight_strat(rx, holding_range, data_path, False)["aggr"]

# Get the corresponding descriptives
descr_pfct_all = taf.descriptives(perfect_all, 1)

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




