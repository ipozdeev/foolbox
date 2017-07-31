from foolbox.api import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
from wp_tabs_figs.wp_settings import settings
from foolbox.visuals import *

"""Plots returns of a universe of strategies aiming to predict policy rates
changes, for different holding horizons and threshold levels.
"""
# Set the output path, input data and sample
out_path = data_path + settings["fig_folder"]
input_dataset = settings["fx_data"]
start_date = settings["sample_start"]
end_date = settings["sample_end"]

# Set the broomstick-backtest parameters
# Set up the parameters of trading strategies
holding_range = np.arange(1, 16, 1)
threshold_range = np.arange(1, 26, 1)

# Forecast, and forecast consistency parameters
avg_impl_over = 5    # smooth implied rate
avg_refrce_over = 5  # smooth reference rate

# Import the FX data
with open(data_path+input_dataset, mode="rb") as fname:
    data = pickle.load(fname)

# Get the individual currenices
spot_mid = data["spot_mid"][start_date:end_date].drop(["dkk", "jpy", "nok"],
                                                      axis=1)
rx = np.log(spot_mid/spot_mid.shift(1))

# Import the all fixing times for the dollar index
with open(data_path+"fx_by_tz_sp_fixed.p", mode="rb") as fname:
    data_all_fix = pickle.load(fname)

# Construct a pre-set fixing time dollar index
data_usd = data_all_fix["spot_mid"].loc[:, :, settings["usd_fixing_time"]]\
    .drop(["dkk"], axis=1)
rx_usd = -np.log(data_usd/data_usd.shift(1))[start_date:end_date].mean(axis=1)

# Add it to the data
rx["usd"] = rx_usd

# Lag to ensure trading before the announcements
rx = rx.shift(1)

# Run the backtests
btest = pe_backtest(rx, holding_range, threshold_range, data_path,
                    avg_impl_over=avg_impl_over,
                    avg_refrce_over=avg_refrce_over, sum=True)
aggr = btest["aggr"]
cumret = aggr.dropna(how="all").replace(np.nan, 0).cumsum()

# Plot the results and save 'em
fig = broomstick_plot(cumret * 100)
fig.savefig(out_path + "broomstick_plot_spot.pdf")
