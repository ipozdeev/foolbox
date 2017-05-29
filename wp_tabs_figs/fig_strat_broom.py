from foolbox.api import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
from wp_tabs_figs.wp_settings import settings

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

# Import the FX data, prepare dollar index
with open(data_path+input_dataset, mode="rb") as fname:
    data = pickle.load(fname)

# Lag returns to ensure pre-announcements
rx = data["spot_ret"][start_date:end_date].drop(["dkk"], axis=1).shift(1)
rx["usd"] = -1 * rx.mean(axis=1)      # construct the dollar index
rx = rx.drop(["jpy", "nok"], axis=1)  # no ois data for these gentlemen

# Run the backtests
btest = pe_backtest(rx, holding_range, threshold_range, data_path,
                    avg_impl_over=avg_impl_over,
                    avg_refrce_over=avg_refrce_over)
aggr = btest["aggr"]
cumret = aggr.dropna(how="all").replace(np.nan, 0).cumsum()

# Plot the results and save 'em
fig = broomstick_plot(cumret)
fig.savefig(out_path + "broomstick_plot.pdf")
