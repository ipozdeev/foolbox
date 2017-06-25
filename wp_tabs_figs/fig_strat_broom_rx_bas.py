from foolbox.api import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
from wp_tabs_figs.wp_settings import settings
from foolbox.trading import EventTradingStrategy
from foolbox.utils import *

"""Broomstick plots for swap points- and bid-ask spreads-adjusted returns
"""
# Set the output path, input data and sample
out_path = data_path + settings["fig_folder"]
input_dataset = settings["fx_data"]
start_date = settings["sample_start"]
end_date = settings["sample_end"]

# Set up the parameters of trading strategies
holding_range = np.arange(1, 16, 5)
threshold_range = np.arange(1, 26, 5)

# Policy expectations keyword arguments
pol_exp_args = {"avg_impl_over": 5,
                "avg_refrce_over": 5,
                "bday_reindex": True}

# Import the FX data
with open(data_path+input_dataset, mode="rb") as fname:
    data = pickle.load(fname)

# Import the all fixing times for the dollar index
with open(data_path+"fx_by_tz_sp_fixed.p", mode="rb") as fname:
    data_usd = pickle.load(fname)

# Get the individual currenices, spot rates:
spot_mid = data["spot_mid"][start_date:end_date].drop(["dkk", "jpy", "nok"],
                                                      axis=1)
spot_bid = data["spot_bid"][start_date:end_date].drop(["dkk", "jpy", "nok"],
                                                      axis=1)
spot_ask = data["spot_ask"][start_date:end_date].drop(["dkk", "jpy", "nok"],
                                                      axis=1)
swap_ask = data["tnswap_ask"][start_date:end_date].drop(["dkk", "jpy", "nok"],
                                                        axis=1)
swap_bid = data["tnswap_bid"][start_date:end_date].drop(["dkk", "jpy", "nok"],
                                                        axis=1)

# Construct a pre-set fixing time dollar index
spot_mid_us = data_usd["spot_mid"].loc[:, :, settings["usd_fixing_time"]]\
    .drop(["dkk"], axis=1)[start_date:end_date]
spot_bid_us = data_usd["spot_bid"].loc[:, :, settings["usd_fixing_time"]]\
    .drop(["dkk"], axis=1)[start_date:end_date]
spot_ask_us = data_usd["spot_ask"].loc[:, :, settings["usd_fixing_time"]]\
    .drop(["dkk"], axis=1)[start_date:end_date]
swap_ask_us = data_usd["tnswap_ask"].loc[:, :, settings["usd_fixing_time"]]\
    .drop(["dkk"], axis=1)[start_date:end_date]
swap_bid_us = data_usd["tnswap_bid"].loc[:, :, settings["usd_fixing_time"]]\
    .drop(["dkk"], axis=1)[start_date:end_date]

# Align and ffill the data, first for tz-aligned countries
(spot_mid, spot_bid, spot_ask, swap_bid, swap_ask) =\
    align_and_fillna((spot_mid, spot_bid, spot_ask, swap_bid, swap_ask),
                     "B", method="ffill")
# Now for the dollar index
(spot_mid_us, spot_bid_us, spot_ask_us, swap_bid_us, swap_ask_us) =\
    align_and_fillna((spot_mid_us, spot_bid_us, spot_ask_us,
                      swap_bid_us, swap_ask_us),
                     "B", method="ffill")

# Organize the data into dictionaries
fx_data = {"spot_mid": spot_mid, "spot_bid": spot_bid, "spot_ask": spot_ask,
           "tnswap_bid": swap_bid, "tnswap_ask": swap_ask}

fx_data_us = {"spot_mid": spot_mid_us, "spot_bid": spot_bid_us,
              "spot_ask": spot_ask_us,
              "tnswap_bid": swap_bid_us, "tnswap_ask": swap_ask_us}

# Run backtests separately for ex-us and dollar index
ret_xus = event_trading_backtest(fx_data, holding_range, threshold_range,
                                 data_path, fomc=False, **pol_exp_args)["aggr"]
ret_us = event_trading_backtest(fx_data, holding_range, threshold_range,
                                data_path, fomc=True, **pol_exp_args)["aggr"]

# Get the all-events returns
ret_all = ret_xus.add(ret_us, axis=1)\
    .fillna(value=ret_xus).fillna(value=ret_us)[start_date:end_date]
cumret_all = ret_all.dropna(how="all").replace(np.nan, 0).cumsum()

# Get the us returns
cumret_us = ret_us[start_date:end_date]\
    .dropna(how="all").replace(np.nan, 0).cumsum()

# Plot the results and save 'em
fig_all = broomstick_plot(cumret_all)
fig_all.savefig(out_path + "broomstick_plot_rx_bas.pdf")

fig_us = broomstick_plot(cumret_us)
fig_us.savefig(out_path + "broomstick_plot_rx_bas_usd.pdf")







