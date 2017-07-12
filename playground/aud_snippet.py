from foolbox.api import *
from foolbox.fxtrading import FXTrading
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
from wp_tabs_figs.wp_settings import settings
from foolbox.trading import EventTradingStrategy
from foolbox.utils import *

# Set the output path, input data and sample
out_path = data_path + settings["fig_folder"]
input_dataset = settings["fx_data"]
start_date = settings["sample_start"]
end_date = settings["sample_end"]

start_date = "2003-07-21"
end_date = "2003-08-07"

# Set the test currency
test_currency = "aud"

# Set up the parameters of trading strategies
holding_range = np.arange(10, 11, 1)
threshold_range = np.arange(10, 11, 1)

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

# Get the individual currenices, spot rates and tom/next swap points:
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

# Align and ffill the data
(spot_mid, spot_bid, spot_ask, swap_bid, swap_ask) =\
    align_and_fillna((spot_mid, spot_bid, spot_ask, swap_bid, swap_ask),
                     "B", method="ffill")

# Organize the data into dictionaries
fx_data = {"spot_mid": spot_mid, "spot_bid": spot_bid, "spot_ask": spot_ask,
           "tnswap_bid": swap_bid, "tnswap_ask": swap_ask}

for key in fx_data.keys():
    fx_data[key] = fx_data[key][[test_currency]]


# Run the backtests, the legacy one goes first
# Run backtests separately for ex-us and dollar index
ret_old = event_trading_backtest(fx_data, holding_range, threshold_range,
                                 data_path, fomc=False, **pol_exp_args)["aggr"]


# Then for the new one. Organize data first
prices = pd.Panel.from_dict({"bid": fx_data["spot_bid"],
                             "ask": fx_data["spot_ask"]},
                            orient="items")
swap_points = pd.Panel.from_dict({"bid": fx_data["tnswap_bid"],
                                  "ask": fx_data["tnswap_ask"]},
                                 orient="items")

for holding_period in holding_range:
    # Forecast policy one day ahead of opening a position
    lag_expect = holding_period + 2

    for threshold in threshold_range:
        signals = get_pe_signals([test_currency], lag_expect,
                                 threshold, data_path, fomc=False,
                                 **pol_exp_args)
        fx_trading = FXTrading(prices=prices, swap_points=swap_points,
                               signals=signals, settings={"h": holding_period})

        # # Get the flags
        # position_flags = fx_trading.position_flags
        #
        # bool_flags = position_flags.notnull()[test_currency] * 1
        # sparse_flags = position_flags.\
        #     loc[(bool_flags.diff(-1) == -1) |
        #         (bool_flags.diff(-2) == -1) |
        #         (bool_flags.diff(-1) == -1) |
        #         (bool_flags.diff(-2) == -1) |
        #         (bool_flags == 1), :]
        #
        # sparse_position_weights = sparse_flags.divide(
        #     np.abs(sparse_flags).sum(axis=1), axis=0).fillna(0.0)
        #
        # sparse_actions = sparse_position_weights.diff().shift(-1)
        #
        # fx_trading.position_flags = sparse_flags
        # fx_trading.position_weights = sparse_position_weights
        # fx_trading.actions = sparse_actions

        nav = fx_trading.backtest()


