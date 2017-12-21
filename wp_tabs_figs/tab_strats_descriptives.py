from foolbox.api import *
from wp_tabs_figs.wp_settings import settings

"""Generate table with descriptives for a selection of trading strategies (rx
bas-adjusted returns)
"""

if __name__ == "__main__":

    # Set the output path, input data and sample
    out_path = data_path + settings["fig_folder"]
    input_dataset = settings["fx_data"]
    start_date = settings["sample_start"]
    end_date = settings["sample_end"]

    # Set up the parameters of trading strategies
    holding_range = [2, 5, 10, 15]
    threshold_range = [5, 10, 15, 20]
    scale_to = 10  # rescale returns to 'scale_to' holding period

    # Policy expectations keyword arguments
    pol_exp_args = {"avg_impl_over": 5,
                    "avg_refrce_over": 5,
                    "bday_reindex": True}

    # Import the FX data
    data = pd.read_pickle(data_path+input_dataset)

    # Import the all fixing times for the dollar index
    data_usd = pd.read_pickle(data_path+"fx_by_tz_sp_fixed.p")

# Get the individual currenices, spot rates:
spot_mid = data["spot_mid"][start_date:end_date].drop(["dkk", "jpy", "nok"],
                                                      axis=1)
spot_bid = data["spot_bid"][start_date:end_date].drop(["dkk", "jpy", "nok"],
                                                      axis=1)
spot_ask = data["spot_ask"][start_date:end_date].drop(["dkk", "jpy", "nok"],
                                                      axis=1)
swap_ask = data["tnswap_ask"][start_date:end_date].drop(["dkk", "jpy", "nok"],
                                                        axis=1)
swap_ask = remove_outliers(swap_ask, 50)
swap_bid = data["tnswap_bid"][start_date:end_date].drop(["dkk", "jpy", "nok"],
                                                        axis=1)
swap_bid = remove_outliers(swap_bid, 50)

# Construct a pre-set fixing time dollar index
spot_mid_us = data_usd["spot_mid"].loc[:, :, settings["usd_fixing_time"]]\
    .drop(["dkk"], axis=1)[start_date:end_date]
spot_bid_us = data_usd["spot_bid"].loc[:, :, settings["usd_fixing_time"]]\
    .drop(["dkk"], axis=1)[start_date:end_date]
spot_ask_us = data_usd["spot_ask"].loc[:, :, settings["usd_fixing_time"]]\
    .drop(["dkk"], axis=1)[start_date:end_date]
swap_ask_us = data_usd["tnswap_ask"].loc[:, :, settings["usd_fixing_time"]]\
    .drop(["dkk"], axis=1)[start_date:end_date]
swap_ask_us = remove_outliers(swap_ask_us, 50)
swap_bid_us = data_usd["tnswap_bid"].loc[:, :, settings["usd_fixing_time"]]\
    .drop(["dkk"], axis=1)[start_date:end_date]
swap_bid_us = remove_outliers(swap_bid_us, 50)

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