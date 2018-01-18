from foolbox.api import *
from foolbox.fxtrading import FXTrading
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
import matplotlib.lines as mlines
from wp_tabs_figs.wp_settings import settings
from foolbox.utils import *


# Set the output path, input data and sample
out_path = data_path + settings["fig_folder"]
input_dataset = settings["fx_data"]
start_date = settings["sample_start"]
end_date = settings["sample_end"]

# start_date = "2006-03-01"
# end_date = "2006-04-30"

# Set the test currency
test_currency = ["aud", "cad", "chf", "eur", "gbp", "nzd", "sek", "jpy"]
test_currency = ["aud", "cad", "chf", "eur", "gbp", "nzd", "sek",]

# Set up the parameters of trading strategies
holding_range = np.arange(10, 11, 4)
threshold_range = np.arange(20, 22, 4)

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

swap_ask = remove_outliers(swap_ask, 50)
swap_bid = remove_outliers(swap_bid, 50)

# # Get the data for the fixing time, drop DKK
# fixing_time = "LON"
# spot_mid = data_all_tz["spot_mid"].loc[:, :, fixing_time]\
#     .drop(["dkk", "nok", "jpy"], axis=1)[start_date:end_date]
# spot_bid = data_all_tz["spot_bid"].loc[:, :, fixing_time]\
#     .drop(["dkk", "nok", "jpy"], axis=1)[start_date:end_date]
# spot_ask = data_all_tz["spot_ask"].loc[:, :, fixing_time]\
#     .drop(["dkk", "nok", "jpy"], axis=1)[start_date:end_date]
# swap_ask = data_all_tz["tnswap_ask"].loc[:, :, fixing_time]\
#     .drop(["dkk", "nok", "jpy"], axis=1)[start_date:end_date]
# swap_ask = remove_outliers(swap_ask, 50)
# swap_bid = data_all_tz["tnswap_bid"].loc[:, :, fixing_time]\
#     .drop(["dkk", "nok", "jpy"], axis=1)[start_date:end_date]
# swap_bid = remove_outliers(swap_bid, 50)

# Align and ffill the data
(spot_mid, spot_bid, spot_ask, swap_bid, swap_ask) =\
    align_and_fillna((spot_mid, spot_bid, spot_ask, swap_bid, swap_ask),
                     "B", method="ffill")

# Organize the data into dictionaries
fx_data = {"spot_mid": spot_mid, "spot_bid": spot_bid, "spot_ask": spot_ask,
           "tnswap_bid": swap_bid, "tnswap_ask": swap_ask}

for key in fx_data.keys():
    fx_data[key] = fx_data[key][test_currency]


# Run the backtests, the legacy one goes first
# Run backtests separately for ex-us and dollar index
ret_old = event_trading_backtest(fx_data, holding_range, threshold_range,
                                 data_path, fomc=False, **pol_exp_args)["aggr"]

old = ret_old.replace(np.nan, 0).cumsum()


# Then for the new one. Organize data first
prices = pd.Panel.from_dict({"bid": fx_data["spot_bid"],
                             "ask": fx_data["spot_ask"]},
                            orient="items")
swap_points = pd.Panel.from_dict({"bid": fx_data["tnswap_bid"],
                                  "ask": fx_data["tnswap_ask"]},
                                 orient="items")

# The aggregated output is a multiindex
combos = list(itools.product(holding_range, threshold_range))
cols = pd.MultiIndex.from_tuples(combos, names=["holding", "threshold"])
aggr = pd.DataFrame(index=prices.major_axis,
                    columns=cols)
ix = pd.IndexSlice
for holding_period in holding_range:
    # Forecast policy one day ahead of opening a position
    lag_expect = holding_period + 2

    for threshold in threshold_range:
        signals = get_pe_signals(test_currency, lag_expect,
                                 threshold, data_path, fomc=False,
                                 **pol_exp_args)
        signals_us = get_pe_signals(test_currency, lag_expect,
                                    threshold, data_path, fomc=True,
                                    **pol_exp_args)

        fx_trading = FXTrading(prices=prices, swap_points=swap_points,
                               signals=signals,
                               settings={"holding_period": holding_period,
                                         "blackout": 1})
        # Add the US signals, keeping the local ones dominant
        #fx_trading.add_junior_signals(signals_us)

        nav = fx_trading.backtest()
        aggr.loc[:, ix[holding_period, threshold]] = nav
        print(holding_period, threshold, nav.ffill().iloc[-1])

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
        nav_list = list()
        for currency in test_currency:
            fx_trading = FXTrading(prices=prices,
                                   swap_points=swap_points,
                                   signals=signals[[currency]],
                                   settings={"holding_period": holding_period,
                                             "blackout": 1})
            tmp_nav = fx_trading.backtest()
            nav_list.append(tmp_nav)

        nav_df = pd.concat(nav_list, axis=1).ffill().replace(np.nan, 1)
        nav_df.columns = test_currency





df = pd.concat([(nav-1).ffill().replace(np.nan, 0),
                ret_old.replace(np.nan, 0).cumsum()], join="inner", axis=1)
df.columns = ["new", "old"]
df.plot()

df2 = pd.concat([(nav_df-1).sum(axis=1),
                ret_old.replace(np.nan, 0).cumsum()], join="inner", axis=1)
df2.columns = ["new", "old"]
df2.plot()

# with open(data_path + "to_del_old.p", mode="wb") as fname:
#     pickle.dump(ret_old, fname)
# with open(data_path + "to_del_new_corrected_wo_fomc.p", mode="wb") as fname:
#     pickle.dump(aggr, fname)

# from foolbox.api import *
# from foolbox.api import *
# import seaborn as sns
# from wp_tabs_figs.wp_settings import settings
#
#
# with open(data_path + "to_del_old.p", "rb") as fname:
#     old = pickle.load(fname)
# with open(data_path + "to_del_new.p", "rb") as fname:
#     new = pickle.load(fname)
#
#
# data_to_plot_old = old.replace(np.nan, 0).cumsum().iloc[[-1], :].stack()
# data_to_plot_old.index = data_to_plot_old.index.droplevel()
#
# data_to_plot_new = new.iloc[[-1], :].stack()-1
# data_to_plot_new.index = data_to_plot_new.index.droplevel()
#
# fig1, ax = plt.subplots(figsize=(12, 8))
# plt.setp(ax.get_xticklabels(), rotation=90, fontsize=12)
# plt.setp(ax.get_yticklabels(), rotation=90, fontsize=12)
# sns.heatmap(data_to_plot_new, ax=ax, annot=True, center=0.0,
#             annot_kws={"size": 10, "color": "black"})
# plt.xticks(rotation=0)
# plt.yticks(rotation=0)
# plt.ylabel("threshold")
# plt.xlabel("holding period")
# #
# # # Save the bastard
# fig1.tight_layout()
# #fig1.savefig(out_path+"heatmap_wo_fomc.pdf")


# with open(data_path + "to_del_test_good_idx.p", mode="wb") as fname:
#     pickle.dump(aggr, fname)

# with open(data_path + "to_del_test_wo_good_idx.p", mode="wb") as fname:
#     pickle.dump(aggr, fname)

# with open(data_path + "to_del_test_good_idx.p", mode="rb") as fname:
#     good_ix = pickle.load(fname)

# with open(data_path + "ip_rx_wo_fomc.p", mode="rb") as hangar:
#     new = pickle.load(hangar)
#
# from pandas.tseries.offsets import BDay
# data = new["fcast"].loc[:,start_date:,:]
#
# data_flat = data.swapaxes("items", "major_axis").to_frame(
#     filter_observations=False).T
# data_flat = pd.concat((
#     pd.DataFrame(1.0,
#         index=[data_flat.index[0]-BDay()],
#         columns=data_flat.columns),
#     data_flat), axis=0)

# lag_expect = 12
# holding_period = 10
# threshold = 10
#
# signals = get_pe_signals(test_currency, lag_expect,
#                          threshold, data_path, fomc=False,
#                          **pol_exp_args)
# signals_us = get_pe_signals(test_currency, lag_expect,
#                             threshold, data_path, fomc=True,
#                             **pol_exp_args)
#
# fx_trading = FXTrading(prices=prices, swap_points=swap_points,
#                        signals=signals,
#                        settings={"holding_period": holding_period,
#                                  "blackout": 1})
# pf1 = fx_trading.position_flags
#
# fx_trading = FXTrading(prices=prices, swap_points=swap_points,
#                        signals=signals_us,
#                        settings={"holding_period": holding_period,
#                                  "blackout": 1})
# pf2 = fx_trading.position_flags


# from visuals import *
# fig = broomstick_plot(aggr.ffill().fillna(1))
#fig.savefig(out_path + "broomstick_plot_rx_bas_us_only.pdf")


# with open(data_path + "db_rx_fomc.p", mode="rb") as fname:
#     data = pickle.load(fname)