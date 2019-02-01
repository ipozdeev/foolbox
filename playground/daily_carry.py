# from foolbox.api import *
# from wp_tabs_figs.wp_settings import settings
# from foolbox.utils import *
#
#
# # Set the output path, input data and sample
# out_path = data_path + settings["fig_folder"]
# input_dataset = settings["fx_data"]
# start_date = settings["sample_start"]
# end_date = settings["sample_end"]
# fixing_time = "LON"
#
# # Set the test currencies
# currencies = ["aud", "cad", "chf", "eur", "gbp", "jpy", "nzd", "nok", "sek"]
#
#
# # Import the all fixing times for the dollar index
# with open(data_path+"fx_by_tz_sp_fixed.p", mode="rb") as fname:
#     data = pickle.load(fname)
#
# # Get the data for the fixing time, drop DKK
#
# spot_mid = data["spot_mid"].loc[:, start_date:end_date, fixing_time]\
#     .drop(["dkk"], axis=1)
# spot_bid = data["spot_bid"].loc[:, start_date:end_date, fixing_time]\
#     .drop(["dkk"], axis=1)
# spot_ask = data["spot_ask"].loc[:, start_date:end_date, fixing_time]\
#     .drop(["dkk"], axis=1)
# swap_ask = data["tnswap_ask"].loc[:, start_date:end_date, fixing_time]\
#     .drop(["dkk"], axis=1)
# swap_bid = data["tnswap_bid"].loc[:, start_date:end_date, fixing_time]\
#     .drop(["dkk"], axis=1)
# swap_ask = remove_outliers(swap_ask, 50)
# swap_bid = remove_outliers(swap_bid, 50)
#
# # Align and ffill the data
# (spot_mid, spot_bid, spot_ask, swap_bid, swap_ask) =\
#     align_and_fillna((spot_mid, spot_bid, spot_ask, swap_bid, swap_ask),
#                      "B", method="ffill")
#
# swap_mid = 0.5 * (swap_ask + swap_bid)
#
#
# rx = np.log(spot_mid) - np.log((spot_mid+swap_mid).shift(1))
# rx = rx[currencies]
# f_d = np.log(spot_mid) - np.log((spot_mid+swap_mid))
#
# tt = poco.rank_sort_adv(rx,
#                         f_d.rolling(22).sum().shift(1),
#                         3)
# ss = poco.get_factor_portfolios(tt, hml=True)
#
# daily_data = dict()
# daily_data["rx"] = rx
# daily_data["hml"] = ss#[["hml"]]
# daily_data["spot"] = np.log(spot_mid/spot_mid.shift(1))
# with open (data_path+"daily_rx.p", mode="wb") as hunger:
#     pickle.dump(daily_data, hunger)
#
#
# tt = poco.rank_sort_adv(rx.resample("M").sum(),
#                         f_d.resample("M").sum().shift(1),
#                         3)
# ss2 = poco.get_factor_portfolios(tt, hml=True)


import pandas as pd
from foolbox.api import *

from foolbox.wp_tabs_figs.wp_settings import settings
from foolbox.utils import *

from pandas.tseries.offsets import DateOffset
from matplotlib import dates as mdates, ticker

import seaborn as sns
sns.reset_orig()
sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})
sns.set_palette("deep")

# Set the output path, input data and sample
out_path = data_path + settings["fig_folder"]
s_dt = pd.to_datetime(settings["sample_start"])
e_dt = pd.to_datetime(settings["sample_end"])

n_portf = 3
#
# data ----------------------------------------------------------------------
# daily rx
with open(data_path+"daily_rx.p", mode="rb") as fname:
    data_rx = pickle.load(fname)

# rx_d = data_rx["rx"].drop(settings["drop_currencies"], axis=1, errors="ignore")
#
# # daily fx
# with open(data_path+"fx_by_tz_sp_fixed.p", mode="rb") as fname:
#     data_fx = pickle.load(fname)
#
# s_d = data_fx["spot_bid"].add(data_fx["spot_ask"])/2
# s_d = s_d.loc[:,:,"NYC"].drop(settings["drop_currencies"], axis=1)
# s_d = np.log(s_d).diff()
#
# swap_d = data_fx["tnswap_bid"].add(data_fx["tnswap_ask"])/2
# swap_d = swap_d.loc[:,:,"NYC"].drop(settings["drop_currencies"], axis=1)
#
# # align
# rx_d, s_d, swap_d = align_and_fillna(
#     (rx_d, s_d, swap_d), reindex_freq='B', method="ffill")
#
# # carry
# carry_sig = -1*swap_d.rolling(22).mean()
# carry_pfs = poco.rank_sort(rx_d, carry_sig.shift(1), n_portf)
# carry = poco.get_factor_portfolios(carry_pfs, hml=True)
# carry["hml"].sum()

carry = data_rx["hml"]
rx_d = data_rx["rx"]

# events
with open(data_path+"events.p", mode="rb") as fname:
    events_data = pickle.load(fname)

events = events_data["joint_cbs"]

rx_d = rx_d.loc[s_dt:e_dt]
events = events.loc[s_dt:e_dt]
#s_d = s_d.loc[s_dt:e_dt]
#swap_d = swap_d.loc[s_dt:e_dt]

# ---------------------------------------------------------------------------
if __name__ == "__main__":

    this_cb = "usd"
    this_evt = events[this_cb]

    # align
    this_evt = this_evt.reindex(index=rx_d.index)

    # individual currencies
    new_s_d = interevent_quantiles(events=this_evt, df=carry)

    lol = new_s_d.groupby(["_evt", "_q"]).sum()
    new_s_d_rest = new_s_d.reset_index()
    new_s_d_rest = new_s_d_rest.groupby(["_evt", "_q"]).last().loc[:,"index"]
    ticks = new_s_d_rest.index[::len(new_s_d_rest.index)//5]

    idx = pd.IndexSlice

    N = rx_d.shape[1]
    N3 = N//3
    plot_idx = new_s_d.where(new_s_d["_q"] == 0.0).dropna().index

    fig, ax = plt.subplots(3, sharex=True, figsize=(8.27*0.95, 11.3*0.95))

    # set lim to the very last one (lowest) subplot, since the x-axis is shared
    ax[-1].set_xlim((
        s_dt - DateOffset(months=3),
        e_dt + DateOffset(months=3)))

    # set tick location and text appearance via Locator and DateFormatter:
    #   major ticks are 2 years apart, minor - 1 year
    ax[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    ax[-1].xaxis.set_minor_locator(mdates.YearLocator(1))
    ax[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    for p in range(3):
        # p = 2
        to_plot = lol.loc[idx[:,float(p)],:].cumsum()

        # to_plot.index = plot_idx[:-1] if p > 0 else plot_idx
        to_plot.index = new_s_d_rest[idx[:,float(p)]]

        to_plot.plot(ax=ax[p], linewidth=1.5)
        handles, labels = ax[p].get_legend_handles_labels()
        ax[p].legend(handles[(p*N3):((p+1)*N3)], labels[(p*N3):((p+1)*N3)],
            loc="upper left")
        ax[p].grid(axis="both", which="major", alpha=0.33, color='k',
            linestyle="--", linewidth=0.5)
        ax[p].set_title("garch_lags"+str(p) if p > 0 else "event")

    # plt.xticks(ticks, labels=new_s_d_rest.loc[ticks].values)
    plt.setp(ax[-1].xaxis.get_majorticklabels(), rotation="horizontal",
        ha="center")
    plt.suptitle(this_cb.upper(),
        bbox=dict(facecolor="#f4f4f4", edgecolor='k'),
        x=0.20, y=0.965)

    fig.tight_layout(pad=3.0, h_pad=1.0)
    fig.savefig(out_path + "interevent_2q_carry_"+this_cb+".png", dpi=200)

    # correlations
    q1 = lol.loc[idx[:,1.0],:]
    q2 = lol.loc[idx[:,2.0],:]
    q1.index = q1.index.droplevel(1)
    q2.index = q2.index.droplevel(1)

    print(q1.corrwith(q2))
    plt.show()