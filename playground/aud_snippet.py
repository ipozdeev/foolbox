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


from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, \
    GoodFriday, EasterMonday

class SwedenHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday("Epiphany", month=1, day=6),
        GoodFriday,
        EasterMonday,
        Holiday("InternationalLaborDay", month=5, day=1),
        Holiday()
        ]

    , observance=nearest_workday


#
from foolbox.api import *
data_path
pe = PolicyExpectation.from_pickles(data_path,
    currency="usd", events_pickle="ois_project_events.p",
    impl_rates_pickle="implied_rates_bloomberg.p", s_dt=s_dt)

ir = pd.read_clipboard(index_col=0, parse_dates=True, header=None)
ir = ir.squeeze()
ir = ir.loc[sorted(ir.index)]
ir = ir.reindex(index=pd.date_range(ir.index[0], ir.index[-1], freq='B'),
    method="ffill")
pe.rate_expectation = ir
ir.index.isin(pe.rate_expectation.index).sum()

s_dt = "2004-01-01"

this_evt = events.loc[s_dt:, "usd"].dropna()
this_on = on_rates.loc[s_dt:, "usd"]

pe_ois.rate_expectation = pe.rate_expectation.loc[s_dt:end_date]

r_until = OIS.from_iso("usd",
    maturity=DateOffset(months=1)).get_rates_until(this_on, this_evt,
        method="g_average")

# loop over lags, calculate VUS
this_vus = pd.Series(index=lags)

for p in lags:
    this_vus.loc[p] = pe.get_vus(lag=p, ref_rate=r_until)
this_vus.to_clipboard()
both = pd.concat((pe.rate_expectation, ir), axis=1)
both.columns = ["ours", "theirs"]

three = pd.concat(
    (both.shift(10).loc[this_evt.index], events_data["fomc"].loc[:, "rate"]),
    axis=1)

both.dropna().to_clipboard()

this_evt
events_data["fomc"].loc[:, "rate"]

ir.to_clipboard()

pe_fff.rate_expectation = pe_fff.rate_expectation.loc[s_dt:]
pe_fff


# ---------------------------------------------------------------------------
# currency to drop
drop_curs = ["usd","jpy","dkk"]

# window
wa,wb,wc,wd = -10,-1,1,5
window = (wa,wb,wc,wd)

s_dt = settings["sample_start"]
e_dt = settings["sample_end"]

# data ------------------------------------------------------------------
data_path = set_credentials.gdrive_path("research_data/fx_and_events/")
out_path = set_credentials.gdrive_path("opec_meetings/tex/figs/")

# spot returns + drop currencies ----------------------------------------
with open(data_path + settings["fx_data"], mode='rb') as fname:
    fx = pickle.load(fname)
ret = np.log(fx["spot_mid"].drop(drop_curs,axis=1,errors="ignore")).diff()

# events + drop currencies ----------------------------------------------
with open(data_path + settings["events_data"], mode='rb') as fname:
    events_data = pickle.load(fname)

events = events_data["joint_cbs"].drop(drop_curs, axis=1, errors="ignore")
events = events.where(events < 0)

# data = ret["nzd"]
data = ret.copy().loc[s_dt:e_dt]
events = events.loc[s_dt:e_dt]

var_sums = pd.DataFrame(index=events.columns, columns=["var", "count"])

for c in events.columns:
    # c = "nzd"
    es = EventStudy(data=data[c],
        events=events[c].dropna(),
        window=window,
        normal_data=0.0,
        x_overlaps=True)

    evts_used = es.timeline[c].loc[:, "evt_no"].dropna().unique()

    mask = es.timeline[c].loc[:, "inter_evt"].isnull()

    # calculate variance between events
    var_btw = resample_between_events(data[c],
        events=events[c].dropna(),
        fun=lambda x: np.nanmean(x**2),
        mask=mask)

    var_sums.loc[c, "var"] = var_btw.loc[evts_used].sum().values[0]
    var_sums.loc[c, "count"] = var_btw.loc[evts_used].count().values[0]

t = 10
np.sqrt((t*(var_sums.loc[:, "var"] / var_sums.loc[:, "count"]**2)).sum() /\
    var_sums.shape[0]**2)*100*1.95
