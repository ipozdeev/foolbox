from foolbox.api import *
import seaborn as sns

# Import the FX data, prepare dollar index
with open(data_path+"data_wmr_dev_d.p", mode="rb") as fname:
    data = pickle.load(fname)

# Lag returns to ensure pre-announcements
rx = data["spot_ret"].drop(["dkk"], axis=1).shift(1)
rx["usd"] = -1 * rx.mean(axis=1).shift(-1)  # not the case for the dollar tho
rx = rx.drop(["jpy", "nok"], axis=1)        # no ois data for these gentlemen

# Set up the backtest parameters
lag_expect_range = np.arange(2, 16, 3)
lag_bench = 5
threshold_range = np.arange(0.05, 0.31, 0.05)

# Start the backtest
backtest_sharpe = pd.DataFrame(index=threshold_range, columns=lag_expect_range)
backtest_tstat = pd.DataFrame(index=threshold_range, columns=lag_expect_range)

for lag_expect in lag_expect_range:
    # Open position the next day after expectations are formed
    holding_period = lag_expect - 1
    for threshold in threshold_range:

        pooled_signals = list()
        for curr in rx.columns:
            tmp_pe = PolicyExpectation.from_pickles(data_path, curr)
            tmp_fcast = tmp_pe.forecast_policy_change(lag=lag_expect,
                                                      threshold=threshold,
                                                      bench_lag=lag_bench)
            pooled_signals.append(tmp_fcast)

        pooled_signals = pd.concat(pooled_signals, join="outer", axis=1)

        ueber = poco.multiple_timing(rx.rolling(holding_period).sum(),
                                     pooled_signals.replace(0, np.nan),
                                     xs_avg=True)

        tmp_descr = taf.descriptives(ueber, 261/holding_period)
        backtest_sharpe.loc[threshold, lag_expect] =\
            tmp_descr.ix["sharpe"][0]
        backtest_tstat.loc[threshold, lag_expect] =\
            tmp_descr.ix["mean"][0] / tmp_descr.ix["se_mean"][0]

        print("TH:", threshold, "HOLD:", holding_period, "\n",
              tmp_descr, "\n")

backtest_sharpe = backtest_sharpe[sorted(backtest_sharpe.columns)].astype("float")
backtest_tstat = backtest_tstat[sorted(backtest_tstat.columns)].astype("float")

fig1, ax = plt.subplots(figsize=(12, 8))
plt.setp(ax.get_xticklabels(), rotation=90, fontsize=10)
plt.setp(ax.get_yticklabels(), rotation=90, fontsize=10)
sns.heatmap(backtest_sharpe, ax=ax, annot=True,
            center=0.67, annot_kws={"size": 10})
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.ylabel("threshold")
plt.xlabel("holding period")
plt.title("All events, OIS, Sharpe ratios")
fig1.savefig(data_path+"backtest_sharpe.pdf")

fig2, ax = plt.subplots(figsize=(12, 8))
plt.setp(ax.get_xticklabels(), rotation=90, fontsize=10)
plt.setp(ax.get_yticklabels(), rotation=90, fontsize=10)
plt.yticks(rotation=0)
sns.heatmap(backtest_tstat, ax=ax, annot=True,
            center=1.67, annot_kws={"size": 10})
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.ylabel("threshold")
plt.xlabel("holding period")
plt.title("All events, OIS, t-statistics")
fig2.savefig(data_path+"backtest_t_stats.pdf")


meet = tmp_pe.meetings.copy()
meet[meet>0] = 1
meet[meet<0]= -1
t = (meet - pooled_signals.cad)



#==============================================================================
# DISAGGREGATED DATA BY CURRENCY
#==============================================================================
from foolbox.api import *
import itertools as itools

ix = pd.IndexSlice

# Import the FX data, prepare dollar index
with open(data_path+"data_wmr_dev_d.p", mode="rb") as fname:
    data = pickle.load(fname)

# Lag returns to ensure pre-announcements
rx = data["spot_ret"].drop(["dkk"], axis=1).shift(1)
rx["usd"] = -1 * rx.mean(axis=1).shift(-1)  # not the case for the dollar tho
rx = rx.drop(["jpy", "nok"], axis=1)        # no ois data for these gentlemen

# Set up the backtest parameters
lag_expect_range = np.arange(2, 17, 1)
lag_bench = 2
threshold_range = np.arange(1, 26, 1)

# Set up the output structure
out = dict()
out["aggr"] = dict()
out["disaggr"] = dict()

combos = list(itools.product(lag_expect_range-1, threshold_range))

cols = pd.MultiIndex.from_tuples(combos, names=["holding", "threshold"])
aggr = pd.DataFrame(index=rx["2000":].index, columns=cols)


for lag_expect in lag_expect_range:

    # Open position the next day after expectations are formed
    holding_period = lag_expect - 1
    out["disaggr"][str(holding_period)] = dict()

    for threshold in threshold_range:

        pooled_signals = list()
        for curr in rx.columns:
            tmp_pe = PolicyExpectation.from_pickles(data_path, curr)
            tmp_fcast = tmp_pe.forecast_policy_change(lag=lag_expect,
                                                      threshold=threshold/100,
                                                      avg_impl_over=2,
                                                      avg_refrce_over=lag_bench)
            pooled_signals.append(tmp_fcast)

        pooled_signals = pd.concat(pooled_signals, join="outer", axis=1)

        ueber = poco.multiple_timing(rx.rolling(holding_period).sum(),
                                     pooled_signals.replace(0, np.nan),
                                     xs_avg=False)
        ueber_avg = ueber.mean(axis=1)

        print("TH:", threshold, "HOLD:", holding_period, "\n")

        out["disaggr"][str(holding_period)][str(threshold)] = ueber
        aggr.loc[:, ix[holding_period, threshold]] = ueber_avg

out["aggr"] = aggr



import matplotlib.pyplot as plt
import matplotlib.lines as mlines
plt.rc("font", family="serif", size=12)
gr_1 = "#8c8c8c"


def broomstick_plot(data, ci=(0.1, 0.9)):
    """Given an input array of data, produces a broomstick plot, with all
    series in gray, the mean of the series in black (along with confidence
    interval). The data are intended to be cumulative returns on a set of
    trading strategies

    Parameters
    ----------
    return_data: pd.DataFrame
        of the cumulative returns data to plot
    ci: tuple
        of floats specifying lower and upper quantiles for empirical confidence
        interval. Default is (0.1, 0.9)

    Returns
    -------
    figure: matplotlib.pyplot.plot
        with plotted data


    """

    # Check the input data
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Nope, the input should be a DataFrame")

    # Drop absent observations if any
    data = data.dropna()

    # Get the mean and confidence bounds
    cb_u = data.quantile(ci[1], axis=1)
    cb_l = data.quantile(ci[0], axis=1)
    avg = data.mean(axis=1)
    stats = pd.concat([avg, cb_u, cb_l], axis=1)
    stats.columns = ["mean", "cb_u", "cb_l"]

    # Start plotting
    fig, ax = plt.subplots(figsize=(8.4, 11.7/3))
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    # Grid
    ax.grid(which="both", alpha=0.33, linestyle=":")
    ax.plot(data, color=gr_1, lw=0.75, alpha=0.25)

    ax.plot(stats["mean"], color="k", lw=2)
    ax.plot(stats[["cb_l", "cb_u"]], color="k", lw=2, linestyle="--")
    #ax.legend_.remove()

    # Construct lines for the custom legend
    solid_line = mlines.Line2D([], [], color='black', linestyle="-",
                               lw=2, label="Mean")

    ci_label = "{}th and {}th percentiles"\
        .format(int(ci[0]*100), int(ci[1]*100))
    dashed_line = mlines.Line2D([], [], color='black', linestyle="--",
                                lw=2, label=ci_label)
    ax.legend(handles=[solid_line, dashed_line], loc="upper left")

    return fig



ix = pd.IndexSlice
data = aggr.loc[:,  ix[:, :]].dropna(how="all").replace(np.nan, 0).cumsum()
fig = broomstick_plot(data)

fig.savefig(data_path+"parameter_uncertainty.pdf", bbox_inches="tight")

from foolbox.api import *

# Import the FX data, prepare dollar index
with open(data_path+"data_wmr_dev_d.p", mode="rb") as fname:
    data = pickle.load(fname)

# Lag returns to ensure pre-announcements
rx = data["spot_ret"].drop(["dkk"], axis=1).shift(1)
rx["usd"] = -1 * rx.mean(axis=1).shift(-1)  # not the case for the dollar tho
rx = rx.drop(["jpy", "nok"], axis=1)        # no ois data for these gentlemen

# Set up the backtest parameters
holding_range = np.arange(1, 4, 1)
threshold_range = np.arange(7, 14, 1)

test_curr = "gbp"

btest = pe_backtest(rx[["usd"]], holding_range, threshold_range, data_path,
                    avg_impl_over=5, avg_refrce_over=5)

btest2 = pe_perfect_foresight_strat(rx[["usd"]], holding_range, data_path, False,
                                    smooth_burn=5)

aggr = btest["aggr"]
aggr2 = btest2["aggr"]
cumret = aggr.dropna(how="all").replace(np.nan, 0).cumsum()
cumret2 = aggr2.dropna(how="all").replace(np.nan, 0).cumsum()

fig = broomstick_plot(cumret2)
#pd.concat([cumret, cumret2], axis=1).ffill().fillna(0).plot()

fig.tight_layout()
fig.savefig(data_path +"parameter_uncertainty2.pdf")



from foolbox.api import *

# Import the FX data, prepare dollar index
with open(data_path+"fx_by_tz_d.p", mode="rb") as fname:
    data = pickle.load(fname)
with open(data_path+"data_wmr_dev_d.p", mode="rb") as fname:
    data2 = pickle.load(fname)

# Dollar factor
rx_usd = data2["spot_ret"].mean(axis=1).to_frame() * -1
rx_usd.columns = ["usd"]

# Lag returns to ensure pre-announcements
rx = data["spot_ret"].drop(["dkk", "jpy", "nok"], axis=1).shift(1)
rx = pd.concat([rx, rx_usd], axis=1)

# Set up the backtest parameters
holding_range = np.arange(1, 15, 1)
threshold_range = np.arange(1, 25, 1)

test_curr = "usd"
#rx = rx[[test_curr]]

btest = pe_backtest(rx[["usd"]], holding_range, threshold_range, data_path,
                    avg_impl_over=5, avg_refrce_over=5)

btest2 = pe_perfect_foresight_strat(rx[["usd"]], holding_range, data_path, False,
                                    smooth_burn=5)

aggr = btest["aggr"]
aggr2 = btest2["aggr"]
cumret = aggr.dropna(how="all").replace(np.nan, 0).cumsum()
cumret2 = aggr2.dropna(how="all").replace(np.nan, 0).cumsum()


fig = broomstick_plot(cumret)
fig2 = broomstick_plot(cumret2)

