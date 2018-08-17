from foolbox.api import *
from foolbox.wp_ois.wp_settings import ois_start_dates, end_date
from itertools import combinations_with_replacement, product
from foolbox.visuals import broomstick_plot
import seaborn as sns
from utils import align_and_fillna
from foolbox.wp_tabs_figs.wp_settings import settings

if __name__ == "__main__":

    # Test for combinations of these maturities. e.g. 1Y-6M, 1Y-1M, 6M-3M...
    test_maturities = ["1W", "2W", "1M", "3M", "6M", "9M", "1Y"]

    test_maturities = [x.lower() for x in test_maturities]

    test_currencies = ['aud', 'cad', 'chf', 'eur', 'gbp', 'jpy', 'nzd', 'sek']
    # test_currencaies = ["usd"]
    # Get all combinations of maturities
    maturity_combos = list(combinations_with_replacement(test_maturities, 2))

    # Drop repetitions like 1M-1M, 6M-6M, etc.
    for position, value in enumerate(maturity_combos):
        if value[0] == value[1]:
            maturity_combos.pop(position)

    # Lookback periods and smoothing parameter
    signal_lag = np.arange(2, 22, 1)
    signal_smooth = 5  # take rolling mean of rates over this period

    # Import the data
    fx_data = pd.read_pickle(data_path+"daily_rx.p")
    # with open(data_path+"daily_rx.p", mode="rb") as hunger:
    #     fx_data = pickle.load(hunger)

    ois_data = pd.read_pickle(data_path+"ois_bloomi_1w_30y.p")
    # with open(data_path + "ois_bloomberg.p", mode="rb") as halupa:
    #     ois_data = pickle.load(halupa)

    ois_data = align_and_fillna(ois_data, method="ffill")

    # with open(data_path+"data_dev_d.p", mode="rb") as fname:
    #     stock_data = pickle.load(fname)["msci_ret"]

    stock_data = pd.read_pickle(data_path+"data_dev_d.p")

    # Select currencies
    rx = fx_data["rx"]
    rx["usd"] = -rx.mean(axis=1)
    test_rx = rx[test_currencies]


    # cols = test_rx.columns
    # test_rx = pd.concat([test_rx.usd for col in test_rx.columns], axis=1)
    # test_rx.columns = cols
    # test_rx = test_rx[test_currencies]

    # # Unhedged stocks
    # test_rx = stock_data[test_currencies]

    # Hedged stocks
    # test_rx = stock_data[test_currencies] - rx[test_currencies] + \
    #     fx_data["spot"][test_currencies]

    # test_rx = pd.concat([stock_data.usd for curr in test_currencies], axis=1)
    # test_rx.columns = test_currencies

    test_rx = test_rx#["2004":"2010"]

    # Multiindex df with combos of maturity differentials and lookbacks
    mix = product(maturity_combos, signal_lag)  # brut force product of tuples
    mix = [item[0] + (item[1],) for item in mix]  # unpack inner maturity tuple

    # Create the multiindex
    mix = pd.MultiIndex.from_tuples(mix).sort_values()
    mix.names = ["lower maturity", "upper maturity", "lookback"]

    # Preallocate output
    backtest = pd.DataFrame(index=rx.index, columns=mix)

    # Loop over combination of lower, higher maturity, and lookback
    for mat_low, mat_high, lookback in backtest.columns:

        # Signals, based on differential between OIS of different maturities
        signals = pd.DataFrame(index=rx.index, columns=test_currencies)

        for curr in test_currencies:
            # For example 6-month minus 1 month OIS fixed rates
            # signals[curr] = \
            #     ois_data[curr][mat_high][ois_start_dates[curr]:end_date] - \
            #     ois_data[curr][mat_low][ois_start_dates[curr]:end_date]

            signals[curr] = \
                ois_data[mat_high.lower()][curr][ois_start_dates[curr]:end_date] - \
                ois_data[mat_low.lower()][curr][ois_start_dates[curr]:end_date]
            # signals[curr] = signals[curr] - \
            #     (ois_data["usd"][mat_high][ois_start_dates[curr]:end_date] -
            #      ois_data["usd"][mat_low][ois_start_dates[curr]:end_date])

            # other_currs = list(set(test_currencies).difference(curr))
            # tmp_other = \
            #     pd.concat(
            #         [ois_data[k][mat_high][ois_start_dates[curr]:end_date]
            #          for k in other_currs], axis=1).mean(axis=1) - \
            #     pd.concat(
            #         [ois_data[k][mat_low][ois_start_dates[curr]:end_date]
            #          for k in other_currs], axis=1).mean(axis=1)
            #
            # signals[curr] -= tmp_other

            # signals[curr] = \
            #     ois_data["usd"][mat_high][ois_start_dates[curr]:end_date] - \
            #     ois_data["usd"][mat_low][ois_start_dates[curr]:end_date]

            # if curr == "usd":
            #     signals[curr] = \
            #         ois_data["usd"][mat_high][ois_start_dates[curr]:end_date] - \
            #         ois_data["usd"][mat_low][ois_start_dates[curr]:end_date]
            #     other_currs = list(set(test_currencies).difference(curr))
            #     tmp_other = \
            #         pd.concat(
            #             [ois_data[k][mat_high][ois_start_dates[curr]:end_date]
            #              for k in other_currs], axis=1).mean(axis=1) - \
            #         pd.concat(
            #             [ois_data[k][mat_low][ois_start_dates[curr]:end_date]
            #              for k in other_currs], axis=1).mean(axis=1)
            #
            #     signals[curr] -= tmp_other



        # Lag and roll
        signals = signals.rolling(signal_smooth).mean().shift(lookback)

        # Populate the output
        backtest.loc[:, (mat_low, mat_high, lookback)] = \
            multiple_timing(test_rx, signals, xs_avg=True).squeeze()

        # test_rx = test_rx.resample("M").sum()
        # signals = signals.resample("M").last().shift(1)
        # backtest.loc[:, (mat_low, mat_high, lookback)] = \
        #     poco.get_factor_portfolios(
        #         poco.rank_sort_adv(
        #             test_rx.reindex(signals.index), signals, 3),
        #         hml=True).hml
        # print(mat_low, mat_high, lookback)

    # Plot the strategy
    fig = broomstick_plot(backtest.cumsum())
    ix = pd.IndexSlice
    # backtest.loc[:, ix["1m", "9m", 5]].cumsum().plot()

    # Plot the cumulative return heatmap
    # Arrange mean return: maturity differential vs lookback
    avg_ret = backtest.dropna(how="all").mean().unstack() * 100 * 252
    avg_ret.index = avg_ret.index.tolist()

    # Create a new order -.-
    new_order = list(combinations_with_replacement(test_maturities[::-1], 2))

    for position, value in enumerate(new_order):
        # Drop duplicates
        if value[0] == value[1]:
            new_order.pop(position)

    # Swap maturities
    new_order = [(value[1], value[0]) for value in new_order]

    # Flip the new order ^.^
    new_order = new_order[::-1]

    # Finally, arrange the data
    avg_ret = avg_ret.loc[new_order]

    # Plot the map
    fig1, ax = plt.subplots(figsize=(12, 8))
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=12)
    plt.setp(ax.get_yticklabels(), rotation=90, fontsize=12)
    sns.heatmap(avg_ret.round(2), ax=ax, annot=True,
                center=avg_ret.unstack().mean(),
                annot_kws={"size": 10, "color": "black"})
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.ylabel("maturity differential")
    plt.xlabel("lookback period")
    plt.title("Mean return in percent p.a.")

    # Plot the average return across all
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    ax2.plot(avg_ret.mean(axis=1).values)
    ax2.set_xticklabels(avg_ret.index)
    ax2.set_ylabel("mean return in percent p.a")
    ax2.set_xlabel("maturity differential")
    plt.title("average return in percent p.a. by maturity differential over "
              "all lookback horizons")
    plt.show()
