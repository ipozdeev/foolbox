from foolbox.api import *
from foolbox.wp_ois.wp_settings import ois_start_dates, end_date
from itertools import combinations_with_replacement, product
from foolbox.visuals import broomstick_plot
import seaborn as sns
from foolbox.wp_tabs_figs.wp_settings import settings

if __name__ == "__main__":

    # Test for combinations of these maturities. e.g. 1Y-6M, 1Y-1M, 6M-3M...
    test_maturities = ["ON", "1M", "3M", "6M", "9M", "1Y"]
    test_currencies = ['aud', 'cad', 'chf', 'eur', 'gbp', 'jpy', 'nzd', 'sek']

    # Get all combinations of maturities
    maturity_combos = list(combinations_with_replacement(test_maturities, 2))

    # Drop repetitions like 1M-1M, 6M-6M, etc.
    for position, value in enumerate(maturity_combos):
        if value[0] == value[1]:
            maturity_combos.pop(position)

    # Lookback periods and smoothing parameter
    signal_lag = np.arange(2, 44, 1)
    signal_smooth = 5  # take rolling mean of rates over this period

    # Import the data
    with open(data_path+"daily_rx.p", mode="rb") as hunger:
        rx = pickle.load(hunger)["rx"]

    with open(data_path + "ois_bloomberg.p", mode="rb") as halupa:
        ois_data = pickle.load(halupa)

    # Select currencies
    test_rx = rx[test_currencies]

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
            signals[curr] = \
                ois_data[curr][mat_high][ois_start_dates[curr]:end_date] - \
                ois_data[curr][mat_low][ois_start_dates[curr]:end_date]

        # Lag and roll
        signals = signals.rolling(signal_smooth).mean().shift(lookback)

        # Populate the output
        backtest.loc[:, (mat_low, mat_high, lookback)] = \
            multiple_timing(test_rx, signals, xs_avg=True).squeeze()

    # Plot the strategy
    fig = broomstick_plot(backtest.cumsum())

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