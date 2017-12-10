import pandas as pd
from foolbox.api import *
import matplotlib
matplotlib.interactive(True)  # plot after every command
from matplotlib import dates as mdates, ticker
from foolbox.utils import *
from wp_ois.wp_settings import *
from linear_models import PureOls
import statsmodels.api as sm

if __name__ == "__main__":

    # Settings
    currencies = ["aud", "cad", "chf", "eur", "gbp", "jpy", "nzd", "sek",
                  "usd"]
    ois_maturity = ["1m"]  # which ois rx is forecast

    # Select regressors
    regressors = ["ted", "vix", "msci", "libor_ois"]
    # regressors = ["ted", "vix", "msci", "term_5_2", "term_7_5", "term_10_7",
    #               "libor_ois"]
    # regressors = ["ted", "vix", "msci"]

    # Minmum datapoints to produce first out-of-sample forecast from the
    # expanding window regression
    min_periods = 252

    # Import and prepare data -------------------------------------------------
    ois_rx = data_path + "ois_rx_w_day_count.p"
    ois_all_maturities = data_path + "ois_all_maturities_bloomberg.p"
    libor = data_path + "libor_spliced_2000_2007_d.p"
    msci_ted_vix = data_path + "msci_ted_vix_d.p"
    yields = data_path + "bond_yields_all_maturities_bloomberg.p"

    with open(ois_rx, mode="rb") as hangar:
        ois_rx = pickle.load(hangar)

    with open(ois_all_maturities, mode="rb") as hut:
        ois_data = pickle.load(hut)

    with open(libor, mode="rb") as hangar:
        libor = pickle.load(hangar)

    with open(msci_ted_vix, mode="rb") as hangar:
        msci_ted_vix = pickle.load(hangar)

    with open(yields, mode="rb") as hangar:
        yields = pickle.load(hangar)

    # Select the ois rx to be forecast (in basis points)
    ois_rx = {
        curr: this_rx.loc[ois_start_dates[curr]:end_date, ois_maturity] * 100
        for curr, this_rx in ois_rx.items()
        }

    # Compute stock returns, get vix and ted spread
    stocks = (np.log(msci_ted_vix["msci_pi"]).diff() * 100).loc[:end_date, :]
    vix = msci_ted_vix["ted_vix"].loc[:end_date, "vix"]
    ted = msci_ted_vix["ted_vix"].loc[:end_date, "ted"]

    # Compute 1-month Libor-OIS spreads
    ois_1m = [ois_data[curr]["1m"][ois_start_dates[curr]:end_date]
              for curr in currencies]
    ois_1m = pd.concat(ois_1m, axis=1)
    ois_1m.columns = currencies
    libor_ois_spr = libor["1m"].loc[:end_date, currencies] - ois_1m

    # Compute term spreads
    term_spr = dict()
    for curr in currencies:

        this_spr = pd.DataFrame(columns=["term_5_2", "term_7_5", "term_10_7"])

        this_spr["term_5_2"] = yields[curr]["5y"] - yields[curr]["2y"]
        this_spr["term_7_5"] = yields[curr]["7y"] - yields[curr]["5y"]
        this_spr["term_10_7"] = yields[curr]["10y"] - yields[curr]["7y"]

        # this_spr = pd.DataFrame(columns=["term_10_2"])
        # this_spr["term_10_2"] = yields[curr]["10y"] - yields[curr]["2y"]

        term_spr[curr] = this_spr[ois_start_dates[curr]:end_date]

    # # Resample to monthly frequency
    # ois_rx = {curr: rx_daily.resample("M").first()
    #           for curr, rx_daily in ois_rx.items()}
    # stocks = stocks.resample("M").sum()  # monthly return
    # vix = vix.resample("M").last()       # other data is end-of-month
    # ted = ted.resample("M").last()
    # libor_ois_spr = libor_ois_spr.resample("M").last()
    # term_spr = {curr: term_spr_daily.resample("M").last()
    #             for curr, term_spr_daily in term_spr.items()}

    # -------------------------------------------------------------------------

    # Forecasting regressions--------------------------------------------------

    y_hats_and_y = dict()  # to store the forecast values

    for curr in currencies:
        # Get ois rx as a dependent variable
        this_y = ois_rx[curr]

        # Construct the matrix of regressors
        this_X = pd.concat([ted, vix, stocks[curr], term_spr[curr],
                            libor_ois_spr[curr]], join="outer", axis=1)
        this_X.columns = ["ted", "vix", "msci", "term_5_2", "term_7_5",
                          "term_10_7", "libor_ois"]
        this_X = this_X.loc[:, regressors].shift(1)

        # Ensure availability of observations
        this_y, this_X = this_y.dropna().align(this_X.dropna(), join="inner",
                                               axis=0)

        # Output dataframes for predicted values and exp-window params
        y_hat = pd.DataFrame(index=this_y.index, columns=this_y.columns)
        exp_param = pd.DataFrame(index=this_X.index,
                                 columns=["const"] + list(this_X.columns))

        # Set the forecast sample, leaving min_periods for initial estimation
        forecast_sample = this_y.index[min_periods:]

        # Run expanding window OLS
        for t in forecast_sample:
            # Select the data
            tmp_y = this_y.loc[:t, :]
            tmp_X = this_X.loc[:t, :]

            # Construtc and fit the model
            mod = PureOls(tmp_y, tmp_X, add_constant=True)
            mod.fit()

            # Store coefficient estimates
            exp_param.loc[t, :] = mod.coef.squeeze()

        # Get the current period's predicted values, mind that X was lagged
        y_hat = sm.add_constant(this_X.shift(-1)) \
            .mul(exp_param, axis=0).sum(axis=1).shift(1)

        # Store predicted and the corresponding realized values
        y = this_y.squeeze()
        y, y_hat = y.align(y_hat.dropna(), join="inner")  # set the same sample
        tmp_out = pd.concat([y_hat, y], axis=1)
        tmp_out.columns = ["y_hat", "y"]
        y_hats_and_y[curr] = tmp_out

    # -------------------------------------------------------------------------

    # Plot results ------------------------------------------------------------
    # Construct data for plotting
    data_to_plot = dict()
    for curr, data in y_hats_and_y.items():
        # Compute cumulative rMSPE against forecast value in percent
        rmspe_fcst = \
            ((data["y_hat"] - data["y"]).pow(2).cumsum()).pow(0.5) / 100

        # And against a naive benchmark
        rmspe_naive = (data["y"].pow(2).cumsum()).pow(0.5) / 100

        # Aggregate into a dataframe
        tmp_rmspe = pd.concat([rmspe_fcst, rmspe_naive], axis=1)
        tmp_rmspe.columns = ["fcst", "naive"]
        data_to_plot[curr] = tmp_rmspe

    # Plot the stuff
    fig, ax = plt.subplots(nrows=3, ncols=3, sharex=True, figsize=(8.4, 8.4))

    # Vectorize the ax matrix for looping
    flat_ax = ax.flatten()

    # Set x-limits to the earlies and latest data poits over entire dataset
    x_limits = \
        (min([data_to_plot[curr].index[0] for curr in data_to_plot.keys()]),
         max([data_to_plot[curr].index[-1] for curr in data_to_plot.keys()]))

    # Initialize the plot counter
    count = 0

    # Start plotting
    for curr in currencies:

        # Shortcuts to ax and data
        this_ax = flat_ax[count]
        this_data = data_to_plot[curr]

        # Plot data
        this_data["fcst"].plot(ax=this_ax, x_compat=True, color=new_blue,
                               linewidth=1.5, label="model forecast")
        this_data["naive"].plot(ax=this_ax, x_compat=True, color=new_red,
                                linewidth=1.5, label="zero-mean benchmark")

        # Set the same x-axis for all subplots. Sharex is fucked.
        this_ax.set_xlim(x_limits[0], x_limits[1])

        # Maximum y-limit is the maximum value in the data plus 10%
        y_limit = max(this_data.iloc[-1, :]) * 1.1
        this_ax.set_ylim(0, y_limit)

        # Make the plot look nicer
        this_ax.xaxis.set_major_locator(mdates.YearLocator(4))
        this_ax.xaxis.set_minor_locator(mdates.YearLocator(1))
        this_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.setp(this_ax.get_xticklabels(), rotation=0, ha="center")

        this_ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        this_ax.grid(axis='y', which="major", alpha=0.66)
        this_ax.grid(axis='x', which="major", alpha=0.66)
        this_ax.grid(axis='x', which="minor", alpha=0.33)

        # Set the plot title
        this_ax.set_title(curr)

        # Place legend on the first plot
        if count == 0:
            this_ax.legend(loc="lower right", fontsize=9, frameon=False)

        # Update the counter
        count += 1

    # Label the y-axis
    fig.text(0.0, 0.5, "cumulative root mspe in percent",
             va='center', rotation='vertical')

    # Save the figure
    fig.tight_layout(h_pad=0.5, pad=1.05)
    fig.savefig(data_path + "wp_figures_limbo/" + "forecast_vs_zero_mean.pdf")
