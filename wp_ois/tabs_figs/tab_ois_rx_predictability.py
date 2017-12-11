import pandas as pd
from foolbox.api import *
from foolbox.utils import *
from wp_ois.wp_settings import *
from linear_models import PureOls
import itertools as itools



out_path = set_credentials.set_path("../ois/", which="local")
# out_path = set_credentials.set_path("", which="local")

if __name__ == "__main__":

    order = ["1w", "2w", "1m", "3m", "6m", "9m", "1y"]

    msci_ted_vix = data_path + "msci_ted_vix_d.p"
    yields = data_path + "bond_yields_all_maturities_bloomberg.p"
    ois_rx = data_path + "ois_rx_w_day_count.p"
    libor = data_path + "libor_spliced_2000_2007_d.p"
    ois_all_maturities = data_path + "ois_all_maturities_bloomberg.p"

    with open(ois_all_maturities, mode="rb") as hut:
        ois_data = pickle.load(hut)

    with open(libor, mode="rb") as hangar:
        libor = pickle.load(hangar)

    with open(msci_ted_vix, mode="rb") as hangar:
        msci_ted_vix = pickle.load(hangar)

    stocks = np.log(msci_ted_vix["msci_pi"]).diff() * 100

    vix = msci_ted_vix["ted_vix"]["vix"]
    ted = msci_ted_vix["ted_vix"]["ted"]

    with open(yields, mode="rb") as hangar:
        yields = pickle.load(hangar)

    with open(ois_rx, mode="rb") as hangar:
        ois_rx = pickle.load(hangar)

    # policy rates
    events = pd.read_pickle(data_path + "ois_project_events.p")

    # Libor-OIS spreads
    ois_1m = [ois_data[curr]["1m"] for curr in libor["1m"].columns]
    ois_1m = pd.concat(ois_1m, axis=1)
    ois_1m.columns = libor["1m"].columns
    libor_ois_spr = libor["1m"] - ois_1m


    all_currencies = ois_1m.columns.tolist()
    all_currencies = ["gbp"]
    all_ois_maturities = ["1m"]

    reindex_monthly = True

    out_coefs = pd.DataFrame(index=["const", "ted", "vix", "stocks",
                                    "term_5_2", "term_7_5", "term_10_7",
                                     "adj r2"],
                             columns=all_currencies) #"libor_ois"

    out_r2 = pd.Series(index=all_currencies)

    out_tstats = pd.DataFrame(index=out_coefs.index, columns=out_coefs.columns)

    for curr, maturity in list(itools.product(all_currencies,
                                              all_ois_maturities)):

        this_y = ois_rx[curr][[maturity]].loc[ois_start_dates[curr]:, ] \
                     .dropna() * 100
        this_y.columns = [col[::-1] for col in this_y.columns]

        term_10_7 = yields[curr]["10y"] - yields[curr]["7y"]
        term_7_5 = yields[curr]["7y"] - yields[curr]["5y"]
        term_5_2 = yields[curr]["5y"] - yields[curr]["2y"]

        this_libor_ois_spr = libor_ois_spr[curr]

        stock_ret = stocks[curr]

        this_X = pd.concat([ted, vix, stock_ret, term_5_2, term_7_5, term_10_7,
                            this_libor_ois_spr],
                           join="inner", axis=1).loc[this_y.index, :]
        x_cols = ["ted", "vix", "stocks", "term_5_2", "term_7_5",
                  "term_10_7", "libor_ois"]

        this_X.columns = x_cols

        this_y, this_X = align_and_fillna([this_y, this_X], "B", "inner",
                                          method="ffill")

        if reindex_monthly is True:
            this_y = this_y.resample("M").first()
            this_X1 = this_X[["ted", "vix", "term_5_2", "term_7_5",
                             "term_10_7", "libor_ois"]].resample("M").last()
            this_X2 = this_X[["stocks"]].resample("M").sum()

            this_X = pd.concat([this_X1, this_X2], axis=1).loc[:, x_cols]

        this_y = this_y.loc[:end_date, :]
        this_X = this_X.loc[:end_date, :].drop(["libor_ois"], axis=1)


        mod = PureOls(this_y, this_X.shift(1), add_constant=True)

        estimates = mod.get_diagnostics(HAC=True)

        out_coefs.loc[estimates.columns, curr] = estimates.loc["coef", :]
        out_coefs.loc["adj r2", curr] = estimates.loc["adj r2", "const"]
        out_tstats.loc[estimates.columns, curr] = estimates.loc["tstat", :]

        # Do the bloody expanding OLS
        # from foolbox.econometrics import expanding_ols
        # exp_params, resids = expanding_ols(this_y, this_X, 20, constant=True)
        min_periods = 24

        y_hat = pd.DataFrame(index=this_y.index, columns=this_y.columns)
        exp_param = pd.DataFrame(index=this_X.index, columns=estimates.columns)

        forecast_sample = this_y.index[min_periods:]
        for t in forecast_sample:
            tmp_y = this_y.loc[:t, :]
            tmp_X = this_X.loc[:t, :].shift(1)

            mod = PureOls(tmp_y, tmp_X, add_constant=True)
            y_hat.loc[t, :] = mod.get_yhat().iloc[-1]
            exp_param.loc[t, :] = mod.coef.squeeze()

        import statsmodels.api as sm
        y_hat = sm.add_constant(this_X).mul(exp_param,
                                            axis=0).sum(axis=1).shift(1)

        lol = pd.concat([(this_y.sub(y_hat, axis=0)).pow(2), (this_y).pow(2)],
                        axis=1).dropna()

        lol.columns = ["model", "naive"]

        out = (this_y.sub(y_hat, axis=0)).pow(2).mean()/(this_y).pow(2).mean()

        out_r2.loc[curr] = 1-out[0]

        out = (this_y.sub(y_hat, axis=0)).pow(2).expanding().mean()/\
              ((this_y).pow(2)).expanding().mean()

        (1-out).plot()
        plt.show()


    # out_path = set_credentials.set_path("/ois/", which="local")
    #
    # to_better_latex(
    #     df_coef=out_coefs,
    #     df_tstat=out_tstats,
    #     fmt_coef="{:3.2f}", fmt_tstat="{:3.2f}",
    #     buf=out_path+"tex/tabs/ois_predictability_monthly_h_1.tex",
    #     column_format="l"+"W"*len(out_coefs.columns))






