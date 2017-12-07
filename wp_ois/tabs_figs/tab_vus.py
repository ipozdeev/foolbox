import pandas as pd
from pandas.tseries.offsets import DateOffset
import numpy as np
import pickle
from foolbox.api import PolicyExpectation, set_credentials
from foolbox.wp_ois.wp_settings import central_banks_start_dates, end_date, \
    cb_fx_map, fx_cb_map


def vus_table(data_path, currencies, proxy_rate_pickle,
              e_proxy_rate_pickle, lags=None):
    """

    Returns
    -------

    """
    # functions to transform rates (smooth)
    map_proxy_rate = lambda x: x.rolling(5, min_periods=1).mean().shift(1)
    map_expected_rate = lambda x: x.rolling(5, min_periods=1).mean().shift(1)

    # lags
    if lags is None:
        lags = np.arange(1, 11)

    # loop over currencies
    vuss = dict()

    for c in currencies:
        start_dt = central_banks_start_dates[fx_cb_map[c]]
        pe = PolicyExpectation.from_pickles(data_path, c, start_dt,
            proxy_rate_pickle=proxy_rate_pickle,
            e_proxy_rate_pickle=e_proxy_rate_pickle,
            meetings_pickle="ois_project_events.p",
            ffill=True)

        # first valid date
        s_dt_meet = pe.meetings.first_valid_index()
        s_dt_rate = pe.expected_proxy_rate.first_valid_index()

        print("{}: {:d} meetings since {}, implied rates since {}".format(
            c, len(pe.meetings), s_dt_meet, s_dt_rate))

        # loop over lags, calculate vus
        this_vus = pd.Series(index=lags)

        for p in lags:
            this_vus.loc[p] = pe.get_vus(lag=p,
                map_proxy_rate=map_proxy_rate,
                map_expected_rate=map_expected_rate)

        vuss[c] = this_vus

    res = pd.DataFrame(vuss)

    return res


if __name__ == "__main__":

    path_to_data = set_credentials.set_path("research_data/fx_and_events/")
    currencies = ['cad', 'aud', 'usd', 'nzd', 'chf', 'eur', 'sek', 'gbp']
    proxy_rate_pickle = "libor_spliced_2000_2007_d.p"
    e_proxy_rate_pickle = "implied_rates_from_1m_ois.p"

    vuss = vus_table(path_to_data, currencies, proxy_rate_pickle,
        e_proxy_rate_pickle)

    path_to_latex = set_credentials.set_path(
        "ois/tex/tabs/", which="local")
    # vuss.round(4).to_latex(path_to_latex + "tab_vus_from_libor.tex")

    # # plus fff --------------------------------------------------------------
    # s_dt_ois = pe_ois.rate_expectation.dropna().index[0]
    # pe_fff = PolicyExpectation.from_pickles(data_path, c,
    #     impl_rates_pickle="implied_rates_ffut.p")
    # pe_fff.rate_expectation = pe_fff.rate_expectation.loc[s_dt_ois:end_date]
    # r_until_fff = OIS.from_iso(c,
    #     maturity=DateOffset(months=1)).get_rates_until(this_on, this_evt,
    #         method="a_average")
    #
    # this_vus = pd.Series(index=lags)
    #
    # for p in lags:
    #     this_vus.loc[p] = pe_fff.get_vus(lag=p, ref_rate=r_until_fff)
    #
    # vuss.update({"usd (ois)": vuss["usd"]})
    # vuss.pop("usd")
    #
    # vuss_out = pd.DataFrame.from_dict(vuss)
    #
    # vuss_out.loc[:, "usd (fff)"] = this_vus
    #
    # vuss_out.columns = [p.upper() for p in vuss_out.columns]
    #
    # out_path = set_credentials.set_path("../projects/ois/tex/tabs/",
    #     which="local")
    # vuss_out.to_latex(buf=out_path+"tab_vus.tex",
    #     float_format="{:5.4f}".format,
    #     column_format="l"+"W"*len(vuss_out.columns))

    # with open(data_path + "overnight_rates.p", mode='wb') as hangar:
    #     pickle.dump(on_rates, hangar)
