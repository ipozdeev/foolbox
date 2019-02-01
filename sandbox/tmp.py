import pandas as pd
import numpy as np
from foolbox.api import set_credentials as set_cred
from foolbox.api import PolicyExpectation

path_to_data = set_cred.set_path("research_data/fx_and_events/")


def get_data(path_to_data):
    """

    Parameters
    ----------
    path

    Returns
    -------

    """
    events_data = pd.read_pickle(path_to_data + "events.p")
    fomc_meetings = events_data["joint_cbs"].loc[:, "usd"].dropna()
    policy_rate = events_data["joint_cbs_lvl"].loc[:, "usd"].dropna()

    on_rates_data = pd.read_pickle(path_to_data + "overnight_rates.p")
    ff_rate = on_rates_data.loc[:, "usd"]

    ff = pd.read_pickle(path_to_data + "fed_funds_futures_settle.p")

    pe = PolicyExpectation.from_funds_futures(meetings=fomc_meetings,
                                              policy_rate=policy_rate,
                                              proxy_rate=ff_rate,
                                              funds_futures=ff)

    with pd.HDFStore(path_to_data + "fomc_expected_rate_1993_2017_d.h5",
                     mode='w') as hangar:
        hangar.put("e_rate", pe.expected_proxy_rate, format='fixed')

    return pe.expected_proxy_rate
    
    
if __name__ == "__main__":
    path = "c:/Users/Igor/Documents/HSG/hedge_funds_stock_pickers/data/new/"
    eqt_fname = "pivoted.h5"

    # data = get_data(path + eqt_fname)

    with pd.HDFStore(path + eqt_fname, mode='r') as hangar:
        # print(hangar.keys())
        idx = hangar.get("retx").index

    print(idx)
    # print(stock_r.index)

    res = get_data(path_to_data)

    print(res)