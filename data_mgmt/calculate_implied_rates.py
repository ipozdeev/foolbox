import pandas as pd
from pandas.tseries.offsets import DateOffset
from foolbox.api import data_path, PolicyExpectation
from foolbox.fixed_income import OIS, LIBOR
import pickle

def calculate_implied_rates(ois_data, on_data, events_data):
    """
    """
    # space for implied rates
    implied_rates = dict()

    for c in ois_data.keys():
        # c = "usd"
        print(c)

        # OIS
        ois_object = OIS.from_iso(c, maturity=DateOffset(months=1))

        # this event, to frame + rename
        this_evt = events_data.loc[:, c].dropna().to_frame("rate_change")

        # this ois
        this_ois_rate = ois_data.loc[:, c].astype(float)

        # this overnight
        this_on = on_data.loc[:, c].astype(float)

        # align
        s_dt = this_ois_rate.first_valid_index()
        e_dt = this_ois_rate.last_valid_index()
        this_ois_rate, this_on = this_ois_rate.loc[s_dt:e_dt].align(
                this_on, axis=0, join="left")

        # rates expected to prevail before meetings
        rate_on_const = this_on.rolling(5, min_periods=1).mean().shift(1)

        # # rates expected to prevail before meetings
        # rate_on_const = ois_object.get_rates_until(this_on, this_evt,
        #     method="g_average")

        pe = PolicyExpectation.from_ois(
            meetings=this_evt,
            ois_object=ois_object,
            ois_rate=this_ois_rate,
            rate_on_const=rate_on_const)

        # store
        this_ir = pe.expected_proxy_rate.copy()
        this_ir.name = c
        implied_rates[c] = this_ir

    ir = pd.DataFrame.from_dict(implied_rates)

    return ir


def calculate_libor_implied_rates(libor_1m, libor_on, events_data):
    """
    """
    # space for implied rates
    implied_rates = dict()

    for c in libor_1m.keys():
        # c = "usd"
        if c in ['jpy']:
            continue

        print(c)

        # OIS
        this_libor = LIBOR.from_iso(c, maturity=DateOffset(months=1))

        # this event, to frame + rename
        this_evt = events_data.loc[:, c].dropna().to_frame("rate_change")

        # this ois
        this_long_rate = this_libor.reindex_series(libor_1m.loc[:, c],
            method="ffill")

        # this overnight
        this_on = this_libor.reindex_series(libor_on.loc[:, c],
            method="ffill")

        # align
        this_long_rate, this_on = this_long_rate.loc[
            this_long_rate.first_valid_index():\
            this_long_rate.last_valid_index()].align(
                this_on, axis=0, join="left")

        # constant rates
        this_on_const = this_on.rolling(5, min_periods=1).mean().shift(1)

        pe = PolicyExpectation.from_forward_rates(meetings=this_evt,
            rate_object=this_libor,
            rate_long=this_long_rate,
            rate_on_const=this_on_const)

        # store
        this_ir = pe.expected_proxy_rate.copy().dropna()
        this_ir.name = c
        implied_rates[c] = this_ir

    implied_rates = pd.DataFrame.from_dict(implied_rates)

    return implied_rates


if __name__ == "__main__":
    # fetch data ------------------------------------------------------------
    # ois data
    ois_pickle = "ois_all_maturities_bloomberg.p"
    ois_data = pd.read_pickle(data_path + ois_pickle)

    ois_rate = pd.concat(
        [p.loc[:, "1m"].to_frame(c) for c, p in ois_data.items()],
        axis=1)

    ois_rate = ois_rate.drop(["jpy", "dkk"], axis=1, errors="ignore")

    # overnight rates data
    on_pickle = "on_ois_rates_bloomberg.p"

    on_rate = pd.read_pickle(data_path + on_pickle)

    # events
    events_pickle = "ois_project_events.p"
    events_data = pd.read_pickle(data_path + events_pickle)
    events = events_data["joint_cbs"]

    # libor
    libor_pickle = "libor_spliced_2000_2007_d.p"
    libor_data = pd.read_pickle(data_path + libor_pickle)
    libor_1m = libor_data["1m"]
    libor_on = libor_data["on"]

    # from ois --------------------------------------------------------------
    ir = calculate_implied_rates(ois_rate, on_rate, events)

    with open(data_path + "implied_rates_1m_ois.p", mode="wb") as h:
        pickle.dump(ir, h)

    # from libor ------------------------------------------------------------
    ir = calculate_libor_implied_rates(libor_1m, libor_on, events)

    with open(data_path + "implied_rates_1m_libor.p", mode="wb") as h:
        pickle.dump(ir, h)
