import pandas as pd
from pandas.tseries.offsets import DateOffset
from foolbox.finance import PolicyExpectation
from foolbox.data_mgmt import set_credentials as set_cred
from foolbox.fixed_income import OIS, LIBOR
from foolbox.utils import apply_between_events
import pickle

data_path = set_cred.set_path("research_data/fx_and_events/")


def calculate_ois_implied_rates(ois_df, overnight_df, events_df, maturity,
                                map_proxy_rate="rolling", **kwargs):
    """Calculate OIS-implied rates for many currencies for a given maturity.

    Forward-fills ois_df to make sure forecasts are available on all
    business days, even if from previous days.

    Parameters
    ----------
    ois_df : pandas.DataFrame
        with currencies for column names
    overnight_df : pandas.DataFrame
    events_df : pandas.DataFrame
    maturity : str
    map_proxy_rate : str
        "rolling", "expanding_since_previous", "last"

    Returns
    -------
    res : pandas.DataFrame
    """
    assert isinstance(ois_df, pd.DataFrame)

    # from possible strings to pandas.DateOffset ----------------------------
    maturity_map = {
        "1w": DateOffset(weeks=1),
        "2w": DateOffset(weeks=2),
        "1m": DateOffset(months=1),
        "2m": DateOffset(months=2),
    }

    if isinstance(maturity, str):
        maturity = maturity_map[maturity.lower()]

    # default proxy rate map is to take the last value
    if map_proxy_rate is None:
        map_proxy_rate = lambda x: x

    # calculate implied rates -----------------------------------------------
    # space
    implied_rates = list()

    # loop over currencies
    for c in ois_df.columns:
        # c = "usd"
        print(c)

        # OIS
        ois_object = OIS.from_iso(c, maturity=maturity)

        # this event, to frame + rename
        this_evt = events_df.loc[:, c].dropna().to_frame("rate_change")

        # this ois
        this_ois_rate = ois_df.loc[:, c]

        # this overnight
        this_on_rate = overnight_df.loc[:, c]

        # align
        this_ois_rate, this_on_rate = this_ois_rate.dropna().align(
            this_on_rate.dropna(), axis=0, join="inner")

        # reindex by corresponding business day + forward-fill
        this_ois_rate = ois_object.reindex_series(
            this_ois_rate, method="ffill")
        this_on_rate = ois_object.reindex_series(this_on_rate, method="ffill")

        # rates expected to prevail before meetings
        if map_proxy_rate == "rolling":
            proxy_rate = this_on_rate.rolling(**kwargs, min_periods=1).mean()
        elif map_proxy_rate == "expanding_since_previous":
            proxy_rate = apply_between_events(
                this_on_rate,
                this_evt,
                func=lambda x: x.expanding(min_periods=1).mean(),
                lag=ois_object.fixing_lag)
        elif map_proxy_rate == "previous":
            proxy_rate = this_on_rate.copy()

        # shift by the lag at which the rates are published
        proxy_rate = proxy_rate.shift(ois_object.new_rate_lag)

        pe = PolicyExpectation.from_ois(
            meetings=this_evt,
            ois_object=ois_object,
            ois_rate=this_ois_rate,
            rate_on_const=proxy_rate)

        # rename + store
        this_ir = pe.expected_proxy_rate.copy()
        implied_rates.append(this_ir.rename(c))

    # concat, to float (just in case), sort
    ir = pd.concat(implied_rates, axis=1)
    ir = ir.astype(float).loc[:, sorted(ir.columns)]

    return ir


def calculate_libor_implied_rates(libor_1m, libor_on, events_data):
    """
    """
    # space for implied rates
    implied_rates = dict()

    for c in libor_1m.columns:
        # c = "usd"
        if c in ['jpy']:
            continue

        print(c)

        # LIBOR
        this_libor = LIBOR.from_iso(c, maturity=DateOffset(months=1))

        # this event, to frame + rename
        this_evt = events_data.loc[:, c].dropna().to_frame("rate_change")

        # this libor rate
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


def calculate_libor_implied_rates_smart(libor_1m, libor_on, events_data):
    """
    """
    # space for implied rates
    implied_rates = dict()

    for c in libor_1m.columns:
        # c = "usd"
        if c in ['jpy']:
            continue

        print(c)

        # LIBOR
        this_libor = LIBOR.from_iso(c, maturity=DateOffset(months=1))
        this_ois = OIS.from_iso(c, maturity=DateOffset(months=1))
        this_ois.day_count_fix_dnm = this_libor.day_count_dnm
        this_ois.day_count_float_dnm = this_libor.day_count_dnm
        this_ois.day_count_fix_num = this_libor.day_count_num
        this_ois.day_count_float_num = this_libor.day_count_num
        this_ois.value_dt_lag = this_libor.value_dt_lag
        this_ois.day_roll = this_libor.day_roll
        this_ois.fixing_lag = 1
        this_ois.new_rate_lag = 0

        # this event, to frame + rename
        this_evt = events_data.loc[:, c].dropna().to_frame("rate_change")

        # this ois
        this_ois_rate = libor_1m.loc[:, c].astype(float)

        # this overnight
        this_on = libor_on.loc[:, c].astype(float)

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
            ois_object=this_ois,
            ois_rate=this_ois_rate,
            rate_on_const=rate_on_const)

        # store
        this_ir = pe.expected_proxy_rate.copy()
        this_ir.name = c
        implied_rates[c] = this_ir

    ir = pd.DataFrame.from_dict(implied_rates)

    return ir


if __name__ == "__main__":

    # fetch data ------------------------------------------------------------
    # ois data
    maturity = "1m"
    ois_pkl = "ois_merged_4.p"

    ois_out_pkl = "implied_rates_from_" + maturity + "_ois_4_since.p"
    on_pkl = "overnight_rates.p"
    map_proxy_rate = lambda x: x.rolling(5).mean()

    ois_data = pd.read_pickle(data_path + ois_pkl)

    ois_rate = ois_data[maturity].drop(["jpy", "nok", "dkk"],
        axis=1, errors="ignore")

    # overnight rates data
    on_rate = pd.read_pickle(data_path + on_pkl)

    # events
    events_pickle = "events.p"
    events_data = pd.read_pickle(data_path + events_pickle)
    events = events_data["joint_cbs"]

    # # libor
    # libor_pickle = "libor_spliced_2000_2007_d.p"
    # libor_data = pd.read_pickle(data_path + libor_pickle)
    # libor_1m = libor_data["1m"]
    # libor_on = libor_data["on"]

    # from ois --------------------------------------------------------------
    ir = calculate_ois_implied_rates(ois_rate, on_rate, events,
        maturity=maturity,
        map_proxy_rate="expanding_since_previous", window=5)

    with open(data_path + ois_out_pkl, mode="wb") as hangar:
        pickle.dump(ir, hangar)

    # ir = pd.read_pickle(data_path + "implied_rates_from_1m_ois.p")

    # # from libor ------------------------------------------------------------
    # ir = calculate_libor_implied_rates(libor_1m, libor_on, events)
    #
    # with open(data_path + "implied_rates_from_1m_libor.p", mode="wb") as h:
    #     pickle.dump(ir, h)

    # with open(data_path + "implied_rates_1m_libor.p", mode="wb") as h:
    #     pickle.dump(ir, h)
