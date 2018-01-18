from foolbox.api import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
import matplotlib.lines as mlines
from foolbox.wp_tabs_figs.wp_settings import settings
from foolbox.trading import EventTradingStrategy
from foolbox.utils import *


from foolbox.api import *
from foolbox.wp_tabs_figs.wp_settings import settings

if __name__ == "__main__":
    # Set the sample
    start_date = settings["sample_start"]
    end_date = settings["sample_end"]

    currencies = ["aud", "cad", "chf", "eur", "gbp", "nzd", "sek", "usd"]

    # Set up the parameters of rate forecasts
    lag = 10              # actually the holding period,
    lag_expect = lag + 2  # forecast rate one day in advance of trading

    # Forecast, and forecast consistency parameters
    avg_impl_over = 5    # smooth implied rate
    avg_refrce_over = 5  # smooth reference rate

    # Transformation applied to reference and implied rates
    map_expected_rate = lambda x: x.rolling(avg_impl_over,
                                            min_periods=1).mean()
    map_proxy_rate = lambda x: x.rolling(avg_refrce_over,
                                         min_periods=1).mean()

    # Get the implied future rates
    implied_rates_old = list()
    implied_rates_new = list()
    meetings = list()

    for curr in currencies:

        # New implied rates are from "implied_rates_from_1m_ois.p"
        new_pe = PolicyExpectation.from_pickles(data_path, curr, ffill=True)

        # Also get the old ones
        old_pe = PolicyExpectation.from_pickles(
            data_path, curr, ffill=True,
            e_proxy_rate_pickle="implied_rates_bloomberg_1m.p")

        # Smooth old forecast rates, lag them by 12 periods, locate meetings
        this_implied_old = \
            map_expected_rate(old_pe.expected_proxy_rate).shift(lag_expect)
        this_implied_old = this_implied_old.loc[old_pe.meetings.index]
        implied_rates_old.append(this_implied_old)

        # Same for the new rates
        this_implied_new = \
            map_expected_rate(new_pe.expected_proxy_rate).shift(lag_expect)
        this_implied_new = this_implied_new.loc[new_pe.meetings.index]
        implied_rates_new.append(this_implied_new)

        meetings.append(new_pe.meetings)

    # Locate the meetings in the sample
    implied_rates_old = \
        pd.concat(implied_rates_old, axis=1).loc[start_date:end_date, :]
    implied_rates_new = \
        pd.concat(implied_rates_new, axis=1).loc[start_date:end_date, :]
    meetings = pd.concat(meetings, axis=1).loc[start_date:end_date, :]

    # Print some descriptives for discrepancies
    diff_bps = implied_rates_old.sub(implied_rates_new)*100
    print(diff_bps.round(2).describe())

    # END OF THE DIFFERENCE IN IMPLIED RATES SNIPPET --------------------------

    # Set the output path, input data and sample
    out_path = data_path + settings["fig_folder"]
    input_dataset = settings["fx_data"]
    start_date = settings["sample_start"]

    #start_date = "2010-10-03"

    end_date = settings["sample_end"]

    # Set up the parameters of trading strategies
    lag = 10              # actually the holding period,
    lag_expect = lag + 2  # forecast rate one day in advance of trading
    threshold = 10        # threshold in bps

    # Forecast, and forecast consistency parameters
    avg_impl_over = 5    # smooth implied rate
    avg_refrce_over = 5  # smooth reference rate
    smooth_burn = 5      # discard number of periods corresponding to smoothing
                         # for the forecast-consistent perfect foresight

    # Transformation applied to reference and implied rates
    map_expected_rate = lambda x: x.rolling(avg_impl_over,
                                            min_periods=1).mean()
    map_proxy_rate = lambda x: x.rolling(avg_refrce_over,
                                         min_periods=1).mean()

    # EventTradingStrategy() settings
    trad_strat_settings = {
        "horizon_a": -lag,
        "horizon_b": -1,
        "bday_reindex": True
        }

    currencies = ["aud"]

    # matplotlib settings -------------------------------------------------------
    # font, colors
    plt.rc("font", family="serif", size=12)
    # locators
    minor_locator = mdates.YearLocator()
    major_locator = mdates.YearLocator(2)

    # Import the FX data
    # with open(data_path+input_dataset, mode="rb") as fname:
    #     data = pickle.load(fname)
    data = pd.read_pickle(data_path+input_dataset)

    # Get the individual currenices, spot rates:
    spot_mid = data["spot_mid"].loc[start_date:end_date, currencies]
    spot_bid = data["spot_bid"].loc[start_date:end_date, currencies]
    spot_ask = data["spot_ask"].loc[start_date:end_date, currencies]

    # And swap points
    swap_ask = data["tnswap_ask"].loc[start_date:end_date, currencies]
    swap_ask = remove_outliers(swap_ask, 50)
    swap_bid = data["tnswap_bid"].loc[start_date:end_date, currencies]
    swap_bid = remove_outliers(swap_bid, 50)

    # Import the all fixing times for the dollar index
    # with open(data_path+"fx_by_tz_sp_fixed.p", mode="rb") as fname:
    #     data_usd = pickle.load(fname)
    data_usd = pd.read_pickle(data_path+"fx_by_tz_sp_fixed.p")

    # Construct a pre-set fixing time dollar index
    us_spot_mid = data_usd["spot_mid"].loc[:, :, settings["usd_fixing_time"]]\
        .drop(["dkk"], axis=1)[start_date:end_date]
    us_spot_bid = data_usd["spot_bid"].loc[:, :, settings["usd_fixing_time"]]\
        .drop(["dkk"], axis=1)[start_date:end_date]
    us_spot_ask = data_usd["spot_ask"].loc[:, :, settings["usd_fixing_time"]]\
        .drop(["dkk"], axis=1)[start_date:end_date]
    us_swap_ask = data_usd["tnswap_ask"].loc[:, :, settings["usd_fixing_time"]]\
        .drop(["dkk"], axis=1)[start_date:end_date]
    us_swap_ask = remove_outliers(us_swap_ask, 50)
    us_swap_bid = data_usd["tnswap_bid"].loc[:, :, settings["usd_fixing_time"]]\
        .drop(["dkk"], axis=1)[start_date:end_date]
    us_swap_bid = remove_outliers(us_swap_bid, 50)

    # Align and ffill the data, first for tz-aligned countries
    (spot_mid, spot_bid, spot_ask, swap_bid, swap_ask) =\
        align_and_fillna((spot_mid, spot_bid, spot_ask, swap_bid, swap_ask),
                         "B", method="ffill")
    # Now for the dollar index
    (us_spot_mid, us_spot_bid, us_spot_ask, us_swap_bid, us_swap_ask) =\
        align_and_fillna((us_spot_mid, us_spot_bid, us_spot_ask,
                          us_swap_bid, us_swap_ask),
                         "B", method="ffill")

    # Get signals for all countries except for the US
    policy_fcasts = list()
    for curr in spot_mid.columns:
        # Get the predicted change in policy rate
        tmp_pe = PolicyExpectation.from_pickles(data_path, curr, ffill=True)
        policy_fcasts.append(
            tmp_pe.forecast_policy_direction(
                        lag=lag_expect, h_high=threshold/100,
                        map_proxy_rate=map_proxy_rate,
                        map_expected_rate=map_expected_rate))

    # Put individual predictions into a single dataframe
    signals = pd.concat(policy_fcasts, join="outer", axis=1)[start_date:end_date]
    signals.columns = spot_mid.columns

    # Get the trading strategy
    strat = EventTradingStrategy(
        signals=signals,
        prices={"mid": spot_mid, "bid": spot_bid, "ask": spot_ask},
        settings=trad_strat_settings)

    strat_bas_adj = strat.bas_adjusted().roll_adjusted(
        {"bid": swap_bid, "ask": swap_ask})

    strat_ret = strat_bas_adj._returns.dropna(how="all")

    strat_ret.cumsum().plot()


    new = pd.read_pickle(data_path + "implied_rates_from_1m_ois.p")
    old = pd.read_pickle(data_path + "implied_rates_bloomberg_1m.p")

    df = (100*old.sub(new).dropna(how="all"))\
             .loc[tmp_pe.meetings.index, currencies]
    df.plot()

    curr = 'cad'
    new_pe = PolicyExpectation.from_pickles(data_path, curr, ffill=True)
    old_pe = PolicyExpectation.from_pickles(data_path, curr, ffill=True,
                                            e_proxy_rate_pickle="implied_rates_bloomberg_1m.p")

    new = new_pe.forecast_policy_direction(lag=lag_expect, h_high=threshold/100,
                        map_proxy_rate=map_proxy_rate,
                        map_expected_rate=map_expected_rate)

    old = old_pe.forecast_policy_direction(lag=lag_expect, h_high=threshold/100,
                        map_proxy_rate=map_proxy_rate,
                        map_expected_rate=map_expected_rate)
    act = new_pe.meetings

    act = np.sign(act)

    pd.concat([old-act, new-act], axis=1, join="inner").plot()
    print((old-act).pow(2).sum(), (new-act).pow(2).sum())

    lol = pd.read_pickle(data_path + "temp_all_strats.p")






