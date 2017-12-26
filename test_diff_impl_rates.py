from foolbox.api import *
from foolbox.wp_tabs_figs.wp_settings import settings

if __name__ == "__main__":
    # Set the deviation sensitivity in bps
    deviation_threshold = 50

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
    diff_bps = implied_rates_old.sub(implied_rates_new) * 100
    print("These bastards do deviate: \n",
          diff_bps.where(np.abs(diff_bps) >
                         deviation_threshold).dropna(how="all"))
