import pandas as pd
import numpy as np
from scipy.optimize import fsolve
from foolbox.bankcalendars import *
from pandas.tseries.offsets import DateOffset, CustomBusinessDay, BDay


class FixedIncome():
    """docstring for FixedIncome.

    Parameters
    ----------
    calendar : pandas.tseries.offsets.DateOffset
        calendar with holidays
    value_dt_lag : int
        number of days after the quote to determine the value date (the date
        to which rates relate)
    maturity : pandas.tseries.offsets.DateOffset
        maturity of the contract, e.g. DateOffset(months=1)
    day_roll : str
        convention to move the payment date when it falls on a holiday.
        Typical conventions are: "previous", following", "modified following".
        Currently only the latter are implemented.
    day_count : str
        (case-insensitive) such as "ACT/365" or "30/360"
    """
    def __init__(self, calendar, value_dt_lag, maturity, day_roll, day_count):
        """
        """
        self.calendar = calendar
        self.value_dt_lag = value_dt_lag
        self.maturity = maturity
        self.day_roll = day_roll
        self.day_count = day_count

        # property-based
        self._quote_dt = None
        self._value_dt = None
        self._end_dt = None

        self.b_day = CustomBusinessDay(calendar=calendar)

        # day count
        num, dnm = self.recover_day_count(day_count)

        self.day_count_num = num
        self.day_count_dnm = np.float(dnm)

    # value date ------------------------------------------------------------
    @property
    def value_dt(self):
        if self._value_dt is None:
            raise ValueError("Define start date first!")
        return self._value_dt

    @value_dt.setter
    def value_dt(self, value):
        """Set value date `value_dt`.

        Sets value date to the provided value, then calculates end date by
        adding maturity to value date, then creates a sequence of business
        days from value to end dates, then for each of these days calculates
        the number of subsequent days over which the rate will stay the same,
        finally, calculates the lifetime of OIS.

        Parameters
        ----------
        value : str/numpy.datetime64
            date to use as start date
        """
        # set value date as is
        self._value_dt = pd.to_datetime(value)

        # set payment date by adding maturity to start date and rolling to bday
        self._end_dt = self.roll_day(self._value_dt + self.maturity)

        # # set end date as the date preceding the payment date
        # # TODO: offsetting by 1 day necessary?
        # self._end_dt = self._payment_dt - self.b_day

        # calculation period: range of business days from start to end dates
        calculation_period = pd.date_range(
            self._value_dt,
            self._end_dt,
            freq=self.b_day)

        # number of days to multiply rate with: uses function from utils.py
        self.lengths_of_overnight = \
            self.get_lengths_of_overnight(calculation_period)

        # exclude the last day from calculation period
        self.calculation_period = calculation_period[:-1]

        # calculation period length
        self.lifetime = self.lengths_of_overnight.sum()

    # end date --------------------------------------------------------------
    @property
    def end_dt(self):
        """Date when the instrument matures.

        This date is set off the start date.
        """
        if self._end_dt is None:
            raise ValueError("Define start date first!")
        return self._end_dt

    # quote date ------------------------------------------------------------
    @property
    def quote_dt(self):
        if self._quote_dt is None:
            raise ValueError("Define quote date first!")
        return self._quote_dt

    @quote_dt.setter
    def quote_dt(self, value):
        """Set quote date.

        Sets quote date, then sets start date as T+`self.value_dt_lag`. In
        doing so calls to the setter of start date.
        """
        # set quote date as is
        self._quote_dt = pd.to_datetime(value)

        # evoke value_dt setter
        self.value_dt = self._quote_dt + self.b_day * self.value_dt_lag

    @staticmethod
    def get_lengths_of_overnight(calculation_period):
        """
        Parameters
        ----------
        calculation_period : pd.DatetimeIndex
            consisting of business days only
        """
        tmp_idx = pd.Series(index=calculation_period, data=calculation_period)

        res = tmp_idx.diff().shift(-1) / np.timedelta64(1, 'D')

        # do not need the last date, as it is the payment date
        res = res.iloc[:-1]

        return res

    @staticmethod
    def recover_day_count(day_count):
        """

        Parameters
        ----------
        day_count : str
            such as 'act/365'

        Returns
        -------

        """
        # day count
        num, dnm = day_count.lower().split('/')

        if num != "act":
            raise NotImplementedError("only 'act' is currently implemented!")

        day_count_num = num
        day_count_dnm = np.float(dnm)

        return day_count_num, day_count_dnm

    def roll_day(self, dt):
        """Offset `dt` making sure that the result is a business day.

        Parameters
        ----------
        dt : numpy.datetime64
            date to offset

        Returns
        -------
        res : numpy.datetime64
            offset date
        """
        # try to adjust to the working day
        if self.day_roll == "previous":
            # Roll to the previous business day
            res = dt - self.b_day * 0

        elif self.day_roll == "following":
            # Roll to the next business day
            res = dt + self.b_day * 0

        elif self.day_roll == "modified following":
            # Try to roll forward
            tmp_dt = dt + self.b_day * 0

            # If the dt is in the following month roll backwards instead
            if tmp_dt.month == dt.month:
                res = dt + self.b_day * 0
            else:
                res = dt - self.b_day * 0

        else:
            raise NotImplementedError(
                "{} date rolling is not supported".format(self.day_roll))

        return res

    def get_accrual_factor(self, day_count_num, day_count_dnm=None):
        """
        """
        if day_count_num == "act":
            num = self.lifetime
        else:
            raise NotImplementedError()

        if day_count_dnm is None:
            dnm = self.day_count_dnm
        else:
            dnm = day_count_dnm

        res = num / dnm

        return res

    def reindex_series(self, series, **kwargs):
        """
        """
        idx = pd.date_range(series.index[0], series.index[-1], freq=self.b_day)
        res = series.reindex(index=idx, **kwargs)

        return res


class LIBOR(FixedIncome):
    """
    """
    def get_end_payment(self, rate):
        """
        Parameters
        ----------
        rate : float
            rate, annualized, in percent

        Returns
        -------
        res : float
            total payment on 1 dollar invested, in frac of 1, not annualized
        """
        # accrual factor
        accr_fact = self.get_accrual_factor(self.day_count_num)

        # multiply, to fractions of one
        res = rate / 100 * accr_fact

        return res

    def annualize(self, value, accr_fact=None):
        """
        """
        if accr_fact is None:
            accr_fact = self.get_accrual_factor(self.day_count_num)

        res = value / accr_fact

        return res

    def get_forward_rate(self, rate, forward_dt, rate_until):
        """
        Parameters
        ----------
        forward_dt : str/numpy.datetime64
            date
        rate : float
            main rate until the contract matures
        rate_until : float
            rate to prevail < `forward_dt`, annualized, in percent

        Returns
        -------
        res : float
            implied post-`forward_dt` rate, annualized, in percent
        """
        # get end payment from `rate` ----------------------------------------
        rate_total = self.get_end_payment(rate)

        # get the pre-forward_dt part of that payment ------------------------
        # index before `forward_dt`
        pre_forward_dt = pd.to_datetime(forward_dt) - DateOffset(days=1)

        rate_pre = pd.Series(
            data=rate_until,
            index=self.calculation_period).loc[:pre_forward_dt]

        rate_pre = (1 + rate_pre / 100 / self.day_count_dnm *\
            self.lengths_of_overnight).prod()

        # get the post-forward_dt part of that payment -----------------------
        # index after `forward_dt`
        accr_fact = self.lengths_of_overnight.loc[forward_dt:].sum() /\
            self.day_count_dnm

        # divide, annualize, to percent --------------------------------------
        res = self.annualize((1 + rate_total) / rate_pre - 1, accr_fact)*100

        return res

    @classmethod
    def from_iso(cls, iso, maturity):
        """Return OIS class instance with specifications of currency `iso`."""
        # calendars
        calendars = {
            "usd": USTradingCalendar(),
            "eur": EuropeTradingCalendar()}

        all_settings = {

            "eur": {
                "value_dt_lag": 0,
                "day_roll": "modified following",
                "day_count": "act/360"},

            "usd": {
                "value_dt_lag": 2,
                "day_roll": "modified following",
                "day_count": "act/360"},

            "chf": {
                "value_dt_lag": 2,
                "day_roll": "modified following",
                "day_count": "act/360"},

            "gbp": {
                "value_dt_lag": 0,
                "day_roll": "modified following",
                "day_count": "act/365"},

            "jpy": {
                "value_dt_lag": 2,
                "day_roll": "modified following",
                "day_count": "act/360"},

            "aud": {
                "value_dt_lag": 0,
                "day_roll": "modified following",
                "day_count": "act/365"},

            "cad": {
                "value_dt_lag": 0,
                "day_roll": "modified following",
                "day_count": "act/365"},

            "nzd": {
                "value_dt_lag": 0,
                "day_roll": "modified following",
                "day_count": "act/365"},

            "sek": {
                "value_dt_lag": 2,
                "day_roll": "modified following",
                "day_count": "act/360"},

            }

        this_setting = all_settings[iso]
        this_setting.update({"maturity": maturity})
        this_setting.update({"calendar": calendars.get(iso)})

        return cls(**this_setting)


class OIS(FixedIncome):
    """Perform basic operations with OIS.

    Keeps all information relevant for an OIS: maturity, offsets, conventions
    etc. Methods allow to calculate floating and fixed leg return, rate
    implied before event etc. Construction of the class is easiest
    performed through usage of `.from_iso` factory.

    To each contract, there is quote date, start (=effective) date and end
    date. Quote date is when the OIS price (=OIS rate) is negotiated and
    reported. Start date is usually determined as T+2 and corresponds to the
    first date when the floating rate is fixed (accrues) for the first time.
    End date is the last date when the floating rate is fixed, and is usually
    determined as the last trading date before T+2+maturity.

    Parameters
    ----------
    calendar : Calendar
        calendar influencing all business day-related date offsets
    value_dt_lag : int
        number of business days until the first accrual of floating rate
    fixing_lag : int
        number of periods to shift returns _*forward*_, that is, if the
        underlying rate is reported with one period lag, today's rate on
        the floating leg is determined tomorrow. The value is negative if
        today's rate is determined yesterday (as with CHF)
    day_roll : str
        specifying date rolling convention if the end date of the contract
        is not a business day. Typical conventions are: "previous",
        "following", and "modified following". Currently only the latter
        two are implemented
    maturity : pandas.tseries.offsets.DateOffset
        maturity of the contract, e.g. DateOffset(months=1)
    day_count_float : int
        e.g. act/360 or act/365 day (for the floating leg)
    day_count_fix : int
        e.g. act/360 or act/365 day
    new_rate_lag : int
        lag with which newly announced rates (usually on monetary policy
        meetings) become effective; this impacts calculation of implied rates
    """
    def __init__(self, calendar, value_dt_lag, fixing_lag, new_rate_lag,
                 maturity, day_roll, day_count_float, day_count_fix=None):
        """
        """
        if day_count_fix is None:
            day_count_fix = day_count_float

        self.calendar = calendar
        self.value_dt_lag = value_dt_lag
        self.fixing_lag = fixing_lag
        self.new_rate_lag = new_rate_lag
        self.maturity = maturity
        self.day_roll = day_roll

        self.b_day = CustomBusinessDay(calendar=calendar)

        self.day_count_float_num, self.day_count_float_dnm = \
            self.recover_day_count(day_count_float)
        self.day_count_fix_num, self.day_count_fix_dnm = \
            self.recover_day_count(day_count_fix)

        # property-based
        self._quote_dt = None
        self._value_dt = None
        self._end_dt = None

    def get_return_of_floating(self, on_rate, dt_since=None, dt_until=None):
        """Calculate return on the floating leg.

        The payoff of the floating leg is determined as the cumproduct of the
        underlying rate. For days followed by K non-business days, the rate on
        that day is multiplied by (K+1): for example, the 0.35% p.a. rate on
        Friday will contribute (1+0.35/100/365*3) to the floating leg payoff.

        Parameters
        ----------
        on_rate : float/pandas.Series
            overnight rate (in which case will be broadcast over the
            calculation period) or series thereof, in percent p.a.
        dt_since : str/date
            date to use instead of `self.value_dt` because... reasons!
        dt_until : str/date
            date to use instead of `self.end_dt` because... reasons!

        Returns
        -------
        res : float
            cumulative payoff, in frac of 1, not annualized
        """
        if dt_until is None:
            dt_until = self.calculation_period[-1]
        if dt_since is None:
            dt_since = self.value_dt

        # three possible data types for on_rate: float, NDFrame and anything
        # else if float, convert to pd.Series
        if isinstance(on_rate, (np.floating, float)):
            if np.isnan(on_rate):
                return np.nan
            on_rate_series = pd.Series(on_rate, index=self.calculation_period)
        elif isinstance(on_rate, pd.core.generic.NDFrame):
            # for the series case, need to ffill if missing and shift by the
            #   fixing lag
            on_rate_series = on_rate.shift(self.fixing_lag).ffill()
        else:
            return np.nan

        # return nan if the end date has already happened
        if dt_until not in on_rate_series.index:
            return np.nan
        if dt_since not in on_rate_series.index:
            return np.nan

        # Reindex to calendar day, carrying rates forward over non b-days
        # NB: watch over missing values in the data: here is the last chance
        #   to keep them as is
        tmp_ret = on_rate_series.reindex(index=self.calculation_period)

        # deannualize etc.
        tmp_ret /= (100 * self.day_count_float_dnm)

        # Compute the cumulative floating leg return over the period
        # res = self.cumprod_with_mult(tmp_ret / self.day_count_float_dnm / 100)
        ret_mult = (1 + tmp_ret * self.lengths_of_overnight)
        res = ret_mult.loc[dt_since:dt_until].prod() - 1

        # # annualize etc.
        # res *= (100 * self.day_count_float_dnm / self.lifetime)

        return res

    def get_return_on_fixed(self, swap_rate):
        """Calculate return on the fixed leg over the lifetime of OIS.

        Parameters
        ----------
        swap_rate : float
            in percent per year
        """
        # accrual factor
        accr_fact = self.get_accrual_factor(self.day_count_fix_num,
                                            self.day_count_fix_dnm)

        # multiply, to fractions of one
        res = swap_rate / 100 * accr_fact

        return res

    def get_implied_rate(self, event_dt, swap_rate, rate_until):
        """Calculate the rate expected to prevail after `event_dt`.

        First, given `rate_until`, calculates the more or less sure part of
        the total floating leg payoff that will have accrued until the new
        rate enters the floating leg for the first time
        (at `event_dt`+`self.fixing_lag`+`self.new_rate_lag`). Then, divides
        the sure payoff of the fixed leg through this to arrive at the
        expected floating leg payoff that will have accrued after the new rate
        is in place.

        Parameters
        ----------
        event_dt : str/numpy.datetime64
            date of event
        on_rate : pandas.Series
            containing rates around `event_dt`
        rate_until : float
            rate to prevail before event; default is to take correct today's
        """
        event_dt = pd.to_datetime(event_dt)

        # number of days between: for the US case, the new rate is effective
        #   one day after announcement, and also there is one day fixing lag
        # days_until = (event_dt - self.value_dt).days + self.fixing_lag + \
        #     self.new_rate_lag
        # TODO: set rule to determine since when new rate becomes effective
        dt_until = event_dt + \
            self.b_day*(self.fixing_lag + self.new_rate_lag - 1)

        # total return from entering this OIS
        cumprod_total = self.get_return_on_fixed(swap_rate) + 1

        # part prior to event -----------------------------------------------
        # cumulative rate before the new rate will be introduced
        cumprod_until = self.get_return_of_floating(rate_until,
            dt_until=dt_until) + 1

        # expected part after event -----------------------------------------
        # expected_cumprod_after = cumprod_total / cumprod_until

        # implied daily rate after event
        dt_since = event_dt + self.b_day*(self.fixing_lag + self.new_rate_lag)
        obj_fun = lambda x: cumprod_total - \
            (1 + self.get_return_of_floating(x[0], dt_since=dt_since)) *\
                cumprod_until

        # solve thingy
        res = fsolve(obj_fun, x0=np.array([swap_rate]), xtol=1e-05)[0]
        if np.abs(res - swap_rate) < 1e-06:
            res = np.nan

        return res

    def get_rates_until(self, on_rate, meetings, method="g_average"):
        """Calculate rates to be considered as prevailing until next meeting.

        Parameters
        ----------
        on_rate : pandas.Series
            of underlying rates
        meetings : pandas.Series/pandas.DataFrame
            of meetings; only index matters, values can be whatever
        method : str
            "average" to calculate the prevailing rate as geometric avg since
            the last meeting;

        Returns
        -------
        res : float
            rate, in percent p.a.
        """
        on_rate = on_rate.copy() / 100

        # with this method, take the fixing rate at `self.quote_dt`
        if method == "last":
            res = on_rate.shift(self.fixing_lag)

        elif method == "g_average":
            # expanding geometric average starting from events
            res = on_rate * np.nan
            for t in list(meetings.index)[::-1] + list(on_rate.index[[0]]):
                to_fill_with = np.exp(
                    np.log(1 + on_rate.shift(-self.new_rate_lag).loc[t:])\
                        .expanding().mean())
                res.fillna(to_fill_with.shift(self.new_rate_lag), inplace=True)

            res -= 1.0

        elif method == "a_average":
            # expanding geometric average starting from events
            res = on_rate * np.nan
            for t in list(meetings.index)[::-1] + list(on_rate.index[[0]]):
                to_fill_with = on_rate.shift(-self.new_rate_lag).loc[t:]\
                        .expanding().mean()
                res.fillna(to_fill_with.shift(self.new_rate_lag), inplace=True)

        else:
            raise ValueError(
                "Method not implemented: choose 'g_average' or " +
                "'a_average' or 'last'")

        res *= 100

        return res

    def annualize(self, value, relates_to_float=True):
        """Annualize and multiply by 100."""
        res = value * (
            self.day_count_float_dnm if relates_to_float
            else self.day_count_fix_dnm) * 100

        return res

    @staticmethod
    def implied_rate_formula(rate_before, swap_rate, days_until, days_after):
        """
        Parameters
        ----------
        rate_before : float
            rate prevailing before the possible change, in frac of 1 per period
        swap_rate : float
            swap rate, in frac of 1 per period
        days_until : int
            days before the possible change when `rate_before` applies
        days_after : int
            days after the possible change when the new rate applies
        """
        res = (
            (1+swap_rate*days_until)/(1+rate_before)**days_until
            )**(1/days_after)-1

        return res

    @classmethod
    def from_iso(cls, iso, maturity):
        """Return OIS class instance with specifications of currency `iso`."""
        # calendars
        calendars = {
            "usd": USTradingCalendar(),
            "aud": AustraliaTradingCalendar(),
            "cad": CanadaTradingCalendar(),
            "chf": SwitzerlandTradingCalendar(),
            "eur": EuropeTradingCalendar(),
            "gbp": UKTradingCalendar(),
            "jpy": None,
            "nzd": NewZealandTradingCalendar(),
            "sek": SwedenTradingCalendar()
        }

        all_settings = {
            "aud": {"value_dt_lag": 1,
                    "fixing_lag": 0,
                    "day_count_float": "act/365",
                    "day_roll": "modified following",
                    "new_rate_lag": 1},

            "cad": {"value_dt_lag": 0,
                    "fixing_lag": 1,
                    "day_count_float": "act/365",
                    "day_roll": "modified following",
                    "new_rate_lag": 0},

            "chf": {"value_dt_lag": 2,
                    "fixing_lag": -1,
                    "day_count_float": "act/360",
                    "day_roll": "modified following",
                    "new_rate_lag": 0},

            "eur": {"value_dt_lag": 2,
                    "fixing_lag": 0,
                    "day_count_float": "act/360",
                    "day_roll": "modified following",
                    "new_rate_lag": 1},

            "gbp": {"value_dt_lag": 0,
                    "fixing_lag": 0,
                    "day_count_float": "act/365",
                    "day_roll": "modified following",
                    "new_rate_lag": 1},

            "jpy": {"value_dt_lag": 2,
                    "fixing_lag": 1,
                    "day_count_float": "act/365",
                    "day_roll": "modified following",
                    "new_rate_lag": None},

            "nzd": {"value_dt_lag": 2,
                    "fixing_lag": 0,
                    "day_count_float": "act/365",
                    "day_roll": "modified following",
                    "new_rate_lag": 0},

            "sek": {"value_dt_lag": 2,
                    "fixing_lag": -1,
                    "day_count_float": "act/360",
                    "day_roll": "modified following",
                    "new_rate_lag": 1},

            "usd": {"value_dt_lag": 2,
                    "fixing_lag": 1,
                    "day_count_float": "act/360",
                    "day_roll": "modified following",
                    "new_rate_lag": 1}
        }

        this_setting = all_settings[iso]
        this_setting.update({"maturity": maturity})
        this_setting.update({"calendar": calendars.get(iso)})

        return cls(**this_setting)


if __name__ == "__main__":
    libor = LIBOR.from_iso("usd", DateOffset(months=1))
    libor.quote_dt = "2017-12-02"
    print(libor.quote_dt)
    print(libor.value_dt)
    print(libor.end_dt)
    print(libor.calculation_period)
    print(libor.lengths_of_overnight)
    print(libor.day_count_num)
    print((libor.end_dt - libor.value_dt).days)
    print(sum(libor.lengths_of_overnight))
    print(libor.get_forward_rate(1.5, "2017-12-15", 1.49918))
