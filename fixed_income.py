import pandas as pd
import numpy as np
from foolbox.bankcalendars import *
from pandas.tseries.offsets import DateOffset, BDay, CustomBusinessDay

class LIBOR():
    """docstring for FixedIncome.

    Parameters
    ----------
    calendar : pandas.tseries.offsets.DateOffset
        calendar with holidays
    value_dt_lag : int
        number of days after the fixing to determine the value date (the date
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
        num, dnm = day_count.lower().split('/')

        if num != "act":
            raise NotImplementedError("only 'act' is currently implemented!")

        self.day_count_num = num
        self.day_count_dnm = np.float(dnm)


    # value date -------------------------------------------------------------
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

        # set payment date by adding maturity to start date and rolling to b/day
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

        # exclude the last day from clacultion period
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

        Sets quote date, then sets start date as T+`self.start_offset`. In
        doing so calls to the setter of start date.
        """
        # set quote date as is
        self._quote_dt = pd.to_datetime(value)

        # envoke value_dt setter
        self.value_dt = self._quote_dt + self.b_day*(self.value_dt_lag)

    @staticmethod
    def get_lengths_of_overnight(calculation_period):
        """
        Parameters
        ----------
        calculation_period : pd.DatetimeIndex
            cosisting of business days only
        """
        tmp_idx = pd.Series(index=calculation_period, data=calculation_period)

        res = tmp_idx.diff().shift(-1) / np.timedelta64(1, 'D')

        # do not need the last date, as it is the payment date
        res = res.iloc[:-1]

        return res

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
            res = dt - self.b_day*(0)

        elif self.day_roll == "following":
            # Roll to the next business day
            res = dt + self.b_day*(0)

        elif self.day_roll == "modified following":
            # Try to roll forward
            tmp_dt = dt + self.b_day*(0)

            # If the dt is in the following month roll backwards instead
            if tmp_dt.month == dt.month:
                res = dt + self.b_day*(0)
            else:
                res = dt - self.b_day*(0)

        else:
            raise NotImplementedError(
                "{} date rolling is not supported".format(self.day_roll))

        return res

    def get_accrual_factor(self):
        """
        """
        if self.day_count_num == "act":
            num = self.lifetime
        else:
            raise NotImplementedError()

        res = num / self.day_count_dnm

        return res

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
        accr_fact = self.get_accrual_factor()

        # multiply, to fractions of one
        res = rate / 100 * accr_fact

        return res

    def get_forward_rate(self, rate, forward_dt, rate_until):
        """
        Parameters
        ----------
        forward_dt : str/numpy.datetime64
            date
        rate : float
            main rate until the conract matures
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

        rate_pre = (1 + rate_pre / 100 / self.day_count_dnm \
            * self.lengths_of_overnight).prod()

        # get the post-forward_dt part of that payment -----------------------
        # index after `forward_dt`
        accr_fact = self.lengths_of_overnight.loc[forward_dt:].sum() /\
            self.day_count_dnm

        # divide, annualize, to percent --------------------------------------
        res = self.annualize((1 + rate_total) / rate_pre - 1, accr_fact)*100

        return res

    def annualize(self, value, accr_fact=None):
        """
        """
        if accr_fact is None:
            accr_fact = self.get_accrual_factor()

        res = value / accr_fact

        return res

    def reindex_series(self, series, **kwargs):
        """
        """
        idx = pd.date_range(series.index[0], series.index[-1], freq=self.b_day)
        res = series.reindex(index=idx, **kwargs)

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


if __name__ == "__main__":
    import ipdb
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
    # ipdb.set_trace()
    print(libor.get_forward_rate(1.5, "2017-12-15", 1.49918))
