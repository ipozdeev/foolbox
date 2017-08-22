from pandas.tseries.offsets import *
from pandas.tseries.holiday import *

class USTradingCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('NewYearsDay', month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday('USIndependenceDay', month=7, day=4,
            observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday('Christmas', month=12, day=25, observance=nearest_workday)
    ]

class UKTradingCalendar(AbstractHolidayCalendar):
    """
    Taken from http://mapleoin.github.io/perma/python-uk-business-days
    """
    rules = [
        Holiday('NewYearsDay',
            month=1, day=1, observance=next_monday),
        GoodFriday,
        EasterMonday,
        Holiday('EarlyMayBankHoliday',
            month=5, day=1, offset=DateOffset(weekday=MO(1))),
        Holiday('SpringBankHoliday',
            month=5, day=31, offset=DateOffset(weekday=MO(-1))),
        Holiday('SummerBankHoliday',
            month=8, day=31, offset=DateOffset(weekday=MO(-1))),
        Holiday('ChristmasDay',
            month=12, day=25, observance=next_monday),
        Holiday('Boxing Day',
            month=12, day=26, observance=next_monday_or_tuesday)
        ]

class AustraliaTradingCalendar(AbstractHolidayCalendar):
    """
    Bank close days are those in NSW.

    Public holidays:
        https://www.legislation.nsw.gov.au/#/view/act/2010/115/part2/sec4
    Bank close days:
        https://www.legislation.nsw.gov.au/#/view/act/2008/49/part3a/sec14b
    """
    rules = [
        Holiday('NewYearsDay',
            month=1, day=1, observance=next_monday),
        # NB: only from 2011 observance is next monday; what to do?
        Holiday('AustraliaDay',
            month=1, day=26, observance=next_monday),
        GoodFriday,
        EasterMonday,
        Holiday('ANZACDay',
            month=4, day=25, observance=next_monday),
        Holiday('QueensBirthday',
            month=6, day=1, offset=DateOffset(weekday=MO(2))),
        Holiday('AugustBankHoliday',
            month=8, day=1, offset=DateOffset(weekday=MO(1))),
        Holiday('LabourDay',
            month=5, day=1, offset=DateOffset(weekday=MO(1))),
        Holiday('Christmas Day',
            month=12, day=25, observance=next_monday),
        Holiday('Boxing Day',
            month=12, day=26, observance=next_monday_or_tuesday)
        ]

class SwedenTradingCalendar(AbstractHolidayCalendar):
    """
    Riksbank: http://www.riksbank.se/en/Calender/Bank-Holidays/2017/
    Nordea: https://www.nordea.com/en/about-nordea/contact/bank-holidays/

    NB: are there observances in Sweden?
    """
    rules = [
        # NB: no observance for January 1st (cf year 2017)
        Holiday('NewYearsDay',
            month=1, day=1),
            # NB: no observance for Epiphany (cf year 2013)
        Holiday('Epiphany',
            month=1, day=6),
        GoodFriday,
        EasterMonday,
        Holiday('LabourDay',
            month=5, day=1),
        Holiday('AscensionDay',
            month=1, day=1, offset=[Easter(), Day(39)]),
        Holiday('NationalDay',
            month=6, day=6),
        Holiday('MidsummersEve',
            month=6, day=25, offset=DateOffset(weekday=FR(-1))),
        Holiday('ChristmasEve',
            month=12, day=24),
        Holiday('ChristmasDay',
            month=12, day=25),
        Holiday('BoxingDay',
            month=12, day=26),
        Holiday('NewYearsEve',
            month=12, day=31)
        ]


if __name__ == "__main__":
    SwedenTradingCalendar().holidays(start="2017-01-01", end="2017-12-31")
