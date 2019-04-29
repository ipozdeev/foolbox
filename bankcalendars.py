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


class NewZealandTradingCalendar(AbstractHolidayCalendar):
    """
    From https://www.nzfma.org/Site/practices_standards/market_conventions.aspx
    """
    rules = [
        Holiday('NewYearsDay',
            month=1, day=1, observance=next_monday),
        Holiday('DayAfterNewYearsDay',
            month=1, day=2, observance=next_monday_or_tuesday),
        Holiday('WellingtonAnniversaryDay',
            month=1, day=22, offset=DateOffset(weekday=MO(0))),
        Holiday('AucklandAnniversaryDay',
            month=1, day=29, offset=DateOffset(weekday=MO(0))),
        Holiday('WaitangiDay',
            month=2, day=6),
        GoodFriday,
        EasterMonday,
        Holiday('ANZACDay',
            month=4, day=25),
        Holiday('QueensBirthday',
            month=6, day=1, offset=DateOffset(weekday=MO(1))),
        Holiday('LabourDay',
            month=10, day=1, offset=DateOffset(weekday=MO(4))),
        Holiday('ChristmasDay',
            month=12, day=25, observance=next_monday),
        Holiday('BoxingDay',
            month=12, day=26, observance=next_monday_or_tuesday)
    ]


class EuropeTradingCalendar(AbstractHolidayCalendar):
    """
    From https://www.ecb.europa.eu/press/pr/date/2000/html/pr001214_4.en.html
    """
    rules = [
        Holiday('NewYearsDay',
            month=1, day=1),
        GoodFriday,
        EasterMonday,
        Holiday('LabourDay',
            month=5, day=1),
        Holiday('ChristmasDay',
            month=12, day=25),
        Holiday('BoxingDay',
            month=12, day=26)
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
        Holiday('BoxingDay',
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
            month=10, day=1, offset=DateOffset(weekday=MO(1))),
        Holiday('Christmas Day',
            month=12, day=25, observance=next_monday),
        Holiday('BoxingDay',
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


class SwitzerlandTradingCalendar(AbstractHolidayCalendar):
    """
    Examples:
    https://www.six-securities-services.com/en/home/clearing/member-information/market-information.html
    More examples:
    https://www.six-interbank-clearing.com/dam/downloads/en/payment_services/sic/banking_holidays.pdf
    """
    rules = [
        # NB: no observance for January 1st (cf year 2017)
        Holiday('NewYearsDay',
            month=1, day=1),
        Holiday('SwissBankHoliday',
            month=1, day=2),
        GoodFriday,
        EasterMonday,
        Holiday('LabourDay',
            month=5, day=1),
        Holiday('AscensionDay',
            month=1, day=1, offset=[Easter(), Day(39)]),
        Holiday('WhitMonday',
            month=1, day=1, offset=[Easter(), Day(50)]),
        Holiday('NationalDay',
            month=8, day=1),
        Holiday('ChristmasDay',
            month=12, day=25),
        Holiday('BoxingDay',
            month=12, day=26)
        ]


class CanadaTradingCalendar(AbstractHolidayCalendar):
    """
    Bank close days are those in Toronto (ON).

    BoC:
        http://www.bankofcanada.ca/about/contact-information/bank-of-canada-holiday-schedule
    Wiki:
        https://en.wikipedia.org/wiki/Public_holidays_in_Canada#Ontario
    A good way to check is to look at daily CORRA:
        http://www.bankofcanada.ca/rates/interest-rates/canadian-interest-rates
    """
    rules = [
        Holiday('NewYearsDay',
            month=1, day=1, observance=next_monday),
        Holiday('FamilyDay',
            month=2, day=1, offset=DateOffset(weekday=MO(3))),
        GoodFriday,
        Holiday('VictoriaDay',
            month=5, day=24, offset=DateOffset(weekday=MO(-1))),
        Holiday('CanadaDay',
            month=7, day=1, observance=next_monday),
        Holiday('AugustCivicHoliday',
            month=8, day=1, offset=DateOffset(weekday=MO(1))),
        Holiday('LabourDay',
            month=9, day=1, offset=DateOffset(weekday=MO(1))),
        Holiday('ThanksgivingDay',
            month=10, day=1, offset=DateOffset(weekday=MO(2))),
        Holiday('RemembranceDay',
            month=11, day=11, observance=next_monday),
        Holiday('Christmas Day',
            month=12, day=25, observance=next_monday),
        Holiday('BoxingDay',
            month=12, day=26, observance=next_monday_or_tuesday)
        ]

def get_bank_calendar(iso):
    """Retrieve calendar for currency `iso`.

    Parameters
    ----------
    iso : str
    """
    calendar_dict = {
        "usd": USTradingCalendar,
        "chf": SwitzerlandTradingCalendar,
        "eur": EuropeTradingCalendar,
        "cad": CanadaTradingCalendar,
        "nzd": NewZealandTradingCalendar,
        "aud": AustraliaTradingCalendar,
        "gbp": UKTradingCalendar,
        "sek": SwedenTradingCalendar
    }

    return calendar_dict[iso]


if __name__ == "__main__":
    CanadaTradingCalendar().holidays(start="2013-11-29", end="2015-01-05")
