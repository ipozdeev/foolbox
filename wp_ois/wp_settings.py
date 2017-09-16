ois_start_dates =\
    {"aud": "2001-10", "cad": "2002-05", "chf": "2000-08",
     "eur": "2000-01", "gbp": "2000-12", "jpy": "2002-03",
     "nzd": "2002-09", "sek": "2004-08", "usd": "2001-12"}

central_banks_start_dates =\
    {"rba": "2001-10", "boc": "2002-05", "snb": "2000-08",
     "ecb": "2000-01", "boe": "2000-12", "rbnz": "2002-09",
     "riks": "2004-08", "fomc": "2001-12"}

end_date = "2017-06-30"

# Effective day offset in business days, for countries with no series available
cb_effective_date_offset = \
    {"boc": 0, "snb": 0, "boe": 1, "rbnz": 0, "norges": 1, "fomc": 1}

cb_fx_map = \
    {"rba": "aud", "boc": "cad", "snb": "chf", "ecb": "eur", "boe": "gbp",
     "rbnz": "nzd", "riks": "sek", "fomc": "usd"}
