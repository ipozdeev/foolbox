"""Here dwell downloading and pickling of fed funds futures data from quandl
"""
from foolbox.api import *
import itertools as itools
import quandl

quandl.ApiConfig.api_key = "TQTWuU5e53sYEykyzzjW"

out_name = "fed_funds_futures_raw.p"

# Define the name pattern
name_pattern = "CME/FF"

# Set the desired sample
first_year = 1989
last_year = 2018

# Futures are available for each month
months = ["F", "G", "H", "J", "K", "M", "N", "garch_lags", "U", "V", "X", "Z"]
years = [str(year) for year in np.arange(first_year, last_year+1)]

# Create a list to iterate requests over
combos = list(itools.product(months, years))

# Apend November and December 1988 manually, FU that's why
combos.append(("X", "1988"))
combos.append(("Z", "1988"))

# Download the data
raw_data = dict()
for month, year in combos:
    raw_data[month+year] = quandl.get(name_pattern+month+year)

# Pickle the bastards
with open(data_path + out_name, "wb") as fname:
    pickle.dump(raw_data, fname)



def fx_market_factors(returns, keep_base_currency=True):
    """Function's behaviour assumes log returns for aedditivity

    Parameters
    ----------
    returns: pd.DataFrame
        of (excess) FX returns in terms of another currency. Say returns of
        AUD, CAD and NZD in USD
    keep_base_currency: bool
        specifying whether the returns shall be kept in the original currency
        if True or expressed in the counter currencies if False. Default is
        True

    Returns
    -------
    market_factors: pd.DataFrame
        with the same structure as the input. Each column contains return on
        market portfolio of currencies either in terms of their own currency

    """
    # Temp check get the FX market factors form spot rates
    out = pd.DataFrame(index=returns.index, columns=returns.columns)
    for col in returns.columns:
        tmp_ret = returns.copy()
        tmp_ret["usd"] = 1  # add us dollar
        tmp_ret = returns.divide(returns[col], axis=0)
        out[col] = np.log(tmp_ret/tmp_ret.shift(1))\
            .drop(col, axis=1).mean(axis=1)


    return out

s_d = data["spot_ret"]
out = pd.DataFrame(index=s_d.index, columns=s_d.columns)
for col in s_d.columns:
    tmp_df = s_d.copy()
    tmp_df = tmp_df.subtract(tmp_df[col], axis=0)
    tmp_df[col] = tmp_df[col] - s_d[col]
    out[col] = tmp_df.mean(axis=1)



from foolbox.api import *

with open(data_path + "data_wmr_dev_d.p", mode="rb") as fname:
    data = pickle.load(fname)

s_d = data["spot_ret"]
spot = data["spot_mid"].pow(-1)  # convert to dollars per unit of FX

# Option 1: construct from spot rates
out1 = pd.DataFrame(index=spot.index, columns=spot.columns)
for col in spot.columns:
    tmp_rate = spot.copy()
    tmp_rate["usd"] = 1  # add us dollar, 1 for one
    tmp_rate = tmp_rate.divide(tmp_rate[col], axis=0)
    out1[col] = np.log(tmp_rate/tmp_rate.shift(1))\
        .drop(col, axis=1).mean(axis=1)

# Option 2: construct from log returns
out2 = pd.DataFrame(index=s_d.index, columns=s_d.columns)
for col in s_d.columns:
    tmp_df = s_d.copy()
    tmp_df = tmp_df.subtract(tmp_df[col], axis=0)
    tmp_df[col] = tmp_df[col] - s_d[col]
    out2[col] = tmp_df.mean(axis=1)



# Generate events descriptives
from foolbox.api import *

with open(data_path + "events_new.p", mode="rb") as fname:
    events = pickle.load(fname)

banks = ["rba", "boc", "ecb", "rbnz", "norges", "riks", "snb", "boe", "fomc"]
out = pd.DataFrame(index=banks, columns=["total", "hikes", "cuts"])

for bank in banks:
    tmp_policy = events[bank].change
    out.loc[bank, "total"] = tmp_policy.count()
    out.loc[bank, "hikes"] = tmp_policy.where(tmp_policy > 0).count()
    out.loc[bank, "cuts"] = tmp_policy.where(tmp_policy < 0).count()


from foolbox.api import *
start = "2000-11-01"
with open(data_path + "events.p", mode="rb") as fname:
    events = pickle.load(fname)

banks = ["rba", "boc", "ecb", "rbnz", "norges", "riks", "snb", "boe", "fomc"]
out = pd.DataFrame(index=banks, columns=["total", "hikes", "cuts"])

for bank in banks:
    tmp_policy = events[bank].change[start:]
    out.loc[bank, "total"] = tmp_policy.count()
    out.loc[bank, "hikes"] = tmp_policy.where(tmp_policy > 0).count()
    out.loc[bank, "cuts"] = tmp_policy.where(tmp_policy < 0).count()

tab_meetings_descr = out



























