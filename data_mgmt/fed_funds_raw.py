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
months = ["F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"]
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
