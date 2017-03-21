"""Here dwells parsing of the raw fed funds futures data
"""
from foolbox.api import *

with open(data_path + "fed_funds_futures_raw.p", "rb") as fname:
    ff_raw = pickle.load(fname)

column_to_parse = "Settle"  # parses settlement prices
out_name = "fed_funds_futures_"+column_to_parse.lower()+".p"

# Set range of futures to parse
start_date = "1988-11"
end_date = "2019-01"

# Dictionary providing a map between futures' names and months
naming_map = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6, "N": 7, "Q": 8, "U": 9,
    "V": 10, "X": 11, "Z": 12,
}

# Loop over the futures, appending the list to be further concatenated
out_list = []
for key in ff_raw.keys():
    # Get the expiry date as a string
    month = str(naming_map[key[0]])
    year = key[1:]
    # Offset to month end, corresponding to the axtual expiry date
    exp_date = pd.Timestamp(year + "-" + month) + pd.offsets.MonthEnd()

    # Append the output to list
    tmp_df = ff_raw[key][[column_to_parse]]
    tmp_df.columns = [exp_date]
    out_list.append(tmp_df)

# Concatenate futures of different expiries, sort in chronological order
out = pd.concat(out_list, join="outer", axis=1)
out = out[sorted(out.columns)]

# Pickle 'em
with open(data_path+out_name, mode="wb") as fname:
    pickle.dump(out, fname)
