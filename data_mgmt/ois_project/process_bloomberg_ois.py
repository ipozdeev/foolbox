from foolbox.api import *
from utils import parse_bloomberg_excel
# Parse the following bugger
file_to_parse = data_path + "ois_2000_2017_d.xlsx"
out_name = "ois_all_maturities_bloomberg.p"
out_name_on = "on_ois_rates_bloomberg.p"  # separate file for overnight rates

currencies_to_parse = \
    ["aud", "cad", "chf", "eur", "gbp", "jpy", "nzd", "sek", "usd"]

# Parse the data
ois_diff_maturities = parse_bloomberg_excel(file_to_parse, "tenor",
                                            currencies_to_parse)

# Make columns lower case
for currency in ois_diff_maturities.keys():
    ois_diff_maturities[currency].columns = \
     [col.lower() for col in ois_diff_maturities[currency].columns]

# Extract the overnight rates for a separate file
on_rates = pd.concat(
    [ois_diff_maturities[curr]["on"] for curr in currencies_to_parse], axis=1)
on_rates.columns = currencies_to_parse

# Dump all maturities together
with open(data_path + out_name, mode="wb") as halupa:
    pickle.dump(ois_diff_maturities, halupa)

# And on rates separately
with open(data_path + out_name_on, mode="wb") as halupa:
    pickle.dump(on_rates, halupa)
