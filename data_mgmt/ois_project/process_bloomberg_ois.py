from foolbox.api import *
from utils import parse_bloomberg_excel
# Parse the following bugger
file_to_parse = data_path + "ois_2000_2017_d.xlsx"
out_name = "ois_bloomberg.p"

currencies_to_parse = \
    ["usd", "eur", "jpy", "chf", "gbp", "aud", "cad", "nzd", "sek"]
currencies_to_parse = ["chf"]

ois_diff_maturities = dict()
for curr in currencies_to_parse:
    print(curr)
    ois_diff_maturities[curr] = \
        parse_bloomberg_excel(file_to_parse, curr, "tenor")


with open(data_path + out_name, mode="wb") as halupa:
    pickle.dump(ois_diff_maturities, halupa)
