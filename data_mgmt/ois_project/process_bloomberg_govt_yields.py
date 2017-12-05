from foolbox.api import *
from utils import parse_bloomberg_excel
# Parse the following bugger
file_to_parse = data_path + "govt_yld_2000_2017_d.xlsx"
out_name = "bond_yields_all_maturities_bloomberg.p"

maturities_to_parse = \
    ["1y", "2y", "3y", "4y", "5y", "7y", "10y"]

bonds_diff_maturities = parse_bloomberg_excel(file_to_parse, "iso",
                                              maturities_to_parse)

# Restructure dictionary to contain currencies as keys
out_data = dict()
# Loop over currencies
for curr in bonds_diff_maturities["1y"].columns:
    # Loop over maturities
    for maturity in maturities_to_parse:
        out_data[curr] = pd.concat(
            [bonds_diff_maturities[m][curr] for m in maturities_to_parse],
            axis=1)
    out_data[curr].columns = maturities_to_parse

    # Convert to float
    out_data[curr] = out_data[curr].astype(float)

# Dump
with open(data_path + out_name, mode="wb") as halupa:
    pickle.dump(out_data, halupa)
