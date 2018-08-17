"""Parses JP Morgan bond indices.
"""

import pandas as pd
import pickle
from foolbox.data_mgmt import set_credentials as set_cred

if __name__ == "__main__":
    # Set the user contingent path
    path = set_cred.gdrive_path("research_data/fx_and_events/")

    # Set the i/o names
    file_name = "bond_index_data.xlsm"
    out_name = "bond_index_data.p"

    # Parse the data, throw out the request table sheet
    raw_data = pd.read_excel(path + file_name, sheet_name=None, index_col=0,
                             parse_dates=True)
    raw_data.pop("REQUEST_TABLE")

    # Assign column names
    col_names = raw_data["maturity_map"].columns
    raw_data.pop("maturity_map")
    for key, df in raw_data.items():
        df.columns = col_names

    # Save the data
    with open(path + out_name, mode="wb") as halupa:
        pickle.dump(raw_data, halupa)
