import pandas as pd
import pickle

# Set the user contingent path
from foolbox.data_mgmt import set_credentials as set_cred
path = set_cred.gdrive_path("research_data/fx_and_events/")

def parse_bond_data():
    """

    Returns
    -------

    """

    # Set the i/o names
    file_name = "bond_data.xlsm"
    out_name = "bond_data.p"

    # Parse the data, throw out the request table sheet
    raw_data = pd.read_excel(path + file_name, sheetname=None, index_col=0)
    raw_data.pop("REQUEST_TABLE")

    # Save the data
    with open(path + out_name, mode="wb") as halupa:
        pickle.dump(raw_data, halupa)


def parse_bond_futures_data():
    """

    Returns
    -------

    """
    # Set the i/o names
    file_name = "bond_cont_futures_2000_2018_d.xlsm"
    out_name = "bond_cont_futures_2000_2018_d.p"

    # Parse the data, throw out the request table sheet
    raw_data = pd.read_excel(path + file_name, sheet_name=None, index_col=0)
    raw_data.pop("REQUEST_TABLE")

    for k, v in raw_data.items():
        v.index = v.index.to_datetime()
        raw_data[k] = v.rename(columns={v.columns[0]: "short",
                                        v.columns[1]: "long"})

    # Save the data
    with open(path + out_name, mode="wb") as hangar:
        pickle.dump(raw_data, hangar)


if __name__ == "__main__":
    parse_bond_futures_data()