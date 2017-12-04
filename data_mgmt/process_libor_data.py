import pandas as pd
import pickle
from foolbox.data_mgmt.set_credentials import set_path
from foolbox.data_mgmt.parse_bloomberg_excel import *

def fetch_libor_data(path_to_data=None, filename=None, data_sheet=None):
    """
    """
    if path_to_data is None:
        path_to_data = set_path("research_data/fx_and_events/")

    if filename is None:
        filename = "libor_2000_2017_d.xlsx"

    if data_sheet is None:
        data_sheet = ["on", "1m"]

    # parse the following bugger
    file_to_parse = path_to_data + filename

    # read in ---------------------------------------------------------------
    raw_data = parse_bloomberg_excel(
        filename=file_to_parse,
        colnames_sheet="iso",
        data_sheet=["on", "1m"])

    return raw_data

def fetch_libor_sub_data(path_to_data=None, filename=None, data_sheet=None):
    """
    """
    if path_to_data is None:
        path_to_data = set_path("research_data/fx_and_events/")

    if filename is None:
        filename = "libor_sub_2000_2017_d.xlsx"

    if data_sheet is None:
        data_sheet = ["on", "1m"]

    # parse the following bugger
    file_to_parse = path_to_data + filename

    # read in ---------------------------------------------------------------
    raw_data = parse_bloomberg_excel(
        filename=file_to_parse,
        colnames_sheet="iso",
        data_sheet=data_sheet)

    return raw_data

def fetch_on_data(path_to_data=None, filename=None):
    """
    """
    if path_to_data is None:
        path_to_data = set_path("research_data/fx_and_events/")

    if filename is None:
        filename = "ois_bloomberg.p"

    # parse the following bugger
    file_to_parse = path_to_data + filename

    raw_data = pd.read_pickle(file_to_parse)

    on_rates = pd.concat([v["ON"].rename(k) for k, v in raw_data.items()],
        axis=1)

    return on_rates

def splice_libor_data(libor, libor_sub, on_rates):
    """
    """
    res = dict()

    for k, v in libor.items():
        this_data = v.fillna(libor_sub[k])

        if k == "on":
            this_data = this_data.fillna(on_rates)

        res[k] = this_data.astype(float)

    return res


if __name__ == "__main__":

    # define file name for pickling
    pickle_name = "libor_spliced_2000_2007_d.p"

    libor = fetch_libor_data()
    libor_sub = fetch_libor_sub_data()
    on_rates = fetch_on_data()

    spliced_data = splice_libor_data(libor, libor_sub, on_rates)

    with open(path_to_data + pickle_name, mode="wb") as fname:
        pickle.dump(spliced_data, fname)
