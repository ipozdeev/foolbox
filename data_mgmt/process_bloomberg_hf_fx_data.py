import pandas as pd
import re

from foolbox.api import set_credentials as set_cred
from foolbox.utils import parse_bloomberg_excel


def main(path_to_data, filename):
    """

    Parameters
    ----------
    path_to_data : str
    filename : str
        ending in .xls*

    Returns
    -------

    """
    wrong_cur = ["cad", "chf", "jpy", "nok", "sek"]

    bid_ask = parse_bloomberg_excel(path_to_data + filename,
                                    colnames_sheet="colnames",
                                    data_sheets=None, space=1, skiprows=5)

    for k, v in bid_ask.items():
        v.loc[:, wrong_cur] = 1/v.loc[:, wrong_cur]
        v.index = v.index.tz_localize("US/Eastern")
        bid_ask[k] = v

    bid_ask.update({"mid": (bid_ask["ask"] + bid_ask["bid"])/2})

    pd.to_pickle(bid_ask, path_to_data + re.sub("xlsx", "p", filename))


if __name__ == '__main__':
    path_to_xlsx = set_cred.set_path("research_data/fx_and_events/",
                                     which="gdrive")
    filename = "fx_hf_ba_2017_2018.xlsx"

    main(path_to_xlsx, filename)
