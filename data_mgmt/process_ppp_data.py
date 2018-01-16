import pandas as pd
import pickle

from foolbox.data_mgmt.set_credentials import set_path


def process_ppp_data():
    """
    """
    path_to_data = set_path("research_data/fx_and_events/")

    # load
    ppp_data = pd.read_excel(path_to_data + "ppp_1990_2017_y.xlsx",
                             index_col=0, header=0, sheetname="ppp")

    # to have all data in for xxxusd
    ppp_data = 1 / ppp_data

    # save
    with open(path_to_data + "ppp_1990_2017_y.p", mode="wb") as hangar:
        pickle.dump(ppp_data, hangar)

    return


if __name__ == "__main__":
    process_ppp_data()
