from foolbox.api import *
from foolbox.utils import parse_bloomberg_excel

def fetch_datastream_ois_data(data_path=None):
    """
    """
    if data_path is None:
        data_path = set_credentials.set_path("research_data/fx_and_events/")

    # Parse the following bugger
    file_to_parse = data_path + "ois_tr_icap_1999_2017_d.xlsm"

    # Define the file name for pickling
    pickle_name = "ois_tr_icap_1m_3m.p"

    # Load sheets from excel into a dictionary of dataframes
    raw_data = pd.read_excel(file_to_parse, sheetname=None, index_col=0)

    # columns to lowercase
    for key in raw_data.keys():
        raw_data[key].columns = [col.lower() for col in raw_data[key].columns]

    # Populate the output
    ois_data = dict()

    ois_data["tr"] = {
        "1m": raw_data["ois_1m_tr"],
        "3m": raw_data["ois_3m_tr"]
    }
    ois_data["icap"] = {
        "1m": raw_data["ois_1m_icap"],
        "3m": raw_data["ois_3m_icap"]
    }

    # Pickle the bastard
    with open(data_path + pickle_name, mode="wb") as fname:
        pickle.dump(ois_data, fname)


def fetch_bloomberg_ois_data(data_path=None):
    """
    NB: something is wrong here, some keys in output have empty values...
    """
    if data_path is None:
        data_path = set_credentials.set_path("research_data/fx_and_events/")

    fname = data_path + "ois_2000_2017_d.xlsx"

    curs = ["aud", "cad", "chf", "eur", "gbp", "jpy", "nzd", "sek", "usd"]

    # read in ---------------------------------------------------------------
    ois_data = parse_bloomberg_excel(
        filename=fname,
        colnames_sheet="tenor",
        data_sheets=curs)

    # pivot -----------------------------------------------------------------
    ois_data_xarr = {(tau.lower(), c.lower()): ois_data[c].loc[:, tau]
                     for c in ois_data.keys() for tau in ois_data[c].columns}
    ois_data_xarr = pd.DataFrame(ois_data_xarr).astype(float)

    ois_data_dict = {
        k: ois_data_xarr.loc[:, k] for k in ois_data_xarr.columns.levels[0]
    }

    # Pickle the bastard ----------------------------------------------------
    # Define the file name for pickling
    on_pickle_name = "overnight_rates.p"
    ois_pickle_name = "ois_bloomi_1w_30y.p"

    with open(data_path + on_pickle_name, mode="wb") as fname:
        pickle.dump(ois_data_dict.pop("on"), fname)
    with open(data_path + ois_pickle_name, mode="wb") as fname:
        pickle.dump(ois_data_dict, fname)


def fetch_tr_ois_data(data_path=None):
    """

    Parameters
    ----------
    data_path

    Returns
    -------

    """
    if data_path is None:
        data_path = set_credentials.set_path("research_data/fx_and_events/")

    fname = data_path + "tr_ois_2000_2017_d.xlsx"

    curs = ["aud", "cad", "chf", "eur", "gbp", "jpy", "nzd", "sek", "usd"]

    # read in ---------------------------------------------------------------
    ois_data = parse_bloomberg_excel(filename=fname, colnames_sheet="tenor",
                                     data_sheets=curs, space=1, skiprows=2,
                                     header=None)

    # pivot -----------------------------------------------------------------
    ois_data_xarr = {(tau.lower(), c.lower()): ois_data[c].loc[:, tau]
                     for c in ois_data.keys() for tau in ois_data[c].columns}

    ois_data_xarr = pd.DataFrame(ois_data_xarr).astype(float)

    ois_data_dict = {
        k: ois_data_xarr.loc[:, k] for k in ois_data_xarr.columns.levels[0]
    }

    # Pickle the bastard ----------------------------------------------------
    # Define the file name for pickling
    on_pickle_name = "overnight_rates_tr.p"
    ois_pickle_name = "ois_tr_1w_30y.p"

    with open(data_path + on_pickle_name, mode="wb") as fname:
        pickle.dump(ois_data_dict.pop("on"), fname)
    with open(data_path + ois_pickle_name, mode="wb") as fname:
        pickle.dump(ois_data_dict, fname)


def merge_ois_data(*args):
    """
    Parameters
    ----------
    priority : str
        of 3 letter: 'i' for ICAP, 't' for Thomson Reuters and 'b' for Bloomi;
        the order that the letters obey determines the priority
    """
    # if data_path is None:
    #     data_path = set_credentials.set_path("research_data/fx_and_events/")
    #
    # if ds_pkl is None:
    #     ds_pkl = data_path + "ois_tr_icap_1m_3m.p"
    #
    # if bloomi_pkl is None:
    #     bloomi_pkl = data_path + "ois_bloomi_1w_30y.p"

    # # read in both
    # ois_ds = pd.read_pickle(ds_pkl)
    # ois_bloomi = pd.read_pickle(bloomi_pkl)

    # # for each data provider (bloomi already a dict)
    # ois_ds_tr = ois_ds["tr"]
    # ois_ds_icap = ois_ds["icap"]

    data = args[0]

    for mat, v in data.items():

        for d in args[1:]:

            new_df = d.get(mat, pd.DataFrame())
            v, _ = v.align(new_df, axis=0, join="outer")
            v.fillna(new_df, inplace=True)

    # # loop over maturities
    # for m in ois_bloomi.keys():
    #
    #     # collect into dictionary to be able to prioritize
    #     all_three = {
    #         'i': ois_ds_icap.get(m, pd.DataFrame()),
    #         't': ois_ds_tr.get(m, pd.DataFrame()),
    #         'b': ois_bloomi.get(m, pd.DataFrame())}
    #
    #     # reindex data (according to what is given highest priority)
    #     priority = list(priority)
    #
    #     new_idx = all_three['b'].index \
    #         .union(all_three['i'].index.union(all_three['t'].index))
    #     new_col = all_three['b'].columns \
    #         .union(all_three['i'].columns.union(all_three['t'].columns))
    #
    #     new_data = all_three[priority[0]].reindex(index=new_idx, columns=new_col) \
    #         .fillna(all_three[priority[1]]) \
    #         .fillna(all_three[priority[2]])
    #
    #     # dropna
    #     data[m] = new_data.dropna(how="all").astype(float)

    # pickle
    pickle_name = "ois_merged_{}.p".format(len(args))

    with open(data_path + pickle_name, mode="wb") as fname:
        pickle.dump(data, fname)

    return data


if __name__ == "__main__":

    # fetch_datastream_ois_data(data_path=data_path)
    # fetch_bloomberg_ois_data(data_path=data_path)
    # res = merge_ois_data(priority="bit")
    # fetch_tr_ois_data()

    ois_tr = pd.read_pickle(data_path + "ois_tr_1w_30y.p")
    ois_bl = pd.read_pickle(data_path + "ois_bloomi_1w_30y.p")
    ois_two = pd.read_pickle(data_path + "ois_tr_icap_1m_3m.p")
    ois_ic = ois_two["icap"]
    ois_tr_old = ois_two["tr"]

    new_merged = merge_ois_data(ois_bl, ois_ic, ois_tr_old, ois_tr)
    # old_merged = pd.read_pickle(data_path + "ois_bloomi_1w_30y.p")


