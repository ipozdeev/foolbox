from foolbox.api import *
from foolbox.utils import parse_bloomberg_excel

def fetch_datastream_ois_data(data_path=None):
    """
    """
    if data_path is None:
        data_path = set_credentials.set_path("research_data/fx_and_events/")

    # Parse the following bugger
    file_to_parse = data_path + "ois_data.xlsm"

    # Define the file name for pickling
    pickle_name = "ois_datastream.p"

    # Load sheets from excel into a dictionary of dataframes
    raw_data = pd.read_excel(file_to_parse, sheetname=None, index_col=0)

    # Populate the output
    ois_data = dict()

    ois_data["tr_1m"] = raw_data["ois_1m_tr"]
    ois_data["tr_3m"] = raw_data["ois_3m_tr"]
    ois_data["icap_1m"] = raw_data["ois_1m_icap"]
    ois_data["icap_3m"] = raw_data["ois_3m_icap"]

    for key in ois_data.keys():
        ois_data[key].columns = [col.lower() for col in ois_data[key].columns]

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

    # read in ---------------------------------------------------------------
    ois_data = parse_bloomberg_excel(
        filename=fname,
        colnames_sheet="tenor",
        data_sheets=["aud","cad","chf","gbp","nzd","sek","usd","eur"])

    # Pickle the bastard ----------------------------------------------------
    # Define the file name for pickling
    pickle_name = "ois_bloomberg.p"

    with open(data_path + pickle_name, mode="wb") as fname:
        pickle.dump(ois_data, fname)

def merge_ois_data(datastream_pkl=None, bloomberg_pkl=None, maturity='1M',
    priority="itb"):
    """
    Parameters
    ----------
    priority : str
        of 3 letter: 'i' for ICAP, 't' for Thomson Reuters and 'b' for Bloomi;
        the order that the letters obey determines the priority
    """
    if datastream_pkl is None:
        data_path = set_credentials.set_path("research_data/fx_and_events/")
        datastream_pkl = data_path + "ois_datastream.p"

    if bloomberg_pkl is None:
        data_path = set_credentials.set_path("research_data/fx_and_events/")
        bloomberg_pkl = data_path + "ois_bloomberg.p"

    # read in both
    with open(datastream_pkl, mode='rb') as fname:
        ois_datastream = pickle.load(fname)
    with open(bloomberg_pkl, mode='rb') as fname:
        ois_bloomberg = pickle.load(fname)

    # for each data provider, select maturity
    ois_ds_tr = ois_datastream["tr_"+maturity.lower()]
    ois_ds_icap = ois_datastream["icap_"+maturity.lower()]
    ois_bl = pd.concat(
        [p.loc[:,maturity].to_frame(c) for c, p in ois_bloomberg.items()],
        axis=1)

    # collect into dictionary to be able to prioritize
    all_three = {
        'i': ois_ds_icap,
        't': ois_ds_tr,
        'b': ois_bl}

    # reindex data (according to what is given highest priority)
    priority = list(priority)

    new_idx = ois_bl.index.union(ois_ds_icap.index.union(ois_ds_tr.index))
    new_col = \
        ois_bl.columns.union(ois_ds_icap.columns.union(ois_ds_tr.columns))

    new_data = all_three[priority[0]].reindex(index=new_idx, columns=new_col)\
            .fillna(all_three[priority[1]])\
            .fillna(all_three[priority[2]])

    # dropna
    new_data.dropna(how="all", inplace=True)

    # pickle
    pickle_name = "ois_merged_" + maturity.lower() + ".p"

    with open(data_path + pickle_name, mode="wb") as fname:
        pickle.dump(new_data, fname)


if __name__ == "__main__":
    fetch_bloomberg_ois_data(data_path=data_path)
    merge_ois_data(priority="bit")
