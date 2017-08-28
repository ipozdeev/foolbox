import pandas as pd

def parse_bloomberg_excel(filename, data_sheet, colnames_sheet):
    """
    filename = fname_1w_settl
    Returns
    -------
    this_data : dict
        with "NYC", "LON", "TOK" as keys, pandas.DataFrames as values
    """
    colnames = pd.read_excel(filename, sheetname=colnames_sheet, header=0)
    colnames = colnames.columns

    N = len(colnames)

    converters = {}
    for p in range(N):
        converters[p*3] = lambda x: pd.to_datetime(x)

    # s = "NYC"
    data = pd.read_excel(
        io=filename,
        sheetname=data_sheet,
        skiprows=2,
        header=None,
        converters=converters)

    # take every third third column: these are the values
    data = [data.ix[:,(p*3):(p*3+1)].dropna() for p in range(N)]

    # pop dates as index
    for p in range(N):
        data[p].index = data[p].pop(p*3)

    # `data` is a list -> concat to a df
    data = pd.concat(data, axis=1, ignore_index=False)

    # columns are iso letters
    data.columns = colnames

    return data

if __name__ == "__main__":
    from foolbox.api import *
    fname = data_path + "ois_2000_2017_d.xlsx"
    lol = parse_bloomberg_excel(fname, data_sheet="sek", colnames_sheet="tenor")
