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

    # converters ------------------------------------------------------------
    def converter(x):
        try:
            res = pd.to_datetime(x)
        except:
            res = pd.NaT
        return res

    converters = {}
    for p in range(N):
        converters[p*3] = converter

    # read in all sheets ----------------------------------------------------
    if not isinstance(data_sheet, (list, tuple)):
        data_sheet = [data_sheet, ]

    all_sheets = pd.read_excel(
        io=filename,
        sheetname=data_sheet,
        skiprows=2,
        header=None,
        converters=converters)

    all_data = {}

    for k, v in all_sheets.items():
        # k, v = "jpy", all_sheets["jpy"]
        # take every third third column: these are the values
        data = [v.iloc[:,(p*3):(p*3+1+1)].dropna() for p in range(N)]

        # pop dates as index
        for p in range(N):
            data[p].index = data[p].pop(p*3)

        # `data` is a list -> concat to a df
        data = pd.concat(data, axis=1, ignore_index=False)

        # columns are iso letters
        data.columns = colnames

        if len(data_sheet) < 2:
            return data
        else:
            all_data[k] = data

    return all_data

if __name__ == "__main__":
    from foolbox.api import *
    fname = data_path + "ois_2000_2017_d.xlsx"
    lol = parse_bloomberg_excel(filename=fname, data_sheet=["sek", "jpy"],
        colnames_sheet="tenor")
