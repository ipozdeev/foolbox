import pandas as pd


def parse_bloomberg_excel(filename, colnames_sheet, data_sheets, space=1,
                          colnames=None, **kwargs):
    """Parse .xlsx file constructed with Bloomberg Excel API.

    Parameters
    ----------
    data_sheets : list or string or None
        if None, all sheets are loaded
    colnames : list-like
    space : int
        number of empty columns between neighboring data series

    Returns
    -------
    this_data : pandas.DataFrame
        of data
    """
    if isinstance(data_sheets, str):
        data_sheets = [data_sheets, ]
        flag_single_input = True
    else:
        flag_single_input = False

    # converter for date
    def converter(x):
        try:
            res = pd.to_datetime(x)
        except Exception:
            res = pd.NaT
        return res

    # colnames_sheet
    if colnames is None:
        colnames = pd.read_excel(filename, sheet_name=colnames_sheet, header=0)
        colnames = colnames.columns

    # float converters
    float_conv = {
        k: float for k in list(range(1, len(colnames)*(2+space), 2+space))
    }

    # if data_sheets is None, read in all sheets
    if data_sheets is None:
        data_dict_full = pd.read_excel(filename, sheet_name=None,
                                       converters=float_conv, verbose=True,
                                       **kwargs)

        # remove the sheet with colnames from this dict
        data_dict_full.pop(colnames_sheet, None)

    else:
        data_dict_full = pd.read_excel(filename, sheet_name=data_sheets,
                                       converters=float_conv, verbose=True,
                                       **kwargs)

    # loop over sheetnames
    all_data = dict()

    for s, data_df in data_dict_full.items():
        # loop over triplets, map dates, extract
        new_data_df = []
        for p in range((data_df.shape[1]+1)//(space+2)):
            # this triplet
            this_piece = data_df.iloc[1:, p*(space+2):(p+1)*(space+2)-space]

            # map date
            this_piece.iloc[:, 0] = this_piece.iloc[:, 0].map(converter)

            # drop nans (with dates)
            this_piece = this_piece.dropna()

            # extract date as index
            this_piece = this_piece.set_index(this_piece.columns[0])

            # rename
            this_piece.columns = [colnames[p]]
            this_piece.index.name = "date"

            # drop duplicates from the index
            this_piece = this_piece.loc[
                ~this_piece.index.duplicated(keep='first'), :]

            # store
            new_data_df += [this_piece, ]

        # concat
        all_data[s] = pd.concat(new_data_df, axis=1, join="outer")

    # if only one sheet was asked for
    if flag_single_input:
        all_data = list(all_data.values())[0]

    return all_data