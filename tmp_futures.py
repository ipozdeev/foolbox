"""Here dwell functions for getting futures data from quandl and performing
some basic operations like rollover on them.
"""
# TODO: where to move this module, shal it be OOPed?

from foolbox.api import *
import quandl
quandl.ApiConfig.api_key = "TQTWuU5e53sYEykyzzjW"


def load_quandl_futures_data(name_pattern, start_date, end_date):
    """Downloads raw futures data for individual contracts from quandl.

    Parameters
    ----------
    name_pattern: str
        specifying the quandl contract prefix, eg.
            'CME/CL'  - CME WTI crude oil futures
            'CME/FF'  - CME federal funds futures
            'CBOE/VX' - CBOE VIX index futures
    start_date: str
        of the format Month-YYYY, e.g. March-2009 or Mar-2009, specifying the
        first futures contract to load
    end_date: str
        of the format Month-YYYY, e.g. March-2009 or Mar-2009, specifying the
        last futures contract to load
    Returns
    -------
    raw_data: dict
        with keys 'month - year' corresponding to the expiry date of each
        contract, i.e. 'V2017' - October 2017, 'Z2018' - December 2018

    """
    # Define the map from month's number to futures contract letter code
    name_map = {
        "1": "F", "2": "G", "3": "H", "4": "J", "5": "K", "6": "M", "7": "N",
        "8": "Q", "9": "U", "10": "V", "11": "X", "12": "Z"
               }

    # Get the months' numbers and years
    sample = pd.period_range(start_date, end_date, freq="M")
    years = sample.year.tolist()
    months = sample.month.tolist()

    # Convert months into futures contracts codes
    months = [name_map[str(month_num)] for month_num in months]

    # Construct requests
    requests = [month + str(year) for month, year in list(zip(months, years))]

    raw_data = dict()

    # Get the data
    for request in requests:
        try:
            raw_data[request] = quandl.get(name_pattern + request)

        except quandl.errors.quandl_error.NotFoundError:
            # If a particular contract is not available, don't panic:
            # some contracts expiry every quarter, loop goes over months
            print("Contract {} not found in the Quandl DB".format(request))

    return raw_data


def get_futures_datatype(futures_raw_data, datatype):
    """Loops over keys in 'futures_raw_data', extracting the requested datatype
    and returning a structured dataset 'data' where data for individual
    contracts is ordered chronologically according to contracts' expiry dates

    Parameters
    ----------
    futures_raw_data: dict
        with keys 'month - year' corresponding to the expiry date of each
        contract, i.e. 'V2017' - October 2017, 'Z2018' - December 2018
    datatype: str
        specifying which datatype to return "Settle", "Open", "Volume", etc.

    Returns
    -------
    data: pd.DataFrame
        with columns being 'pd.Timestamp's corresponding to futures' expiry,
        containing the data series for the requested datatype

    """

    naming_map = {
        "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6, "N": 7, "Q": 8, "U": 9,
        "V": 10, "X": 11, "Z": 12,
        }

    # Loop over the futures, appending the list to be further concatenated
    data = []
    for key in futures_raw_data.keys():
        # Get the expiry date as a string
        month = str(naming_map[key[0]])
        year = key[1:]
        # Offset to month end, corresponding to the actual expiry date
        exp_date = pd.Timestamp(year + "-" + month) + pd.offsets.MonthEnd()

        # Append the output to list
        tmp_df = futures_raw_data[key][[datatype]]
        tmp_df.columns = [exp_date]
        data.append(tmp_df)

    # Concatenate futures of different expiries, sort in chronological order
    data = pd.concat(data, join="outer", axis=1)
    data = data[sorted(data.columns)]

    return data


def continuos_futures(returns, roll_dates, depth=0):
    """Constructs a continuous futures returns, given a dataframe 'data' with
    returns and 'roll dates' specifying when the contract is to be rolled

    Parameters
    ----------
    returns: pd.DataFrame
        with each columns containing prices of individual futures contracts
        ordered chronologically by expiry date
    roll_dates: pd.Series
        of 'pd.Timestamp's which are a subset of the index of 'data' and
        ordered chronologically. Indexed by individual contracts
    depth: int
        controlling 'frontmonthsness' of the continuous futures series, with
        0 corresponding to the rolling the frontmonth contracts;
        1 - to the continuos rolling of the contracts a level deeper that the
        nearest maturity and so on

    Returns
    -------
    cont_futures: pd.DataFrame
        with simple returns to continuously rolled futures

    """
    # Construct the dataframe summarizing the rolling rules:
    # Each contract is held between dates in the "from" and "to" columns
    roll_df = \
        pd.concat([roll_dates.shift(1), roll_dates], axis=1).dropna()
    roll_df.columns = ["from", "to"]

    # The contract which is held depends on depth: 0 - is nearest to expire
    roll_df["contract"] = roll_df.index
    roll_df["contract"] = roll_df["contract"].shift(-depth)

    # Drop nan's appearing from shifting the series
    roll_df = roll_df.where(roll_df["from"] != roll_df["to"]).dropna()

    # Structure the output
    out_name = "cont_"+str(depth)
    cont_futures = \
        pd.DataFrame(index=returns.index, columns=[out_name], dtype="float")

    # Simply gather the return of each contract between the 'from' and 'two'
    # dates into single series. Loop over each contract:
    for index, roll_info in roll_df.iterrows():
        # Some shortcut notation
        this_contract = roll_info["contract"]
        hold_from = roll_info["from"]
        hold_to = roll_info["to"]

        # Hold the last available contract til the end of the dataset
        if index == roll_df.index[-1]:
            cont_futures.loc[hold_from:, out_name] = \
                returns.loc[hold_from:, this_contract]
        else:
            # Otherwise, hold between the specified dates
            cont_futures.loc[hold_from:hold_to, out_name] = \
                returns.loc[hold_from:hold_to, this_contract]

    return cont_futures


if __name__ == "__main__":

    # # =========================================================================
    # # EXAMPLE: Load and save S&P500 futures data
    # name_pattern = "CME/SP"
    # start_date = "Jan-1993"
    # end_date = "Dec-2018"
    #
    # sp_data = load_quandl_futures_data(name_pattern, start_date, end_date)
    #
    # filename = data_path + "sp_futures_raw.p"
    # with open(filename, mode="wb") as hut:
    #     pickle.dump(sp_data, hut)
    # #==========================================================================

    # name_pattern = "MX/CGB"
    # start_date = "Jan-1993"
    # end_date = "Dec-2018"
    # _data = load_quandl_futures_data(name_pattern, start_date, end_date)
    # filename = data_path + "cgb10_futures_raw.p"
    # with open(filename, mode="wb") as hut:
    #     pickle.dump(_data, hut)
    # =========================================================================
    # EXAMPLE: get settlement prices of S&P500, construct continuous futures,
    # run an event study

    # Feel free to test the stuff for oil futures, or for the fucking goddamn
    # LEAN HOGS
    filename = data_path + "sp_futures_raw.p"
    filename = data_path + "wti_futures_raw.p"
    filename = data_path + "cgb10_futures_raw.p"
    # filename = data_path + "gold_futures_raw.p"
    # filename = data_path + "lean_hogs_futures_raw.p"
    # filename = data_path + "fed_funds_futures_raw.p"
    # filename = data_path + "nzd_futures_raw.p"
    # filename = data_path + "aud_futures_raw.p"
    # filename = data_path + "ftse100_futures_raw.p"
    # filename = data_path + "vix_futures_raw.p"
    # filename = data_path + "nikkei_futures_raw.p"
    # filename = data_path + "coffee_futures_raw.p"
    # filename = data_path + "brent_futures_raw.p"
    # with open(filename, mode="rb") as hut:
    #     sp_data = pickle.load(hut)

    sp_data = pd.read_pickle(filename)

    # Get the settlement prices
    settle = get_futures_datatype(sp_data, "Settlement Price")
    volume = get_futures_datatype(sp_data, 'Volume')

    # Get the roll dates: roll contracts 10 days before the last trading day
    roll_dates = settle.apply(lambda x: x.dropna().index[-10])

    # Get the continuous returns for contracts maturing up to a year ahead
    # (SP futures expire every quarter 4-layer depth should span a year)
    cunt_rat = [continuos_futures(settle.pct_change(),
                                  roll_dates, k) for k in range(4)]
    cunt_rat = pd.concat(cunt_rat, join="outer", axis=1).dropna(how="all")*100
    # cunt_rat = cunt_rat.sub(cunt_rat["cont_0"], axis=0).drop(["cont_0"],
    #                                                          axis=1)

    cunt_rat = [continuos_futures(volume.pct_change(),
                                  roll_dates, k) for k in range(1)]
    cunt_rat = pd.concat(cunt_rat, join="outer", axis=1).dropna(how="all")*100

    # Load the events
    with open(data_path + settings["events_data"], mode='rb') as fname:
        events_data = pickle.load(fname)

    s_dt = "1995-01"
    e_dt = "2017-06"
    wa, wb, wc, wd = -10, -1, 1, 5
    window = (wa, wb, wc, wd)

    events = events_data["rba"]["change"]
    events = events.loc[s_dt:e_dt]

    # Make the same event for each contract
    events = pd.concat([events for k in cunt_rat.columns], axis=1)
    events.columns = cunt_rat.columns

    # Select the data for the event study
    data = cunt_rat.loc[s_dt:e_dt]
    test_events = events.copy().where(events < 0)

    normal_data = data.rolling(66).mean().shift(1)

    es = EventStudy.with_normal_data(events, test_events, wind,
                                     mean_type="count_weighted",
                    norm_data_method="between_events", x_overlaps=True)

    # es = EventStudy(data, test_events, wind, mean_type="count_weighted",
    #                 normal_data=normal_data, x_overlaps=True)

    ci_boot_c = es.get_ci(ps=(0.025, 0.975), method="boot", n_blocks=10, M=10)
    es.plot()
