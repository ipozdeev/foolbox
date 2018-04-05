import requests
from socketIO_client import SocketIO
import pandas as pd
from foolbox.data_mgmt import set_credentials as set_cred
import pickle

# API settings
TRADING_API_URL = "https://api-demo.fxcm.com:443"
WEBSOCKET_PORT = 443
socketIO = None
ACCESS_TOKEN = "eb2ebc9abd4790aa4d2fe22a6107b2fb45081998"

socketIO = SocketIO(TRADING_API_URL, WEBSOCKET_PORT,
                    params={"access_token": ACCESS_TOKEN})

bearer_access_token = "Bearer {0}{1}".format(socketIO._engineIO_session.id,
                                             ACCESS_TOKEN)

headers = {"User-Agent": "request",
           "Accept": "application/json",
           "Content-Type": "application/x-www-form-urlencoded",
           "Authorization": bearer_access_token}


def fx_pair_to_fxcm_id(fx_pairs):
    """Maps currency pairs to fxcm integer id"s

    Parameters
    ----------
    fx_pairs: iterable
        of strings with the format "Base currency/Counter Currency", e.g.
        "EUR/USD"

    Returns
    -------
    fxcm_id_map: pd.Series
        indexed by "fx_pairs" with values being integer fxcm id"s

    """
    # Get the instruments the current token is subscribed to
    offers = requests.get(TRADING_API_URL+"/trading/get_model",
                          headers=headers, params={"models": "Offer"}).json()
    offers = pd.DataFrame(offers["offers"])

    # Fetch the map
    fxcm_id_map = offers.set_index("currency").loc[fx_pairs, "offerId"]

    return fxcm_id_map


def get_fxcm_data_by_chunks(id, start_date, end_date, frequency):
    """Given fxcm instrument id and request range and frequency, fetch the
    data by chunks. This is a utility function and is not intended to be used
    outside the 'get_fcxm_data()' function

    Parameters
    ----------
    id: int
        identificator of FXCM instrument
    start_date: int
        unix Epoch stamp
    end_date:
        unix Epoch stamp
    frequency: str
        specifying data frequency from one to thirty minutes: m1, m5, m15, m30

    Returns
    -------
    data: pd.DataFrame
        of raw fxcm request, integer columns correspond to "bid_open",
        "bid_close", "bid_high", "bid_low", "ask_open", "ask_close",
        "ask_high", "ask_low", "num_ticks" but to be explicitly processed
        outside of this function

    """
    # FXCM can deliver roughly 150k datapoints per request, specify the
    # dictionary mapping fxcm frequency to date range of a request in seconds
    frequency_map = {"m1": int(150e3 * 60),
                     "m5": int(150e3 * 60 * 5),
                     "m15": int(150e3 * 60 * 15),
                     "m30": int(150e3 * 60 * 30),
                     }

    # Get the request step
    rqst_step = frequency_map[frequency]

    # Pool requests in this list, initialize the request loop
    data = list()
    this_start = start_date
    this_end = start_date + rqst_step

    if this_end > end_date:
        this_end = end_date

    while this_end <= end_date:

        # If the last date has already been reached break the while loop
        if this_start == this_end:
            break

        # Get the data
        this_data = requests.get(
            TRADING_API_URL+"/candles/"+str(id)+"/"+frequency,
            headers=headers, params={"num": 1, "from": this_start,
                                     "to": this_end}
             ).json()["candles"]

        # Update iterators
        this_start = this_end
        this_end = this_start + rqst_step

        # Check if there is enough data for the last step
        if this_end > end_date:
            this_end = end_date

        # Filter out None's from the output
        this_data = [x for x in this_data if x is not None]

        # Check if the list is empty
        if not this_data:
            print("FXCM ID {} has returned no data for the period from {}"
                  " to {}.".format(id,
                                   pd.Timestamp.utcfromtimestamp(this_start),
                                   pd.Timestamp.utcfromtimestamp(this_end)))
            continue

        # Append the output
        data.append(pd.DataFrame(this_data))

    # Aggregate and drop duplicate stamps, they are in the first column
    data = pd.concat(data, ignore_index=True).drop_duplicates([0])

    return data


def get_fcxm_data(fx_pairs, frequency, start_date, end_date, num_periods=None):
    """Given an iterable of currency pairs and request settings, fetches FXCM
    data.

    Parameters
    ----------
    fx_pairs: iterable
        of strings with the format "Base currency/Counter Currency", e.g.
        "EUR/USD"
    frequency: str
        specifying data frequency from one minute to one month: m1, m5, m15,
        m30, H1, H2, H3, H4, H6, H8, D1, W1, M1
    start_date: datelike
        start date of the request
    end_date: datelike
        end date of the request
    num_periods:
        number of periods in requested data. If not None, then "start_date" and
        "end_date" parameters are redundant and not used

    Returns
    -------
    data: dict
        of pd.DataFrame"s with "fx_pairs" as columns. Dictionary has the
        following keys, each containing the corresponding datatype: "bid_open",
        "bid_close", "bid_high", "bid_low", "ask_open", "ask_close",
        "ask_high", "ask_low", "num_ticks"

    """
    # Convert start and end dates to timestamps
    if start_date is not None and end_date is not None:
        # FXCM accpets integer POSIX stamps, "to_pydatetime()" ensures that
        # timestamp is not automatically adjusted to the timezone difference
        start_date = int(pd.Timestamp(start_date).to_pydatetime().timestamp())
        end_date = int(pd.Timestamp(end_date).to_pydatetime().timestamp())

        # Even if start and end dates are provided, FXCM API requires
        # num_periods to be an integer, arbitrarily set it to 1
        num_periods = 1

    elif num_periods is None:
        raise ValueError("Provide either num_periods or both start and end "
                         "dates.")

    # Map currency pairs to FXCM ID's
    id_map = fx_pair_to_fxcm_id(fx_pairs)

    # Set up the output structure
    bid_open = list()
    bid_close = list()
    bid_high = list()
    bid_low = list()
    ask_open = list()
    ask_close = list()
    ask_high = list()
    ask_low = list()
    num_ticks = list()

    # Loop over the currency pairs and fetch the data
    for curr, id in id_map.iteritems():

        # For high frequency requests query the data in chunks
        if frequency in ["m1", "m5", "m15", "m30"]:
            print("FFFFFUUUUUUU! Invoking the high frequency mode. The "
                  "requests are gonna be chunkized. Currency is {}, its ID "
                  "is {}.".format(curr, id))

            this_data = \
                get_fxcm_data_by_chunks(id, start_date, end_date, frequency)

        else:
            # Normal mode: pull the data in a single request
            this_data = requests.get(
             TRADING_API_URL+"/candles/"+str(id)+"/"+frequency,
                headers=headers, params={"num": num_periods,
                                         "from": start_date, "to": end_date}
             ).json()["candles"]

            # Filter out None's from the retrieved data
            this_data = [x for x in this_data if x is not None]

            # Check if the list is empty
            if not this_data:
                print("{} has returned no data for the period from {} "
                      "to {}.".format(
                    curr, pd.Timestamp.utcfromtimestamp(start_date),
                    pd.Timestamp.utcfromtimestamp(end_date))
                    )
                continue
            else:
                print("Data for {} has been downloaded.".format(curr))

        # Transform to dataframe, assign column, names, index by stamps
        this_data = pd.DataFrame(this_data)
        this_data.columns = ["stamp", "bid_open", "bid_close", "bid_high",
                             "bid_low", "ask_open", "ask_close", "ask_high",
                             "ask_low", "num_ticks"]

        # Convert to utc timestamps
        this_data["stamp"] = this_data["stamp"].apply(
            lambda x: pd.Timestamp.utcfromtimestamp(x).tz_localize("UTC"))
        this_data = this_data.set_index("stamp")

        # Append the output lists
        bid_open.append(this_data["bid_open"].rename(curr))
        bid_close.append(this_data["bid_close"].rename(curr))
        bid_high.append(this_data["bid_high"].rename(curr))
        bid_low.append(this_data["bid_low"].rename(curr))
        ask_open.append(this_data["ask_open"].rename(curr))
        ask_close.append(this_data["ask_close"].rename(curr))
        ask_high.append(this_data["ask_high"].rename(curr))
        ask_low.append(this_data["ask_low"].rename(curr))
        num_ticks.append(this_data["num_ticks"].rename(curr))

    # Construct the output
    data = {
        "bid_open": pd.concat(bid_open, axis=1),
        "bid_close": pd.concat(bid_close, axis=1),
        "bid_high": pd.concat(bid_high, axis=1),
        "bid_low": pd.concat(bid_low, axis=1),
        "ask_open": pd.concat(ask_open, axis=1),
        "ask_close": pd.concat(ask_close, axis=1),
        "ask_high": pd.concat(ask_high, axis=1),
        "ask_low": pd.concat(ask_low, axis=1),
        "num_ticks": pd.concat(num_ticks, axis=1)
        }

    return data


if __name__ == "__main__":
    # Define the currency pairs of interest, data frequency and request period
    fxcm_currs = ["AUD/USD", "USD/CAD", "USD/CHF", "EUR/USD", "GBP/USD",
                  "USD/JPY", "USD/NOK", "NZD/USD", "USD/SEK"]
    data_frequency = "m15"
    s_dt = "1999-12-31"
    e_dt = "2018-03-31"

    # Set output path and output names
    out_path = set_cred.gdrive_path("research_data/fx_and_events/")
    out_raw_name = "fxcm_raw_" + data_frequency + ".p"
    out_counter_usd_name = "fxcm_counter_usd_" + data_frequency + ".p"

    data_raw = get_fcxm_data(fx_pairs=fxcm_currs, frequency=data_frequency,
                             start_date=s_dt, end_date=e_dt)

    # Convert datapoints to USD being countercurrency if it's not
    data_usd_counter = dict()
    for key, df in data_raw.items():
        # Don't convert ticks though. That would be silly
        if key != "num_ticks":
            tmp_cols = list()
            tmp_df = df.copy()

            # Convert currencies where USD is the base
            for col in df.columns:
                if col[:3] == "USD":
                    tmp_df[col] = tmp_df[col].pow(-1)
                    tmp_cols.append(col[-3:].lower())

                else:
                    tmp_cols.append(col[:3].lower())

            # Set the columns as the lower case single currency names
            tmp_df.columns = tmp_cols
            data_usd_counter[key] = tmp_df

        else:
            # The number of ticks is the same irrespectively of quoting
            # convention
            data_usd_counter[key] = df

    # Dump the data
    with open(out_path+out_raw_name, mode="wb") as hut:
        pickle.dump(data_raw, hut)
    with open(out_path+out_counter_usd_name, mode="wb") as hut:
        pickle.dump(data_usd_counter, hut)

    print("kek")

    # Define the currency pairs of interest, data frequency and request period
    # fxcm_currs = ["EUR/USD"]
    # data_frequency = "m5"
    # s_dt = "2018-04-04"
    # e_dt = "2018-04-05"
    #
    # # Set output path and output names
    # out_path = set_cred.gdrive_path("research_data/fx_and_events/")
    #
    # data_raw = get_fcxm_data(fx_pairs=fxcm_currs, frequency=data_frequency,
    #                          start_date=s_dt, end_date=e_dt)
    #
    # mid = (data_raw["ask_close"] + data_raw["bid_close"])/2
    # mid.plot()

