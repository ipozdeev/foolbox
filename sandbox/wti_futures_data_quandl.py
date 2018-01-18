"""Here dwell downloading and pickling of fed funds futures data from quandl
"""
from foolbox.api import *
import quandl
quandl.ApiConfig.api_key = "TQTWuU5e53sYEykyzzjW"


def load_ff_futures_data(filename, start_date, end_date):
    """Downloads or updates fed funds futures data on selection of contracts
    creating a new or updating an existing pickle file

    Parameters
    ----------
    filename: str
        path and filename to the pickle file containing raw fed fund futures
        data
    start_date: str
        of the format Month-YYYY, e.g. March-2009 or Mar-2009, specifying the
        first futures contract to load/update
    end_date: str
        of the format Month-YYYY, e.g. March-2009 or Mar-2009, specifying the
        last futures contract to load/update

    Returns
    -------
    pickle file in the specified path, containting a dictionary of dataframes
    with Quandl data on the requested futures contracts

    """
    # Check if file exists (update mode)
    try:
        with open(filename, mode="rb") as fname:
            ff_data = pickle.load(fname)
    except FileNotFoundError:
        print("pickle file not found, new file will be created")
        ff_data = dict()

    # Define the name pattern for quandl request
    name_pattern = "CME/CL"

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

    # Get the data
    for request in requests:
        ff_data[request] = quandl.get(name_pattern + request)
        print(request)

    with open(filename, mode="wb") as fname:
        pickle.dump(ff_data, fname)


def main():
    # Run the update
    filename = data_path + "gsci_futures_raw.p"
    start_date = "Jun-1996"
    end_date = "Dec-2018"

    load_ff_futures_data(filename, start_date, end_date)


if __name__ == '__main__':
    main()
