import pandas as pd
import matplotlib.pyplot as plt
from pandas.tseries.offsets import MonthEnd, QuarterEnd
from foolbox.data_mgmt import set_credentials as setcred

path_to_data = setcred.set_path("research_data/fx_and_events/")

def get_cpi():
    """

    Returns
    -------

    """
    cpi_x = pd.read_excel(path_to_data + "fx_and_macro_data.xlsm",
                          index_col=0, parse_dates=True, skiprows=4,
                          header=None, sheet_name="CPI_sadj_x_aud_nzd")

    cols = ["USD", "GBP", "CHF", "JPY", "CAD", "SEK", "NOK", "DKK", "EUR",
            "DEM"]

    cpi_x.columns = [p.lower() for p in cols]
    cpi_x.index = cpi_x.index.map(lambda x: x + MonthEnd())

    cpi_y = pd.read_excel(path_to_data + "fx_and_macro_data.xlsm",
                          index_col=0, parse_dates=True, skiprows=4,
                          header=None, sheet_name="CPI_sadj_aud_nzd")
    cpi_y.columns = ["aud", "nzd"]
    cpi_y.index = cpi_y.index.map(lambda x: x + QuarterEnd())

    cpis = pd.concat((cpi_x, cpi_y), axis=1)

    with pd.HDFStore(path_to_data + "cpi_1961_2017_m.h5") as hangar:
        hangar.put("cpi", cpis)

    return cpis

if __name__ == "__main__":
    with pd.HDFStore(path_to_data + "strategies_m.h5", mode='r') as hangar:
        data = hangar.get("strats")

    data["vrp"].plot()
    plt.show()

    # res = get_cpi()

