import pandas_datareader.data as web
from foolbox.api import *

if __name__ == "__main__":

    out_path = set_credentials.set_path("ois_project/figs/")

    # data ------------------------------------------------------------------
    # fed funds futures
    with open(data_path + "fed_funds_futures_settle.p", mode="rb") as hangar:
        fff_data = pickle.load(hangar)

    # # overnight rate
    # with open(data_path + "overnight_rates.p", mode="rb") as hangar:
    #     on_rates_data = pickle.load(hangar)
    #
    # on_rate = on_rates_data["usd"]

    on_rate = web.DataReader("DFF", data_source="fred", start="1988-01-01")

    on_rate_m = on_rate.reindex(
        index=pd.date_range(on_rate.index[0], on_rate.index[-1], freq='D'),
        method="ffill").resample('M').mean()
    on_rate_m.name = "on"

    # take end-of-month fff
    eom_fff = fff_data.resample('M').last().shift(1)

    fff_m = pd.Series(
        data=[eom_fff.loc[p,p] for p in eom_fff.columns if p in eom_fff.index],
        index=[p for p in eom_fff.columns if p in eom_fff.index])

    fff_m.name = "fff"

    # concatenate
    both = pd.concat((on_rate_m, 100-fff_m), axis=1)
    both.diff(axis=1).plot()

    desc = taf.descriptives(both.diff(axis=1).loc["2001":, ["fff"]] * 100, 1)

    print(desc)
    print(desc.loc["mean"] / desc.loc["se_mean"])
