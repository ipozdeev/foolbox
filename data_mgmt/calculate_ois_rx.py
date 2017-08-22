import pandas as pd
from foolbox.api import *

if __name__ == "__main__":

    cur = "usd"

    # data ------------------------------------------------------------------
    # ois data
    with open(data_path + "ois_bloomberg.p", mode="rb") as hangar:
        ois_data = pickle.load(hangar)

    ois_data = ois_data[cur].astype(float)

    # overnight rates
    on_rate = ois_data.pop("ON")

    # space for implied rates
    rp = dict()

    for m in ois_data.columns:
        # m = "3M"
        print(m)

        this_ois_quotes = ois_data[m]

        # OIS
        ois = OIS.from_iso(cur, maturity=DateOffset(months=1))

        # space for data
        rx = this_ois_quotes.dropna()
        rx *= np.nan

        for t in rx.index:
            ois.quote_dt = t

            # calculate return
            fwd_ret = ois.get_return_of_floating(on_rate)
            fwd_ret *= (ois.day_cnt_fix / ois.lifetime * 100)

            rx.loc[t] = fwd_ret - this_ois_quotes.loc[t]

        taf.descriptives(rx.loc[:"2005"].to_frame()*100, scale=1)
        rx.count()

        rx.loc["2009":].rolling(5).mean().plot()

        lol = rx.rolling(window=504, min_periods=40).apply(
            lambda x: x.mean()/x.std()*np.sqrt(len(x)))
        lol.plot()


        ois.quote_dt = "2009-03-24"
        ois.get_return_of_floating(on_rate)
