import pandas as pd
from foolbox.api import *
from foolbox.wp_tabs_figs.wp_settings import *

my_blue = "#2f649e"
%matplotlib

# assets
with open(data_path + "fx_by_tz_aligned_d.p", mode='rb') as fname:
    fx = pickle.load(fname)
spot_ask = fx["spot_ask"].drop(["jpy","dkk","nok"],
    axis=1).loc["2000-11":,:]
spot_mid = fx["spot_mid"].drop(["jpy","dkk","nok"],
    axis=1).loc["2000-11":,:]
spot_bid = fx["spot_bid"].drop(["jpy","dkk","nok"],
    axis=1).loc["2000-11":,:]

tnswap_ask = fx["tnswap_ask"].drop(["jpy","dkk","nok"],
    axis=1).loc["2000-11":,:]
tnswap_bid = fx["tnswap_bid"].drop(["jpy","dkk","nok"],
    axis=1).loc["2000-11":,:]


# settings --------------------------------------------------------------
settings = {
    "horizon_a": -10,
    "horizon_b": -1,
    "bday_reindex": True}


# policy expectations ---------------------------------------------------
policy_fcasts = dict()
for c in ['aud', 'cad', 'chf', 'eur', 'gbp', 'nzd', 'sek']:
    # c = "aud"
    pe = PolicyExpectation.from_pickles(data_path, c)
    policy_fcasts[c] = pe.forecast_policy_change(
        lag=12,
        threshold=0.10,
        avg_impl_over=5,
        avg_refrce_over=5,
        bday_reindex=True)

policy_fcasts = pd.DataFrame.from_dict(policy_fcasts).loc["2000-11":]

# strategies ------------------------------------------------------------
# simple strategy: no leverage, no bas adjustment -----------------------
ts = EventTradingStrategy(
    signals=policy_fcasts,
    prices={"mid": spot_mid, "bid": spot_bid, "ask": spot_ask},
    settings=settings)

# leverage
levg = ts.position_flags.dropna(how="all").count(axis=1)

fig, ax = plt.subplots()
levg.rolling(10).mean().plot(ax=ax, linewidth=1.5, color=my_blue)
levg.describe()
(levg <= 2).mean()
