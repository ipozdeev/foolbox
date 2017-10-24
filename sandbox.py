import numpy as np
import pandas as pd
from foolbox import EventStudy
from assetpricing import portfolio_construction as poco
%matplotlib

#
usr = "pozdeev"
path = "c:/Users/"+usr+"/Google Drive/Personal/opec_meetings/"
path_to_ret = "c:/Users/pozdeev/Google Drive/Personal/option_implied_betas_project/data/raw/returns/"

s_d = pd.read_csv(path+"data/s_d.csv", index_col=0, parse_dates=True)
s_d.columns = [p.lower() for p in s_d.columns]

these_cur = sorted(
    ['aud', 'eur', 'gbp', 'nzd', 'cad', 'chf', 'jpy', 'nok', 'sek', 'dkk'])

s_d = s_d[these_cur]

s_d["chf"].loc["2015":"2016"].cumsum().plot()
dol_f = s_d.mean(axis=1).to_frame()

opec = pd.read_clipboard(index_col=0, parse_dates=True, header=None)
opec = opec.where(opec.iloc[:,1].notnull()).dropna(how="all")
opec = opec.iloc[:,0]
opec = opec.to_frame()

opec.to_csv()
fomc = pd.read_excel(path+"fomc_meetings_1994_2015.xlsx", sheetname=0,
    indeX_col=0)

#
fwd_d = pd.read_csv(path_to_ret+"fwd_disc_d.csv", index_col=0,
    parse_dates=True)
fwd_d.columns = [p.lower() for p in fwd_d.columns]
fwd_d = fwd_d[these_cur]

# carry
p_carry = poco.rank_sort(returns=s_d, signals=fwd_d.shift(22), n_portfolios=3)
carry_ps = poco.get_factor_portfolios(p_carry, hml=True)
carry_hml = carry_ps.hml.to_frame()

p_mom = poco.rank_sort(returns=s_d,
    signals=s_d.rolling(window=22).mean().shift(1), n_portfolios=3)
mom_ps = poco.get_factor_portfolios(p_mom, hml=True)
mom_ps.cumsum().plot()

#
evt = EventStudy.EventStudy(data=mom_ps[["hml"]],
    events=opec, window=[-22,-1,0,22])
evt.plot(ps=0.9, method="simple")


#
with open(data_path + "implied_rates_from_1m.p", mode="rb") as hangar:
    ir = pickle.load(hangar)

ir_usd = ir["usd"].dropna()

with open(data_path + "ois_", mode="rb") as hangar:
    ir = pickle.load(hangar)

ir_usd = ir["usd"].dropna()



ois_us = OIS.from_iso("usd", maturity="1m")


# carry ----------------------------------------------------------------------
from foolbox.api import *
from foolbox.wp_tabs_figs.wp_settings import settings
from foolbox.utils import *

# Set the output path, input data and sample
start_date = pd.to_datetime(settings["sample_start"])
end_date = pd.to_datetime(settings["sample_end"])
avg_impl_over = settings["avg_impl_over"]
avg_refrce_over = settings["avg_refrce_over"]
settings["base_holding_h"] = 5
base_lag = settings["base_holding_h"] + 2
base_th = settings["base_threshold"]

# carry
with open(data_path+"fx_by_tz_sp_fixed.p", mode="rb") as hangar:
    data_d = pickle.load(hangar)

fx_tr_env = FXTradingEnvironment.from_scratch(
    spot_prices={
        "bid": data_d["spot_bid"].loc[:, start_date:, "NYC"],
        "ask": data_d["spot_ask"].loc[:, start_date:, "NYC"]},
    swap_points={
        "bid": data_d["tnswap_bid"].loc[:, start_date:, "NYC"],
        "ask": data_d["tnswap_ask"].loc[:, start_date:, "NYC"]}
        )

fx_tr_env.drop(labels=["dkk"], axis="minor_axis", errors="ignore")
fx_tr_env.remove_swap_outliers()
fx_tr_env.reindex_with_freq('D')
fx_tr_env.align_spot_and_swap()
fx_tr_env.fillna(which="both", method="ffill")

fdisc_d = fx_tr_env.swap_points.mean(axis="items")
spot_d = fx_tr_env.spot_prices.mean(axis="items")

dr = -1*np.log((fdisc_d + spot_d)/spot_d).rolling(22).mean()

pf = poco.rank_sort(dr, dr, 3)
pos_fl = (pf["portfolio3"].notnull().where(pf["portfolio3"].notnull()))\
    .fillna(pf["portfolio1"].notnull().where(pf["portfolio1"].notnull())*-1)
pos_fl = pos_fl.astype(float).shift(1)
strat_carry = FXTradingStrategy.from_position_flags(pos_fl, leverage="net")


# fx_tr_str_carry = FXTradingStrategy.monthly_from_daily_signals(
#     signals_d=dr.drop("dkk", axis=1), n_portf=3, leverage="net")

# ois-implied forecasts
curs = [p for p in dr.columns if p not in ["dkk", "nok", "jpy"]]

# forecast direction
signals_fcast = get_pe_signals(curs, base_lag, base_th*100, data_path,
    fomc=False,
    avg_impl_over=avg_impl_over,
    avg_refrce_over=avg_refrce_over,
    bday_reindex=True)

signals_fcast.loc[:, "nok"] = np.nan
signals_fcast.loc[:, "jpy"] = np.nan

signals_fcast = signals_fcast.loc[start_date:end_date].reindex(
    index=pd.date_range(start_date, end_date, freq='B'))

strategy_fcast = FXTradingStrategy.from_events(signals_fcast,
    blackout=1, hold_period=5, leverage="net")

combined_strat = strat_carry + strategy_fcast
combined_strat.position_flags.to_clipboard()

fx_tr = FXTrading(environment=fx_tr_env, strategy=strat_carry)
# fx_tr = FXTrading(environment=fx_tr_env, strategy=strategy_fcast)
fx_tr = FXTrading(environment=fx_tr_env, strategy=combined_strat)

res_unr_comb = fx_tr.backtest("unrealiz")
res_unr_comb.dropna().plot()

res_bal = fx_tr.backtest("balance")
# %matplotlib inline
res_unr.dropna().plot()

# ----------------------------------------------------------------------------
sig_mom = np.log(spot_d).diff().rolling(22).sum().shift(1)
pf = poco.rank_sort(sig_mom, sig_mom, 3)
pos_fl = (pf["portfolio3"].notnull().where(pf["portfolio3"].notnull()))\
    .fillna(pf["portfolio1"].notnull().where(pf["portfolio1"].notnull())*-1)
pos_fl = pos_fl.astype(float)
strat_mom = FXTradingStrategy.from_position_flags(-1*pos_fl, leverage="net")
fx_tr = FXTrading(environment=fx_tr_env, strategy=strat_mom)
res_mom = fx_tr.backtest("unrealiz")
res_mom.plot()


signals_fcast.to_clipboard()
