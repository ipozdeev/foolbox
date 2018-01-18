from foolbox.api import *
from EventStudy import signal_from_events

with open(data_path+"data_dev_d.p", mode='rb') as fname:
    data = pickle.load(fname)
with open(data_path+"data_wmr_dev_d.p", mode='rb') as fname:
    data_wmr = pickle.load(fname)


with open(data_path+"events.p", mode='rb') as fname:
    events = pickle.load(fname)
with open(data_path+"data_dev_m.p", mode='rb') as fname:
    data_m = pickle.load(fname)
with open(data_path+"mom_dev_d_dir_s_3p.p", mode='rb') as fname:
    ir_mom = pickle.load(fname)

with open(data_path+"data_wmr_dev_d_dir_s_3p.p", mode='rb') as fname:
    ir_mom = pickle.load(fname)


#start = "1988-01-30"
s_d = data["spot_ret"]
f_d = data["fwd_disc"]
s = np.log(data["spot_mid"])
f = np.log(data["fwd_mid"])
rx = data_m["rx"]


carry_lag = np.arange(0, 20, 1)
carries = pd.DataFrame(index=s_d.resample("M").last().index, columns=carry_lag)

for lag in carry_lag:
    tmp_f = f.shift(lag).resample("M").last()
    tmp_s = s.shift(lag).resample("M").last()
    tmp_rx = tmp_f.shift(1)-tmp_s
    signal = (tmp_f-tmp_s).shift(1)

    carry = poco.rank_sort(tmp_rx, signal, 2)
    carries[lag] = poco.get_factor_portfolios(carry, hml=True).hml
    print(lag)
    #carry = poco.rank_sort()



signals = f_d.shift(31)

carry = poco.rank_sort_adv(-f.shift(31)+s, signals, 5, 30)
carry_p = poco.get_factor_portfolios(carry, hml=True)
(carry_p/12).cumsum().plot()


fomc = events["fomc"].squeeze()

#signals = signal_from_events(s_d, fomc, (-10, -2), lambda x: max(x.cumsum()))
#signals = signal_from_events(s_d, fomc, (-10, -2))
#df = data["rx"][["GBP", "AUD", "JPY", "CHF"]].tail(12)
#test_idx = pd.DatetimeIndex(['2016-02-29', '2016-05-31', '2016-08-31',
#                             '2016-11-30'], dtype='datetime64[ns]',
#                            name='Internal ID', freq='3M')

# def custom_signal(returns, events, sum_hor):
#     out = pd.DataFrame(columns=returns.columns)
#     for event in events.index:
#         vals = returns.values[returns.index.get_loc(event):returns.index.get_loc(event)+sum_hor].sum()
#         out =

fomc2 = fomc.copy()
fomc2 = fomc2.where(fomc2.diff()>0).dropna()
shenanigan = s_d[::-1].shift(1).rolling(22).mean()[::-1]
signals = shenanigan.loc[fomc.index, :].shift(1).rolling(1).sum()

shenanigan = s_d.shift(1).rolling(22).mean()
signals = shenanigan.loc[fomc.index, :].rolling(2).sum()


test = poco.rank_sort_adv(s_d, signals, 5, 22, signals.index)


p = poco.get_factor_portfolios(test, hml=True)
p.dropna().cumsum().plot()

taf.descriptives(p.dropna(), scale=26)


# Pre-announcement strategy
with open(data_path+"data_dev_d.p", mode='rb') as fname:
    data = pickle.load(fname)
with open(data_path+"events.p", mode='rb') as fname:
    events = pickle.load(fname)

s_d = data["spot_ret"]
fomc = events["fomc"].squeeze()

get_rb_dates = s_d.copy()
get_rb_dates["dates"] = get_rb_dates.index

get_rb_dates = get_rb_dates.shift(5)

rb_dates = pd.DatetimeIndex(get_rb_dates.loc[fomc.index, :].dates.values)


signals = s_d.shift(1).rolling(5).mean()
signals = signals.loc[fomc.index, :].shift(1).rolling(24, 6).sum()

signals.index = rb_dates

strat = poco.rank_sort_adv(s_d, signals, 5, None, rb_dates)

p = poco.get_factor_portfolios(strat, hml=True)
p.dropna().cumsum().plot()

s_d = data["spot_ret"]
fomc = events["fomc"].squeeze()
get_rb_dates = s_d.copy()
get_rb_dates["dates"] = get_rb_dates.index
get_rb_dates = get_rb_dates.shift(22)
rb_dates = pd.DatetimeIndex(get_rb_dates.loc[fomc.index, :].dates.values)
signals = s_d.shift(1).rolling(22).mean()
signals = signals.loc[fomc.index, :].shift(1).rolling(12, 6).sum()
signals.index = rb_dates
strat = poco.rank_sort_adv(s_d, signals, 5, 22, rb_dates)
p = poco.get_factor_portfolios(strat, hml=True)
p["2010":].dropna().cumsum().plot()




taf.descriptives(p.dropna(), scale=1)


with open(data_path+"mom_dev_m_s_rx.p", mode="rb") as fname:
    mom = pickle.load(fname)

with open(data_path+"data_dev_d.p", mode="rb") as fname:
    data = pickle.load(fname)

with open(data_path+"events.p", mode='rb') as fname:
    events = pickle.load(fname)
s_d = data["spot_ret"]
f_d = data["fwd_disc"]
fomc = events["fomc"].squeeze()
#fomc = fomc.where(fomc.diff() > 0).dropna()
get_rb_dates = s_d.copy()
get_rb_dates["dates"] = get_rb_dates.index
get_rb_dates = get_rb_dates.shift(-12)
rb_dates = pd.DatetimeIndex(get_rb_dates.loc[fomc.index, :].dates.values)
#signals = s_d.shift(-22).rolling(21).mean()

signals = f_d.diff(1).shift(-11).rolling(22).mean()

signals = signals.loc[fomc.index, :]#.shift(1)#.rolling(12, 3).sum()
signals.index = rb_dates
strat = poco.rank_sort_adv(s_d, signals, 3, 5, rb_dates)
p = poco.get_factor_portfolios(strat, hml=True)
p["dol"] = p.drop(["hml"], axis=1).mean(axis=1)
p[:"2015"].dropna().cumsum().plot()

taf.descriptives(p.dropna(), scale=261)
# ACHTUNG



s_d = data["spot_ret"]
fomc = events["fomc"].squeeze()
get_rb_dates = s_d.copy()
get_rb_dates["dates"] = get_rb_dates.index
get_rb_dates = get_rb_dates.shift(14)
rb_dates = pd.DatetimeIndex(get_rb_dates.loc[fomc.index, :].dates.values)
signals = s_d.shift(-14).rolling(13).sum()
signals = signals.loc[fomc.index, :].shift(1).rolling(1, 1).sum()
signals.index = rb_dates
strat = poco.rank_sort_adv(s_d, signals, 3, 14, rb_dates)
p = poco.get_factor_portfolios(strat, hml=True)

p["2001":"2015"].dropna().cumsum().plot()




from foolbox.api import *
import pickle

with open(data_path+"data_dev_d.p", mode="rb") as fname:
    data = pickle.load(fname)

with open(data_path+"events.p", mode='rb') as fname:
    events = pickle.load(fname)
s_d = data["spot_ret"]
f_d = data["fwd_disc"]
fomc = events["fomc"].squeeze()
#fomc = fomc.where(fomc.diff() > 0).dropna()
# get_rb_dates = s_d.copy()
# get_rb_dates["dates"] = get_rb_dates.index
# get_rb_dates = get_rb_dates.shift(-22)
# rb_dates = pd.DatetimeIndex(get_rb_dates.loc[fomc.index, :].dates.values)
#
# signals = f_d.diff(22).shift(1)
# signals = signals.loc[fomc.index, :].shift(1)

get_rb_dates = s_d.copy()
get_rb_dates["dates"] = get_rb_dates.index
get_rb_dates = get_rb_dates.shift(-5)
rb_dates = pd.DatetimeIndex(get_rb_dates.loc[fomc.index, :].dates.values)

#signals = f_d.diff(261).shift(1+23)
signals = (f_d.rolling(12).mean()-f_d.rolling(22).mean()).rolling(12).max().shift(1)
signals = signals.loc[rb_dates, :]#.shift(1)
signals.index = rb_dates

strat = poco.rank_sort_adv(s_d, signals, 5, 22, None)
p = poco.get_factor_portfolios(strat, hml=True)
p[:].dropna().cumsum().plot()

taf.descriptives(p.dropna(), scale=261)


with open(data_path+"data_dev_d.p", mode="rb") as fname:
    data = pickle.load(fname)
s_d = data["spot_ret"]
f_d = data["fwd_disc"]

with open(data_path+"data_dev_m.p", mode="rb") as fname:
    data_m = pickle.load(fname)
rx = data_m["rx"]

signals = (f_d.rolling(22).mean()-f_d.rolling(66).mean()).rolling(22).max()
#signals = (np.abs(f_d.diff()).rolling(130).mean())
signal_m = signals.resample("M").last().shift(1)

test = poco.rank_sort_adv(rx, signal_m, 3)
pp = poco.get_factor_portfolios(test, hml=True)
pp.cumsum().plot()


signals = (f_d.rolling(22).mean() - f_d.rolling(66).mean()).rolling(22).max().shift(1)

#signals = signals.loc[s_d.resample("M").last().index, :] s_d.resample("M").last().index
strat = poco.rank_sort_adv(s_d, signals, 5, 22)
p = poco.get_factor_portfolios(strat, hml=True)
p[:].dropna().cumsum().plot()

ff = dict()
for key in strat.keys():
    ff[key] = strat[key].resample("M").sum()

for key in ff.keys():
    if key[:4] == "port":
        ff[key] = ff[key] + f_d.resample("M").last().shift(1)

for key in ff.keys():
    if len(key) == 2:
        ff[key] = ff["portfolio"+key[1]].mean(axis=1).to_frame()
        ff[key].columns = [key]

ptest = poco.get_factor_portfolios(ff, hml=True)
ptest.cumsum().plot()


















with open(data_path+"carry_dev_d_fwd_disc_s.p", mode="rb") as fname:
    carry = pickle.load(fname)

carry.sum().where(carry.sum() == carry.sum().max()).dropna()




riks = pd.read_csv(data_path+"riksbank_meetings_1994_2017.txt", sep=",",
                   index_col=0, parse_dates=True, header=0).rate.to_frame()
riks.columns = ["rate"]
riks = riks.squeeze()
riks = riks.where(riks.diff() == 0).dropna()

evenk = EventStudy(ir.sek.diff(), riks, [-10, -1, 0, 10])
evenk.get_ci(ps=0.9, method="simple")
evenk.plot()



with open(data_path+"events.p", mode='rb') as fname:
    events = pickle.load(fname)

fomc = events["fomc"].squeeze()
fomc = fomc.where(fomc.diff() == 0).dropna()

evenk = EventStudy(ir_usd.usd.diff(), fomc, [-10, -1, 0, 10])
evenk.get_ci(ps=0.9, method="simple")
evenk.plot()


with open(data_path+"events.p", mode='rb') as fname:
    events = pickle.load(fname)

boe = events["boe"].squeeze()
boe = boe.where(boe.diff() < 0).dropna()

evenk = EventStudy(s_d.gbp, boe, [-10, -1, 0, 10])
evenk.get_ci(ps=0.9, method="simple")
evenk.plot()



norg = pd.read_csv(data_path+"norges_bank_meetings_1993_2017.txt", sep=",",
                   index_col=0, parse_dates=True, header=0).rate.to_frame()
norg.columns = ["rate"]
norg = norg.squeeze()
norg = norg.where(norg.diff() == 0).dropna()

evenk = EventStudy(ir.nok.diff(), norg, [-10, -1, 0, 10])
evenk.get_ci(ps=0.9, method="simple")
evenk.plot()


rba = pd.read_csv(data_path+"rba_meetings_1990_2017.txt", sep=",",
                   index_col=0, parse_dates=True, header=0).rate.to_frame()
rba.columns = ["rate"]
rba = rba.squeeze()
rba = rba.where(rba.diff() > 0).dropna()

evenk = EventStudy(s_d.aud, rba["1998":], [-2, -1, 0, 22])
evenk.get_ci(ps=0.9, method="simple")
evenk.plot()


rbnz = pd.read_csv(data_path+"rbnz_meetings_1999_2017.txt", sep=",",
                   index_col=0, parse_dates=True, header=0).rate.to_frame()
rba.columns = ["rate"]
rbnz = rbnz.squeeze()
rbnz = rbnz.where(rbnz.diff() < 0).dropna()

evenk = EventStudy(s_d.nzd, rbnz, [-10, -1, 0, 10])
evenk.get_ci(ps=0.9, method="simple")
evenk.plot()

with open(data_path+"events.p", mode='rb') as fname:
    events = pickle.load(fname)

with open(data_path+"data_dev_d.p", mode='rb') as fname:
    data = pickle.load(fname)

s_d = data["spot_ret"]

fomc = events["fomc"]
fomc = fomc.squeeze()
fomc = fomc.where(fomc.diff() < 0).dropna()

evenk = EventStudy(libor_us.us.diff()["2004-06-30":], fomc, [-10, -1, 0, 10])
evenk.get_ci(ps=0.9, method="simple")
evenk.plot()

with open(data_path+"events.p", mode='rb') as fname:
    events = pickle.load(fname)

ecb = events["ecb"]
ecb = ecb[["refinancing"]]
ecb = ecb.where(ecb.diff() > 0).dropna()

evenk = EventStudy(ir.eur, ecb, [-1, -1, 0, 1])
evenk.get_ci(ps=0.9, method="simple")
evenk.plot()



op = ir_mom.loc[:, (slice(3, 15), slice(1, 5))].mean(axis=1)
fomc = events["riks"].rate
fomc = fomc.squeeze()
fomc = fomc.where(fomc.diff() > 0).dropna()

evenk = EventStudy(op["1998":], fomc["1998":], [-10, -1, 0, 10])
evenk.get_ci(ps=0.9, method="simple")
evenk.plot()



with open(data_path + "fed_funds_futures_settle.p", "rb") as fname:
    ff = pickle.load(fname)

ff.loc["Mar-2017":, "Mar-2017"]









from foolbox.api import *

with open(data_path+"data_dev_d.p", mode='rb') as fname:
    data = pickle.load(fname)
with open(data_path+"data_wmr_dev_d.p", mode='rb') as fname:
    data_wmr = pickle.load(fname)
with open(data_path+"ir.p", mode='rb') as fname:
    data_ir = pickle.load(fname)

start = "1994"
sd_b = data["spot_ret"][start:]
sd_w = data_wmr["spot_ret"][start:]
fd_b = data["fwd_disc"][start:]
fd_w = data_wmr["fwd_disc"][start:]
ir = data_ir[start:]

with open(data_path+"events.p", mode='rb') as fname:
    events = pickle.load(fname)

fomc = events["rba"][start:]
fomc = fomc.squeeze()
fomc = fomc.where(fomc.diff() < 0).dropna()

evenk = EventStudy(sd_w.aud.rolling(261).sum().shift(-261), fomc, [-10, -1, 0, 10])
evenk.get_ci(ps=0.9, method="simple")
evenk.plot()


from foolbox.api import *
from pandas.tseries.offsets import DateOffset, MonthBegin, MonthEnd,\
    relativedelta

with open(data_path + "fed_funds_futures_settle.p", mode="rb") as fname:
    ff_data = pickle.load(fname)
with open(data_path + "events.p", mode="rb") as fname:
    fomc = pickle.load(fname)["fomc"]


effr = 100*\
    fomc.reindex(ff_data.index, method="ffill")["1994":].fillna(method="bfill")
ff_data = ff_data["1994":]

# Set the implied rate according to EFFR
impl_rate = effr.copy()*np.nan
# Start looping over each date. Why? Fuck vectorization, that's why
for t in impl_rate.index:
    # Locate the closest future FOMC meeting
    next_fomc = fomc.index[fomc.index.get_loc(
        t+relativedelta(minutes=1), method="bfill")]
    # Get the corresponding month's beginning, end and number of days
    next_fomc_month_start = MonthBegin().rollback(next_fomc)
    next_fomc_month_end = MonthEnd().rollforward(next_fomc)
    days_in_month = next_fomc.daysinmonth
    # CASE 1: the FOMC meeting is in the month following t
    impl_rate.loc[t] = \
        ((100-ff_data.loc[t,next_fomc_month_end]) - \
        effr.loc[t]*(
        (next_fomc-next_fomc_month_start).days+1)/days_in_month)/\
        ((next_fomc_month_end-next_fomc).days/days_in_month)


