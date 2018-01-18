from foolbox.api import *

with open(data_path+"data_wmr_dev_d.p", mode="rb") as fname:
    data = pickle.load(fname)
with open(data_path+"data_wmr_dev_m.p", mode="rb") as fname:
    data_m = pickle.load(fname)
with open(data_path+"events.p", mode="rb") as fname:
    events = pickle.load(fname)

fomc = events["fomc"]
joint = events["joint_cbs"]

fd = data["fwd_disc"]
sd = data["spot_ret"]

rx = data_m["rx"]

carry = poco.rank_sort(rx, data_m["fwd_disc"].shift(1),5)
carry_d = poco.upsample_portfolios(carry, sd["1997":])
p_d = poco.get_factor_portfolios(carry_d, hml=True)

mom = poco.rank_sort(rx, rx.rolling(12).sum().shift(1),5)
mom_d = poco.upsample_portfolios(mom, sd["1997":])
p_d = poco.get_factor_portfolios(mom_d, hml=True)

dol = rx.mean(axis=1).to_frame()
dol.columns = ["dol"]

signal_dol = fd.resample("M").last().shift(1).mean(axis=1).to_frame()
signal_dol.columns = ["dol"]

dol_carry_m = dol.where(signal_dol.dol > 0, -dol)

signal = fd.rolling(22).mean().shift(1).mean(axis=1)
dol_d = data["spot_ret"].mean(axis=1)
dol_carry = dol_d.where(signal > 0, -dol_d)
dol_carry = dol_carry["1997":]


mom = poco.rank_sort(sd, sd.rolling(261).sum().shift(1), 5)
p_d = poco.get_factor_portfolios(mom, hml=True)["1997":]
p_d.cumsum().plot()

evts = joint.mean(axis=1)["1997":]
evts = joint[["nzd", "aud",]].mean(axis=1)["1997":]
evts = evts.where(evts > 0).dropna()["1997":]


evts = joint.mean(axis=1)["1997":]
evts = evts.where(evts < 0).dropna()["1997":]

evts = fomc.diff().dropna()["1997":]
evts = evts.where(evts > 0).dropna()["1997":]

event_study_wrapper(sd.mean(axis=1)["1997":], evts,
                    direction="all",
                    window=[-10, -1, 0, 10],
                    ci_method="simple")


with open(data_path+"mom_dev_d_dir_s_3p.p", mode='rb') as fname:
    ir_mom = pickle.load(fname)

op = ir_mom.loc[:, (slice(3, 15), slice(1, 5))].mean(axis=1)["1997":]