from foolbox.api import *
from wp_tabs_figs.wp_settings import settings
import seaborn as sns

# Set starting date since interest rate data are available for all currencies
start_date = settings["sample_start"]
end_date = settings["sample_end"]
currencies = ["aud", "cad", "chf", "eur", "gbp", "nok", "nzd", "sek"]
n_portfolios = 3
lag_events = 0


# Import the data
with open(data_path + "data_dev_m.p", mode="rb") as fname:
    fx_data_m = pickle.load(fname)

with open(data_path + "data_dev_d.p", mode="rb") as fname:
    fx_data = pickle.load(fname)

s_d = fx_data["spot_ret"].loc[start_date:end_date, currencies]
f_d = fx_data["fwd_disc"].loc[start_date:end_date, currencies]
f_m = fx_data_m["fwd_disc"].loc[start_date:end_date, currencies]
rx = fx_data_m["rx"].loc[start_date:end_date, currencies]

# Construct a portfolio to have a look at around events, e.g. carry:
signal = f_d.resample("M").mean().shift(1)
carry = poco.rank_sort_adv(rx, signal, n_portfolios)
carry_p = poco.get_factor_portfolios(carry, hml=True)

# Get the monetary policy action data
with open(data_path + "events.p", mode='rb') as fname:
    events = pickle.load(fname)["joint_cbs"]

# Aggregate events by month, in such a way that on average the target rate was
# reduced within a month this is a cut month
events = events.loc[start_date:end_date, currencies + ["usd"]]
events = np.sign(events.resample("M").mean().shift(lag_events))

# Get the hikes and cuts dummies
hikes = events.where(events > 0, 0)
cuts = -1 * events.where(events < 0, 0)
no_change = (events.where(events == 0, np.nan) + 1).fillna(0)
no_meeting = events.isnull().astype("float")


# Treat the FOMC announcements separately
events_us = events[["usd"]]
events = events[currencies]

hikes_us = hikes[["usd"]]
hikes = hikes[currencies]

cuts_us = cuts[["usd"]]
cuts = cuts[currencies]

no_change_us = no_change[["usd"]]
no_change = no_change[currencies]

no_meeting_us = no_meeting[["usd"]]
no_meeting = no_meeting[currencies]


# Locate events within the portfolios, get the high and low portfolios first
p_high = carry["portfolio"+str(n_portfolios)]
p_low = carry["portfolio"+str(1)]

# Note that if events are lagged (forwarded) the intuition is past (future)
# monetary policy actions in the current portfolio
hikes_in_high = hikes.where(p_high.notnull(), 0)
cuts_in_high = cuts.where(p_high.notnull(), 0)

hikes_in_low = hikes.where(p_low.notnull(), 0)
cuts_in_low = cuts.where(p_low.notnull(), 0)


tt = pd.concat([hikes_in_high, -cuts_in_high],
               axis=1).sum(axis=1).replace(np.nan, 0).to_frame()
tt.columns = ["change_in_h"]

tt2 = pd.concat([hikes_in_low, -cuts_in_low],
               axis=1).sum(axis=1).replace(np.nan, 0).to_frame()
tt2.columns = ["change_in_l"]

mx = pd.DataFrame(index=[-1, 0, 1], columns=[-1, 0, 1])
for (h, l), df in pd.concat(
        [carry_p, np.sign(tt), np.sign(tt2)], axis=1).groupby(
        ["change_in_h", "change_in_l"]):
    mx.loc[h, l] = df.mean().hml*100 #df.count().hml
    print(df.mean()*100, "\n")

mx = mx.astype("float")
print(mx)

# Plot the stuff
fig1, ax = plt.subplots(figsize=(12, 8))
plt.setp(ax.get_xticklabels(), rotation=90, fontsize=20)
plt.setp(ax.get_yticklabels(), rotation=90, fontsize=20)
sns.heatmap(mx, ax=ax, annot=True, center=0.0,
            annot_kws={"size": 20, "color": "black"})
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.ylabel("MP action in high portfolio", fontsize=20)
plt.xlabel("MP action in low portfolio", fontsize=20)
plt.title("Events are lagged by {} period(s). Number of portfolios is "
          "{}".format(lag_events, n_portfolios),
          fontsize=20)



mx = pd.DataFrame(index=[-1, 0, 1], columns=[-1, 0, 1])
for (h, l), df in pd.concat(
        [carry_p, np.sign(tt), np.sign(tt2)], axis=1).groupby(
        ["change_in_h", "change_in_l"]):
    mx.loc[h, l] = df.count().hml
    print(df.mean()*100, "\n")

mx = mx.astype("float")
print(mx)

# Plot the stuff
fig2, ax = plt.subplots(figsize=(12, 8))
plt.setp(ax.get_xticklabels(), rotation=90, fontsize=20)
plt.setp(ax.get_yticklabels(), rotation=90, fontsize=20)
sns.heatmap(mx, ax=ax, annot=True, center=0.0,
            annot_kws={"size": 20, "color": "black"})
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.ylabel("MP action in high portfolio", fontsize=20)
plt.xlabel("MP action in low portfolio", fontsize=20)
plt.title("Events are lagged by {} period(s). Number of portfolios is "
          "{}".format(lag_events, n_portfolios),
          fontsize=20)

# high_hikes = carry["portfolio"+str(n_portfolios)].where(hikes!=0)
# high_cuts = carry["portfolio"+str(n_portfolios)].where(cuts!=0)
# high_no_change = carry["portfolio"+str(n_portfolios)].where(hikes==0)
#
# hikes_in_high = hikes.where(high_hikes.notnull()).mean(axis=1).to_frame()
# hikes_in_high.columns = ["h_in_h"]
#
# cuts_in_high = cuts.where(high_cuts.notnull()).mean(axis=1).to_frame()
# cuts_in_high.columns = ["c_in_h"]
#
# noch_in_high = \
#     no_change.where(high_no_change.notnull()).mean(axis=1).to_frame()
# noch_in_high.columns = ["n_in_h"]





# low_hikes = carry["portfolio"+str(1)].where(hikes!=0)
# low_cuts = carry["portfolio"+str(1)].where(cuts!=0)
# low_no_change = carry["portfolio"+str(1)].where(hikes==0)
#
# hikes_in_low = hikes.where(low_hikes.notnull()).mean(axis=1).to_frame()
# hikes_in_low.columns = ["h_in_l"]
#
# cuts_in_low = cuts.where(low_cuts.notnull()).mean(axis=1).to_frame()
# cuts_in_low.columns = ["c_in_l"]
#
# tt2 = pd.concat([hikes_in_low, -cuts_in_low],
#                axis=1).mean(axis=1).replace(np.nan, 0).to_frame()
# tt2.columns = ["change_in_l"]
#
#
# mx = pd.DataFrame(index=[-1, 0, 1], columns=[-1, 0, 1])
# for (h, l), df in pd.concat([carry_p, tt, tt2], axis=1).groupby(
#         ["change_in_h", "change_in_l"]):
#     mx.loc[h, l] = df.mean().hml*100
#     print(df.mean()*100, "\n")
#
# mx = mx.astype("float")
# print(mx)
#
#
# # Plot the stuff
# fig1, ax = plt.subplots(figsize=(12, 8))
# plt.setp(ax.get_xticklabels(), rotation=90, fontsize=20)
# plt.setp(ax.get_yticklabels(), rotation=90, fontsize=20)
# sns.heatmap(mx, ax=ax, annot=True, center=0.0,
#             annot_kws={"size": 20, "color": "black"})
# plt.xticks(rotation=0)
# plt.yticks(rotation=0)
# plt.ylabel("MP action in high portfolio", fontsize=20)
# plt.xlabel("MP action in low portfolio", fontsize=20)
# plt.title("Events are lagged by {} period(s). Number of portfolios is "
#           "{}".format(lag_events, n_portfolios),
#           fontsize=20)
#
#
# for (u), df in pd.concat([carry_p, events_us], axis=1).groupby(["usd"]):
#     print(df.mean()*100, "\n")
#
# mx = mx.astype("float")
#
#
# ff = pd.concat([hikes_in_high, cuts_in_low],
#                axis=1).mean(axis=1).to_frame()
# ff.columns = ["ff"]
#
# ff2 = pd.concat([cuts_in_high, hikes_in_low],
#                axis=1).mean(axis=1).to_frame()
# ff2.columns = ["ff"]
#
# ff3 = events_us.replace(0, np.nan)
# ff3.columns = ["ff"]
#
# test = pd.concat([hikes_in_high, hikes_in_low, -cuts_in_high, -cuts_in_low],
#                  axis=1).sum(axis=1)
#
# hml = carry_p[["hml"]]
# hml.columns=[["usd"]]
# st = poco.timed_strat(hml,
#                       pd.concat([test, ff3], axis=1).sum(axis=1)\
#                       .to_frame().shift(1)
#                       ).to_frame()
# st.columns = ["usd"]
# st.replace(np.nan, 0).cumsum().plot()
#
# st2 = st.where(st.notnull(), hml)
# st2.columns = ["timed carry"]
#
# df = pd.concat([hml, st2], axis=1)
# df.cumsum().plot()
#
# lol = pd.concat([ff, -ff2, ff3], axis=1).sum(axis=1)\
#                       .to_frame().shift(1).replace(-2, -1).replace(2, 1)
# lol.columns = ["lol"]
#
#
#
#
#
#
# combo = (high_hikes.replace(np.nan, 0).sum(axis=1) +
#          low_hikes.replace(np.nan, 0).sum(axis=1) -
#          high_cuts.replace(np.nan, 0).sum(axis=1)-
#          low_cuts.replace(np.nan, 0).sum(axis=1)).to_frame()
# combo.columns = ["combo"]
# #
# # taf.ts_ap_tests(carry_p, combo.replace(0, np.nan))
# #
# #
# # aggr_hikes = hikes.sum(axis=1).to_frame()
# # aggr_hikes.columns = ["hikes"]
# #
# # aggr_cuts = cuts.sum(axis=1).to_frame()
# # aggr_cuts.columns = ["cuts"]
# #
# # taf.ts_ap_tests(carry_p, pd.concat([aggr_cuts, aggr_hikes], axis=1))
# #
# # for lol, df in pd.concat(
# #         [carry1_p[start_date:end_date], cuts], axis=1).groupby(["chf"]):
# #     print(df.mean()*100, "\n")
# #
# #
# # p5 = carry1["portfolio5"]
# # p5 =
# #
# # signal = events.replace(0, np.nan).ffill().ewm(3).mean()
# # carry = poco.rank_sort_adv(rx, signal, 2)
# # carry_p = poco.get_factor_portfolios(carry, hml=True)
# # carry_p.cumsum().plot()
#
# # signals = pd.DataFrame(index=events_us.index, columns=events.columns)
# # for col in currencies:
# #     signals[col] = events_us
# #
# # signals = events - signals
# # # st = poco.multiple_timing(rx, signals)
# # # st.cumsum().plot()
# #
# #
signals = (tot.ewm(halflife=6).mean()-tot.ewm(halflife=18).mean()).shift(1)
carry = poco.rank_sort_adv(rx, signals, 2)
carry_p = poco.get_factor_portfolios(carry, hml=True)
carry_p.cumsum().plot()

days = []
for col in events.columns:
    loc_evts = pd.Series(events[col].dropna().index,
                         index=events[col].dropna().index)
    days.append(loc_evts)
df = pd.concat(days, axis=1)
df.columns = currencies + ["usd"]
df = df.ffill().resample("M").last()

current_month = df.copy()
for curr in current_month.columns:
    current_month[curr] = df.index

diff = current_month-df
diff = df.sub(df.usd, axis=0)

signals = diff.apply(lambda x: x.dt.days).shift(1)[currencies]
signals = signals.mask((signals<-100) | (signals > 100))
carry = poco.rank_sort_adv(rx, signals, 2)
carry_p = poco.get_factor_portfolios(carry, hml=True)
carry_p.cumsum().plot()




# Get the monetary policy action data
with open(data_path + "events.p", mode='rb') as fname:
    events = pickle.load(fname)["fomc"][["change"]]

events.columns = ["usd"]

currency = ['jpy']

days = []
for col in events.columns:
    loc_evts = pd.Series(events[col].dropna().index,
                         index=events[col].dropna().index)
    days.append(loc_evts)
df = pd.concat(days, axis=1)
df.columns = ["usd"]


def count_gaps(series):
    counter = 0
    out = pd.Series(index=series.index)
    for ix, val in series.loc[series.first_valid_index():].isnull().iteritems():
        if val:
            counter += 1
        else:
            counter = 0

        out.loc[ix] = counter


    return out


events = events.asfreq("B")
day_count = count_gaps(events.usd)
qcut = pd.qcut(day_count, 4).to_frame("qcut")

spot = s_d.asfreq("B")
#
# spot_carry = poco.rank_sort_adv(spot, f_d.rolling(22).mean().shift(1), 2)
# portf = poco.get_factor_portfolios(spot_carry, hml=True)

portf = spot.sek.to_frame("hml")


tt = pd.concat([portf, qcut, np.sign(events)], join="inner", axis=1).ffill()
tt = tt[start_date:end_date]

out = pd.DataFrame(index=qcut.drop_duplicates().qcut,
                   columns=[-1, 0, 1])
for (qc, evt), df in tt.groupby(["qcut", "usd"]):
    print(df.mean().hml)
    out.loc[qc, evt] = df.hml.mean() * 10000

out = out.astype("float")

fig1, ax = plt.subplots(figsize=(12, 8))
plt.setp(ax.get_xticklabels(), rotation=90, fontsize=20)
plt.setp(ax.get_yticklabels(), rotation=90, fontsize=20)
sns.heatmap(out, ax=ax, annot=True, center=0.0,
            annot_kws={"size": 20, "color": "black"})
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.ylabel("QCUT", fontsize=20)
plt.xlabel("FOMC action", fontsize=20)
