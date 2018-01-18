from foolbox.api import *
from wp_tabs_figs.wp_settings import settings
#import seaborn as sns

# Set starting date since interest rate data are available for all currencies
start_date = settings["sample_start"]
end_date = settings["sample_end"]
currencies = ["aud", "cad", "chf", "eur", "gbp", "nok", "nzd", "sek", "jpy"]
test_curr = "sek"
test_cb = "riks"
n_quantiles = 2


with open(data_path+"carry_spot_d_3p.p", mode="rb") as hunger:
    carry_spot = pickle.load(hunger)

hml = carry_spot[["hml"]]

# Import the data
with open(data_path + "data_dev_m.p", mode="rb") as fname:
    fx_data_m = pickle.load(fname)

with open(data_path + "data_dev_d.p", mode="rb") as fname:
    fx_data = pickle.load(fname)

s_d = fx_data["spot_ret"].loc[start_date:end_date, currencies].asfreq("B")
f_d = fx_data["fwd_disc"].loc[start_date:end_date, currencies].asfreq("B")
f_m = fx_data_m["fwd_disc"].loc[start_date:end_date, currencies]
rx = fx_data_m["rx"].loc[start_date:end_date, currencies]

with open (data_path+"daily_rx.p", mode="rb") as hunger:
    daily_data = pickle.load(hunger)

rx = daily_data["rx"].loc[start_date:end_date, currencies]
hml = daily_data["hml"].loc[start_date:end_date].hml.to_frame("hml")
s_d = rx



# Get the monetary policy action data
with open(data_path + "events.p", mode='rb') as fname:
    events = pickle.load(fname)[test_cb][["change"]]
events.columns = [test_cb]


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


def simple_tstat(series):
    return series.mean()/series.std() * np.sqrt(series.count())


events = events.asfreq("B")
day_count = count_gaps(events[test_cb])
qcut = pd.qcut(day_count, n_quantiles).to_frame("qcut")


if test_curr == "hml":
    test_asset = hml
else:
    test_asset = s_d[[test_curr]]

tt = pd.concat([test_asset, qcut, np.sign(events)],
               join="inner", axis=1).ffill()
tt = tt[start_date:end_date]

# out = pd.DataFrame(index=qcut.drop_duplicates().qcut,
#                    columns=[-1, 0, 1])
# for (qc, evt), df in tt.groupby(["qcut", test_cb]):
#     print(df.mean()[test_curr])
#     out.loc[qc, evt] = \
#         simple_tstat(df[test_curr])#df[test_curr].mean() * 10000
#
# out = out.astype("float")
#
# fig1, ax = plt.subplots(figsize=(12, 8))
# plt.setp(ax.get_xticklabels(), rotation=90, fontsize=20)
# plt.setp(ax.get_yticklabels(), rotation=90, fontsize=20)
# sns.heatmap(out, ax=ax, annot=True, center=0.0,
#             annot_kws={"size": 20, "color": "black"})
# plt.xticks(rotation=0)
# plt.yticks(rotation=0)
# plt.ylabel("QCUT", fontsize=20)
# plt.xlabel("{} action".format(test_cb.upper()), fontsize=20)
# plt.title("{} cycle around {} actions".format(test_curr.upper(),
#                                               test_cb.upper()), fontsize=20)
#
#
#
# out2 = pd.DataFrame(index=qcut.drop_duplicates().qcut,
#                    columns=["total"])
# for qc, df in tt.groupby(["qcut"]):
#     print(df.mean()[test_curr])
#     out2.loc[qc] = \
#         simple_tstat(df[test_curr])# df[test_curr].mean() * 10000
#
# out2 = out2.astype("float")
#
# fig2, ax = plt.subplots(figsize=(12, 8))
# plt.setp(ax.get_xticklabels(), rotation=90, fontsize=20)
# plt.setp(ax.get_yticklabels(), rotation=90, fontsize=20)
# sns.heatmap(out2, ax=ax, annot=True, center=0.0,
#             annot_kws={"size": 20, "color": "black"})
# plt.xticks(rotation=0)
# plt.yticks(rotation=0)
# plt.ylabel("QCUT", fontsize=20)
# plt.xlabel("{} action".format(test_cb.upper()), fontsize=20)
# plt.title("{} cycle around {} actions".format(test_curr.upper(),
#                                               test_cb.upper()), fontsize=20)
#
#
# out3 = pd.DataFrame(columns=[-1, 0, 1], index=["total"])
# for action, df in tt.groupby([test_cb]):
#     print(df.mean()[test_curr])
#     out3.loc[:, action] = \
#         simple_tstat(df[test_curr]) #df[test_curr].mean() * 10000 #
#
# out3 = out3.astype("float")
#
# fig3, ax = plt.subplots(figsize=(12, 8))
# plt.setp(ax.get_xticklabels(), rotation=90, fontsize=20)
# plt.setp(ax.get_yticklabels(), rotation=90, fontsize=20)
# sns.heatmap(out3, ax=ax, annot=True, center=0.0,
#             annot_kws={"size": 20, "color": "black"})
# plt.xticks(rotation=0)
# plt.yticks(rotation=0)
# plt.ylabel("action", fontsize=20)
# plt.xlabel("{}".format(test_cb.upper()), fontsize=20)
# plt.title("{} cycle around {} actions".format(test_curr.upper(),
#                                               test_cb.upper()), fontsize=20)



# Make a standard carry
# signal = f_d.resample("M").mean().shift(1)
# signal = signal.reindex(s_d.index).ffill()
# carry = poco.rank_sort_adv(s_d, signal, 3)
# carry_p = poco.get_factor_portfolios(carry, hml=True)
#
# with open(data_path+"carry_spot_d_3p.p", mode="wb") as hunger:
#     pickle.dump(carry_p, hunger)

t = tt.dropna()
q_list = []
for k in np.arange(0, n_quantiles):
    q_list.append(
        t.loc[t.qcut == qcut.drop_duplicates().dropna().qcut[k]].dropna()[test_curr])
h = t[test_curr]
q_list.append(h)

df = pd.concat(q_list, axis=1).fillna(0)
df.columns = \
    ["q"+str(k+1) for k in range(len(q_list)-1)] + [test_curr+"_"+test_cb]
df.cumsum().plot()


