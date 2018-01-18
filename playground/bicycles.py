from foolbox.api import *
from wp_tabs_figs.wp_settings import settings
import seaborn as sns

# Set starting date since interest rate data are available for all currencies
start_date = settings["sample_start"]
end_date = settings["sample_end"]
currencies = ["aud", "cad", "chf", "eur", "gbp", "nok", "nzd", "sek", "jpy"]
test_curr = "sek"
test_cbs = ["fomc", "riks"]
n_quantiles = 2

joint = test_cbs[0]+"_"+test_cbs[1]

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
hml = daily_data["hml"].hml
s_d = rx


# Get the monetary policy action data
with open(data_path + "events.p", mode='rb') as fname:
    evt_data = pickle.load(fname)

events = []
for cb in test_cbs:
    evt = evt_data[cb]["change"].to_frame(cb)
    events.append(evt)
events = pd.concat(events, axis=1)
for col in events.columns:
    if col == "fomc":
        events[col] *= -1
events[joint] = events.mean(axis=1)
events = events[[joint]]


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


events = events.asfreq("B")[start_date:end_date]
day_count = count_gaps(events[joint])
qcut = pd.qcut(day_count, n_quantiles).to_frame("qcut")


if test_curr == "hml":
    test_asset = hml
else:
    test_asset = s_d[[test_curr]]

tt = pd.concat([test_asset, qcut, np.sign(events)], join="inner", axis=1).ffill()
tt = tt[start_date:end_date]

# out = pd.DataFrame(index=qcut.drop_duplicates().qcut,
#                    columns=[-1, 0, 1])
# for (qc, evt), df in tt.groupby(["qcut", joint]):
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
# plt.xlabel("{} action".format(joint.upper()), fontsize=20)
# plt.title("{} cycle around {} actions".format(test_curr.upper(),
#                                               joint.upper()), fontsize=20)
#
#
#
# out2 = pd.DataFrame(index=qcut.drop_duplicates().qcut,
#                    columns=["total"])
# for qc, df in tt.groupby(["qcut"]):
#     print(df.mean()[test_curr])
#     out2.loc[qc] = \
#         simple_tstat(df[test_curr])#df[test_curr].mean() * 10000
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
# plt.xlabel("{} action".format(joint.upper()), fontsize=20)
# plt.title("{} cycle around {} actions".format(test_curr.upper(),
#                                               joint.upper()), fontsize=20)
#
#
# out3 = pd.DataFrame(index=[-1, 0, 1], columns=["total"])
# for action, df in tt.groupby([joint]):
#     print(df.mean()[test_curr])
#     out3.loc[action] = \
#         simple_tstat(df[test_curr])#df[test_curr].mean() * 10000
#
# out3 = out3.astype("float")
#
# fig3, ax = plt.subplots(figsize=(12, 8))
# plt.setp(ax.get_xticklabels(), rotation=90, fontsize=20)
# plt.setp(ax.get_yticklabels(), rotation=90, fontsize=20)
# sns.heatmap(out3.T, ax=ax, annot=True, center=0.0,
#             annot_kws={"size": 20, "color": "black"})
# plt.xticks(rotation=0)
# plt.yticks(rotation=0)
# plt.ylabel("action", fontsize=20)
# plt.xlabel("{}".format(joint.upper()), fontsize=20)
# plt.title("{} cycle around {} actions".format(test_curr.upper(),
#                                               joint.upper()), fontsize=20)


t = tt.dropna()
q_list = []
for k in np.arange(0, n_quantiles):
    q_list.append(
        t.loc[t.qcut == qcut.drop_duplicates().dropna().qcut[k]].dropna()[test_curr])
h = t[test_curr]
q_list.append(h)

df = pd.concat(q_list, axis=1).fillna(0)
df.columns = \
    ["q"+str(k+1) for k in range(len(q_list)-1)] + [test_curr+"_"+joint]
df.cumsum().plot()