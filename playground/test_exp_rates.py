from foolbox.api import *

exp_rates = pd.read_csv(data_path + "temp_impl_rate.csv", index_col=0,
                        parse_dates=True)

with open(data_path + "data_wmr_dev_d.p", mode="rb") as fname:
    fx_data = pickle.load(fname)

with open(data_path + "events.p", mode="rb") as fname:
    events = pickle.load(fname)

with open (data_path + "policy_rates.p", mode="rb") as fname:
    policy_rates = pickle.load(fname)

with open(data_path + "implied_rates_from_deposit.p", mode="rb") as fname:
    dep = pickle.load(fname)["deposit"]

dep.drop(["dkk"], axis=1)

rx = fx_data["rx_5d"]
rx["usd"] = -rx.mean(axis=1)

joint = events["joint_cbs"]
joint_lvl = events["joint_cbs_lvl"]


icap_1m = policy_rates["icap_1m"].drop(["dkk"], axis=1)
impl = icap_1m.shift(6).reindex(joint_lvl.index)
impl_diff = impl - joint_lvl.shift(6).drop(["nok", ], axis=1)

# Align data
exp_rates.columns = ["cad", "gbp", "jpy", "eur", "usd", "aud", "nzd", "sek",
                     "chf"]
exp_rates = exp_rates[sorted(exp_rates.columns)]

exp_rates, joint = exp_rates.align(joint, join="inner", axis=1)

rx = rx[exp_rates.columns]

rx = rx.drop(["eur", "chf", "jpy", "nzd"], axis=1)["2003":]
exp_rates = exp_rates.drop(["eur", "jpy"], axis=1)["2003":]

rx, impl_diff = rx.align(impl_diff, axis=1)

ueber = poco.multiple_timing(rx,
                             impl_diff.where(abs(impl_diff)>=0.1),
                             xs_avg=True)

ueber.dropna().cumsum().plot(lw=2.25, c='red', title='In your face, pal!')
taf.descriptives(ueber.dropna(), 52)

ueber2 = poco.multiple_timing(rx,
                             impl_diff.where(abs(impl_diff)>=0.25),
                             xs_avg=False)

ueber2.replace(np.nan, 0).cumsum().plot()







from foolbox.api import *

with open(data_path + "data_wmr_dev_d.p", mode="rb") as fname:
    fx_data = pickle.load(fname)

with open(data_path + "events.p", mode="rb") as fname:
    events = pickle.load(fname)

with open (data_path + "policy_rates.p", mode="rb") as fname:
    p_rates = pickle.load(fname)


# Construct market portfolios for each currency
tmp_ret = fx_data["spot_ret"]

out2 = pd.DataFrame(index=tmp_ret.index, columns=tmp_ret.columns)
for col in tmp_ret.columns:
    tmp_df = tmp_ret.copy()
    tmp_df = tmp_df.subtract(tmp_df[col], axis=0)
    tmp_df[col] = tmp_df[col] - tmp_ret[col]
    out2[col] = tmp_df.mean(axis=1)

lag = 5
# Get the five-day rx, add the dollar column
rx = out2
rx["usd"] = -1 * tmp_ret.mean(axis=1)     # add usd
rx = -1 * rx.rolling(lag).sum().shift(0)[["usd"]]

joint = events["joint_cbs"]
joint_lvl = events["joint_cbs_lvl"].fillna(method="ffill")

p_rate = p_rates["tr_1m"]

# Get intersection of column names for each of the inputs
common_cols = rx.columns.intersection(joint_lvl.columns)\
    .intersection(p_rate.columns)

# Select the common sample
rx = rx[common_cols]
joint = joint[common_cols]
joint_lvl = joint_lvl[common_cols]
p_rate = p_rate[common_cols]

impl = p_rate.shift(lag).reindex(joint_lvl.index)
impl_diff = impl - joint_lvl.shift(lag)


ueber = poco.multiple_timing(rx,
                             impl_diff.where(abs(impl_diff)>=0.15),
                             xs_avg=True)

ueber.dropna().cumsum().plot(lw=2.25, c='red', title='In your face, pal!')
taf.descriptives(ueber.dropna(), 52)

ueber2 = poco.multiple_timing(rx,
                             impl_diff.where(abs(impl_diff)>=0.2),
                             xs_avg=False)

ueber2.replace(np.nan, 0).cumsum().plot()
