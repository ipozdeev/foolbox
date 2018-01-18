from foolbox.api import *

# Import the data
with open(data_path + "data_dev_m.p", mode="rb") as fname:
    fx_data_m = pickle.load(fname)

with open(data_path + "data_dev_d.p", mode="rb") as fname:
    fx_data = pickle.load(fname)

with open(data_path + "ir.p", mode="rb") as fname:
    ir_data = pickle.load(fname)

with open(data_path + "mom_dev_d_dir_s_3p.p", mode="rb") as fname:
    ir_mom = pickle.load(fname)

with open(data_path + "mom_dev_m_rx_rx.p", mode="rb") as fname:
    mom = pickle.load(fname)


# Set starting date since interest rate data are available for all currencies
# since 1997-04-01
start_date = "1997-04-01"
s_d = fx_data["spot_ret"]["1997-04-01":]
f_d = fx_data["fwd_disc"]["1997-04-01":]
rx = fx_data_m["rx"]["1997-04-01":]
ir_data = ir_data["1997-04-01":]

# Get explicit interest rate differentials
ir_differentials = ir_data.copy()
for col in ir_differentials.columns:
    ir_differentials[col] = ir_differentials[col]-ir_differentials["usd"]

ir_usd = ir_data[["usd"]]
ir_diff = ir_differentials.drop(["usd"], axis=1)
ir = ir_data.drop(["usd"], axis=1)


n_portfolios = 3

signals = ir.diff().rolling(5).mean().shift(1)
test1 = poco.rank_sort_adv(s_d, signals, n_portfolios, 5)
pp1 = poco.get_factor_portfolios(test1, hml=True)
pp1.cumsum().plot()

ir_mom.loc[:, (slice(3, 15), slice(1, 5))].cumsum().plot()

# n_portfolios = 3
#
# signals = ir.diff(22).shift(1)
# test1 = poco.rank_sort_adv(s_d+ir_diff/26100, signals, n_portfolios, 5)
# pp1 = poco.get_factor_portfolios(test1, hml=True)
# pp1.cumsum().plot()


# Test those rolling max bastards
signals_fd = (f_d.rolling(22).mean() -
              f_d.rolling(66).mean()).rolling(22).max().shift(1)
signals_id = (ir_diff.rolling(22).mean() -
              ir_diff.rolling(66).mean()).rolling(22).max().shift(1)

test1 = poco.rank_sort_adv(s_d, signals_fd, n_portfolios, 22)
pp1 = poco.get_factor_portfolios(test1, hml=True)
pp1.cumsum().plot()

test2 = poco.rank_sort_adv(s_d, signals_id, n_portfolios, 22)
pp2 = poco.get_factor_portfolios(test2, hml=True)
pp2.cumsum().plot()




signals = ir.diff().resample("M").sum()
test1 = poco.rank_sort_adv(rx, signals, n_portfolios)
pp1 = poco.get_factor_portfolios(test1, hml=True)
pp1.cumsum().plot()

