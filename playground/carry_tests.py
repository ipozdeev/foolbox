from foolbox.api import *

# Import the data
with open(data_path + "data_dev_m.p", mode="rb") as fname:
    fx_data_m = pickle.load(fname)

with open(data_path + "data_dev_d.p", mode="rb") as fname:
    fx_data = pickle.load(fname)

with open(data_path + "ir.p", mode="rb") as fname:
    ir_data = pickle.load(fname)


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

ir_differentials = ir_differentials.drop(["usd"], axis=1)


# Construct carry trade portfolios according to fwd_discs or ir_diffs
n_portfolios = 5

# Carry, based on previous month's average fwd_disc, held next month
signal = f_d.resample("M").mean().shift(1)
carry1 = poco.rank_sort_adv(rx, signal, n_portfolios)
carry1_p = poco.get_factor_portfolios(carry1, hml=True)
carry1_p.cumsum().plot()


# Carry, based on previous month's'last fwd_disc, held for the next month
signal = fx_data_m["fwd_disc"].shift(1)
carry2 = poco.rank_sort_adv(rx, signal, n_portfolios)
carry2_p = poco.get_factor_portfolios(carry2, hml=True)
carry2_p.cumsum().plot()


# Carry, based on previous month's average ir_diff, held next month
signal = ir_differentials.resample("M").mean().shift(1)
signal, _ = signal.align(rx, axis=0)
carry3 = poco.rank_sort_adv(rx.sort_index(axis=1), signal, n_portfolios)
carry3_p = poco.get_factor_portfolios(carry3, hml=True)
carry3_p.cumsum().plot()


# Carry, based on previous month's last ir_diff, held next month
signal = ir_differentials.resample("M").last().shift(1)
signal, _ = signal.align(rx, axis=0)
carry4 = poco.rank_sort_adv(rx.sort_index(axis=1), signal, n_portfolios)
carry4_p = poco.get_factor_portfolios(carry4, hml=True)
carry4_p.cumsum().plot()


