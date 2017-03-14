"""Here dwell generation and saving of momentum strategies based on interest
rate or forward discounts changes
"""
from foolbox.api import *
import itertools as itools

# Set sample, frequency, signals' and returns' keys
sample = "dev"        # set to "all" for al countries
freq = "d"            # set to "d" to daily frequency
ret_key = "spot_ret"  # spot returns for portfolio formation


# Set up the number of portoflios
n_portfolios = 3
start_date = "1997-04-01"  # interest rates are not available before this day
                           # for some currencies

# Set up the lookback, holdng and burn periods as a numpy ranges
lookback = np.arange(1, 23, 1)  # over how many past performance is evaluated
holding = np.arange(1, 23, 1)   # for how many periods the portfolio is held

# Set output name
out_name = "mom_dev_d_dir_s_3p.p"  # developed, monthly, spot, spot

# Get the FX data
with open(data_path+"data_"+sample+"_"+freq+".p", mode='rb') as fname:
    data = pickle.load(fname)

# Get and process interest rate data
with open(data_path + "ir.p", mode="rb") as fname:
    ir_data = pickle.load(fname)

s_d = data["spot_ret"][start_date:]
f_d = data["fwd_disc"][start_date:]
ir_data = ir_data[start_date:]

# Get explicit interest rate differentials
ir_differentials = ir_data.copy()
for col in ir_differentials.columns:
    ir_differentials[col] = ir_differentials[col]-ir_differentials["usd"]

ir_usd = ir_data[["usd"]]
ir_diff = ir_differentials.drop(["usd"], axis=1)
ir = ir_data.drop(["usd"], axis=1)


# Construct a multiindex dataframe to store the strategies
combos = list(itools.product(lookback, holding))

cols = pd.MultiIndex.from_tuples(combos, names=["lookback", "holding"])
mom = pd.DataFrame(index=s_d.index, columns=cols)

# Iterate over lookbacks, holdning horizons, and burns, generating strategies
t = 0
N = len(combos)
for lb, h in combos:
    # Construct momentum signal, offsetting it by burn
    signal = ir.diff().rolling(lb).mean().shift(1)
    # Get portfolios
    portf = poco.rank_sort_adv(s_d, signal, n_portfolios, h)
    # Append the multiindex frame
    mom.loc[:, (lb, h)] = poco.get_factor_portfolios(portf, hml=True).hml

    t = t + 1
    print("Finished run "+str(t)+" out of " + str(N))

# Save the output
with open(data_path+out_name, "wb") as fname:
    pickle.dump(mom, fname)
