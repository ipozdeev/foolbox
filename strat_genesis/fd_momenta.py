"""Here dwell generation and saving of momentum strategies based on interest
rate or forward discounts changes
"""
from foolbox.api import *
import itertools as itools

# Set sample, frequency, signals' and returns' keys
sample = "wmr_dev"        # set to "all" for al countries
freq = "d"            # set to "d" to daily frequency
ret_key = "spot_ret"  # spot returns for portfolio formation


# Set up the number of portoflios
n_portfolios = 3

# Set up the lookback, holdng and burn periods as a numpy ranges
lookback = np.arange(2, 5, 1)  # over how many past performance is evaluated
holding = np.arange(1, 5, 1)   # for how many periods the portfolio is held

# Set output name
out_name = "mom_dev_d_dfd_s_3p.p"  # developed, monthly, spot, spot

# Get the FX data
with open(data_path+"data_"+sample+"_"+freq+".p", mode='rb') as fname:
    data = pickle.load(fname)

start = "1997"

s_d = data["spot_ret"][start:]
f_d = data["fwd_disc"][start:]

# Construct a multiindex dataframe to store the strategies
combos = list(itools.product(lookback, holding))

cols = pd.MultiIndex.from_tuples(combos, names=["lookback", "holding"])
mom = pd.DataFrame(index=s_d.index, columns=cols)

# Iterate over lookbacks, holdning horizons, and burns, generating strategies
t = 0
N = len(combos)
for lb, h in combos:
    # Construct momentum signal, offsetting it by burn
    signal = f_d.diff().rolling(lb).mean().shift(1)
    # Get portfolios
    portf = poco.rank_sort_adv(s_d, signal, n_portfolios, h)
    # Append the multiindex frame
    mom.loc[:, (lb, h)] = poco.get_factor_portfolios(portf, hml=True).hml

    t = t + 1
    print("Finished run "+str(t)+" out of " + str(N))

# # Save the output
# with open(data_path+out_name, "wb") as fname:
#     pickle.dump(mom, fname)
