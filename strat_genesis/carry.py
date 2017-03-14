"""Here dwells generation and saving of carry strategies
"""
from foolbox.api import *
import itertools as itools
import pickle


# Set sample, frequency, signals' and returns' keys
sample = "dev"        # set to "all" for al countries
freq = "d"            # set to "d" to daily frequency
sig_key = "fwd_disc"  # spot returns for signals
ret_key = "spot_ret"  # spot returns for portfolio formation

# Set up the number of portoflios
n_portfolios = 5

# Set up the lookback and holdng periods as a numpy ranges
lookback = np.arange(1, 66, 3)  # over how many past performance is evaluated
holding = np.arange(1, 66, 3)   # for how many periods the portfolio is held

# Set output name
out_name = "carry_dev_d_fwd_disc_s.p" # developed, daily, fwd_disc, spot

# Get the data
with open(data_path+"data_"+sample+"_"+freq+".p", mode='rb') as fname:
    data = pickle.load(fname)

sig_data = data[sig_key]
ret_data = data[ret_key]

# Construct a multiindex dataframe to store the strategies
# Get combinations of lokback, holding, and burn periods
combos = list(itools.product(lookback, holding))

cols = pd.MultiIndex.from_tuples(combos, names=["lookback", "holding"])
carry = pd.DataFrame(index=ret_data.index, columns=cols)

# Iterate over lookbacks, holdning horizons, and burns, generating strategies
t = 0
N = len(combos)
for lb, h in combos:
    # Construct momentum signal, offsetting it by burn
    signal = sig_data.diff(lb).shift(1)
    # Get portfolios
    portf = poco.rank_sort_adv(ret_data, signal, n_portfolios, h)
    # Append the multiindex frame
    carry.loc[:, (lb, h)] = poco.get_factor_portfolios(portf, hml=True).hml

    t = t + 1
    print("Finished run "+str(t)+" out of " + str(N))

# Save the output
# with open(data_path+out_name, "wb") as fname:
#     pickle.dump(carry, fname)
