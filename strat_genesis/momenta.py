"""Here dwell generation and saving of momentum strategies
"""
from foolbox.api import *
import itertools as itools
import pickle


# Set sample, frequency, signals' and returns' keys
sample = "dev"        # set to "all" for al countries
freq = "m"            # set to "d" to daily frequency
sig_key = "spot_ret"  # spot returns for signals
ret_key = "rx"  # spot returns for portfolio formation

# Set up the number of portoflios
n_portfolios = 5

# Set up the lookback, holdng and burn periods as a numpy ranges
lookback = np.arange(1, 36, 1)  # over how many past performance is evaluated
holding = np.arange(1, 12, 1)   # for how many periods the portfolio is held
burn = np.arange(0, 12, 1)       # how many recent periods before rebalancing
                                # are to be discarded

# Set output name
out_name = "mom_dev_m_s_rx.p" # developed, monthly, spot, spot

# Get the data
with open(data_path+"data_"+sample+"_"+freq+".p", mode='rb') as fname:
    data = pickle.load(fname)

sig_data = data[sig_key]
ret_data = data[ret_key]

# Construct a multiindex dataframe to store the strategies
# First get combinations of lokback, holding, and burn periods
all_combos = list(itools.product(lookback, holding, burn))
# Second, filter out combinations where lookback is less or equal burn
combos = list()
for combo in all_combos:
    if combo[0] > combo[2]:
        combos.append(combo)

cols = pd.MultiIndex.from_tuples(combos, names=["lookback", "holding", "burn"])
mom = pd.DataFrame(index=ret_data.index, columns=cols)

# Iterate over lookbacks, holdning horizons, and burns, generating strategies
t = 0
N = len(combos)
for lb, h, b in combos:
    # Construct momentum signal, offsetting it by burn
    signal = sig_data.rolling(lb-b).sum().shift(1+b)
    # Get portfolios
    portf = poco.rank_sort_adv(ret_data, signal, n_portfolios, h)
    # Append the multiindex frame
    mom.loc[:, (lb, h, b)] = poco.get_factor_portfolios(portf, hml=True).hml

    t = t + 1
    print("Finished run "+str(t)+" out of " + str(N))

# Save the output
# with open(data_path+out_name, "wb") as fname:
#     pickle.dump(mom, fname)
