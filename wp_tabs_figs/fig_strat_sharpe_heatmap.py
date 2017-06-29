from foolbox.api import *
import seaborn as sns
from wp_tabs_figs.wp_settings import settings

with open(data_path + 'broom_ret_rx_bas.p', mode='rb') as fname:
    backtest_ret = pickle.load(fname)

# Some settings: choose subset of thresholds to avoid clutter
thresholds_to_plot = np.arange(2, 26, 2)
round_to = 2  # round sharpe ratios to 'round_to' number of decimals

# Normalize Sharpe ratios to a certain horizon, for comparability
normalize = True
normalization_period = 10

# plot settings
# font, colors
plt.rc("font", family="serif", size=12)
out_path = data_path + settings["fig_folder"]

# Get the Sharpe ratios
backtest_sharpe = backtest_ret.mean()/backtest_ret.std()
backtest_sharpe = backtest_sharpe.unstack()  # Get rid of the time dimension

# Normalize if needed
if normalize:

    # Tile the holding horizon to the shape of the Sharpe ratio df
    normalization_matrix = pd.DataFrame(
        [backtest_sharpe.index.tolist() for col in backtest_sharpe.columns],
        index=backtest_sharpe.columns,
        columns=backtest_sharpe.index).T

    # Populate the matrix with normalization factors
    normalization_matrix = normalization_matrix.pow(0.5) / \
        np.sqrt(normalization_period)

    # Transform every entry into a 'normalization period'-Sharpe ratio
    backtest_sharpe = backtest_sharpe / normalization_matrix


# Prepare the data to plot, choose thresholds, round the numbers
data_to_plot = backtest_sharpe.loc[:, thresholds_to_plot]. \
    sort_index(axis=1, ascending=False).T.round(round_to)

# Plot the stuff
fig1, ax = plt.subplots(figsize=(12, 8))
plt.setp(ax.get_xticklabels(), rotation=90, fontsize=12)
plt.setp(ax.get_yticklabels(), rotation=90, fontsize=12)
sns.heatmap(data_to_plot, ax=ax, annot=True, center=0.034,
            annot_kws={"size": 10, "color": "black"})
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.ylabel("threshold")
plt.xlabel("holding period")

# Save the bastard
fig1.tight_layout()
fig1.savefig(out_path + "rx_bas_sharpe.pdf")
