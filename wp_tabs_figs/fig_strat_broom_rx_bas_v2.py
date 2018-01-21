from foolbox.wp_tabs_figs.wp_settings import settings
from foolbox.data_mgmt import set_credentials as set_cred
from foolbox.visuals import broomstick_plot
import pandas as pd
import numpy as np

"""Broomstick plots for swap points- and bid-ask spreads-adjusted returns
"""
# Set the output path, input data and sample
path_to_data = set_cred.set_path("research_data/fx_and_events/")
out_path = path_to_data + settings["fig_folder"]
pkl_all = "broomstick_rx_data_v2.p"
pkl_fomc = "broomstick_rx_data_fomc_v2.p"
s_dt = settings["sample_start"]
e_dt = settings["sample_end"]

# Import the data
data_all = pd.read_pickle(path_to_data + pkl_all)
data_fomc = pd.read_pickle(path_to_data + pkl_fomc)

# Sample and reindex in logs
data_all = data_all.loc[s_dt:e_dt, :]
data_all = (np.log(data_all).diff() * 100).replace(np.nan, 0).cumsum()

data_fomc = data_fomc.loc[s_dt:e_dt, :]
data_fomc = (np.log(data_fomc).diff() * 100).replace(np.nan, 0).cumsum()

# Plot the results and save 'em
fig_all = broomstick_plot(data_all)
fig_all.savefig(out_path + "broomstick_plot_rx_bas.pdf")

fig_fomc = broomstick_plot(data_fomc)
fig_fomc.savefig(out_path + "broomstick_plot_rx_bas_usd.pdf")
