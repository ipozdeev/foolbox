"""Settings shared by all tables and figures, e.g. sample start,
currency data set"""

settings = {
    "drop_currencies": ["dkk",],
    "no_ois_currencies": ["jpy", "nok"],
    "sample_start": "2000-11-01",
    "sample_end": "2017-03-31",
    "fx_data": "fx_by_tz_aligned_d.p",
    "fx_data_fixed": "fx_by_tz_sp_fixed.p",
    "events_data": "events.p",
    "fig_folder": "wp_figures_limbo/",
    "usd_fixing_time": "NYC",
    "avg_impl_over": 5,
    "avg_refrce_over": 5,
    "bday_reindex": True,
    "base_threshold": 0.10,
    "base_holding_h": 10,
    "base_blackout": 1,
    }

# matplotlib settings -------------------------------------------------------
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# colors
new_gray = "#8c8c8c"
new_red = "#ce3300"
new_blue = "#2f649e"

my_palette = [new_red, new_blue, new_gray]
my_cmap = LinearSegmentedColormap.from_list("my_cmap", my_palette)

# settings
font_settings = {
    "family": "serif",
    "size": 12}
fig_settings = {
    "figsize": (8.27,11.3/3)}
tick_settings = {
    "labelsize": 12}
axes_settings = {
    "grid": True}
grid_settings = {
    "alpha": 0.66}

# parse all
plt.rc("xtick", **tick_settings)
plt.rc("ytick", **tick_settings)
plt.rc("figure", **fig_settings)
plt.rc("font", **font_settings)
plt.rc("axes", **axes_settings)
plt.rc("grid", **grid_settings)
