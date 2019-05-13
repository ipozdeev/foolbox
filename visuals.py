import pandas as pd
from seaborn import color_palette, diverging_palette

import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.lines as mlines


# colors
palette = color_palette("deep", 8)
color_gray = "#afafaf"
color_blue = colors.to_hex(palette[0])
color_red = colors.to_hex(palette[1])

new_blue_hsluv = 248.2
new_red_hsluv = 26.8
heatmap_cmap = diverging_palette(new_blue_hsluv, new_red_hsluv, 85, 54,
                                 n=15, as_cmap=True)

# figsizes
figsize_single = (8.27-2, (11.69-2) / 3)
figsize_double = (8.27-2, (11.69-2) / 2)
figsize_full = (8.27-2, (11.69-2) / 1.25)


def set_style_paper(fontsize=10):
    """
    """
    # settings
    font_settings = {
        "family": "serif",
        "size": fontsize}
    fig_settings = {
        "figsize": figsize_single}
    tick_settings = {
        "labelsize": fontsize}
    axes_settings = {
        "grid": True}
    grid_settings = {
        "linestyle": '-',
        "alpha": 0.75}
    legend_settings = {
        "fontsize": fontsize}

    # apply all
    plt.rc("xtick", **tick_settings)
    plt.rc("ytick", **tick_settings)
    plt.rc("figure", **fig_settings)
    plt.rc("font", **font_settings)
    plt.rc("axes", **axes_settings)
    plt.rc("grid", **grid_settings)
    plt.rc("legend", **legend_settings)


def broomstick_plot(data, ci=(0.1, 0.9)):
    """Given an input array of data, produces a broomstick plot, with all
    series in gray, the mean of the series in black (along with confidence
    interval). The data are intended to be cumulative returns on a set of
    trading strategies

    Parameters
    ----------
    data: pd.DataFrame
        of the cumulative returns to plot
    ci: tuple
        of floats specifying lower and upper quantiles for empirical confidence
        interval. Default is (0.1, 0.9)

    Returns
    -------
    figure: matplotlib.pyplot.plot
        with plotted data


    """
    # Check the input data
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Nope, the input should be a DataFrame")

    # Drop absent observations if any
    data = data.dropna()

    # Get the mean and confidence bounds
    cb_u = data.quantile(ci[1], axis=1)
    cb_l = data.quantile(ci[0], axis=1)
    avg = data.mean(axis=1)
    stats = pd.concat([avg, cb_u, cb_l], axis=1)
    stats.columns = ["mean", "cb_u", "cb_l"]

    # Start plotting
    fig, ax = plt.subplots(figsize=(8.4, 11.7/3))
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    # Grid
    ax.grid(which="both", alpha=0.33, linestyle=":")
    ax.plot(data, color=color_gray, lw=0.75, alpha=0.25)

    ax.plot(stats["mean"], color="k", lw=2)
    ax.plot(stats[["cb_l", "cb_u"]], color="k", lw=2, linestyle="--")

    # Construct lines for the custom legend
    solid_line = mlines.Line2D([], [], color='black', linestyle="-",
                               lw=2, label="Mean")

    ci_label = "{}th and {}th percentiles"\
        .format(int(ci[0]*100), int(ci[1]*100))
    dashed_line = mlines.Line2D([], [], color='black', linestyle="--",
                                lw=2, label=ci_label)
    ax.legend(handles=[solid_line, dashed_line], loc="upper left", fontsize=10)
    ax.set_ylabel("cumulative return, in percent", visible=True)

    return fig


if __name__ == '__main__':
    pass
