import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

plt.rc("font", family="serif", size=12)
new_gray = "#8c8c8c"


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
    ax.plot(data, color=new_gray, lw=0.75, alpha=0.25)

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
