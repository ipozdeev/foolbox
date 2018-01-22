import pandas as pd
import numpy as np
from foolbox.linear_models import PureOls
import matplotlib.pyplot as plt
import seaborn as sns
from foolbox.data_mgmt import set_credentials as set_cred


def spanning_tests(target, factors, scale=1):
    """

    Parameters
    ----------
    target
    factors
    scale

    Returns
    -------

    """
    res_individ = dict()
    for f_name, f in factors.iteritems():
        mod = PureOls(y0=target, X0=f.rename("beta"), add_constant=True)
        res_individ[f_name] = mod.get_diagnostics(HAC=True)

    res_individ = pd.concat(res_individ, axis=1)

    # all together
    mod = PureOls(y0=target, X0=factors, add_constant=True)
    res_together = mod.get_diagnostics(HAC=True)

    return res_individ, res_together

def plot_strats_on_grid(strats):
    """

    Returns
    -------

    """
    # plot
    g = sns.PairGrid(strats.dropna(how="any"), size=1)

    # on-diagonal elements are densities
    g.map_diag(sns.kdeplot)

    # off-diaf elements are scatterplots
    g.map_offdiag(plt.scatter, marker=".", alpha=0.25)

    # turn off ticks:
    #   changes apply to both axes
    #   both major and minor ticks are affected
    #   ticks along the bottom edge are off
    #   ticks along the top edge are off
    #   labels along the bottom edge are off
    for ax in g.axes.flat:
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(
            axis='both',
            which='both',
            bottom='off',
            top='off',
            labelbottom='off')
    plt.rcParams['axes.unicode_minus'] = False

    # add correlation
    cormat = strats.corr()

    for p, q in zip(*np.triu_indices_from(g.axes, 1)):
        this_ax = g.axes[p, q]
        xmin, xmax = np.array(this_ax.get_xlim())
        ymin, ymax = np.array(this_ax.get_ylim())
        dy = ymax - ymin
        this_ax.clear()
        this_ax.text(xmax - abs(0.15 * xmin), ymax - dy * 0.5,
                     "{:+3.2f}".format(cormat.iloc[p, q]),
                     fontsize=11,
                     horizontalalignment="right",
                     verticalalignment="center")

        plt.setp(this_ax.get_yticklabels(), visible=False)
        this_ax.tick_params(
            axis='both',
            which='both',
            bottom='off',
            top='off',
            labelbottom='off')

        this_ax.set_axis_bgcolor('white')

    return g

if __name__ == "__main__":
    path_to_data = set_cred.set_path("research_data/fx_and_events/",
                                     which="gdrive")
    path_to_out = path_to_data + "wp_figures_limbo/"

    # data ------------------------------------------------------------------
    with pd.HDFStore(path_to_data + "strategies.h5", mode="r") as hangar:
        strats = pd.DataFrame.from_dict(
            {k[1:]: hangar.get(k) for k in hangar.keys()}
        )

    # # plot
    # strats_grid = plot_strats_on_grid(path_to_data)
    #
    # # save
    # strats_grid.savefig(path_to_out + "strats_grid.pdf", bbox_inches="tight")

    # spanning tests
    res_i, res_t = spanning_tests(target=strats.loc[:, "saga"],
                                  factors=strats.drop("saga", axis=1))

    print("lol")