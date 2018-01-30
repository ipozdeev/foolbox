import pandas as pd
import numpy as np
from foolbox.utils import diag_table_of_regressions
from foolbox.linear_models import PureOls
import matplotlib.pyplot as plt
import seaborn as sns
from foolbox.data_mgmt import set_credentials as set_cred
from foolbox.utils import to_better_latex

path_to_data = set_cred.set_path("research_data/fx_and_events/",
                                 which="gdrive")
path_to_out = path_to_data + "wp_figures_limbo/"


def data_for_spanning_tests(n_portf=3):
    """

    Returns
    -------

    """
    # fx
    with pd.HDFStore(path_to_data + "strategies_m.h5", mode="r") as hangar:
        fx_strats = hangar["strats"]

    # to logs
    fx_strats = np.log(fx_strats).diff().replace(0.0, np.nan)\
                    .dropna(how="all") * 100

    # saga
    with pd.HDFStore(path_to_data + "strategies.h5", mode="r") as hangar:
        saga_strat = hangar["saga"]

    # to logs
    saga_strat = np.log(saga_strat).diff().replace(0.0, np.nan).dropna() * 100
    saga_strat.name = "saga"

    # Fama-French
    ff_x = pd.read_csv(path_to_data +
                       "F-F_Research_Data_5_Factors_2x3_daily.csv", skiprows=3,
                       index_col=0, parse_dates=True)
    ff_x = ff_x.drop("RF", axis=1)
    ff_x.columns = [p.lower() for p in ff_x.columns]

    # all
    ix = pd.IndexSlice
    fx_x = fx_strats.loc[:, ix[:, n_portf]]
    fx_x.columns = fx_x.columns.droplevel(1)

    x = pd.concat((fx_x, ff_x), axis=1)

    return saga_strat, x


def spanning_tests(target, factors, scale=10):
    """

    Parameters
    ----------
    target
    factors
    scale

    Returns
    -------

    """
    # factor by factor ------------------------------------------------------
    res_individ = dict()

    # loop over factors
    for f_name, f in factors.iteritems():
        mod = PureOls(y0=target, X0=f.rename("beta"), add_constant=True)
        res_individ[f_name] = mod.get_diagnostics(HAC=True)

    # concat individual results
    res_individ = pd.concat(res_individ, axis=1)

    # all factors together --------------------------------------------------
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


def make_diagonals(res_i, res_t):
    """
    Parameters
    ----------
    res_i
    res_t

    Returns
    -------

    """
    beta_d = diag_table_of_regressions(
        individ=res_i.loc["coef", :].unstack(level=0),
        joint=res_t.loc["coef", :])
    tstat_d = diag_table_of_regressions(
        individ=res_i.loc["tstat", :].unstack(level=0),
        joint=res_t.loc["tstat", :])

    return beta_d, tstat_d


def wrapper_spanning_tests(n_portf):
    """

    Parameters
    ----------
    n_portf

    Returns
    -------

    """
    # load data
    y, x = data_for_spanning_tests(n_portf=n_portf)

    renames = {
        "carry": "CAR",
        "cma": "CMA",
        "dollar_carry": "DCAR",
        "smb": "SMB",
        "hml": "HML",
        "mkt-rf": "MKT",
        "momentum": "MOM",
        "value": "VAL",
        "vrp": "VRP",
        "rmw": "RMW",
        "dollar_index": "DOL"
    }

    x = x.rename(columns=renames, level=0) * 100
    y = y * 100

    fx_fact = ["DOL", "CAR", "DCAR", "MOM", "VAL", "VRP"]
    eq_fact = ["MKT", "SMB", "HML", "CMA", "RMW"]

    res_i, res_t = spanning_tests(target=y, factors=x.loc[:, fx_fact+eq_fact])
    res_i_fx, res_t_fx = spanning_tests(target=y, factors=x.loc[:, fx_fact])
    res_i_eq, res_t_eq = spanning_tests(target=y, factors=x.loc[:, eq_fact])

    tab_b, tab_ts = make_diagonals(res_i, res_t)
    tab_b_fx, tab_ts_fx = make_diagonals(res_i_fx, res_t_fx)
    tab_b_eq, tab_ts_eq = make_diagonals(res_i_eq, res_t_eq)

    bta = pd.concat((tab_b.loc[:, fx_fact],
                     tab_b_fx.loc[:, "joint"].rename("fx"),
                     tab_b.loc[:, eq_fact],
                     tab_b_eq.loc[:, "joint"].rename("eq"),
                     tab_b.loc[:, "joint"].rename("fx+eq")), axis=1)
    bta = bta.loc[fx_fact + eq_fact + ["const"], :]

    tst = pd.concat((tab_ts.loc[:, fx_fact],
                     tab_ts_fx.loc[:, "joint"].rename("fx"),
                     tab_ts.loc[:, eq_fact],
                     tab_ts_eq.loc[:, "joint"].rename("eq"),
                     tab_ts.loc[:, "joint"].rename("fx+eq")), axis=1)
    tst = tst.loc[fx_fact + eq_fact + ["const"], :]

    # add r-squared
    ix = pd.IndexSlice
    r_sq_i = res_i.loc[["adj r2", "nobs"]].loc[:, ix[:, "const"]]
    r_sq_i.columns = r_sq_i.columns.droplevel(1)
    r_sq_fx = res_t_fx.loc[["adj r2", "nobs"], "const"].rename("fx")
    r_sq_eq = res_t_eq.loc[["adj r2", "nobs"], "const"].rename("eq")
    r_sq_j = res_t.loc[["adj r2", "nobs"], "const"].rename("fx+eq")
    r_squared = pd.concat((r_sq_i.loc[:, fx_fact],
                           r_sq_fx,
                           r_sq_i.loc[:, eq_fact],
                           r_sq_eq,
                           r_sq_j), axis=1)

    res = to_better_latex(bta, tst, r_squared, '{:3.2f}', '{:3.2f}',
                          buf=path_to_out + "spanning_tests.tex")


if __name__ == "__main__":

    res = wrapper_spanning_tests(n_port=5)