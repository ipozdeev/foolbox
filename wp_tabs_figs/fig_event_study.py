import pandas as pd
import numpy as np
from foolbox.api import *
from foolbox.finance import PolicyExpectation
from foolbox.wp_tabs_figs.wp_settings import *

import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

minorLocator = MultipleLocator(1)
majorLocator = MultipleLocator(5)

# data path -------------------------------------------------------------
data_path = set_credentials.gdrive_path("research_data/fx_and_events/")
out_path = set_credentials.gdrive_path(
    "research_data/fx_and_events/wp_figures_limbo/")

# currency to drop
drop_curs = settings["drop_currencies"] + ["usd"]

# dates to trim to
s_dt = settings["sample_start"]
e_dt = settings["sample_end"]


def prepare_bond_data(maturity="1m"):
    """
    Returns
    -------

    """
    bonds_data = pd.read_pickle(data_path + "msci_ted_vix_d.p")

    # res = pd.concat(bonds_data, axis=1).xs(maturity, axis=1, level=1,
    #                                        drop_level=True)
    res = bonds_data["msci_pi"]

    # res = res.drop(drop_curs, axis=1, errors="ignore")

    res = np.log(res).diff() * 252

    return res


def prepare_events_data(perfect_foresight=False):
    """

    Parameters
    ----------
    perfect_foresight

    Returns
    -------

    """
    # window
    wa, wb, wc, wd = -10, -1, 1, 5

    # events ----------------------------------------------------------------
    events_data = pd.read_pickle(data_path + settings["events_data"])
    events = events_data["joint_cbs"]
    # events = events.loc[s_dt:e_dt].drop(drop_curs, axis=1, errors="ignore")

    if not perfect_foresight:
        events_exp = dict()

        # drop no-ois currencies
        events = events.drop(
            settings["drop_currencies"] + settings["no_ois_currencies"],
            axis=1, errors="ignore")

        # mappers
        map_proxy_rate = lambda x: x.rolling(5, min_periods=1).mean()
        map_expected_rate = map_proxy_rate

        for c in events.columns:
            pe = PolicyExpectation.from_pickles(data_path, c,
                                                start_dt=events.index[0])

            this_exp = pe.forecast_policy_direction(
                lag=-wa + 2, h_low=-0.1, h_high=0.1,
                map_proxy_rate=map_proxy_rate,
                map_expected_rate=map_expected_rate)

            events_exp[c] = this_exp

        events = pd.concat(events_exp, axis=1)

    events = events.dropna(how="all")

    return events


def prepare_fx_data():
    """

    Parameters
    ----------
    perfect_foresight

    Returns
    -------

    """
    # spot returns ----------------------------------------------------------
    fx_data = pd.read_pickle(data_path + settings["fx_data"])
    spot = fx_data["spot_mid"]

    # fx_data = pd.read_pickle(data_path + "fxcm_counter_usd_h1.p")
    # spot = (fx_data["bid_close"] + fx_data["ask_close"]) / 2
    # spot.index = spot.index.tz_convert("US/Eastern")
    # spot = spot.loc[spot.index.hour.isin([16, 17])]
    # spot = spot.resample('B').last()

    ret = np.log(spot).diff() * 100
    ret = ret.loc[s_dt:e_dt, :]
    ret.index = ret.index.tz_localize(None)

    return ret


def fig_event_study(events, ret, direction, mean_type="count_weighted",
                    window=(-10, -1, 1, 5), ci_kwds=None, ci_exog=None):
    """
    """
    if ci_kwds is None:
        ci_kwds = {"ps": 0.95, "method": "boot", "n_iter": 100}

    # unpack window
    wa, wb, wc, wd = window

    # choose direction
    if direction == "cut":
        events = events.copy().where(events < 0).dropna(how="all")
    elif direction == "hike":
        events = events.copy().where(events > 0).dropna(how="all")
    elif direction == "no_changes":
        events = events.copy().where(events == 0).dropna(how="all")
    else:
        raise ValueError("Not implemented.")

    # event study instance
    es = EventStudy(ret, events, window, mean_type=mean_type)

    # confidence interval
    if (ci_kwds["method"] == "simple") & \
            (ci_kwds.get("variances", None) is None):
        variances = (es.data**2).where(es.mask_between_events).ewm(
            alpha=0.4).mean()
        variances = variances.where(es.events.notnull()).reindex(
            index=es.events.index)
        ci_kwds["variances"] = variances

    if ci_exog is None:
        _ = es.get_ci(**ci_kwds)
    else:
        es.ci = ci_exog

    # car -------------------------------------------------------------------
    cumsums = es.car.mean(axis=1, level="assets")

    # insert zero row
    cumsums.loc[0, :] = np.nan
    cumsums = cumsums.sort_index()

    # mean ------------------------------------------------------------------
    the_mean = es.the_mean.copy()

    # insert zero row
    the_mean.loc[0] = np.nan
    the_mean = the_mean.sort_index()

    # plot ------------------------------------------------------------------
    # plot individual
    fig_individ, ax_individ = plt.subplots(figsize=(8.4, 11.7/3))

    # individual currencies
    cumsums.plot(ax=ax_individ, color=my_gray)

    # black dots at the "inception"
    cumsums.loc[[wb], :].plot(ax=ax_individ, color="k",
                              linestyle="none", marker=".",
                              markerfacecolor="k")
    cumsums.loc[[wc], :].plot(ax=ax_individ, color="k",
                              linestyle="none", marker=".",
                              markerfacecolor="k")

    ax_individ.legend_.remove()

    # plot ------------------------------------------------------------------
    # plot average
    fig_avg, ax_avg = plt.subplots(figsize=(8.4, 11.7/3))

    the_mean.plot(ax=ax_avg, color='k', linestyle="-", linewidth=1.5)

    # plot confidence interval
    ax_avg.fill_between(es.the_mean.index, es.ci.iloc[:, 0].values,
                        es.ci.iloc[:, 1].values,
                        color=my_gray, alpha=0.5, label="conf. interval")

    # ax_avg.fill_between(es.the_mean.index,
    #     ci_2.iloc[:,0].values,
    #     ci_2.iloc[:,1].values,
    #     color=gr_2, alpha=0.5, label="conf. interval")

    # final retouche --------------------------------------------------------
    furbish_plot(ax_individ, arrows=True,
                 tot_curs={"a": cumsums.iloc[0, :], "d": cumsums.iloc[-1, :]})

    furbish_plot(ax_avg, set_xlabel=True, arrows=False)

    black_line = mlines.Line2D([], [], linewidth=1.5, color='k',
                               label="cross-currency CAR")

    gray_patch = mpatches.Patch(color=my_gray, label='95% conf. int.')

    # gray_patch_2 = mpatches.Patch(color=gr_2, label='90% conf. int.')
    # lines = [black_line, gray_patch, gray_patch_2]
    lines = [black_line, gray_patch]
    labels = [line.get_label() for line in lines]
    ax_avg.legend(lines, labels, fontsize=12, loc="upper right")

    # y-label
    ax_individ.set_ylabel("cumulative return, in percent")
    ax_avg.set_ylabel("cumulative return, in percent")

    return fig_individ, fig_avg


def furbish_plot(ax, set_xlabel=False, arrows=False, tot_curs=None):
    """
    f_i
    ax=f_i.axes[0]
    """
    # limits ----------------------------------------------------------------
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    dylim = ylim[1]-ylim[0]
    ax.set_xlim((xlim[0]-1.75, xlim[1]+1.75))

    # arrow, text -----------------------------------------------------------
    if arrows:
        ax.arrow(x=-1, y=ylim[1]*0.7, dx=-3.3, dy=0,
                 length_includes_head=True,
                 head_width=dylim/30,
                 head_length=0.2,
                 linewidth=2,
                 fc='k', ec='k')
        # ax.arrow(x=-1, y=ylim[0]+0.1, dx=-3.3, dy=0,
        #     length_includes_head=True,
        #     head_width=dylim/30,
        #     head_length=0.2,
        #     linewidth=2,
        #     fc='k', ec='k')
        ax.text(-1, ylim[1]*0.75, r"hold from x to -1",
                verticalalignment="bottom", horizontalalignment="right",
                fontsize=12)
        # ax.text(-1, ylim[0]+0.15, r"hold from x to -1",
        #     verticalalignment="bottom", horizontalalignment="right",
        #     fontsize=12)

        ax.arrow(x=1, y=ylim[1]*0.7, dx=2.3, dy=0,
                 length_includes_head=True,
                 head_width=dylim/30,
                 head_length=0.2,
                 linewidth=2,
                 fc='k', ec='k')
        # ax.arrow(x=1, y=ylim[0]+0.1, dx=2.3, dy=0,
        #     length_includes_head=True,
        #     head_width=dylim/30,
        #     head_length=0.2,
        #     linewidth=2,
        #     fc='k', ec='k')
        ax.text(1, ylim[1]*0.75, r"hold from 1 to x",
                verticalalignment="bottom", horizontalalignment="left",
                fontsize=12)
        # ax.text(1, ylim[0]+0.15, r"hold from 1 to x",
        #     verticalalignment="bottom", horizontalalignment="left",
        #     fontsize=12)

    # zero line -------------------------------------------------------------
    # ax.xaxis.set_ticks(cumsums.dropna().index)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1.0, alpha=0.75)

    # ticks -----------------------------------------------------------------
    ax.xaxis.set_major_locator(majorLocator)
    ax.xaxis.set_minor_locator(minorLocator)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # grid ------------------------------------------------------------------
    ax.grid(which="both", alpha=0.33, linestyle=":")

    # labels ----------------------------------------------------------------
    if set_xlabel:
        ax.set_xlabel("days after event", fontsize=12)

    # individual currencies -------------------------------------------------
    ax.set_ylim(ylim)
    if tot_curs is not None:
        # ax = f_i.axes[0]
        # sort in descending order
        tot_curs["a"] = tot_curs["a"].sort_values(ascending=False)
        tot_curs["d"] = tot_curs["d"].sort_values(ascending=False)

        cnt = 0
        for c, p in tot_curs["a"].iteritems():
            this_x_pos = -10.1 - 0.6*(cnt % 2)
            ax.annotate(c, xy=(this_x_pos, p),
                        horizontalalignment='right',
                        verticalalignment='center',
                        fontsize=12)
            cnt += 1

        cnt = 0
        for c, p in tot_curs["d"].iteritems():
            this_x_pos = 5.1 + 0.6*(cnt % 2)
            ax.annotate(c, xy=(this_x_pos, p),
                        horizontalalignment='left',
                        verticalalignment='center',
                        fontsize=12)
            cnt += 1

    return


def prepare_rates_data(maturity="1m"):
    """

    Returns
    -------

    """
    bonds_data = pd.read_pickle(data_path + "ois_bloomi_1w_30y.p")

    res = pd.concat(bonds_data, axis=1).xs(maturity, axis=1, level=0,
                                           drop_level=True)

    # res = res.drop(drop_curs, axis=1, errors="ignore")

    return res


def prepare_futures_data(which="carry"):
    """
    Parameters
    ---------
    which : str
        'short', 'long' or 'carry'

    Returns
    -------

    """
    data = pd.read_pickle(data_path + "bond_cont_futures_2000_2018_d.p")

    data = pd.concat(data, axis=1)

    if which != "carry":
        res = data.xs(which, axis=1, level=1, drop_level=True)
    else:
        res = data.xs("long", axis=1, level=1, drop_level=True) / \
              data.xs("short", axis=1, level=1, drop_level=True)

    res = np.log(res).diff()
    
    res = res.where(np.abs(res).lt(
        pd.DataFrame(1, index=res.index, columns=res.columns).mul(
            2*res.quantile([0.01, 0.99]).diff().iloc[-1], axis=1))) * 100

    return res


if __name__ == "__main__":

    curs = ["aud", "cad", "eur", "gbp", "nzd", ]

    # rates = prepare_rates_data()
    events = prepare_events_data(perfect_foresight=False)
    # data = prepare_bond_data(maturity="1m")
    data = prepare_futures_data("short")

    window = (-10, -1, 1, 5)

    # normal data
    temp_es = EventStudy(data.loc[:, curs], events.loc[:, curs], window)
    # normal_data = temp_es.data.where(temp_es.mask_between_events).ewm(
    #     alpha=0.075).mean().where(~temp_es.mask_between_events).shift(1)\
    #     .replace(np.nan, 0.0)
    normal_data = 0

    # data_mean = temp_es.data.where(temp_es.mask_between_events).ewm(
    #     alpha=0.4).mean().shift(1)
    # data_dmd = data - data_mean

    # normal_data, variances = EventStudyFactory().get_normal_data_exog(
    #     data=data.loc[:, curs], events=events.loc[:, curs],
    #     exog=pd.concat({c: data.drop(c, axis=1).mean(axis=1) for c in curs},
    #                    axis=1),
    #     window=window, add_constant=True)

    ds = data - normal_data
    # ds = data.copy()

    events = events.loc[:, curs]
    ds = ds.loc[:, curs]

    # # ci
    # temp_es = EventStudy(ds, events, window)
    # ci = temp_es.get_ci(ps=0.95, method="boot", n_iter=125, what="car")

    # loop over direction of rate decision: "ups" or "downs"
    for d in ["cut", "hike"]:
        # d = "hike"
        # and over counter currencies
        for c in ["usd"]:
            # c = "usd"
            # trim returns
            # this_ret = into_currency(ds, c).drop(
            #     [p for p in drop_curs if p != c], axis=1, errors="ignore")

            # and events accordingly
            this_evt, this_ret = events.align(ds, axis=1, join="inner")

            ci_kwds = {"method": "simple", "ps": 0.95}
            # ci_kwds = None

            # event study ---------------------------------------------------
            f_i, f_a = fig_event_study(this_evt, this_ret,
                                       direction=d, mean_type="count_weighted",
                                       ci_kwds=ci_kwds, window=window)
            plt.show()

            # # save ----------------------------------------------------------
            # f_i.tight_layout()
            # f_a.tight_layout()
            #
            # f_i.savefig(out_path +\
            #     '_'.join(("xxx", c,  "before", d, "wght_indv.pdf")))
            # f_a.savefig(out_path +\
            #     '_'.join(("xxx", c,  "before", d, "wght_avg.pdf")))

            # # fomc ----------------------------------------------------------
            # if c == "usd":
            #     f_i_fomc, f_a_fomc = fig_event_study(
            #         fomc, ret_fomc, direction=d, mean_type="count_weighted",
            #         ci_width=0.95, window=window)

                # f_i_fomc.tight_layout()
                # f_a_fomc.tight_layout()
                # f_i_fomc.savefig(out_path +\
                #     '_'.join(
                #         ("xxx", c,  "before", "fomc", d, "wght_indv.pdf")))
                # f_a_fomc.savefig(out_path +\
                #     '_'.join(
                #         ("xxx", c,  "before", "fomc", d, "wght_avg.pdf")))

