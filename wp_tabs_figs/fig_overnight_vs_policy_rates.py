import pandas as pd
from matplotlib.ticker import FixedLocator
from matplotlib import lines as mlines, patches as mpatches
import seaborn as sns
from foolbox.wp_tabs_figs.wp_settings import *
plt.rcParams["axes.edgecolor"] = new_gray
plt.rcParams["axes.linewidth"]  = 2.0

from foolbox.api import *
from foolbox.linear_models import PureOls
from foolbox.wp_tabs_figs.wp_settings import *

def fig_implied_rates_unbiasedness(tgt_rate_diff, on_rate_diff, ols_eq=False):
    """
    """
    # rename, just in case
    y = on_rate_diff.rename("on")
    x = tgt_rate_diff.rename("tgt")

    # concatenate, drop na
    data_to_boxplot = pd.concat((x, y), axis=1).dropna()

    # unique target rate changes; need sorted because will plot on text axis
    unq_tgt_change = sorted(data_to_boxplot["tgt"].unique())

    # start figure
    fig, ax = plt.subplots(figsize=(8.27,8.27))

    # boxplots --------------------------------------------------------------
    bp = sns.boxplot(data=data_to_boxplot, x="tgt", y="on", linewidth=1.5,
        color=new_blue, saturation=.9, fliersize=3, width=0.4, ax=ax)

    # plot bisector-like points ---------------------------------------------
    # (45 degree line showing unbiasedness)
    for p in range(len(unq_tgt_change)):
        ax.scatter(p, unq_tgt_change[p], marker='D',
            color=new_red, edgecolor='k', s=65)

    # number of cases -------------------------------------------------------
    # in a gray box below
    # nee ylim to determine the 'below'
    ylim = ax.get_ylim()

    cnt = 0
    for p, q in data_to_boxplot.groupby("tgt"):
        ax.annotate(str(q.on.count()), xy=(cnt, ylim[0] + np.diff(ylim)[0]/25),
            fontsize=12, bbox=dict(facecolor=new_gray, edgecolor='k'),
            horizontalalignment='center', verticalalignment='center')
        cnt += 1

    # artists ---------------------------------------------------------------
    # grid
    bp.grid(axis='y', linestyle=':', color='k', alpha=0.45)
    plt.setp(bp.artists, alpha=.90)

    # limits
    ax.yaxis.set_major_locator(FixedLocator(np.arange(
        np.ceil(ax.get_ylim()[0]/0.25)*0.25,
        np.ceil(ax.get_ylim()[-1]/0.25)*0.25,
        0.25)))
    ax.set_ylim((ylim[0]-0.05, ylim[-1]+0.05))

    # ticks
    bp.tick_params(axis='y', labelright=True)
    bp.tick_params(axis='both', labelsize=12)

    # labels
    ax.set_xlabel('target rate', fontsize=12)
    ax.set_ylabel('o/n rate', fontsize=12)
    bp.yaxis.label.set_size(12)

    # legend
    solid_line = mlines.Line2D([], [], color=new_red, linestyle="none",
        marker='D', markersize=8,
        label=r"($\alpha=0$, $\beta=1$) points")
    gray_patch = mpatches.Patch(color=new_gray, label="number of cases")

    lg = ax.legend(handles=[solid_line, gray_patch], loc='upper right',
        bbox_to_anchor=(0.375, 0.95), fontsize=12, frameon=True)
    lg.get_frame().set_facecolor('w')
    lg.get_frame().set_alpha(1.0)

    if ols_eq:
        this_b, _, _ = light_ols(y, x, add_constant=True, ts=True)

        tex_message = \
            r"$\widehat{{\Delta i}} = {:3.2f} + {:3.2f} \Delta r_{{tgt}}$"

        ax.annotate(tex_message.format(*this_b, tgt="tgt"),
            xy=(0.175, 0.7),
            horizontalalignment='center',
            verticalalignment='center',
            bbox=dict(facecolor='w', alpha=1.0, edgecolor='w'),
            xycoords='axes fraction',
            fontsize=12)

    ax.set_title(c.upper())

    return fig, ax

if __name__ == "__main__":

    out_path = set_credentials.set_path("ois_project/figs/")

    # data ------------------------------------------------------------------
    # overnight rates
    with open(data_path + "ois_bloomberg.p", mode="rb") as hangar:
        ois_data = pickle.load(hangar)

    on_rates = pd.concat(
        [p.loc[:, "ON"].to_frame(c) for c, p in ois_data.items()],
        axis=1).astype(float)

    # policy rates
    with open(data_path + "events.p", mode="rb") as hangar:
        events_data = pickle.load(hangar)

    tgt_rate_changes = events_data["joint_cbs"].astype(float)
    tgt_rate_changes = tgt_rate_changes.drop("nok", axis=1)

    # # lag settings
    # all_settings = {
    #     "aud": 1,
    #     "cad": 0,
    #     "chf": 0,
    #     "eur": 1,
    #     "gbp": 1,
    #     "nzd": 0,
    #     "sek": 1,
    #     "usd": 1
    #     }

    # OLS regression --------------------------------------------------------
    coef = dict()
    se = dict()

    for c in tgt_rate_changes.columns:
        # c = "usd"
        this_ois = OIS.from_iso(c, DateOffset(months=1))

        this_tgt_rate_change = tgt_rate_changes.loc[:, c].dropna()
        this_on_rate = on_rates.loc[:, c]
        this_on_rate_avg = this_ois.get_rates_until(this_on_rate,
            meetings=this_tgt_rate_change,
            method="average")
        this_on_rate_diff = (
            this_on_rate_avg.shift(-10).loc[this_tgt_rate_change.index] -
            this_on_rate_avg.shift(5).loc[this_tgt_rate_change.index])

        fig, ax = fig_implied_rates_unbiasedness(
            tgt_rate_diff=this_tgt_rate_change,
            on_rate_diff=this_on_rate_diff,
            ols_eq=True)

        fig.tight_layout()
        fig.savefig(out_path + "unbias_" + c + ".png",
            dpi=120)

        # regression
        y0 = this_on_rate_diff.rename("on")
        x0 = this_tgt_rate_change.rename("tgt")

        mod = PureOls(y0*100, x0*100, add_constant=True)
        diagnostics = mod.get_diagnostics(HAC=True)

        coef[c] = diagnostics.loc["coef"]
        se[c] = diagnostics.loc["se"]

    coef = pd.DataFrame.from_dict(coef)
    coef.index = ["alpha", "beta"]
    se = pd.DataFrame.from_dict(se)
    se.index = ["alpha", "beta"]

    out_path = set_credentials.set_path("../projects/ois/tex/tabs/",
        which="local")
    to_better_latex(coef, se, fmt_coef="{:3.2f}", fmt_tstat="{:3.2f}",
        buf=out_path+"tab_overnight_vs_policy_rates.tex",
        column_format="l"+"W"*len(se.columns))
