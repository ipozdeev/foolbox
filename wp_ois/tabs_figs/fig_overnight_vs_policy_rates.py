import pandas as pd
from matplotlib.ticker import FixedLocator
from matplotlib import lines as mlines, patches as mpatches
import seaborn as sns
sns.set_style("white")

from foolbox.wp_tabs_figs.wp_settings import *
plt.rcParams["axes.edgecolor"] = new_gray
plt.rcParams["axes.linewidth"] = 2.0

from foolbox.api import *
from foolbox.linear_models import PureOls
from foolbox.wp_tabs_figs.wp_settings import *

from utils import resample_between_events

from wp_ois.wp_settings import central_banks_start_dates, end_date, cb_fx_map

def fig_implied_rates_unbiasedness(tgt_rate_diff, on_rate_diff, ols_eq=False):
    """
    """
    # rename, just in case
    y = on_rate_diff.rename("on")*100
    x = tgt_rate_diff.rename("tgt")*100

    # concatenate, drop na
    data_to_boxplot = pd.concat((x, y), axis=1).dropna()

    # unique target rate changes; need sorted because will plot on text axis
    unq_tgt_change = sorted(data_to_boxplot["tgt"].unique())

    # start figure
    fig, ax = plt.subplots(figsize=(8.27,8.27/1.66))

    # boxplots --------------------------------------------------------------
    bp = sns.boxplot(data=data_to_boxplot, x="tgt", y="on", linewidth=1,
        color=new_blue, saturation=.75, fliersize=3, width=0.4, ax=ax)

    # plot bisector-like points ---------------------------------------------
    # (45 degree line showing unbiasedness)
    for p in range(len(unq_tgt_change)):
        ax.scatter(p, unq_tgt_change[p], marker='D',
            color=new_red, edgecolor='none', s=65)

    # number of cases -------------------------------------------------------
    # in a gray box below
    # nee ylim to determine the 'below'
    ylim = ax.get_ylim()

    cnt = 0
    for p, q in data_to_boxplot.groupby("tgt"):
        ax.annotate(str(q.on.count()),
            xy=(cnt+0.2, ylim[0] + np.diff(ylim)[0]/25),
            fontsize=12,
            bbox=dict(facecolor="#d1d1d1", edgecolor='k'),
            horizontalalignment='center', verticalalignment='center')
        cnt += 1

    # artists ---------------------------------------------------------------
    # grid
    bp.grid(axis='y', linestyle=':', color='k', alpha=0.45)
    plt.setp(bp.artists, alpha=.90)

    # limits
    ax.yaxis.set_major_locator(FixedLocator(np.arange(
        np.ceil(ax.get_ylim()[0]/25)*25,
        np.ceil(ax.get_ylim()[-1]/25)*25,
        25)))
    ax.set_ylim((ylim[0]-5, ylim[-1]+5))

    # ticks
    bp.tick_params(axis='y', labelright=True)
    bp.tick_params(axis='both', labelsize=12)

    # labels
    ax.set_xlabel('target rate change, bps', fontsize=12)
    ax.set_ylabel('underlying rate change, bps', fontsize=12)
    bp.yaxis.label.set_size(12)

    # legend
    solid_line = mlines.Line2D([], [], color=new_red, linestyle="none",
        marker='D', markersize=8,
        label=r"($\alpha=0$, $\beta=1$) points")
    gray_patch = mpatches.Patch(color="#d1d1d1", label="number of cases")

    lg = ax.legend(handles=[solid_line, gray_patch], loc='upper right',
        bbox_to_anchor=(0.375, 0.96), fontsize=12, frameon=True)
    lg.get_frame().set_facecolor('w')
    lg.get_frame().set_alpha(1.0)

    if ols_eq:
        this_b = PureOls(y, x, add_constant=True).fit()

        tex_message = \
            r"$\widehat{{\Delta \bar{{r}}}} = {:3.2f} + {:3.2f} \Delta r_{{tgt}}$"

        ax.annotate(tex_message.format(*this_b.values, tgt="tgt"),
            xy=(0.2, 0.65),
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
    with open(data_path + "ois_project_events.p", mode="rb") as hangar:
        events_data = pickle.load(hangar)

    tgt_rate_changes = events_data["joint_cbs_plus_unscheduled_eff"].astype(float)
    tgt_rate_changes = tgt_rate_changes.drop("nok", axis=1)

    # Inverted cb_fx_map {"currency": "corresponding_cb"}
    fx_cb_map = dict((fx, cb) for cb,fx in cb_fx_map.items())

    # OLS regression --------------------------------------------------------
    def fun(x):
        try:
            res = (np.exp(np.nanmean(np.log(1 + x/100/360))) - 1) * 360 * 100
        except:
            res = np.nan

        return res

    coef = dict()
    se = dict()

    # Linear restrictions for the Wald test. Joint hypothesis alpha=0, beta=1
    R = pd.DataFrame([[1, 0], [0, 1]],columns=["const", "tgt"])
    r = pd.Series([0, 1], index=R.index)
    wald = dict()

    for c in tgt_rate_changes.columns:
        # c = "usd"
        # this_ois = OIS.from_iso(c, DateOffset(months=1))

        # Get the sample start for the corresponding cb
        start_date = central_banks_start_dates[fx_cb_map[c]]

        this_tgt_rate_change = tgt_rate_changes.loc[:, c].dropna()\
            .astype(float)
        this_on_rate = on_rates.loc[:, c].astype(float)

        # this_on_rate_avg = this_ois.get_rates_until(this_on_rate,
        #     meetings=this_tgt_rate_change,
        #     method="average")
        # this_on_rate_diff = (
        #     this_on_rate_avg.shift(-10).loc[this_tgt_rate_change.index] -
        #     this_on_rate_avg.shift(5).loc[this_tgt_rate_change.index])

        mask = pd.Series(True, this_tgt_rate_change.index).reindex(
            index=this_on_rate.index)
        mask.fillna(method="ffill", limit=2, inplace=True)
        mask.fillna(method="bfill", limit=2, inplace=True)

        this_cumul_rate = resample_between_events(
            data=np.log(this_on_rate/100/360+1),
            events=this_tgt_rate_change,
            fun=np.nanmean, mask=mask)

        this_cumul_rate = (np.exp(this_cumul_rate) - 1)*360*100

        # Take the sample
        this_on_rate_diff = \
            this_cumul_rate.diff().squeeze()[start_date:end_date]
        this_tgt_rate_change = this_tgt_rate_change[start_date:end_date]

        fig, ax = fig_implied_rates_unbiasedness(
            tgt_rate_diff=this_tgt_rate_change,
            on_rate_diff=this_on_rate_diff,
            ols_eq=True)

        fig.tight_layout()
        #fig.savefig(out_path + "unbias_" + c + ".pdf")

        # # regression
        y0 = this_on_rate_diff.rename("on")
        x0 = this_tgt_rate_change.rename("tgt")

        mod = PureOls(y0*100, x0*100, add_constant=True)
        diagnostics = mod.get_diagnostics(HAC=True)

        coef[c] = diagnostics.loc["coef"]
        se[c] = diagnostics.loc["se"]

        # Compute Wald test for the joint hypothesis: alpha=0, beta=1
        wald[c] = mod.linear_restrictions_test(R, r)


    coef = pd.DataFrame.from_dict(coef)
    coef.index = ["alpha", "beta"]
    se = pd.DataFrame.from_dict(se)
    se.index = ["alpha", "beta"]

    wald = pd.DataFrame.from_dict(wald)

    # Append coef and se dfs for making the table
    coef.loc["chi_sq", :] = wald.loc["chi_sq", :]
    se.loc["chi_sq", :] = wald.loc["p_val", :]


    out_path = set_credentials.set_path("../projects/ois/tex/figs/",
        which="local")

    # out_path = set_credentials.set_path("",
    #     which="local")

    to_better_latex(coef, se, fmt_coef="{:3.2f}", fmt_tstat="{:3.2f}",
        buf=out_path+"tab_overnight_vs_policy_rates.tex",
        column_format="l"+"W"*len(se.columns))


    # this_tgt_rate_change.where(this_tgt_rate_change < -0.5).dropna()
    # this_tgt_rate_change.loc["2008":"2009"]
    # this_on_rate.loc["2008-10-30":"2008-12-15"].mean()

    # events_data["fomc"].loc["2008",:]
