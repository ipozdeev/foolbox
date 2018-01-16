import pandas as pd
import numpy as np
import pickle
from scipy.stats import norm
from foolbox.api import *

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.ticker import MultipleLocator, FixedLocator, FormatStrFormatter

minorLocator = MultipleLocator(1)
majorLocator = MultipleLocator(5)

# matplotlib settings -------------------------------------------------------
plt.rc("font", family="serif", size=12)
gr_1 = "#8c8c8c"
# gr_2 = "#595959"

# %matplotlib

def fig_event_study(events, ret, direction,
    mean_type="count_weighted",
    ci_width=0.95,
    normal_data=0.0,
    window=(-10,-1,1,5)):
    """
    """
    # unpack window
    wa, wb, wc, wd = window

    # choose direction
    if direction == "cut":
        events = events.copy().where(events < 0).dropna(how="all")
    elif direction == "hike":
        events = events.copy().where(events > 0).dropna(how="all")
    else:
        raise ValueError("Not implemented.")

    es = EventStudy(data=ret,
        events=events,
        window=window,
        mean_type=mean_type,
        normal_data=normal_data,
        x_overlaps=True)

    cumsums = es.evt_avg_ts_sum.mean(axis="items")
    cumsums.loc[0, :] = np.nan
    cumsums = cumsums.sort_index()

    # ci_2 = es.get_ci(ps=0.9, method="simple")
    ci = es.get_ci(ps=ci_width, method="simple")
    # ci.loc[0, :] = np.nan
    # ci = ci.sort_index()

    the_mean = es.the_mean.copy()
    the_mean.loc[0] = np.nan
    the_mean = the_mean.sort_index()

    # plot ------------------------------------------------------------------
    # plot individual
    fig_individ, ax_individ = plt.subplots(figsize=(8.4,11.7/3))

    # individual currencies
    cumsums.plot(ax=ax_individ, color=gr_1)

    # black dots at the "inception"
    cumsums.loc[[wb],:].plot(ax=ax_individ, color="k",
        linestyle="none", marker=".", markerfacecolor="k")
    cumsums.loc[[wc],:].plot(ax=ax_individ, color="k",
        linestyle="none", marker=".", markerfacecolor="k")

    ax_individ.legend_.remove()

    # plot ------------------------------------------------------------------
    # plot average
    fig_avg, ax_avg = plt.subplots(figsize=(8.4,11.7/3))

    the_mean.plot(ax=ax_avg, color='k', linestyle="-", linewidth=1.5)

    # plot confidence interval
    ax_avg.fill_between(es.the_mean.index,
        es.ci.iloc[:,0].values,
        es.ci.iloc[:,1].values,
        color=gr_1, alpha=0.5, label="conf. interval")
    # ax_avg.fill_between(es.the_mean.index,
    #     ci_2.iloc[:,0].values,
    #     ci_2.iloc[:,1].values,
    #     color=gr_2, alpha=0.5, label="conf. interval")

    # final retouche --------------------------------------------------------
    furbish_plot(ax_individ, arrows=True,
        tot_curs={"a": cumsums.iloc[0, :], "d": cumsums.iloc[-1, :]})

    furbish_plot(ax_avg, set_xlabel=True, arrows=False)
    black_line = mlines.Line2D([], [],
        linewidth=1.5, color='k', label="cross-currency CAR")
    gray_patch = mpatches.Patch(color=gr_1, label='95% conf. int.')
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

if __name__ == "__main__":

    # parameters ------------------------------------------------------------
    from foolbox.wp_tabs_figs.wp_settings import *

    # data path -------------------------------------------------------------
    data_path = set_credentials.gdrive_path("research_data/fx_and_events/")
    out_path = set_credentials.gdrive_path(
        "research_data/fx_and_events/wp_figures_limbo/")

    # currency to drop
    drop_curs = settings["drop_currencies"]

    # dates to trim to
    s_dt = settings["sample_start"]
    e_dt = settings["sample_end"]

    # window
    wa, wb, wc, wd = -10, -1, 1, 5
    window = (wa, wb, wc, wd)

    # spot returns ----------------------------------------------------------
    with open(data_path + settings["fx_data"], mode='rb') as fname:
        fx_data = pickle.load(fname)

    ds = np.log(fx_data["spot_mid"]).diff()*100
    ds = ds.loc[s_dt:e_dt, :]

    # events ----------------------------------------------------------------
    with open(data_path + settings["events_data"], mode='rb') as fname:
        events_data = pickle.load(fname)

    events = events_data["joint_cbs"]
    events = events.loc[s_dt:e_dt, :]

    # fomc --------------------------------------------------------------
    with open(data_path + "fx_by_tz_sp_fixed.p", mode='rb') as fname:
        fx_data_us = pickle.load(fname)
    ret_fomc = np.log(fx_data_us["spot_mid"].loc[:, :, "NYC"]\
        .drop(drop_curs, axis=1, errors="ignore")).diff()*100

    fomc = pd.concat([events.loc[:, "usd"], ]*\
        events.drop(drop_curs + ["usd"], axis=1, errors="ignore").shape[1],
            axis=1)*-1
    fomc.columns = events.drop(
        drop_curs + ["usd"], axis=1, errors="ignore").columns

    # loop over direction of rate decision: "ups" or "downs"
    for d in ["hike", "cut"]:
        # d = "hike"
        # and over counter currencies
        for c in ["jpy", "gbp"]:
            # c = "usd"
            # trim returns
            this_ret = into_currency(ds, c).drop(
                [p for p in drop_curs if p != c], axis=1, errors="ignore")

            # and events accordingly
            this_evt = events.loc[:, this_ret.columns]

            # event study ---------------------------------------------------
            f_i, f_a = fig_event_study(this_evt, this_ret,
                direction=d,
                mean_type="count_weighted",
                ci_width=0.95,
                normal_data=0.0,
                window=window)

            # save ----------------------------------------------------------
            f_i.tight_layout()
            f_a.tight_layout()

            f_i.savefig(out_path +\
                '_'.join(("xxx", c,  "before", d, "wght_indv.pdf")))
            f_a.savefig(out_path +\
                '_'.join(("xxx", c,  "before", d, "wght_avg.pdf")))

            # fomc ----------------------------------------------------------
            if c == "usd":
                f_i_fomc, f_a_fomc = fig_event_study(fomc, ret_fomc,
                    direction=d,
                    mean_type="count_weighted",
                    ci_width=0.95,
                    window=window)
                f_i_fomc.tight_layout()
                f_a_fomc.tight_layout()
                f_i_fomc.savefig(out_path +\
                    '_'.join(
                        ("xxx", c,  "before", "fomc", d, "wght_indv.pdf")))
                f_a_fomc.savefig(out_path +\
                    '_'.join(
                        ("xxx", c,  "before", "fomc", d, "wght_avg.pdf")))

    with open(data_path + "implied_rates_bloomberg.p", mode="rb") as hngr:
        ir = pickle.load(hngr)

    with open(data_path + "ois_bloomberg.p", mode="rb") as hngr:
        ois_data = pickle.load(hngr)

    on = pd.DataFrame({k: v.loc[:, "ON"] for k, v in ois_data.items()})

    dr = ir - on
    dr = dr.dropna(axis=1, how="all")


    this_dr = dr.shift(12).where(np.abs(events) > 0.0001)
    this_dr = events.copy().where(np.abs(events) > 0.0001)

    cumret = ds.rolling(10).sum().shift(1).where(np.abs(events) > 0.0001)

    this_dr, cumret = this_dr.align(cumret, axis=1, join="inner")

    y = cumret.stack().rename("return").astype(float)
    x = this_dr.stack().rename("dr").astype(float)

    mod = PureOls(y0=y, X0=x, add_constant=True)
    mod.get_diagnostics()

    dr.where(events > 0.00001).aud.dropna(how="all")
    pd.concat((
        this_dr.where(events > 0.00001).chf.rename("dr"),
        cumret.where(events > 0.00001).chf.rename("return")), axis=1)\
        .plot.scatter(x="dr", y="return")

    dr = dr.shift(12).where(events)

    this_ds = ds.drop(
        [p for p in drop_curs] + ["usd"], axis=1, errors="ignore")

    dr = dr.loc[:, this_ds.columns]

    res = dict()
    for thresh in np.arange(0.1, 0.35, 0.05):
        # thresh = 0.1
        this_evt = dr.where(np.abs(dr) >= thresh)
        this_evt.dropna(how="all")
        es = EventStudy(data=this_ds,
            events=np.sign(this_evt.where(np.sign(this_evt) < 0)),
            window=window,
            mean_type="count_weighted",
            normal_data=0.0,
            x_overlaps=True)

        cumsums = es.evt_avg_ts_sum.mean(axis="items")
        res[thresh] = cumsums.loc[-10, :]

    res = pd.DataFrame(res).T

    res.plot()

    cumret = this_ds.reindex(
        index=pd.date_range(this_ds.index[0], this_ds.index[-1], freq='B'))
    cumret = cumret.rolling(10).sum().shift(1)

    r = cumret.where(events < 0)
    this_dr = dr.shift(12).where(events < 0)
    r = r.stack().astype(float)
    this_dr = this_dr.stack()

    r.head()
    this_dr.head()

    r, this_dr = r.align(this_dr, join="inner")

    r.index = np.arange(r.shape[0])
    this_dr.index = np.arange(this_dr.shape[0])

    r = r.rename("return")
    this_dr = this_dr.rename("dr")

    from foolbox.linear_models import PureOls

    mod = PureOls(r, this_dr, add_constant=True)
    mod.get_diagnostics()

    pd.concat((r, this_dr), axis=1).plot.scatter(x="dr", y="return")




# # event study for all banks in a loop ---------------------------------------
# # space for output
# avg_vars = pd.Series(index=all_evts.columns)*np.nan
# # means = pd.Series(index=all_evts.columns)*np.nan
# cumsums = pd.DataFrame(
#     columns=all_evts.columns,
#     index=np.concatenate((np.arange(wa,wb+1), np.arange(wc,wd+1))))*np.nan
# n_events = pd.Series(index=all_evts.columns)*np.nan
#
# for cur in all_evts.columns:
#     # cur = "jpy"
#     # these datasets
#
#     # # TODO: this is temp
#     # evt = events["fomc"].loc[s_dt:e_dt]
#
#     evt = all_evts.loc[:,cur].dropna()
#
#     # number of events, for weighting
#     n_events.loc[cur] = evt.where(
#         (evt < 0) if direction == "downs" else (evt > 0)).\
#         dropna().count()
#
#     ret = s_d[cur]*100
#
#     # run event study
#     evt_study = event_study_wrapper(ret, evt,
#         reix_w_bday=False,
#         direction=direction,
#         crisis="both",
#         window=[wa,wb,wc,wd], ps=0.9,
#         ci_method="simple",
#         plot=False,
#         impose_mu=0.0)
#
#     cumsums.loc[:,cur] = evt_study.get_cs_ts(evt_study.before, evt_study.after)
#     avg_vars.loc[cur] = evt_study.grp_var.sum()/evt_study.grp_var.count()**2
#     # means.loc[cur] = evt_study.tot_mu
#
# # weight according to the number of events
# # wght = pd.Series(1.0, index=cumsums.columns)/len(cumsums.columns)
# wght = n_events/n_events.sum()
#
# # confidence interval -------------------------------------------------------
# avg_avg_std = np.sqrt(avg_vars.sum()/avg_vars.count()**2)
# # or weighted
# avg_avg_std = np.sqrt(avg_vars.dot(wght**2))
# # mu = means.mean()
#
# q = np.sqrt(np.hstack(
#     (np.arange(-(wa)+(wb)+1,0,-1), np.arange(1,(wd)-(wc)+2))))
#
# ci_lo = norm.ppf(0.05)*q*avg_avg_std
# ci_hi = norm.ppf(0.95)*q*avg_avg_std
#
# ci = pd.DataFrame(
#     index=cumsums.dropna().index,
#     columns=[0.05,0.95])
#
# ci.loc[:,0.05] = ci_lo
# ci.loc[:,0.95] = ci_hi
#
# # reindex a bit
# cumsums = cumsums.reindex(index=np.arange(wa,wd+1))
#
# # plot!
# fig, ax = plt.subplots(figsize=(8.4,11.7/3))
#
# cumsums.plot(ax=ax, color=gr_1)
# # cumsums[drop_what].plot(ax=ax[0], color=gr_1, linestyle='--')
#
# cumsums.loc[[-1],:].plot(ax=ax, color="k",
#     linestyle="none", marker=".", markerfacecolor="k")
# cumsums.loc[[1],:].plot(ax=ax, color="k",
#     linestyle="none", marker=".", markerfacecolor="k")
# ax.legend_.remove()
# cumsums.dot(wght).plot(
#     ax=ax, color='k', linestyle="-", linewidth=1.5)
# # cumsums.drop(drop_what, axis=1).mean(axis=1).plot(
# #     ax=ax[1], color='k', linewidth=1.5)
#
# ax.fill_between(cumsums.dropna().index,
#     ci.iloc[:,0].values,
#     ci.iloc[:,1].values,
#     color=gr_1, alpha=0.5, label="conf. interval")
#
# minorLocator = MultipleLocator(1)
# majorLocator = MultipleLocator(5)
#
# # ax.xaxis.set_ticks(cumsums.dropna().index)
# ax.axhline(y=0, color='r', linestyle='--', linewidth=1.0, alpha=0.75)
# ax.xaxis.set_major_locator(majorLocator)
# ax.xaxis.set_minor_locator(minorLocator)
# ax.grid(which="both", alpha=0.33, linestyle=":")
#
# ax.set_xlabel("days after event", fontsize=12)
#
# fig.savefig(out_path+"xxxusd_before_"+direction+"_weighted_avg_new.pdf",
#     bbox_inches="tight")
# fig.savefig(out_path+"xxxusd_before_"+direction+"_weighted_indiv_new.pdf",
#     bbox_inches="tight")

# events["snb"].to_clipboard()
#
# cumsums
#
# dt_idx = np.logical_or(
#     np.logical_and(s_d.index > "2004-01-01", s_d.index < "2011-09-06"),
#     s_d.index > "2015-01-15")
# es = event_study_wrapper(
#     s_d.loc[:,"chf"].where(dt_idx).dropna(),
#     events["joint_cbs"].loc[:,"chf"].dropna(),
#     reix_w_bday=False,
#     direction="downs", window=(wa,wb,wc,wd), ps=0.9,
#     ci_method="simple",
#     plot=True)

# import pytz
# dt_tok = pd.to_datetime("2016-01-15 20:00:00").\
#     tz_localize(pytz.timezone("Asia/Tokyo"))
# dt_lon = pd.to_datetime("2016-01-15 17:00:00").\
#     tz_localize(pytz.timezone("Europe/London"))
# dt_nyc = pd.to_datetime("2017-03-15 17:00:00").\
#     tz_localize(pytz.timezone("America/New_York"))
#
# dt_tok.tz_convert(pytz.timezone("Europe/Oslo"))
# dt_lon.tz_convert(pytz.timezone("America/New_York"))
# dt_tok.tz_convert(pytz.timezone("America/New_York"))
#
# dt.tz_convert(pytz.timezone("Europe/Berlin"))


x = np.random.normal(size=(1000,)) / 10 + 0.01
y = np.cumprod(1 + x)

import matplotlib.pyplot as plt
plt.plot(y)
