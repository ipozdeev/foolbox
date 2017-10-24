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

# %matplotlib

def fig_event_study(events, ret, direction,
    wght="equal",
    ci_width=0.95,
    window=(-10,-1,1,5)):
    """
    """
    # unpack window
    wa,wb,wc,wd = window

    # space for -------------------------------------------------------------
    # variances
    avg_vars = pd.Series(index=events.columns)*np.nan
    # cumsums
    cumsums = pd.DataFrame(
        columns=events.columns,
        index=np.concatenate((np.arange(wa,wb+1), np.arange(wc,wd+1))))*np.nan
    # number of events
    n_events = pd.Series(index=events.columns)*np.nan

    # loop over currencies --------------------------------------------------
    for cur in events.columns:
        # cur = "jpy"
        # fetch this event
        this_evt = events.loc[:,cur].dropna()

        # signal
        this_sig = (this_evt < 0) if direction == "cuts" else (this_evt > 0)

        # fetch return
        this_ret = ret[cur]*100

        # run event study
        this_es = event_study_wrapper(this_ret, this_evt,
            reix_w_bday=False,
            direction=direction,
            crisis="both",
            window=window,
            ci_method="simple",
            plot=False,
            impose_mu=0.0)

        # store
        cumsums.loc[:,cur] = this_es.get_cs_ts(this_es.before, this_es.after)
        avg_vars.loc[cur] = this_es.grp_var.sum()/this_es.grp_var.count()**2
        n_events.loc[cur] = this_evt.where(this_sig).dropna().count()

    # reindex to delete the missing observarions
    cumsums = cumsums.reindex(index=np.arange(wa,wd+1))
    cumsum_a = cumsums.iloc[0,:]
    cumsum_d = cumsums.iloc[-1,:]

    # average across event-currency pairs -----------------------------------
    if wght == "equal":
        w = pd.Series(1.0, index=cumsums.columns)/len(cumsums.columns)
    elif wght == "by_event":
        w = n_events/n_events.sum()

    # confidence interval ---------------------------------------------------
    cl = (1-ci_width)/2
    ch = ci_width + cl

    # average std of CARs
    # avg_avg_std = np.sqrt(avg_vars.sum()/avg_vars.count()**2)
    avg_avg_std = np.sqrt(avg_vars.dot(w**2))

    # sqrt of multiples
    q = np.sqrt(np.hstack(
        (np.arange(-(wa)+(wb)+1,0,-1), np.arange(1,(wd)-(wc)+2))))

    ci_lo = norm.ppf(cl)*q*avg_avg_std
    ci_hi = norm.ppf(ch)*q*avg_avg_std

    ci = pd.DataFrame(index=cumsums.dropna().index, columns=[cl,ch])

    ci.loc[:,cl] = ci_lo
    ci.loc[:,ch] = ci_hi

    # plot ------------------------------------------------------------------
    # plot individual
    fig_individ, ax_individ = plt.subplots(figsize=(8.4,11.7/3))

    # individual currencies
    cumsums.plot(ax=ax_individ, color=gr_1)
    # cumsums[drop_what].plot(ax=ax[0], color=gr_1, linestyle='--')

    # black dots at the "inception"
    cumsums.loc[[wb],:].plot(ax=ax_individ, color="k",
        linestyle="none", marker=".", markerfacecolor="k")
    cumsums.loc[[wc],:].plot(ax=ax_individ, color="k",
        linestyle="none", marker=".", markerfacecolor="k")

    ax_individ.legend_.remove()

    # plot ------------------------------------------------------------------
    # plot individual
    fig_avg, ax_avg = plt.subplots(figsize=(8.4,11.7/3))

    cumsums.dot(w).plot(ax=ax_avg, color='k', linestyle="-", linewidth=1.5)
    # cumsums.drop(drop_what, axis=1).mean(axis=1).plot(
    #     ax=ax[1], color='k', linewidth=1.5)

    # plot confidence interval
    ax_avg.fill_between(cumsums.dropna().index,
        ci.iloc[:,0].values,
        ci.iloc[:,1].values,
        color=gr_1, alpha=0.5, label="conf. interval")

    # final retouche --------------------------------------------------------
    furbish_plot(ax_individ, arrows=True,
        tot_curs={"a": cumsum_a, "d": cumsum_d})

    furbish_plot(ax_avg, set_xlabel=True, arrows=False)
    black_line = mlines.Line2D([], [],
        linewidth=1.5, color='k', label="cross-currency CAR")
    gray_patch = mpatches.Patch(color=gr_1, label='95% conf. int.')
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
    out_path = set_credentials.gdrive_path("opec_meetings/tex/figs/")

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

    ds = np.log(fx_data["spot_mid"]).diff()
    ds = ds.loc[s_dt:e_dt, :]

    # events ----------------------------------------------------------------
    with open(data_path + settings["events_data"], mode='rb') as fname:
        events_data = pickle.load(fname)

    events = events_data["joint_cbs"]
    events = events.loc[s_dt:e_dt, :]

    # loop over direction of rate decision: "ups" or "downs"
    for d in ["ups", "downs"]:
        # d = "downs"
        # and over counter currencies
        for c in ["usd", "jpy", "gbp"]:
            # c = "jpy"
            ret = into_currency(ds, c).drop(
                [p for p in drop_curs if p != c], axis=1, errors="ignore"))
            ret = ds



        events_perf = events_perf.loc[s_dt:e_dt]

        # # events: forecast ------------------------------------------------------
        # events_fcast = events_perf.drop(["nok",], axis=1)*np.nan
        # for cur in events_fcast.columns:
        #     # cur = "sek"
        #     pe = PolicyExpectation.from_pickles(data_path, cur)
        #     events_fcast.loc[:,cur] = pe.forecast_policy_change(
        #         lag=lag,
        #         threshold=settings["base_threshold"],
        #         avg_impl_over=settings["avg_impl_over"],
        #         avg_refrce_over=settings["avg_refrce_over"])

        # fomc ------------------------------------------------------------------
        with open(data_path + "fx_by_tz_sp_fixed.p", mode='rb') as fname:
            fx_all = pickle.load(fname)
        ret_fomc = np.log(fx_all["spot_mid"].loc[:,:,"NYC"]\
            .drop(drop_curs,axis=1,errors="ignore")).diff()

        fomc = events["joint_cbs"].loc[:,"usd"]
        fomc = pd.concat([fomc,]*events_perf.shape[1], axis=1)
        fomc.columns = events_perf.columns
        fomc = fomc.loc[s_dt:e_dt]

        # event study -----------------------------------------------------------
        f_i, f_a = fig_event_study(events_perf, ret,
            direction=direction,
            wght="by_event",
            ci_width=0.95,
            window=window)

        # fomc ------------------------------------------------------------------
        f_i, f_a = fig_event_study(fomc, ret_fomc,
            direction=direction,
            wght="by_event",
            ci_width=0.95,
            window=window)

        # save ------------------------------------------------------------------
        f_i.tight_layout()
        f_a.tight_layout()

        f_i.savefig(
            out_path+"xxxusd_before_"+direction+"_weighted_indiv.pdf",
            bbox_inches="tight")
        f_a.savefig(
            out_path+"xxxusd_before_"+direction+"_weighted_avg.pdf",
            bbox_inches="tight")



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
