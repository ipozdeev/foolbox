import pandas as pd
from foolbox.api import *
import time
import matplotlib.pyplot as plt
import seaborn.apionly as sns
from matplotlib.colors import ListedColormap
my_cmap = ListedColormap(sns.diverging_palette(220, 20, n=7))

from matplotlib.ticker import MultipleLocator, FormatStrFormatter
majorLocator = MultipleLocator(2)
minorLocator = MultipleLocator(1)
formatter = FormatStrFormatter('%d')

gr_1 = "#8c8c8c"

# %matplotlib

data_path = set_credentials.gdrive_path("research_data/fx_and_events/")
out_path = set_credentials.gdrive_path("opec_meetings/tex/figs/")

def fig_error_plots(settings):
    """
    """
    curs = ['aud', 'cad', 'chf', 'eur', 'gbp', 'nzd', 'sek', 'usd']

    fig_rho, ax_rho = plt.subplots(nrows=3, ncols=3, figsize=(8.4,8.4))
    fig_cmx, ax_cmx = plt.subplots(nrows=3, ncols=3, figsize=(8.4,8.4),
        sharex=True, sharey=True)

    cnt = 0
    for cur in curs:
        # cur = "nzd"

        # select panel of array
        this_ax_rho = ax_rho.flatten()[cnt]
        this_ax_cmx = ax_cmx.flatten()[cnt]

        # policy expectation
        pe = PolicyExpectation.from_pickles(data_path, cur,
            settings["sample_start"])

        plot_one((this_ax_rho, this_ax_cmx), pe, settings)

        # furbish
        furbish_error_plot(this_ax_rho, cur)
        furbish_confusion_plot(this_ax_cmx, cur)

        cnt += 1

    # add fed funds futures -------------------------------------------------
    pe = PolicyExpectation.from_pickles(data_path, "usd",
        pe.rate_expectation.first_valid_index(), use_ffut=True)
    # select panel of array
    this_ax_rho = ax_rho.flatten()[cnt]
    this_ax_cmx = ax_cmx.flatten()[cnt]
    plot_one((this_ax_rho, this_ax_cmx), pe, settings)

    furbish_error_plot(this_ax_rho, cur+", fed funds")
    furbish_confusion_plot(this_ax_cmx, cur+", fed funds")

    # temp
    pe.rate_expectation.dropna()

    return fig_rho, fig_cmx


def plot_one(ax, pe, settings):
    """
    """
    # error plot --------------------------------------------------------
    pe.error_plot(avg_over=settings["avg_impl_over"], ax=ax[0])

    # confusion matrix --------------------------------------------------
    # mind the +1: it is needed to forecast one period before
    cmx = pe.assess_forecast_quality(
        lag=settings["base_holding_h"]+1,
        threshold=settings["base_threshold"],
        avg_impl_over=settings["avg_impl_over"],
        avg_refrce_over=settings["avg_refrce_over"])

    sns.heatmap(cmx, ax=ax[1], cbar=False,
        cmap=plt.get_cmap("Greys"),
        annot=True, linewidths=.75, fmt="d",
        vmax=cmx.loc[0,0]*0.75)

    time.sleep(1.0)

    return

def furbish_confusion_plot(ax, title):
    """
    """
    #
    ax.set_title(title)
    labels = ax.get_yticklabels()
    plt.setp(labels, rotation=0)
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=0)
    ax.tick_params(axis='both', direction='out')

    return

def furbish_error_plot(ax, title):
    """
    """
    ax.xaxis.set_major_locator(majorLocator)
    ax.xaxis.set_minor_locator(minorLocator)
    ax.yaxis.set_major_locator(majorLocator)
    ax.yaxis.set_minor_locator(minorLocator)

    ax.xaxis.set_major_formatter(formatter)

    ax.set_xlabel('', visible=False)
    ax.set_ylabel('', visible=False)
    ax.set_title(title)

    return

if __name__ == "__main__":

    # parameters ------------------------------------------------------------
    from foolbox.wp_tabs_figs.wp_settings import *

    fig_rho, fig_cmx = fig_error_plots(settings=settings)
    fig_rho.tight_layout()
    fig_cmx.tight_layout()

    fig_rho.savefig(
        out_path+"error_plot_"+\
        "thresh{:04.0f}_lag{:d}_ai{:d}_ar{:d}".\
            format(
                settings["base_threshold"]*1e04,
                settings["base_holding_h"],
                settings["avg_impl_over"],
                settings["avg_refrce_over"])+\
        ".pdf", bbox_inches="tight")
    fig_cmx.savefig(
        out_path+"conf_mat_"+\
        "thresh{:04.0f}_lag{:d}_ai{:d}_ar{:d}".\
            format(
                settings["base_threshold"]*1e04,
                settings["base_holding_h"],
                settings["avg_impl_over"],
                settings["avg_refrce_over"])+\
        ".pdf", bbox_inches="tight")




# s_dt = settings["sample_start"]
# e_dt = settings["sample_end"]
#
# # data --------------------------------------------------------------------
# data_path = set_credentials.gdrive_path("research_data/fx_and_events/")
# out_path = set_credentials.gdrive_path("opec_meetings/tex/figs/")
#
# # ois
# with open(data_path + "ois.p", mode='rb') as fname:
#     ois_data = pickle.load(fname)
#
# #
# with open(data_path + "ir.p", mode='rb') as fname:
#     deposit = pickle.load(fname)
#
# # events
# with open(data_path + "events.p", mode='rb') as fname:
#     events = pickle.load(fname)
# evts_lvl = events["joint_cbs_lvl"].loc[s_dt:e_dt,:]
# evts_chg = events["joint_cbs"].loc[s_dt:e_dt,:]
#
# # reference rates
# with open(data_path + "overnight_rates.p", mode='rb') as fname:
#     overnight_rates = pickle.load(fname)
#
# # create instance of PolicyExpectation ------------------------------------
# ois_icap = ois_data["icap_1m"]
# ois_tr = ois_data["tr_1m"]
#
# drop_curs = ["jpy", "dkk", "nok"]
#
# lag = 10
# threshold = 0.10
# avg_refrce_over = 2
# avg_impl_over = 2
#
# ois_icap, ois_tr = ois_icap.align(ois_tr, join="outer")
# ois_mrg = ois_icap.fillna(ois_tr)
# ois = ois_mrg.dropna(how="all")
#
# # ois = ois_data[ois_provider].drop(drop_curs, axis=1, errors="ignore")
# # ois.loc[:,"sek"] = ois_data["tr_1m"].loc[:,"sek"]
#
# instr = ois.copy()
# bench = overnight_rates.copy()
#
# # instr = deposit.copy()/100
# # bench = deposit.copy()/100
#
# instr = instr.drop(drop_curs, axis=1, errors="ignore")
# bench = bench.drop(drop_curs, axis=1, errors="ignore")
#
# fig_rho, ax_rho = plt.subplots(nrows=3, ncols=3, figsize=(8.4,8.4))
# fig_cmx, ax_cmx = plt.subplots(nrows=3, ncols=3, figsize=(8.4,8.4),
#     sharex=True, sharey=True)
#
# cnt = 0
# for c in instr.columns:
#     # c = "cad"
#     this_evt = evts_lvl.loc[:,c].dropna()
#     this_ax_rho = ax_rho.flatten()[cnt]
#     this_ax_cmx = ax_cmx.flatten()[cnt]
#
#     pe = PolicyExpectation.from_pickles(data_path, c)
#     # pe = PolicyExpectation.from_money_market(
#     #     meetings=this_evt,
#     #     instrument=instr.loc[:,c],
#     #     benchmark=bench.loc[:,c],
#     #     tau=1)
#
#     # pe.plot(lag)
#     # pe.policy_exp.dropna()
#
#     # policy expectation to plot
#     to_plot_exp = \
#         pe.rate_expectation.rolling(avg_impl_over, min_periods=1).mean()\
#             .shift(lag).reindex(index=this_evt.index, method="ffill")
#     to_plot_act = pe.reference_rate.rolling(lag, min_periods=1).mean()\
#         .shift(-lag).reindex(index=pe.meetings.index, method="bfill")
#
#     # rename a bit
#     to_plot_exp.name = "policy_exp"
#     to_plot_act.name = "policy_rate"
#
#     # predictive power
#     pd.concat((to_plot_exp, to_plot_act), axis=1).\
#         plot.scatter(
#             ax=this_ax_rho,
#             x="policy_exp",
#             y="policy_rate",
#             alpha=0.75,
#             s=33,
#             color=gr_1,
#             edgecolor='none')
#
#     abs_err = np.abs(pd.concat((to_plot_exp*100, to_plot_act*100), axis=1).\
#         diff(axis=1)).mean().loc["policy_rate"]
#     # err = pd.concat((to_plot_exp*100, to_plot_act*100), axis=1).\
#     #     diff(axis=1).mean().loc["policy_rate"]
#
#     lim_x = this_ax_rho.get_xlim()
#     this_ax_rho.plot(lim_x, lim_x, color='k', linestyle='--')
#     this_ax_rho.set_xlim(lim_x)
#     this_ax_rho.set_ylim(lim_x)
#
#     this_ax_rho.xaxis.set_major_locator(majorLocator)
#     this_ax_rho.xaxis.set_minor_locator(minorLocator)
#     this_ax_rho.yaxis.set_major_locator(majorLocator)
#     this_ax_rho.yaxis.set_minor_locator(minorLocator)
#
#     this_ax_rho.xaxis.set_major_formatter(formatter)
#
#     this_ax_rho.set_xlabel('', visible=False)
#     this_ax_rho.set_ylabel('', visible=False)
#     this_ax_rho.set_title(c)
#
#     # confusion matrix
#     cmx = pe.assess_forecast_quality(
#         lag=lag,
#         threshold=threshold,
#         avg_impl_over=avg_impl_over,
#         avg_refrce_over=avg_refrce_over)
#
#     # pe.forecast_policy_change(
#     #     lag=5,
#     #     threshold=0.1250,
#     #     avg_impl_over=1,
#     #     avg_refrce_over=5).to_clipboard()
#     # pe.meetings.to_clipboard()
#     # pe.policy_exp.loc["2001-08":].to_clipboard()
#     # pe.benchmark.loc["2001-08":].to_clipboard()
#
#     sns.heatmap(cmx, ax=this_ax_cmx, cbar=False,
#         cmap=plt.get_cmap("Greys"),
#         annot=True, linewidths=.75, fmt="d",
#         vmax=cmx.loc[0,0]*0.75)
#
#     time.sleep(1.0)
#
#     this_ax_cmx.set_title(c)
#     labels = this_ax_cmx.get_yticklabels()
#     plt.setp(labels, rotation=0)
#     labels = this_ax_cmx.get_xticklabels()
#     plt.setp(labels, rotation=0)
#     this_ax_cmx.tick_params(axis='both', direction='out')
#
#     this_ax_rho.annotate(r"$|err|={:3.2f}$".format(abs_err),
#         xy=(0.5, 0.05), xycoords='axes fraction')
#     # this_ax_cmx.set_ylabel('predicted', fontsize=12, visible=True)
#
#     cnt += 1
#
# # + fed funds futures
# with open(data_path + "fed_funds_futures_settle.p", mode='rb') as hangar:
#     ff_fut = pickle.load(hangar)
#
# pe = PolicyExpectation.from_pickles(data_path, "usd", s_dt="2001-08",
#     use_ffut=True)
#
# # predictive power
# f, ax = plt.subplots()
# to_plot_exp.name = "policy_exp"
# to_plot_act.name = "policy_rate"
# pd.concat((to_plot_exp, to_plot_act), axis=1).plot.scatter(
#     ax=ax,
#     x="policy_exp",
#     y="policy_rate",
#     alpha=0.66,
#     s=33,
#     color=gr_1,
#     edgecolor='none')
#
# pe.meetings = evts["usd"].dropna().loc["2001-08-22":]
# events["fomc"]["rate"].to_clipboard()
# ois["usd"].dropna()
# pe.meetings.to_clipboard()
# pe.policy_exp.dropna().to_clipboard()
# pe.forecast_policy_change(lag,threshold,avg_impl_over,avg_refrce_over).\
#     to_clipboard()
# this_ax_cmx.clear()
# cmx.loc[1,1] = 20
#
# fig_rho.delaxes(ax_rho.flatten()[cnt])
# fig_rho.tight_layout()
# fig_cmx.delaxes(ax_cmx.flatten()[cnt])
# fig_cmx.tight_layout()
#
# for ax in ax_cmx.flatten():
#     for _, spine in ax.spines.items():
#         spine.set_visible(True)
#
# fig_rho.savefig(
#     out_path+"error_plot_"+\
#     "thresh{:04.0f}_lag{:d}_ai{:d}_ar{:d}".\
#         format(threshold*1e04, lag, avg_impl_over, avg_refrce_over)+\
#     ".pdf", bbox_inches="tight")
# fig_cmx.savefig(
#     out_path+"conf_mat_"+\
#     "thresh{:04.0f}_lag{:d}_ai{:d}_ar{:d}".\
#         format(threshold*1e04, lag, avg_impl_over, avg_refrce_over)+\
#     ".pdf", bbox_inches="tight")
#
# # --------------------- temp
