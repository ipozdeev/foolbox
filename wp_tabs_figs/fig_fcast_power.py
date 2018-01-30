import pandas as pd
from foolbox.finance import PolicyExpectation
from foolbox.data_mgmt import set_credentials as set_cred
import time
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# matplotlib settings
majorLocator = MultipleLocator(2)
minorLocator = MultipleLocator(1)
formatter = FormatStrFormatter('%d')

# colors
gr_1 = "#8c8c8c"
my_greys = plt.get_cmap("Greys")
my_orng_blue = ListedColormap(sns.diverging_palette(220, 20, n=7))

# %matplotlib

data_path = set_cred.gdrive_path("research_data/fx_and_events/")
out_path = set_cred.gdrive_path("opec_meetings/tex/figs/")

def fig_error_plots(settings):
    """
    """
    curs = ['aud', 'cad', 'chf', 'eur', 'gbp', 'nzd', 'sek', 'usd']

    fig_rho, ax_rho = plt.subplots(nrows=3, ncols=3, figsize=(8.4, 8.4))
    fig_cmx, ax_cmx = plt.subplots(nrows=3, ncols=3, figsize=(8.4, 8.4),
        sharex=True, sharey=True)

    cnt = 0
    for cur in curs:
        # cur = "nzd"

        # select panel of array
        this_ax_rho = ax_rho.flatten()[cnt]
        this_ax_cmx = ax_cmx.flatten()[cnt]

        # policy expectation
        pe = PolicyExpectation.from_pickles(data_path,
            currency=cur, start_dt=settings["sample_start"],
            proxy_rate_pickle="overnight_rates.p",
            e_proxy_rate_pickle="implied_rates_from_1m_ois_roll.p",
            meetings_pickle="events.p",
            ffill=True)

        plot_one((this_ax_rho, this_ax_cmx), pe, settings)

        # furbish
        furbish_error_plot(this_ax_rho, cur)
        furbish_confusion_plot(this_ax_cmx, cur)

        cnt += 1

    # add fed funds futures -------------------------------------------------
    pe = PolicyExpectation.from_pickles(data_path,
        currency="usd", start_dt=pe.rate_expectation.first_valid_index(),
        proxy_rate_pickle="overnight_rates.p",
        e_proxy_rates_pickle="implied_rates_ffut.p",
        meetings_pickle="events.p",
        ffill=True)

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
    map_proxy_rate = \
        lambda x: x.rolling(settings["avg_impl_over"], min_periods=1).mean()\
            .shift(1)
    map_expected_rate = lambda x: x.rolling(settings["avg_refrce_over"],
                                            min_periods=1).mean().shift(1)

    policy_fcast = pe.forecast_policy_direction(
        lag=settings["base_holding_h"]+2,
        h_high=settings["base_threshold"],
        map_proxy_rate=map_proxy_rate,
        map_expected_rate=map_expected_rate)

    cmx = pe.assess_forecast_quality(policy_fcast)

    sns.heatmap(cmx, ax=ax[1], cbar=False,
        cmap=my_orng_blue,
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

    plt.show()
    plt.show()

    # fig_rho.savefig(
    #     out_path+"error_plot_"+\
    #     "thresh{:04.0f}_lag{:d}_ai{:d}_ar{:d}".\
    #         format(
    #             settings["base_threshold"]*1e04,
    #             settings["base_holding_h"],
    #             settings["avg_impl_over"],
    #             settings["avg_refrce_over"])+\
    #     ".pdf", bbox_inches="tight")
    # fig_cmx.savefig(
    #     out_path+"conf_mat_"+\
    #     "thresh{:04.0f}_lag{:d}_ai{:d}_ar{:d}".\
    #         format(
    #             settings["base_threshold"]*1e04,
    #             settings["base_holding_h"],
    #             settings["avg_impl_over"],
    #             settings["avg_refrce_over"])+\
    #     ".pdf", bbox_inches="tight")


    # with open(data_path + "implied_rates.p", mode='rb') as hangar:
    #     old = pickle.load(hangar)
    # with open(data_path + "implied_rates_bloomberg_1m.p", mode='rb') as hangar:
    #     new = pickle.load(hangar)
