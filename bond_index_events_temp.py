"""Run event studies for difference in bond index returns vs US for similar
maturities.
"""
import pandas as pd
import seaborn as sns
from foolbox.EventStudy import EventStudy, EventStudyFactory
from foolbox.api import *
from foolbox.wp_tabs_figs import wp_settings

data_path = set_credentials.gdrive_path("research_data/fx_and_events/")
path_out = data_path + "output/"


def load_fut_commitments_data():
    """
    """
    fut_data = pd.read_csv(data_path + "cftc_commitments_1986_2016_w.txt",
                           delimiter=',')
    fut_data.index = fut_data.pop(fut_data.columns[2]).map(pd.to_datetime)
    fut_data = fut_data.sort_index()

    return fut_data


def load_fx_data(drop=None):
    """
    """
    fx_data = pd.read_pickle(data_path + "fx_by_tz_sp_fixed.p")
    spot_ret = np.log(fx_data["spot_mid"].loc[:, :, "NYC"]).diff() * 100

    if drop is not None:
        spot_ret = spot_ret.drop(drop, axis=1, errors="ignore")

    return spot_ret


def load_ois_data(drop=None):
    """
    """
    ois_data = pd.read_pickle(data_path + "ois_bloomberg.p")
    ois = pd.concat(ois_data, axis=1)
    ois.columns = pd.MultiIndex.from_tuples(
        [(p, q.lower()) for p, q in ois.columns])
    ois.columns.names = ["currency", "maturity"]

    if drop is not None:
        ois = ois.drop(drop, axis=1, level="currency", errors="ignore")

    return ois


def load_bonds_data(drop=None):
    """
    """
    # Load bonds data ---------------------------------------------------------
    bonds_data = pd.read_pickle(data_path + "bond_index_data.p")
    bonds = pd.concat(bonds_data, axis=1)
    bonds.columns.names = ["currency", "maturity"]

    bonds = np.log(bonds.resample('B').last()).diff() * 100

    if drop is not None:
        bonds = bonds.drop(drop, axis=1, level="currency",
                           errors="ignore")

    return bonds


def load_cad_f13_data():
    """
    """
    f13_nonres = ["MMNRGOCTB.DALL", "BMNRGOCT.DALL"]

    f13_data = pd.read_csv(data_path + "cad_f13_weekly-sd-1989-01-01.csv",
                           skiprows=28, index_col=0, parse_dates=True,
                           dtype=float, delimiter=',', header=0)
    f13 = f13_data.loc[:, f13_nonres]
    f13.columns = ["tbills", "tbonds"]

    res = np.log(f13).diff()

    return res


def load_events_data(drop=None):
    """
    Returns
    -------
    events_lvl
    events_chg
    """
    events_data = pd.read_pickle(data_path + "events.p")
    events_chg = events_data["joint_cbs"]
    events_lvl = events_data["joint_cbs_lvl"]

    if drop is not None:
        events_chg = events_chg.drop(drop, axis=1, errors="ignore")
        events_lvl = events_lvl.drop(drop, axis=1, errors="ignore")

    return events_lvl, events_chg


def with_events_marking(y, x, events, window):
    """

    Parameters
    ----------
    y : pandas.Series
    x : pandas.Series
    events : pandas.Series
    window : tuple

    Returns
    -------

    """
    y, x = y.align(x, join="inner")

    # marking
    event_marking = EventStudyFactory().mark_windows(y, events, window,
                                                     outside=False)
    event_marking = event_marking.rename('event_window')
    event_marking = event_marking.map(lambda x: "inside" if x else "outside")

    both = pd.concat((y, x, event_marking), axis=1)

    return both


def main0():
    """

    Returns
    -------

    """
    # Event window
    wind = (-10, -1, 0, 5)
    wa, wb, wc, wd = wind

    drop_curs = ["jpy", "dkk", "nok"]

    # Start and end dates
    s_dt = pd.to_datetime(settings["sample_start"])
    e_dt = pd.to_datetime(settings["sample_end"])

    # Compute for fomc or for local cbs
    fomc = False

    # Compute difference in bond returns across these maturities
    mats = ["1m", "3m", "6m", "12m"]
    # mats = ["1-3y", "3-5y", "5-7y", "7-10y", "10+y"]

    # Load bonds data ---------------------------------------------------------
    bond_index_data = pd.read_pickle(data_path + "bond_index_data.p")
    bond_index_data.pop("jpy")

    data = list()
    currs = list()
    for curr, df in bond_index_data.items():
        # Compute average difference in local bond returns vs us bond returns
        data.append(np.log(df[mats]).diff().mean(axis=1) -
                    np.log(bond_index_data["usd"][mats]).diff().mean(axis=1))
        currs.append(curr)

    data = pd.concat(data, axis=1) * 1e2  # to %
    data.columns = currs

    # Drop usd, its return differential is zero by construction
    data = data.drop(["usd"], axis=1)
    data = data.loc[(s_dt - BDay(22)):, :]
    data = data.reindex(
        index=pd.date_range(data.index[0], data.index[-1], freq='B'))

    # Load events data --------------------------------------------------------
    events_data = pd.read_pickle(data_path + settings["events_data"])
    if fomc:
        # Fomc event for every bond return differential
        events = pd.concat([events_data["joint_cbs"]["usd"]
                            for curr in data.columns], axis=1)
        events.columns = data.columns
        events = events.loc[s_dt:e_dt]

    else:
        # Local events
        events = events_data["joint_cbs"].drop(drop_curs + ["usd"], axis=1,
                                               errors="ignore")
        events = events.loc[s_dt:e_dt]

    # Run event studies -------------------------------------------------------
    esh = EventStudy(data=data,
                     events=events.where(events > 0).dropna(how="all"),
                     mean_type="count_weighted",
                     window=wind)
    esl = EventStudy(data=data,
                     events=events.where(events < 0).dropna(how="all"),
                     mean_type="count_weighted",
                     window=wind)
    esn = EventStudy(data=data,
                     events=events.where(events == 0).dropna(how="all"),
                     mean_type="count_weighted",
                     window=wind)

    esh.get_ci(0.95, "boot", n_iter=125)
    esl.get_ci(0.95, "boot", n_iter=125)
    esn.get_ci(0.95, "boot", n_iter=125)
    esh.plot(plot_ci=True)
    esl.plot(plot_ci=True)
    esn.plot(plot_ci=True)

    plt.show()

    print("kek")


def main1():
    """

    Returns
    -------

    """
    drop = ["dkk", "jpy", "nok"]
    events_lvl, events_chg = load_events_data(drop=drop)
    rs = load_fx_data(drop=drop)
    rs_m = rs.resample('M').mean()
    ois = load_ois_data(drop=drop)
    ois_m = ois.resample('M').mean()
    bonds = load_bonds_data(drop=drop)
    bonds_m = bonds.resample('M').mean()

    mat = "1m"
    for c, c_col in rs.iteritems():

        rs_func = lambda x: x.rolling(10, min_periods=1).mean()
        x_func = lambda x: (x - ois[c]["on"]).shift(10)

        df = with_events_marking(
            rs_func(rs[c]).rename("rs"),
            x_func(ois[c][mat]).rename("ois_slope_l"),
            events=events_chg[c].dropna(),
            window=(-10, -1, 1, 10))

        # ax = sns.lmplot(x="ois_slope_l", y="rs",
        #                 data=df,
        #                 fit_reg=True,
        #                 hue="event_window", legend=False,
        #                 markers='.', size=5, aspect=4/3,
        #                 scatter_kws={"alpha": 0.5})

        plt.legend(loc="upper right")
        ax.fig.axes[0].set_ylim((-1, 1))

        ax.fig.tight_layout()
        ax.fig.savefig(path_out + "figure_ois_slope_vs_rs_" + c + ".png")


def main2():
    """
    """
    # Event window
    wind = (-10, -1, 0, 5)
    wa, wb, wc, wd = wind

    drop_curs = ["jpy", "dkk", "nok"]

    # Start and end dates
    s_dt = pd.to_datetime(settings["sample_start"])
    e_dt = pd.to_datetime(settings["sample_end"])

    # Compute for fomc or for local cbs
    fomc = False

    # Compute difference in bond returns across these maturities
    mats = ["1m", "3m", "6m", "12m"]
    # mats = ["1-3y", "3-5y", "5-7y", "7-10y", "10+y"]

    # Load bonds data -------------------------------------------------------
    bonds = load_bonds_data(drop_curs)
    bonds = bonds.loc[:, (slice(None), mats)].mean(axis=1, level="currency")
    bonds = bonds.sub(bonds.pop("usd"), axis=0)

    # Load events data ------------------------------------------------------
    _, events = load_events_data(drop_curs)
    events = events.drop("usd", axis=1).dropna(how="all")

    # Run event studies -----------------------------------------------------
    br_norm, s2 = EventStudyFactory().get_normal_data_ewm(bonds, events, wind,
                                                          alpha=0.1)

    esh = EventStudy(data=bonds - br_norm,
                     events=events.where(events > 0).dropna(how="all"),
                     mean_type="count_weighted",
                     window=wind)
    esl = EventStudy(data=bonds - br_norm,
                     events=events.where(events < 0).dropna(how="all"),
                     mean_type="count_weighted",
                     window=wind)
    esn = EventStudy(data=bonds - br_norm,
                     events=events.where(events == 0).dropna(how="all"),
                     mean_type="count_weighted",
                     window=wind)

    ci = esh.get_ci(0.95, "boot", n_iter=201)
    # esl.get_ci(0.95, "boot", n_iter=125)
    esl.ci = ci
    # esn.get_ci(0.95, "boot", n_iter=125)
    esn.ci = ci
    esh.get_ci(0.95, "simple", variances=s2)
    esl.get_ci(0.95, "simple", variances=s2)
    esn.get_ci(0.95, "simple", variances=s2)

    esh.plot(plot_ci=True)
    esl.plot(plot_ci=True)
    esn.plot(plot_ci=True)


def main3():
    """
    """
    f13 = load_cad_f13_data()
    events_lvl, events_chg = load_events_data()
    evt = events_chg.loc[:, "cad"].dropna().rename("event")

    # roll to the next friday
    f13 = f13.resample("W-FRI").last()
    evt = evt.resample("W-FRI").last().fillna(-999.99)

    both = pd.concat((f13.rolling(2, min_periods=2).sum().shift(1).mean(1),
                      evt), axis=1)

    # lol = both.dropna().melt(value_vars=f13.columns, id_vars=evt.name,
    #                          value_name="dvol", var_name="asset")


    # sns.boxplot(x="event", y="dvol", hue="asset", data=lol)
    ax = sns.boxplot(x="event", y=0, data=both)


    evt.index.weekday


if __name__ == "__main__":

    # main2()
    main3()

