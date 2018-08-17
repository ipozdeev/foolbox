"""Run event studies for difference in bond index returns vs US for similar
maturities.
"""
from foolbox.EventStudy import EventStudy
from foolbox.api import *


if __name__ == "__main__":
    # Settings ----------------------------------------------------------------
    data_path = set_credentials.gdrive_path("research_data/fx_and_events/")

    # Event window
    wind = (-15, -1, 1, 5)
    wa, wb, wc, wd = wind

    drop_curs = ["jpy", "dkk", "nok"]

    # Start and end dates
    s_dt = pd.to_datetime(settings["sample_start"])
    e_dt = pd.to_datetime(settings["sample_end"])

    # Compute for fomc or for local cbs
    fomc = False

    # Compute difference in bond returns across these maturities
    mats = ["1-3y", "3-5y", "5-7y", "7-10y", "10+y"]
    # mats = ["1m", "3m", "6m", "12m", "1-3y", "3-5y", "5-7y", "7-10y", "10+y"]
    mats = ["1m", "3m", "6m", "12m",]
    # Load bonds data ---------------------------------------------------------
    bond_index_data = pd.read_pickle(data_path + "bond_index_data.p")
    bond_index_data.pop("jpy")

    """FISHY DATA:
    AUD, CAD, GBP, JPY, NZD, SEK, USD mat > 1y 2007-06 - 2007-09, flat RI
    NZD 12m: end of year spikes: 2003-2006, 5% jump: May 2008
    NZD 1-3y: Jun-Dec 2009, March 2014- Jan2015, flat RI
    NZD 3-5y: Dec 2003-Aug 2004, Jul - Dec 2006, May 2012-Jan 2013 flat RI
    NZD 5-7y: Nov 2001-Aug 2002, Jul-Dec 2004, Apr 2010- Jan 2011 flat RI
    NZD 10+y: Dec 2007-Jun 2009, Apr 2013-Jul 2014, flat RI
    
    SEK: 5-7y: Feb-Apr 2004,
    """

    # for curr, df in bond_index_data.items():
    #     df.mask(df == df.shift(1), np.nan, inplace=True)

    data = list()
    currs = list()
    for curr, df in bond_index_data.items():
        # Compute average difference in local bond returns vs us bond returns
        # Drop zero returns to exclude the fishy JPM data
        data.append(np.log(df[mats]).diff().replace(0, np.nan).mean(axis=1) -
                    np.log(bond_index_data["usd"][mats]).diff().
                    replace(0, np.nan).mean(axis=1))
        # data.append(np.log(df[mats]).diff().mean(axis=1) -
        #             np.log(bond_index_data["usd"][mats]).diff().mean(axis=1))
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

    esh.get_ci(0.95, "boot", n_iter=33)
    esl.get_ci(0.95, "boot", n_iter=33)
    esn.get_ci(0.95, "boot", n_iter=33)
    esh.plot(plot_ci=True)
    esl.plot(plot_ci=True)
    esn.plot(plot_ci=True)
    plt.show()
    print("kek")
