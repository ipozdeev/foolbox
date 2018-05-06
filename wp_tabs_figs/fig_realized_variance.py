import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from foolbox.finance import EventStudy, realized_variance
from foolbox.api import set_credentials as set_cred


path_to_data = set_cred.set_path("research_data/fx_and_events/")


def main():
    """

    Parameters
    ----------

    Returns
    -------

    """
    # data
    events_data = pd.read_pickle(path_to_data + "events.p")
    events = events_data["joint_cbs"]

    fx_data = pd.read_pickle(path_to_data + "fxcm_counter_usd_m15.p")
    spot = (fx_data["ask_close"] + fx_data["bid_close"]) / 2
    spot_ret = np.log(spot).diff()

    spot_ret, events = spot_ret.align(events, axis=1, join="inner")

    # subsample
    events = events.where(events < 0.0)
    events = events.dropna(how="all")

    spot_ret = spot_ret.loc["2001-11-30":"2017-01-31", :]
    events = events.loc["2001-11-30":"2017-01-31", :]

    # calculate rv honoring the timezones
    rv = dict()

    # time zones
    tz = {
        "aud": "Australia/Sydney",
        "nzd": "Australia/Sydney",
        "jpy": "Asia/Tokyo",
        "cad": "US/Eastern",
        "chf": "Europe/Zurich",
        "eur": "Europe/Zurich",
        "gbp": "Europe/London",
        "sek": "Europe/Stockholm",
        "nok": "Europe/Oslo",
    }

    for c, c_col in spot_ret.iteritems():
        # index to the local timezone
        c_col = c_col.copy()
        c_col.index = c_col.index.tz_convert(tz[c])

        # rv
        this_rv = realized_variance(c_col, freq='B', n_in_day=24 * 4,
                                    r_vola=True)

        # remove tz info to be able to easily merge
        this_rv.index = this_rv.index.tz_localize(None)
        rv[c] = this_rv

    # concat all
    rv = pd.concat(rv, axis=1) * 100

    # deseasonalize by taking the avg vola realized on the same weekday before
    es = EventStudy(rv, events, window=(-10, -1, 0, 5),
                    mean_type="count_weighted")

    # all periods of inter_evt
    inter_evt_mask = es.timeline.xs("inter_evt", axis=1, level=1)

    # subsample rv inbetween events only
    rv_inter_evt = rv.where(inter_evt_mask)

    # loop over weekdays, calculate average expanding return
    rv_ewm_by_wday = list()
    for d in rv_inter_evt.index.weekday.unique():
        this_idx = rv_inter_evt.index.weekday == d
        this_idx_masked = rv_inter_evt.copy().loc[this_idx]
        rv_ewm_by_wday.append(this_idx_masked.ewm(alpha=0.4).mean())

    rv_ewm_by_wday = pd.concat(rv_ewm_by_wday).sort_index(axis=0)

    # abnormal rv
    rv_abnormal = rv - rv_ewm_by_wday

    # event study based on it
    es = EventStudy(rv_abnormal, events, (-10, -1, 0, 5), "count_weighted")
    booted = es.boot_the_mean(what="ar", n_iter=50, fillna=True)

    # plot a bit
    fig, ax = plt.subplots(2, figsize=(8, 8))
    es.ar.mean(axis=1, level="assets").plot(ax=ax[0])
    ax[0].legend(prop={'size': 9})

    es.ar.mean(axis=1).plot(ax=ax[1], linewidth=2.0, color="k")
    booted.quantile([0.005, 0.995], axis=1).T.plot(ax=ax[1], color="gray",
                                                   linestyle='--')
    ax[1].legend_.remove()

    # fig.savefig(path_to_data + "output/rv_evt_study_" + "changes" + ".png")

    fig.show()
    plt.show()


if __name__ == '__main__':
    main()
