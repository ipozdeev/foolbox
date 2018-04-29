from foolbox.events_intraday_sampler import EventDataAggregator, \
    barplot_intraday_by_hour
from foolbox.wp_tabs_figs.wp_settings import *
import numpy as np
import pandas as pd
from foolbox.api import set_credentials, settings
from pandas.tseries.offsets import BDay, Hour
from foolbox.api import my_blue
import seaborn as sns

if __name__ == "__main__":
    # Settings ================================================================
    s_dt = pd.to_datetime(settings["sample_start"])
    e_dt = pd.to_datetime(settings["sample_end"])
    data_path = set_credentials.gdrive_path("research_data/fx_and_events/")

    data_frequency = "H1"
    out_counter_usd_name = "fxcm_counter_usd_" + data_frequency + ".p"

    # Set the desired timezone: the original data in UTC (should be pytz str)
    tz = "Europe/London"  # None = do nothing, "EST", "CET", "Europe/London"

    # Map pytz sting names to strings labeling the x-axis of the graph
    tz_naming_dict ={
        None: "UTC",
        "Europe/London": "London",
        "EST": "EST"
        }
    tz_label = tz_naming_dict[tz]

    # Set up the event window
    pre = (BDay(-10), BDay(-1))
    # Set post subsample inside the pre-event window, making it irrelevant
    post = (pre[1], pre[1])

    # Event type pairs compute difference in means formerminus latter
    event_type_pairs = {
        "No Change vs. No Event": ("no change", "none"),
        "Hike vs. No Event": ("hike", "none"),
        "Cut vs. No Event": ("cut", "none"),
        "Hike vs. Cut": ("hike", "cut")
        }
    plot_order = ["No Change vs. No Event", "Hike vs. No Event",
                  "Cut vs. No Event", "Hike vs. Cut"]

    # Compute statistics for these currenices
    currs = ["aud", "cad", "chf", "eur", "gbp", "nok", "nzd", "sek"]
    events_by = "local_cbs"  # set to 'usd' to compute around fomc to 'eur' for
                             # the ecb, etc

    # Load data ===============================================================
    data = pd.read_pickle(data_path + out_counter_usd_name)

    # Convert to basis point leave a few additional datapoints at the start
    data = np.log((data["ask_close"]+data["bid_close"])/2).diff().loc[
           (s_dt - BDay(22)):, currs] * 1e4
    # data["usd"] = -1 * data.mean(axis=1)

    if tz is not None:
        data.index = [x.astimezone(tz) for x in data.index]
        data.index.name = "stamp"

    events_data = pd.read_pickle(data_path + settings["events_data"])
    if events_by == "local_cbs":
        # Use local announcements
        events = events_data["joint_cbs"][currs]
    else:
        # Use announcements by a particular CB
        events = pd.concat(
            [events_data["joint_cbs"][events_by] for curr in currs], axis=1)
    events.columns = currs
    events = events.loc[s_dt:e_dt]

    # Shift all events to 17:00 UTC of the event day
    events.index = [ix.tz_localize("UTC") + Hour(17) for ix in events.index]
    events = events.dropna(how="all")

    # Estimate differences across event types =================================
    # eda...
    eda = EventDataAggregator(events, data, pre, post)
    stacked = eda.stack_data()

    # Convert stamps to hours
    stacked["stamp"] = stacked["stamp"].apply(lambda x: x.hour)

    event_type_stats = dict()
    for key, (event_type1, event_type2) in event_type_pairs.items():
        event_type_stats[key] = eda.compare_means_by_stamp_and_event(
        stacked, event_type1, event_type2).reset_index()

    # Plot ====================================================================
    fig, ax = plt.subplots(4, figsize=(8.27, 10.0), sharex=True)

    for number, event_type in enumerate(plot_order):
        this_data = event_type_stats[event_type]
        this_ax = ax.flatten()[number]
        sns.barplot("stamp", "diff_means", data=this_data, ci=None, ax=this_ax,
                    color=my_blue, saturation=1, edgecolor="k", linewidth=0.33,
                    yerr=2 * this_data["se"],
                    ecolor="k", capsize=5, error_kw={"elinewidth": 1.75})

        this_ax.set_xlabel("")
        this_ax.set_ylabel("")
        this_ax.set_title(event_type)

    # Get the x-axis labels from integer stamps
    stamps = this_data["stamp"]
    labels = list()
    for stamp in stamps:
        if stamp <= 9:
            labels.append("0"+str(stamp)+":00")
        else:
            labels.append(str(stamp)+":00")

    # Report only every second label
    odd_labels = np.arange(len(labels), step=2) + 1
    for k in odd_labels:
        labels[k] = ""

    # Shift the integer ticks by -0.5 so that bars are not centered and
    # reflect mean return over the period
    this_ax.set_xticks(this_data["stamp"]-0.5)
    this_ax.set_xticklabels(labels)

    this_ax.set_xlabel("{} Time".format(tz_label))

    # Add name for the y-axes
    fig.text(0.0, 0.5, "Difference in hourly mean return in basis points",
             va='center',
             rotation='vertical')

    fig.tight_layout()
    fig.savefig(data_path+"/wp_figures_limbo/" +
                "intraday_patterns_mean_by_hour_events_by_{}.pdf".format(
                    events_by))
    plt.show()
