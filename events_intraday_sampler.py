import pandas as pd
from foolbox.econometrics import simple_se_diff_in_means
from foolbox.wp_tabs_figs.wp_settings import *
from foolbox.api import my_blue
import seaborn as sns


class EventSampler:
    """Class providing functionality to slice and dice returns (or other
    variables) around the central bank announcements.

    """

    def __init__(self, events, returns, pre_evt_offset, post_evt_offset):
        """Instantiate the class.

        Parameters
        ----------
        events: pd.Series or pd.DataFrame
            of events
        returns: pd.Series or pd.DataFrame
            of asset returns
        pre_evt_offset: tuple
            of pd.DateOffset objects sampling the pre event window, e.g.
            (BDay(-5), BDay(-1))
        post_evt_offset: tuple
            of pd.DateOffset objects sampling the pre event window, e.g.
            (Hour(1), BDay(1))

        """
        # If the inputs are dataframe cast to series, enforce single dimension
        if isinstance(events, pd.DataFrame):
            if events.shape[1] > 1:
                raise ValueError("Class accepts one-dimensional inputs only.")
            else:
                events = events.squeeze()

        if isinstance(returns, pd.DataFrame):
            if returns.shape[1] > 1:
                raise ValueError("Class accepts one-dimensional inputs only.")
            else:
                returns = returns.squeeze()

        # As a safety check, assert that there is no NaN's in events
        if events.isnull().any():
            Warning("NaN values in the events, dropping the bastards.")
            events = events.dropna()

        # Assign the parameters
        self.events = events
        self.returns = returns

        # Unpack the tuples
        self.pre_evt_from = pre_evt_offset[0]
        self.pre_evt_to = pre_evt_offset[1]
        self.post_evt_from = post_evt_offset[0]
        self.post_evt_to = post_evt_offset[1]

        # Assign placeholders for properties
        self._event_sample = None
        self._ex_event_sample = None
        self._hikes_sample = None
        self._cuts_sample = None
        self._no_changes_sample = None

    @property
    def event_sample(self):
        """Gets sample spanned by event periods. See
        'event_and_ex_event_subsamples()' for documentation

        Returns
        -------
        self._event_sample: pd.Series
            of returns within the pre- and post-event windows

        """
        if self._event_sample is None:
            self._event_sample, self._ex_event_sample = \
                self._event_and_ex_event_subsamples()

        return self._event_sample

    @property
    def ex_event_sample(self):
        """Gets sample spanned by outside the event windows. See
        'event_and_ex_event_subsamples()' for documentation

        Returns
        -------
        self._event_sample: pd.Series
            of returns within the pre- and post-event windows

        """
        if self._ex_event_sample is None:
            self._event_sample, self._ex_event_sample = \
                self._event_and_ex_event_subsamples()

        return self._ex_event_sample

    @property
    def hikes_sample(self):
        """Gets samples where values in 'self.events' are positive. See
         '_event_type_subsamples(self, event_type)' for further documentation.

        Returns
        -------
        self._hikes_sample: pd.Series
            of returns within the pre- and post-event windows corresponding to
            hikes

        """
        if self._hikes_sample is None:
            self._hikes_sample = self._event_type_subsamples(event_type="hike")

        return self._hikes_sample

    @property
    def cuts_sample(self):
        """Gets samples where values in 'self.events' are negative. See
         '_event_type_subsamples(self, event_type)' for further documentation.

        Returns
        -------
        self._cuts_sample: pd.Series
            of returns within the pre- and post-event windows corresponding to
            cuts

        """
        if self._cuts_sample is None:
            self._cuts_sample = self._event_type_subsamples(event_type="cut")

        return self._cuts_sample

    @property
    def no_changes_sample(self):
        """Gets samples where values in 'self.events' are zero. See
         '_event_type_subsamples(self, event_type)' for further documentation.

        Returns
        -------
        self._no_changes_sample: pd.Series
            of returns within the pre- and post-event windows corresponding to
            no changes

        """
        if self._no_changes_sample is None:
            self._no_changes_sample = self._event_type_subsamples(
                event_type="no change")

        return self._no_changes_sample

    def _event_and_ex_event_subsamples(self):
        """Given the events and pre- and post- offsets breaks the returns into
        two subsamples, one covering the returns within the event window and
        another outside of it. Note, that returns within the 'self.pre_evt_to'-
        'self.post_evt_from' period are dropped.

        Returns
        -------
        event_sample: pd.Series
            containing the returns spanning the event sample that is the
            returns from 'self.pre_evt_from' to 'self.pre_evt_to' and
            'self.post_evt_from' to 'self.post_evt_to'
        ex_event_sample: pd.Series
            containing returns outside the event window

        """
        # Initialize series of booleans, tracking the output samples:
        # Event windows except for the (pre_evt_to, post_evt_from) interval
        event_index = pd.Series(False, index=self.returns.index)

        # Index spanning the [pre_from, post_to] interval. Ex-events are
        # complementary to this index
        event_window_index = pd.Series(False, index=self.returns.index)

        # Loop over the events, filling the indexes
        for stamp, evt in self.events.iteritems():
            event_index.loc[
            stamp + self.pre_evt_from:stamp + self.pre_evt_to] = True
            event_index.loc[
            stamp + self.post_evt_from:stamp + self.post_evt_to] = True
            event_window_index.loc[
            stamp + self.pre_evt_from:stamp + self.post_evt_to] = True

        # Get the samples
        event_sample = self.returns.loc[event_index]
        ex_event_sample = self.returns.loc[~event_window_index]

        return event_sample, ex_event_sample

    def _event_type_subsamples(self, event_type):
        """Assuming that positive values in 'self.events' represent hikes,
        negative - cuts, and zeroes - no changes, get the corresponding
        subsamples.

        Parameters
        ----------
        event_type: str
            'hike', 'cut', or 'no change', controlling sampling 'self.events'
            on its positive, negative or zero values respectively

        Returns
        -------
        event_type_sample: pd.Series
            containing the returns spanning the event sample that is the
            returns from 'self.pre_evt_from' to 'self.pre_evt_to' and
            'self.post_evt_from' to 'self.post_evt_to', where the corresponding
            events are of the 'event_type'

        """
        # Map the strings into boolean indexes
        if event_type == "hike":
            conditional_events = self.events.loc[self.events > 0]
        elif event_type == "cut":
            conditional_events = self.events.loc[self.events < 0]
        elif event_type == "no change":
            conditional_events = self.events.loc[self.events == 0]
        else:
            raise ValueError("Event type {} is not recognized.".format(
                event_type))

        # Loop over the conditional events
        if not conditional_events.empty:
            # Track returns within hikes events
            conditional_index = pd.Series(False, index=self.returns.index)

            for stamp, hike in conditional_events.iteritems():
                conditional_index.loc[
                stamp + self.pre_evt_from:stamp + self.pre_evt_to] = True
                conditional_index.loc[
                stamp + self.post_evt_from:stamp + self.post_evt_to] = True

            event_type_sample = self.returns.loc[conditional_index]

        else:
            # Return None for empty samples
            Warning("Events of the {} type are not in the data".format(
                event_type))
            event_type_sample = None

        return event_type_sample


class EventDataAggregator:
    """Wrapper around the 'EventSampler' class handling aggregation and
    computation of inference across several assets and the corresponding
    events.

    Class Attributes
    ----------------
    event_type_sample_name_map: dict
        mapping event type to the 'EventSampler''s property sampling the data
        for this type. For example {"hike": "hikes_sample",
                                    "none": "ex_event_sample"}
    timestamp_classifiers: dict
        with items being functions that can be applied to pd.Timestamps, and
        keys being the names of these functions. For example:
            {"hour": lambda x: x.hour}

    """
    event_type_sample_name_map = {
        "hike": "hikes_sample", "cut": "cuts_sample",
        "no change": "no_changes_sample", "none": "ex_event_sample"}
    timestamp_classifiers = {"hour": lambda x: x.hour}

    def __init__(self, events, returns, pre_evt_offset, post_evt_offset):
        """Instantiate the class.

        Parameters
        ----------
        events: pd.DataFrame
            of the corresponding events. The columns in 'returns' and 'events'
            should to be named identically
        returns: pd.DataFrame
            of asset returns
        pre_evt_offset: tuple
            of pd.DateOffset objects sampling the pre event window, e.g.
            (BDay(-5), BDay(-1)). Applies to all columns in 'returns'
        post_evt_offset: tuple
            of pd.DateOffset objects sampling the pre event window, e.g.
            (Hour(1), BDay(1)). Applies to all assets in 'returns'

        """
        # Assign attributes
        self.events = events
        self.returns = returns

        # Unpack the tuples
        self.pre_evt_from = pre_evt_offset[0]
        self.pre_evt_to = pre_evt_offset[1]
        self.post_evt_from = post_evt_offset[0]
        self.post_evt_to = post_evt_offset[1]

        self.assets = self.events.columns
        for asset in self.assets:
            # Instantiate 'EventSampler' for each asset
            this_sampler = EventSampler(
                self.events[asset], self.returns[asset], pre_evt_offset,
                post_evt_offset)

            # Set it as an attirbute
            setattr(self, str(asset), this_sampler)

    def stack_data(self, event_types=["hike", "cut", "no change", "none"]):
        """Stacks the return data of each asset around the desired event types.

        Parameters
        ----------
        event_types: iterable
            of strings specifying types of events which should be included into
            the output

        Returns
        -------
        stacked: pd.DataFrame
            with columns 'stamp', 'data', 'event', 'asset', containing stacked
            timestamps, returns, event types, and asset name of each entry
            in 'self.assets' for the r
            equested 'event_types'

        """
        # Soup of two loops: assets -> event_types
        stacked = list()
        for asset in self.assets:

            # Get the sampler for this asset
            this_sampler = getattr(self, asset)

            # Loop over event types
            this_sampler_data = list()
            for event_type in event_types:
                # Map the type to the corresponding porperty of this_sampler's
                # instance and fetch the data
                this_slice = getattr(
                    this_sampler,
                    self.event_type_sample_name_map[event_type]
                    ).reset_index()

                # Assign the event type and append this assets' data
                this_slice["event"] = event_type
                this_sampler_data.append(this_slice)

            # Aggregate for this asset, store its name, and append the output
            this_sampler_data = pd.concat(this_sampler_data, ignore_index=True)
            this_sampler_data["asset"] = asset
            this_sampler_data = this_sampler_data.rename({asset: "data"},
                                                         axis=1)
            stacked.append(this_sampler_data)

        # Aggregate over the assets
        stacked = pd.concat(stacked, ignore_index=True)

        return stacked

    @staticmethod
    def compare_means_by_stamp_and_event(data, event_type1, event_type2,
                                         stamp_classifier=None):
        """Map time stamps into groups (say, to hours) and compare the mean
        values in the dataset across 'event_type1', and 'event_type2' classes.
        Make a comparison for each stamp group. Also compute and report the
        parametric standard error for difference in said means.

        Parameters
        ----------
        data: pd.DataFrame
            with columns 'stamp', 'data', 'event', containing stacked
            timestamps, values to aggregate, event types
        event_type1: str
            compare the average of the data for this event type ...
        event_type2: str
            ... vs. this one
        stamp_classifier: function
            to apply to the 'stamp' column of the 'data'. Default is None

        Returns
        -------
        diff_means: pd.DataFrame
            indexed by 'stamp' groups, with columns  'diff_means' and "se"
            containing estimates of difference in means and standard error of
            this difference for stamp group

        """
        data = data.copy()
        # Classify stamps
        if stamp_classifier is not None:
            data["stamp"] = data["stamp"].apply(stamp_classifier)

        # Compute statistics
        diff_means = pd.DataFrame(index=data.groupby("stamp").groups.keys(),
                                  columns=["diff_means", "se"])
        diff_means.index.name = "stamp"

        for stamp, df in data.groupby("stamp"):
            df = df.set_index("event")

            # Difference in means
            diff_means.loc[stamp, "diff_means"] = \
                df.loc[event_type1, "data"].mean() -\
                df.loc[event_type2, "data"].mean()

            # Its standard error
            diff_means.loc[stamp, "se"] = simple_se_diff_in_means(
                df.loc[event_type1, "data"], df.loc[event_type2, "data"])

        return diff_means


def barplot_intraday_by_hour(data):
    """Wrapper around 'sns.barplot', furbishing the said contraption.

    Parameters
    ----------
    data: pd.DataFrame
        output of the 'EventDataAggregator.compare_means_by_stamp_and_event'
        with reset index

    Returns
    -------
    ax: matplotlib.pyplot.ax
        the plot

    """
    # Get the x-axis labels from integer stamps
    stamps = data["stamp"]
    labels = list()
    for stamp in stamps:
        if stamp <= 9:
            labels.append("0"+str(stamp)+":00")
        else:
            labels.append(str(stamp)+":00")

    ax = sns.barplot("stamp", "diff_means", data=data, ci=None, color=my_blue,
                     saturation=1, yerr=2 * diff["se"], ecolor="k",
                     capsize=5, error_kw={"elinewidth": 1.75})

    # Shift the integer ticks by -0.5 so that bars are not centered and
    # reflect mean return over the period
    ax.set_xticks(data["stamp"]-0.5)
    ax.set_xticklabels(labels)

    ax.set_xlabel("UTC Time")
    ax.set_ylabel("Difference in hourly mean return in bps")

    return ax


if __name__ == "__main__":
    from foolbox.api import set_credentials, settings
    from pandas.tseries.offsets import BDay, Hour

    # Settings ================================================================
    s_dt = pd.to_datetime(settings["sample_start"])
    e_dt = pd.to_datetime(settings["sample_end"])
    data_path = set_credentials.gdrive_path("research_data/fx_and_events/")

    data_frequency = "H1"
    out_counter_usd_name = "fxcm_counter_usd_" + data_frequency + ".p"

    # Set up the event window
    pre = (BDay(-10), BDay(-1))
    # Set post subsample inside the pre-event window, making it irrelevant
    post = (BDay(-1), BDay(-1))

    # Difference between which subsamples to plot. There are 4 subsamples
    # 'hike', 'cut', 'no change' and 'none'.
    event_type1 = "cut"   # compute difference in returns before cuts
    event_type2 = "none"  # and days outside the event window

    # Compute statistics for these currenices
    currs = ["aud", "cad", "chf", "eur", "gbp", "nok", "nzd", "sek"]

    # Compute statistics around these announcements
    events_by = "local_cbs"  # set to 'usd' to compute around fomc to 'eur' for
                             # the ecb, etc

    # Load data ===============================================================
    data = pd.read_pickle(data_path + out_counter_usd_name)
    data = data["ask_close"].pct_change().loc[
           (s_dt - BDay(22)):, currs] * 1e4

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

    # Estimate statistics, and plot the results
    eda = EventDataAggregator(events, data, pre, post)
    stacked = eda.stack_data()
    diff = eda.compare_means_by_stamp_and_event(
        stacked, event_type1, event_type2, lambda x: x.hour).reset_index()

    fig, ax = plt.subplots(figsize=(15, 7))
    ax = barplot_intraday_by_hour(diff)
    plt.title("Difference in hourly returns between '{}' and '{}' subsamples".
              format(event_type1, event_type2))

    fig.tight_layout()
    fig.savefig(data_path+"/wp_figures_limbo/" +
                "intraday_patterns_{}_{}_cb_is_{}.pdf".format(
                    event_type1, event_type2, events_by))
    plt.show()

    print("lol")
