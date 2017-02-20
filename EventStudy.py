import pandas as pd
import numpy as np
from scipy.stats import norm
import random
from pandas.tseries.offsets import DateOffset
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pdb
from assetpricing import portfolio_construction as poco
import matplotlib.pyplot as plt
# %matplotlib

class EventStudy():
    """
    Parameters
    ----------
    data : pandas.DataFrame/pandas.Series
        of assets
    events : list/pandas.DataFrame/pandas.Series
        of events indexed with dates; if DataFrame, must contain one column
        only
    window : tuple of int
        [a,b,c,d] where each element is relative (in periods) date of
        a - start of cumulating returns in `data` before event (negative int);
        b - stop of cumulating returns before event (negative int),
        c - start of cumulating returns in `data` after event,
        d - stop of cumulating returns before event.
        For example, [-5,-1,1,5] means that one starts to look at returns 5
        periods before each event, ends 1 day before, starts again 1 day after
        and stops 5 days after.
    """
    def __init__(self, data, events, window):
        """ Create class, pivot data and center on events.
        """
        ta, tb, tc, td = window

        # if events are a list
        if not isinstance(events, pd.DataFrame):
            events = pd.DataFrame(data=np.arange(len(events)),index=events)

        # if data is a Series:
        if isinstance(data, pd.Series):
            data = data.to_frame()

        # shape
        T,N = data.shape

        # keep record of dimensions (needed for later)
        self.is_1d = N < 2

        # if last event is closer to the end than td - delete last event
        # TODO: write logger for that
        good_evt_idx = events.index.map(
            lambda x: (get_idx(data, x) < (T-td)) & \
                (get_idx(data, x) > -ta-1))
        events = events.ix[good_evt_idx,:]

        # pivot
        before, after = self.pivot_for_event_study(data, events, window)
        # pdb.set_trace()

        self.data = data
        self.events = events
        self.window = window
        self.before = before
        self.after = after
        self.before_idx = before.major_axis
        self.after_idx = after.major_axis
        self.stacked_idx = np.hstack((before.major_axis, after.major_axis))

    @staticmethod
    def pivot_for_event_study(data, events, window):
        """ Pivot `data` to center on event.
        """
        # pdb.set_trace()
        ta, tb, tc, td = window

        # loop over events, save snapshot of returns at -n'th day,
        #   n=1,2,...,window
        before_dict = {}
        after_dict = {}
        for (t, number) in events.iterrows():
            # print(t)
            # fetch index of this event
            this_t = get_idx(data,t)
            # from a to b
            this_df = data.ix[(this_t+ta):(this_t+tb+1),:]
            # index with [a,...,b]
            this_df.index = np.arange(ta, tb+1)
            # store
            before_dict[number.values[0]] = this_df

            # from c to d, same steps
            this_df = data.ix[(this_t+tc):(this_t+td+1),:]
            this_df.index = np.arange(tc, td+1)
            after_dict[number.values[0]] = this_df

        # create panel of before-event dataframes, where minor_axis keeps
        #   individual events, items keeps columns of Y
        before = pd.Panel.from_dict(before_dict, orient="minor")

        # after
        after = pd.Panel.from_dict(after_dict, orient="minor")

        return before, after

    def get_ts_cumsum(self, inplace=False):
        """ Calculate cumulative sum along time.
        """
        # the idea is to buy stuff at relative date a, where T is the event
        #   date and keep it until and including relative date b, so need to
        #   reverse this Panel for further cumulative summation
        ts_mu_before = self.before.ix[:,::-1,:].cumsum().ix[:,::-1,:]
        # after events is ok as is
        ts_mu_after = self.after.cumsum()

        # concat
        ts_mu = pd.concat((ts_mu_before, ts_mu_after), axis=1)

        if inplace:
            self.before = ts_mu_before
            self.after = ts_mu_after

        return ts_mu

    def get_cs_mean(self, inplace=False):
        """ Calculate mean across events
        """
        cs_mu_before = self.before.mean(axis="minor_axis")
        cs_mu_after = self.after.mean(axis="minor_axis")

        # concat
        cs_mu = pd.concat((cs_mu_before, cs_mu_after), axis=0)

        if inplace:
            self.before = cs_mean_before
            self.after = cs_mean_after

        return cs_mu

    def get_ci(self, ps, method="simple", **kwargs):
        """ Calculate confidence bands.
        Parameters
        ----------
        p : float / tuple of floats
            interval width or tuple of (lower bound, upper bound)
        Returns
        -------
        ci : pd.Panel

        """
        ta, tb, tc, td = self.window

        # if `ps` was provided as single float
        if isinstance(ps, float):
            ps = ((1-ps)/2, ps/2+1/2)

        # drop dates around events, 2 times window length
        # TODO: control for overlaps
        boot_from = self.data.copy()
        # pdb.set_trace()
        for (t, number) in self.events.iterrows():
            this_t = get_idx(boot_from, t)
            boot_from.drop(
                boot_from.index[(this_t+ta*2):(this_t+td*2)],
                axis="index",
                inplace=True)

        # calculate confidence band
        if method == "simple":
            ci = self.simple_ci(boot_from=boot_from, ps=ps)

        elif method == "boot":
            ci = self.boot_ci(boot_from=boot_from, ps=ps, **kwargs)

        return ci

    def simple_ci(self, boot_from, ps):
        """
        """
        ta, tb, tc, td = self.window

        mu = boot_from.mean()

        # sd of mean across events is mean outside of events divided
        #   through sqrt(number of events)
        sd_across = boot_from.std()/np.sqrt(len(self.events))
        # sd of cumulative sum of mean across events is sd times sqrt(# of
        #   cumulants)
        # pdb.set_trace()
        q = np.sqrt(np.hstack(
            (np.arange(-ta+tb+1,0,-1), np.arange(1,td-tc+2))))
        q = q[:,np.newaxis]

        # multiply with broadcast, add mean
        ci_lo = norm.ppf(ps[0])*q*sd_across.values[np.newaxis,:] + \
            boot_from.mean().values[np.newaxis,:]*q**2
        ci_hi = norm.ppf(ps[1])*q*sd_across.values[np.newaxis,:] + \
            boot_from.mean().values[np.newaxis,:]*q**2

        # concatenate
        ci = pd.Panel(
            major_axis=self.stacked_idx,
            items=self.data.columns,
            minor_axis=ps)
        ci.loc[:,:,ps[0]] = ci_lo.T
        ci.loc[:,:,ps[1]] = ci_hi.T

        return ci

    def boot_ci(self, boot_from, ps, M=1000):
        """
        """
        # space for df's of pivoted tables
        booted = pd.Panel(
            items=self.data.columns,
            major_axis=self.stacked_idx,
            minor_axis=range(M))

        # possible times should exclude values that are too early or too late
        possible_dates = boot_from.index.tolist()[
            (-self.window[0]):-(self.window[-1]+1)]

        for p in range(M):
            # pdb.set_trace()
            # draw sequence of events, sort them
            events_batch = sorted(
                random.sample(possible_dates, len(self.events)))
            events_batch = pd.DataFrame(
                data=np.arange(len(self.events)), index=events_batch)

            # pivot this batch
            bef, aft = self.pivot_for_event_study(
                boot_from, events_batch, self.window)

            # cumsum
            bef = bef.ix[:,::-1,:].cumsum().ix[:,::-1,:]
            # after events is ok as is
            aft = aft.cumsum()
            # concat (automatically along major_axis)
            tot = pd.concat((bef, aft), axis=1)

            # mean across events
            tot = tot.mean(axis="minor_axis")

            # store
            booted.iloc[:,:,p] = tot

        # quantiles
        # lower
        ci_lo = booted.apply(lambda x: x.quantile(ps[0]), axis="minor")
        # upper
        ci_hi = booted.apply(lambda x: x.quantile(ps[1]), axis="minor")

        # concatenate
        ci = pd.Panel(
            major_axis=self.stacked_idx,
            items=self.data.columns,
            minor_axis=ps)
        ci.loc[:,:,ps[0]] = ci_lo
        ci.loc[:,:,ps[1]] = ci_hi

        return ci

def get_idx(data, t):
    """ Fetch integer index given time index.
    If index `t` is not present in `data`.index, this function finds the
    first present later.
    """
    return data.index.get_loc(t, method="ffill")

def run_event_study(data, events, window, ps, wght=None, ci_method="simple",
    plot=True, **kwargs):
    """
    """
    if wght is not None:
        if wght == "mean":
            data = data.mean(axis=1)
        else:
            data = data.dot(wght)

    evt_study = EventStudy(
        data=data,
        events=events,
        window=window)

    ts_mu = evt_study.get_ts_cumsum(inplace=True)
    cs_mu = evt_study.get_cs_mean()

    ci = evt_study.get_ci(ps=ps, method=ci_method, **kwargs)
    ci = ci.ix[0,:,:]

    # plot ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 10/4*3))
    cs_mu.plot(ax=ax, linewidth=1.5, color='k')
    ax.legend_.remove()
    ax.fill_between(evt_study.stacked_idx,
        ci.iloc[:,0].values,
        ci.iloc[:,1].values,
        color="#b7b7b7", alpha=0.66, label="conf. interval")
    ax.xaxis.set_ticks(evt_study.stacked_idx)
    ax.set_xlabel("days after event")
    ax.set_ylabel("cumulative return, in %")
    # ax.set_title("currencies vs. dollar after opec meetings")
    plt.axhline(y=0, color='k', linestyle='--')
    ax.grid(axis="both", alpha=0.33, linestyle=":")
    # ax.legend(loc="upper right")
    grey_patch = mpatches.Patch(color="#b7b7b7", label="90% ci")
    plt.legend(handles=[ax.get_lines()[0], grey_patch])

    return cs_mu, ci, ax

if __name__ == "__main__":
    pass
