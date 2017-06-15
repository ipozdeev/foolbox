import pandas as pd
import numpy as np
import warnings
import copy
# import ipdb

import matplotlib.pyplot as plt

# TODO: relocate to wp_settings?
font_settings = {
    "family": "serif",
    "size": 12}
fig_settings = {
    "figsize": (8.27,11.3/3)}
tick_settings = {
    "labelsize": 12}
axes_settings = {
    "grid": True}
grid_settings = {
    "alpha": 0.5}

# [p for p in plt.rcParams.keys() if "axes" in p]

plt.rc("xtick", **tick_settings)
plt.rc("ytick", **tick_settings)
plt.rc("figure", **fig_settings)
plt.rc("font", **font_settings)
plt.rc("axes", **axes_settings)
plt.rc("grid", **grid_settings)

from foolbox.api import *

class TradingStrategy():
    """
    """
    def __init__(self):
        """
        """
        pass


class EventTradingStrategy(TradingStrategy):
    """ Trading strategy based on events and signals.

    Parameters
    ----------
    signals : pandas.DataFrame
        of signals {-1, 0, 1}. Assets are held some days (given in
        `settings`) before or after the events defined by `signals`.
    prices : dict of pandas.DataFrame
        (optional) dictionary with "mid", "bid" and "ask", whereby "bid"
        and "ask" are optional; only "mid" will be used to calcualte
        returns, the other two are needed for bas adjustment.
    returns : pandas.DataFrame
        (one of `prices` and `returns` must be specified)
    weights : pandas.DataFrame
        (optional, not tested) weights to use for weighting returns. By
        default equal weights are assumed withing long and short legs.
    settings : dict
        "horizon_a": first day after event to cash in return (possibly negat.),
        "horizon_a": the day to stop cumulating returns (possibly negat.),
        "bday_reindex": True for reindexing returns with business days
            (might expand the sample)
    """
    def __init__(self, signals, prices=None, returns=None, weights=None,
        settings=None):
        """
        """
        if prices is not None:
            if not isinstance(prices, dict):
                prices = {"mid": prices}

        if returns is None:
            returns = np.log(prices["mid"]).diff()

        # first and last valid indices
        s_dt = returns.first_valid_index()
        e_dt = returns.last_valid_index()

        # reindex with business day if necessary
        if settings["bday_reindex"]:
            returns = returns.reindex(
                index=pd.date_range(s_dt, e_dt, freq='B'))

        # warning if some date from `signals`.index is not in `returns`.index
        if not signals.index.isin(returns.index).all():
            warnings.warn(
                "Some dates in `signals` do not correspond to "+
                "rows in `returns`.")
            returns, _ = returns.align(signals, join="outer", axis=0)

        # weights are uniform unless otherwise given
        if weights is None:
            weights = returns.mask(returns.notnull(), 1.0)

        self._returns = returns
        self.signals = signals
        self.settings = settings
        self.prices = prices

        # position flags
        self.position_flags = self.get_position_flags()

        # weights
        self.meta_weights = weights
        self.weights = weights*self.position_flags

        self.e_dt = e_dt
        self.s_dt = s_dt


    def leverage_adjusted(self):
        """ Ensure that weights in the short and long legs sum up to one.

        In case of overlapping holdings, weights in the long leg are made to
        sum up to 100%, and weights in the short leg to -100%. Weighting
        scheme is taken from `self.meta_weights`.

        Returns
        -------
        self_copy : EventTradingStrategy()
            deep copy of `self` with weights reassigned

        Usage
        -----
        ts_lev = EventTradingStrategy(...).leverage_adjusted()

        """
        wght, _ = self.get_historic_weights()

        # copy class instance, replace weights
        self_copy = copy.deepcopy(self)
        self_copy.weights = wght

        return self_copy


    # make returns appear multiplied with 100
    @property
    def returns(self):
        return self._returns

    @returns.getter
    def returns(self):
        return self._returns.loc[self.s_dt:self.e_dt]*100


    def get_position_flags(self):
        """ Create a continuous panel of {+1,0,-1} flags, punched card-esque.

        2001-09-03  -1  0  0
        2001-09-04  -1 +1  0
        2001-09-07  -1 +1  0
        ...
        2001-09-15   0 +1  0
        2001-09-16   0 +1  0

        Returns
        -------
        position_flags : pandas.DataFrame
            of same shape as `self.returns`

        """
        # expand signals by aligning them with returns: will be used for
        #   backward-filling
        signals_reixed, _ = self.signals.align(self._returns, axis=0,
            join="outer")

        # define the end of the horizon_b-horizon_a holding period
        to_b = signals_reixed.shift(self.settings["horizon_b"])

        # fill na backward, limit to (b-a) values
        # NB: this covers the case of too few data points at the beginning
        position_flags = to_b.fillna(method="bfill",
            limit=self.settings["horizon_b"]-self.settings["horizon_a"])

        return position_flags


    def bas_adjusted(self):
        """ Adjust returns for the bid-ask spread.

        Give a panel of continuous position flags, the 'fringes' thereof
        denote the dates where the trades are placed. This method substitutes
        mid prices on these dates with ask prices in case the buy order is
        placed (e.g. when a position goes from -1 to 0) and bid prices in case
        the sell order is placed. Returns are re-calculated, such that returns
        on 'buy' and 'sell' dates are a bit worse.

        Returns
        -------
        self_copy : EventTradingStrategy()
            deep copy of `self` with returns and mid prices reassinged

        Usage
        -----
        ts_bas = EventTradingStrategy(...).bas_adjusted()
        """
        # nan weights are zero weights: going from nan to +1 is a buy
        wght = self.weights.fillna(value=0)

        # difference in weights: in the period preceding such difference an
        #   order was placed
        dwght = wght.diff()

        # fill mid with ask and bids accordingly ----------------------------
        new_mid = self.prices["mid"].copy()

        # buy at ask
        new_mid = new_mid.mask(dwght.shift(-1) > 0, self.prices["ask"])
        # sell at bid
        new_mid = new_mid.mask(dwght.shift(-1) < 0, self.prices["bid"])

        # copy class instance, assign weights -------------------------------
        self_copy = copy.deepcopy(self)
        self_copy._returns = np.log(new_mid).diff()
        self_copy.prices = self.prices.update({"mid": new_mid})

        return self_copy


    def get_historic_weights(self):
        """ Construct summable-to-one portfolio weights in both legs.

        Returns
        -------
        wght : pandas.DataFrame
            of weights, same shape as `returns`
        rest : tuple
            of negative, neutral and positive weights
        """
        flags = self.position_flags

        # rescale negative weights
        w_neg_resc = poco.rescale_weights(
            flags.where(flags < 0)*self.meta_weights)
        # rescale positive weights
        w_pos_resc = poco.rescale_weights(
            flags.where(flags > 0)*self.meta_weights)
        # rescale neutral weights (why? dunno)
        w_nil_resc = poco.rescale_weights(
            flags.where(flags == 0)*self.meta_weights)

        # melt positive and negative legs
        wght = w_neg_resc.fillna(w_pos_resc)

        return wght, (w_neg_resc, w_nil_resc, w_pos_resc)


    def get_strategy_returns(self):
        """ Construct historic strategy returns.

        Returns
        -------
        res : pd.Series
            of returns
        """

        res = poco.weighted_return(self._returns, self.weights)

        return res

    def to_excel(self, filename):
        """ Write stuff to the excel spreadsheet.

        Parameters
        ----------
        filename : string
            full path to the file, with extension

        Returns
        -------
        None

        """
        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(filename, engine='xlsxwriter')

        # Convert dataframes to an XlsxWriter Excel object.
        self.signals.to_excel(writer, sheet_name='signals')
        self.position_flags.to_excel(writer, sheet_name='position_flags')
        self.returns.to_excel(writer, sheet_name='returns')
        self.weights.to_excel(writer, sheet_name='weights')
        self.get_strategy_returns().to_frame().to_excel(writer,
            sheet_name='portf_returns')
        self.prices["bid"].to_excel(writer, sheet_name='bid')
        self.prices["ask"].to_excel(writer, sheet_name='ask')
        self.prices["mid"].to_excel(writer, sheet_name='mid')

        # save and close
        writer.save()

        return


    def plot(self, **kwargs):
        """
        """
        ret = self.get_strategy_returns()
        cum_ret = ret.cumsum()

        fig, ax = plt.subplots()
        cum_ret.plot(ax=ax, **kwargs)

        return fig, ax


if __name__ == "__main__":

    %matplotlib

    # data ------------------------------------------------------------------
    data_path = set_credentials.gdrive_path("research_data/fx_and_events/")

    # assets
    with open(data_path + "mba_by_tz_d.p", mode='rb') as fname:
        fx = pickle.load(fname)
    fx = fx.drop(["jpy","dkk","nok"], axis="items").loc[:,"2000-11":,:]

    # three prices
    mid = fx.loc[:,:,"mid"]
    ask = fx.loc[:,:,"ask"]
    bid = fx.loc[:,:,"bid"]

    # policy expectations
    policy_fcasts = dict()
    for c in ['aud', 'cad', 'chf', 'eur', 'gbp', 'nzd', 'sek']:
        # c = "aud"
        pe = PolicyExpectation.from_pickles(data_path, c)
        policy_fcasts[c] = pe.forecast_policy_change(
            lag=12, threshold=0.10, avg_impl_over=5, avg_refrce_over=5)

    policy_fcasts = pd.DataFrame.from_dict(policy_fcasts).loc["2000-11":]

    # settings --------------------------------------------------------------
    settings = {
        "h": 10,
        "horizon_a": -10,
        "horizon_b": -1,
        "bday_reindex": True}

    # strategies ------------------------------------------------------------
    # simple strategy: no leverage, no bas adjustment
    ts = EventTradingStrategy(
        signals=policy_fcasts,
        prices={"mid": mid, "bid": bid, "ask": ask},
        settings=settings)

    # write to xlsx
    xl_filename = data_path + '../../opec_meetings/calc/insights_simple.xlsx'
    ts.to_excel(xl_filename)
    fig, ax = ts.plot()


    # bas adjusted
    ts_bas = ts.bas_adjusted()
    ts_bas.to_excel(xl_filename)
    fig, ax = ts_bas.plot(color='k')


    # leverage_adjusted
    ts_lev = ts.leverage_adjusted()
    ts_lev.to_excel(xl_filename)
    fig, ax = ts_lev.plot(color='r')


    # adjusted for both
    ts_lev_bas
