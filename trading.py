import pandas as pd
import numpy as np
import warnings
import copy
# import ipdb

import matplotlib.pyplot as plt

gr_1 = "#8c8c8c"

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
            if isinstance(prices, dict):
                prices.update({"actual": prices["mid"]})
            else:
                prices = {"mid": prices, "actual": prices}

        if returns is None:
            returns = np.log(prices["actual"]).diff()

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
            returns, signals = returns.align(signals, join="outer", axis=0)

        # THE index
        the_idx = returns.index

        # weights are uniform unless otherwise given
        if weights is None:
            weights = returns.mask(returns.notnull(), 1.0)
        else:
            weights = weights.reindex(index=the_idx, method="ffill")

        self._returns = returns
        self.signals = signals.replace(0.0, np.nan)
        self.settings = settings
        self.prices = prices

        # position flags
        self.position_flags = self.get_position_flags().replace(0.0, np.nan)

        # weights
        self.meta_weights = weights.copy()
        self.weights = weights*self.position_flags

        self.e_dt = e_dt
        self.s_dt = s_dt

        self.the_idx = the_idx

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
        # # expand signals by aligning them with returns: will be used for
        # #   backward-filling
        # signals_reixed, _ = self.signals.align(self._returns, axis=0,
        #     join="outer")
        signals_reixed = self.signals.copy()

        # define the end of the horizon_b-horizon_a holding period
        to_b = signals_reixed.shift(self.settings["horizon_b"])

        # fill na backward, limit to (b-a) values
        # NB: this covers the case of too few data points at the beginning
        position_flags = to_b.fillna(method="bfill",
            limit=self.settings["horizon_b"]-self.settings["horizon_a"])

        return position_flags

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

        # # align with events -------------------------------------------------
        # res = res.shift(-1*self.settings["horizon_b"])

        return res

    def leverage_adjusted(self):
        """ Ensure that weights in the short and long legs sum up to one.

        In case of overlapping holdings, weights in the long leg are made to
        sum up to 100%, and weights in the short leg to -100%. Weighting
        scheme is taken from `self.meta_weights`.

        Returns
        -------
        new_self : EventTradingStrategy()
            deep copy of `self` with weights reassigned

        Usage
        -----
        ts_lev = EventTradingStrategy(...).leverage_adjusted()

        """
        wght, _ = self.get_historic_weights()

        # copy class instance, replace weights
        new_self = copy.deepcopy(self)
        new_self.weights = wght

        return new_self

    @staticmethod
    def adjust_prices_for_bas(prices, positions):
        """ Substitute prices for bids/asks whenever a position is changed.
        Parameters
        ----------
        prices : dict
            with keys "mid", "bid", "ask"; resp. values are pandas.DataFrame
        positions : pandas.DataFrame
            of continuous positions
        Returns
        -------
        new_prices : pandas.DataFrame
            where values from "mid" are replaced with bids/asks

        """
        # nan positions are zero position: e.g. going from nan to +1 is a buy
        positions_filled = positions.fillna(value=0)

        # difference in positinos: in the period preceding such difference an
        #   order was placed
        d_positions = positions_filled.diff()

        # fill mid with ask and bids accordingly ----------------------------
        new_prices = prices["mid"].copy()

        # buy at ask
        new_prices = new_prices.mask(d_positions.shift(-1) > 0,
            prices["ask"])
        # sell at bid
        new_prices = new_prices.mask(d_positions.shift(-1) < 0,
            prices["bid"])

        return new_prices

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
        new_self : EventTradingStrategy()
            deep copy of `self` with returns and mid prices reassinged

        Usage
        -----
        ts_bas = EventTradingStrategy(...).bas_adjusted()
        """
        p_actual = self.adjust_prices_for_bas(self.prices, self.weights)

        # copy class instance, assign weights -------------------------------
        new_self = copy.deepcopy(self)
        new_self._returns = np.log(p_actual).diff()
        new_self.prices.update({"actual": p_actual})

        return new_self

    def roll_adjusted(self, swap_points):
        """ ForEx: adjust for roll-overs.

        Parameters
        ----------
        swap_points : pandas.DataFrame
            with swap points, in units (e.g. -0.0178 for
            AUDUSD); these should be expressed in the way such that adding
            them to the prices results in correct formulae
        """
        if isinstance(swap_points, dict):
            sp_ask = swap_points["ask"].reindex(
                index=self.the_idx, method="ffill")
            sp_bid = swap_points["bid"].reindex(
                index=self.the_idx, method="ffill")
            # treat short and long positions differently --------------------
            sp = sp_ask.mask(self.position_flags < 0, sp_bid)
        else:
            sp = swap_points.reindex(
                index=self.the_idx, method="ffill")

        # new version. less approximate -------------------------------------
        # horizons: for pure convenience ------------------------------------
        h_a = self.settings["horizon_a"]
        h_b = self.settings["horizon_b"]

        # rolling h-period sum of swap points -------------------------------
        roll_sum_sp = sp.rolling( (h_b-h_a+1) ).sum()

        # adjust prices -----------------------------------------------------
        denom = self.prices["actual"].shift( (h_b-h_a+1) ) + \
            roll_sum_sp.shift(1)

        # recalculate returns -----------------------------------------------
        # these are similar to excess returns in the standard case:
        #   f.shift(1) - s
        numer = self.prices["actual"]

        # align signals with returns
        new_returns = np.log(numer/denom) * self.signals.shift(h_b)

        # return new_returns

        # copy class instance, assign weights -------------------------------
        new_self = copy.deepcopy(self)
        new_self._returns = new_returns
        new_self.swap_points = sp

        return new_self


        # # temp, approximate version -----------------------------------------
        # # roll is credited/debited two days fewer
        # roll_index = self.position_flags.shift(1).notnull() & \
        #     self.position_flags.shift(-1).notnull()
        # swap_points_to_add = sp.where(roll_index).fillna(0.0)
        #
        # prices_adjusted = self.prices["mid"] + swap_points_to_add
        # new_returns = np.log(self.prices["mid"] / prices_adjusted.shift(1))
        #
        # # copy self
        # new_self = copy.deepcopy(self)
        # new_self._returns = new_returns
        # new_self.swap_points = sp
        #
        # return new_self

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
        self.prices["actual"].to_excel(writer, sheet_name='actual')
        try:
            self.swap_points.to_excel(writer, sheet_name='swap_points')
        except:
            pass

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

    def match_forwards(self, settl_dates):
        """ In a cross-section of settl't dates find those of the signals.
        """
        pass

        # idx_union = self.prices["actual"].index.union(
        #     self.signals.index.union(settl_dates.index))
        #
        # sig = self.signals.reindex(index=idx_union)
        # sig = sig.shift(horizon_b)
        #
        # for c in self.signals.columns:
        #     # subset this column, drop duplicated entries
        #     this_col = settl_dates.loc[:,c].drop_duplicates(keep=False)
        #
        #     # match with signals
        #     this_sig_dates = sig.loc[:,c].dropna().index
        #     this_col.isin(this_sig_dates)


def run_this(h, thresh):
    """
    data_path = set_credentials.gdrive_path("research_data/fx_and_events/")

    with open(data_path + "fx_by_tz_aligned_d.p", mode='rb') as fname:
        fx = pickle.load(fname)
    spot_ask = fx["spot_ask"].drop(["jpy","dkk","nok"],
        axis=1).loc["2000-11":,:]
    spot_mid = fx["spot_mid"].drop(["jpy","dkk","nok"],
        axis=1).loc["2000-11":,:]
    spot_bid = fx["spot_bid"].drop(["jpy","dkk","nok"],
        axis=1).loc["2000-11":,:]

    tnswap_ask = fx["tnswap_ask"].drop(["jpy","dkk","nok"],
        axis=1).loc["2000-11":,:]
    tnswap_bid = fx["tnswap_bid"].drop(["jpy","dkk","nok"],
        axis=1).loc["2000-11":,:]

    """
    # policy expectations -----------------------------------------------
    policy_fcasts = dict()
    for c in ['aud', 'cad', 'chf', 'eur', 'gbp', 'nzd', 'sek']:
        # c = "aud"
        pe = PolicyExpectation.from_pickles(data_path, c)
        policy_fcasts[c] = pe.forecast_policy_change(
            lag=h+2,
            threshold=thresh,
            avg_impl_over=5,
            avg_refrce_over=5)

    policy_fcasts = pd.DataFrame.from_dict(policy_fcasts).loc["2000-11":]

    # settings ----------------------------------------------------------
    settings = {
        "horizon_a": -h,
        "horizon_b": -1,
        "bday_reindex": True}

    # strategies --------------------------------------------------------
    # simple strategy: no leverage, no bas adjustment
    ts = EventTradingStrategy(
        signals=policy_fcasts,
        prices={"mid": spot_mid, "bid": spot_bid, "ask": spot_ask},
        settings=settings)

    # advanced strategy: bas + roll
    ts_div_bas = ts.bas_adjusted().roll_adjusted(
        swap_points={"bid": tnswap_bid, "ask": tnswap_bid})

    return ts_div_bas._returns



if __name__ == "__main__":

    # %matplotlib inline

    # data ------------------------------------------------------------------
    data_path = set_credentials.gdrive_path("research_data/fx_and_events/")

    # assets
    with open(data_path + "fx_by_tz_aligned_d.p", mode='rb') as fname:
        fx = pickle.load(fname)
    spot_ask = fx["spot_ask"].drop(["jpy","dkk","nok"],
        axis=1).loc["2000-11":,:]
    spot_mid = fx["spot_mid"].drop(["jpy","dkk","nok"],
        axis=1).loc["2000-11":,:]
    spot_bid = fx["spot_bid"].drop(["jpy","dkk","nok"],
        axis=1).loc["2000-11":,:]

    tnswap_ask = fx["tnswap_ask"].drop(["jpy","dkk","nok"],
        axis=1).loc["2000-11":,:]
    tnswap_bid = fx["tnswap_bid"].drop(["jpy","dkk","nok"],
        axis=1).loc["2000-11":,:]

    # policy expectations
    policy_fcasts = dict()
    for c in ['aud', 'cad', 'chf', 'eur', 'gbp', 'nzd', 'sek']:
        # c = "aud"
        pe = PolicyExpectation.from_pickles(data_path, c)
        policy_fcasts[c] = pe.forecast_policy_change(
            lag=12, threshold=0.10, avg_impl_over=5, avg_refrce_over=5)

    # # dollar policy
    # pe = PolicyExpectation.from_pickles(data_path, "usd")
    # policy_fcast_usd = pe.forecast_policy_change(
    #     lag=12, threshold=0.10, avg_impl_over=5, avg_refrce_over=5)
    #
    policy_fcasts = pd.DataFrame.from_dict(policy_fcasts).loc["2000-11":]

    # for c in policy_fcasts.columns:
    #     policy_fcasts.loc[:,c] = policy_fcasts.loc[:,c].replace(0.0, np.nan)\
    #         .fillna(policy_fcast_usd.replace(0.0, np.nan)*-1)

    # settings --------------------------------------------------------------
    settings = {
        "horizon_a": -10,
        "horizon_b": -1,
        "bday_reindex": True}

    # strategies ------------------------------------------------------------
    # simple strategy: no leverage, no bas adjustment
    ts = EventTradingStrategy(
        signals=policy_fcasts,
        prices={"mid": spot_mid, "bid": spot_bid, "ask": spot_ask},
        settings=settings)

    # # write to xlsx
    # xl_filename = data_path + '../../opec_meetings/calc/insights_simple.xlsx'
    # ts.to_excel(xl_filename)
    # fig, ax = ts.plot()
    #

    # bas adjusted
    ts_bas = ts.bas_adjusted()
    ts_bas.to_excel(xl_filename)
    fig, ax = ts_bas.plot(color='k')


    # leverage_adjusted
    ts_lev = ts.leverage_adjusted()
    ts_lev.to_excel(xl_filename)
    fig, ax = ts_lev.plot(color='r')


    # # adjusted for both
    # ts_lev_bas = ts.leverage_adjusted().bas_adjusted()
    xl_filename = data_path + '../../opec_meetings/calc/insights_lev_bas.xlsx'
    # fig, ax = ts_lev_bas.plot(color='#ddc061')


    # adjusted for dividend -------------------------------------------------
    ts_div_bas = ts.bas_adjusted().roll_adjusted(
        {"bid": tnswap_bid, "ask": tnswap_bid})
    ts_div_bas._returns.sum(axis=1).cumsum().plot()

    # run this --------------------------------------------------------------
    res = run_this(10, 0.100)
