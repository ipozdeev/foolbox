import pandas as pd
import numpy as np
import warnings
import ipdb


class TradingStrategy():
    """
    """

    def __init__(self):
        """
        """
        pass


class EventTradingStrategy(TradingStrategy):
    """
    """
    def __init__(self, signals,
        prices=None,
        returns=None,
        weights=None,
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

        # weights are uniform unless otherwise given
        if weights is None:
            weights = returns.mask(returns.notnull(), 1.0)

        # warning if some date from `signals`.index is not in `returns`.index
        if not signals.index.isin(returns.index).all():
            warnings.warn(
                "Some dates in `signals` do not correspond to "+
                "rows in `returns`.")
            returns, _ = returns.align(signals, join="outer", axis=0)

        self._returns = returns
        self.signals = signals
        self.prices = prices
        self.weights = weights

        self.settings = settings

        self.e_dt = e_dt
        self.s_dt = s_dt


    # make returns appear multiplied with 100
    @property
    def returns(self):
        return self._returns

    @returns.getter
    def returns(self):
        return self._returns.loc[self.s_dt:self.e_dt]*100


    def get_signals_by_period(self):
        """
        """
        signals_reixed, _ = self.signals.align(self._returns, axis=0,
            join="outer")

        to_b = signals_reixed.shift(self.settings["horizon_b"])

        # fill na backward, limit to (b-a) values
        # NB: this covers the case of too few data points at the beginning
        signals_by_period = to_b.fillna(method="bfill",
            limit=self.settings["horizon_b"]-self.settings["horizon_a"])

        return signals_by_period


    def bas_adjusted(self):
        """
        """
        # ipdb.set_trace()
        sigs = self.get_signals_by_period()
        sigs = sigs.fillna(value=0)

        # difference in weights is an action
        dsigs = sigs.diff()

        # fill mid with ask and bids accordingly ----------------------------
        new_mid = self.prices["mid"].copy()

        # buying at ask
        new_mid = new_mid.mask(dsigs.shift(-1) > 0, self.prices["ask"])
        # selling at bid
        new_mid = new_mid.mask(dsigs.shift(-1) < 0, self.prices["bid"])

        new_ts = EventTradingStrategy(
            prices={"mid": new_mid},
            returns=None,
            signals=self.signals,
            settings=self.settings)

        return new_ts


    def get_strategy_returns(self):
        """
        """
        ipdb.set_trace()
        sigs = self.get_signals_by_period()
        res = poco.weighted_return(self._returns, self.weights*sigs)

        return res




if __name__ == "__main__":

    from foolbox.api import *

    data_path = set_credentials.gdrive_path("research_data/fx_and_events/")

    with open(data_path + "mba_by_tz_d.p", mode='rb') as fname:
        fx = pickle.load(fname)
    mid = fx.loc[:,:,"mid"]
    ask = fx.loc[:,:,"ask"]
    bid = fx.loc[:,:,"bid"]

    settings = {
        "h": 10,
        "horizon_a": -10,
        "horizon_b": -1,
        "bday_reindex": True}

    pe = PolicyExpectation.from_pickles(data_path, "aud")
    policy_fcast = pe.forecast_policy_change(lag=7, threshold=0.125)

    ts = EventTradingStrategy(
        signals=policy_fcast,
        prices={"mid": mid["aud"], "bid": bid["aud"], "ask": ask["aud"]},
        settings=settings)
    ts.get_signals_by_period()
    ts_bas = ts.bas_adjusted()
    ts_bas.get_strategy_returns()
