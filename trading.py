import pandas as pd
import numpy as np
import warnings
import copy
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
        self.weights = weights*position_flags

        self.e_dt = e_dt
        self.s_dt = s_dt


    def leverage_adjusted(self):
        """
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
        """
        """
        signals_reixed, _ = self.signals.align(self._returns, axis=0,
            join="outer")

        to_b = signals_reixed.shift(self.settings["horizon_b"])

        # fill na backward, limit to (b-a) values
        # NB: this covers the case of too few data points at the beginning
        position_flags = to_b.fillna(method="bfill",
            limit=self.settings["horizon_b"]-self.settings["horizon_a"])

        return position_flags


    def bas_adjusted(self):
        """
        """
        # ipdb.set_trace()
        wght = self.weights.fillna(value=0)

        # difference in weights is an action
        dwght = wght.diff()

        # fill mid with ask and bids accordingly ----------------------------
        new_mid = self.prices["mid"].copy()

        # buy at ask
        new_mid = new_mid.mask(dsigs.shift(-1) > 0, self.prices["ask"])
        # sell at bid
        new_mid = new_mid.mask(dsigs.shift(-1) < 0, self.prices["bid"])

        self_copy = copy.deepcopy(self)
        self_copy.returns = np.log(new_mid).diff()
        self_copy.prices = self.prices.update({"mid": new_mid})

        return self_copy


    def get_historic_weights(self):
        """
        """
        # ipdb.set_trace()
        flags = self.position_flags

        w_neg_resc = poco.rescale_weights(
            flags.where(flags < 0)*self.meta_weights)
        w_pos_resc = poco.rescale_weights(
            flags.where(flags > 0)*self.meta_weights)
        w_nil_resc = poco.rescale_weights(
            flags.where(flags == 0)*self.meta_weights)

        # ipdb.set_trace()
        wght = w_neg_resc.fillna(w_pos_resc)

        return wght, (w_neg_resc, w_nil_resc, w_pos_resc)


    def get_strategy_returns(self):
        """
        """
        # w, _ = self.get_historic_weights()
        res = poco.weighted_return(self._returns, self.weights)

        return res


    def plot(self, **kwargs):
        """
        """
        pass


if __name__ == "__main__":

    from foolbox.api import *

    data_path = set_credentials.gdrive_path("research_data/fx_and_events/")

    with open(data_path + "mba_by_tz_d.p", mode='rb') as fname:
        fx = pickle.load(fname)
    fx = fx.drop(["jpy","dkk","nok"], axis="items")

    mid = fx.loc[:,:,"mid"]
    ask = fx.loc[:,:,"ask"]
    bid = fx.loc[:,:,"bid"]

    settings = {
        "h": 10,
        "horizon_a": -10,
        "horizon_b": -1,
        "bday_reindex": True}

    policy_fcasts = dict()
    for c in ['aud', 'cad', 'chf', 'eur', 'gbp', 'nzd', 'sek']:
        # c = "aud"
        pe = PolicyExpectation.from_pickles(data_path, c)
        policy_fcasts[c] = pe.forecast_policy_change(
            lag=12, threshold=0.10, avg_impl_over=5, avg_refrce_over=5)

    policy_fcasts = pd.DataFrame.from_dict(policy_fcasts)

    ts = EventTradingStrategy(
        signals=policy_fcasts,
        prices={"mid": mid, "bid": bid, "ask": ask},
        settings=settings)

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(
        data_path + '../../opec_meetings/calc/insights.xlsx',
        engine='xlsxwriter')

    # Convert the dataframe to an XlsxWriter Excel object.
    ts.signals.to_excel(writer, sheet_name='signals')
    ts.get_position_flags().to_excel(writer, sheet_name='position_flags')
    ts.returns.to_excel(writer, sheet_name='returns')
    ts.weights.to_excel(writer, sheet_name='weights')

    ts.bas_adjusted()
    # Close the Pandas Excel writer and output the Excel file.
    writer.save()




    strat_ret = ts.get_strategy_returns()

    ts.get_signals_by_period()
    ts_bas = ts.bas_adjusted()
    ts_bas.get_strategy_returns()
