import pandas as pd
import numpy as np
import warnings

from foolbox.backtesting.sorting import rank_sort
from foolbox.backtesting.weights import position_flags_to_weights


class Backtesting:
    """
    Parameters
    ----------
    r_long
    r_short : pandas.DataFrame
        return on a short position worth 1 unit

    """
    def __init__(self, r_long, r_short=None):
        
        if r_short is None:
            r_short = r_long * -1

        self.r_long = r_long
        self.r_short = r_short

        self.assets = r_long.columns

    def backtest(self, strategy, breakdown=False):
        """Calculate historical series of strategy returns.

        Parameters
        ----------
        strategy : TradingStrategy
        breakdown : bool
            True to break down into high, low and hml

        Returns
        -------
        pandas.Series

        """
        w = strategy.position_weights

        w_long = w.where(w > 0)
        w_short = w.where(w < 0) * -1

        r_long = self.r_long.mul(w_long).sum(axis=1, min_count=1)
        r_short = self.r_short.mul(w_short).sum(axis=1, min_count=1)

        r = r_long + r_short

        if breakdown:
            res = pd.concat({"p_high": r_long, "p_low": -1*r_short, "hml": r},
                            axis=1)
        else:
            res = r

        return res


class TradingStrategy:
    """
    Parameters
    ----------
    actions : pandas.DataFrame
    position_flags : pandas.DataFrame
    position_weights : pandas.DataFrame
    leverage : str
    """
    def __init__(self, actions=None, position_flags=None,
                 position_weights=None, leverage=None):
        """
        """
        self._actions = actions
        self._position_flags = position_flags
        self._position_weights = position_weights
        self._leverage = leverage

    @property
    def position_flags(self):
        return self._position_flags

    @position_flags.getter
    def position_flags(self):
        if self._position_flags is None:
            raise ValueError("No position flags specified!")
        return self._position_flags

    @position_flags.setter
    def position_flags(self, value):
        self._position_flags = value

    @property
    def position_weights(self):
        return self._position_weights

    @position_weights.getter
    def position_weights(self):
        """Get position weights.

        Fills missing values with 0.0: zero position weight. Eases
        computation of statistics, but watch out for fully empty rows!

        Returns
        -------
        pandas.DataFrame

        """
        if self._position_weights is None:
            self._position_weights = \
                position_flags_to_weights(
                    self.position_flags, self._leverage)

        return self._position_weights.fillna(0.0)

    @position_weights.setter
    def position_weights(self, value):
        self._position_weights = value

    @classmethod
    def from_events(cls, events, blackout, hold_period, leverage="net"):
        """
        Parameters
        ----------
        events : pd.DataFrame
            of events: -1,0,1
        blackout : int
            the number of time periods to close the position in: negative
            numbers indicate closing after event
        hold_period : int
            the number of periods to maintain position
        leverage : str
            "unlimited" for allowing unlimited leverage
            "net" for restricting it to 1
            "zero" for restricting that short and long positions net out to
                no-leverage
        """
        # I. position flags -------------------------------------------------
        # position flags are a continuous panel of series of events
        #   indicating where a position is held from dusk till dawn (such that
        #   a return is realized)
        # to be able to fill na backwards, replace zeros with nan's
        cont_events = events.replace(0.0, np.nan)

        # shift by blackout, extend from that `hold_period` periods into the
        #   past to arrive at position flags
        position_flags = cont_events.shift(-blackout)

        # make sure there is one nan at the beginning to be able to open a pos
        if position_flags.iloc[:hold_period, :].notnull().any().any():
            warnings.warn("There are not enough observations at the start; " +
                          "will delete the first events")
            position_flags.iloc[:hold_period, :] = np.nan

        # NB: dimi/
        # # Wipe the intruders who have events inside the first holding period
        # position_flags.iloc[:self.settings["holding_period"], :] = np.nan
        # /dimi

        # fill into the past
        position_flags = position_flags.fillna(method="bfill",
                                               limit=max(hold_period-1, 1))

        # need to open and close it somewhere
        position_flags.iloc[0, :] = np.nan
        position_flags.iloc[-1, :] = np.nan

        return cls(position_flags=position_flags, leverage=leverage)

    @classmethod
    def upsample(cls, freq, **kwargs):
        """Change strategy to a higher frequency without changing the signals.

        Parameters
        ----------
        freq : str
            pandas frequency
        kwargs : dict
            keyword arguments for pandas.resample()

        Returns
        -------
        new_strat : FXTradingStrategy

        """
        assert isinstance(freq, str) and (len(freq) < 2)

        # operate on position flags
        new_weights = cls.position_weights.copy()

        # resample
        new_weights = new_weights.resample(freq, **kwargs).bfill()

        # kill first period, making sure that trading happens on the
        # first timestamp of the new 'month' and NOT on the last one of the
        # previous 'month', as info is still unavailable then
        new_weights = new_weights.shift(1)

        new_strat = cls(position_weights=new_weights)

        return new_strat

    @classmethod
    def long_short(cls, sort_values, n_portfolios=None, legsize=None,
                   leverage="net"):
        """Construct strategy by sorting assets into `n_portfolios`.

        Parameters
        ----------
        sort_values : pandas.DataFrame
            of values to sort, ALREADY SHIFTED AS DESIRED
        n_portfolios : int
            the number of portfolios in portfolio_construction.rank_sort()
        legsize : int
        leverage : str
            'net', 'unlimited' or 'zero'

        Returns
        -------
        res : FXTradingStrategy

        """
        # sort
        pf = rank_sort(sort_values, n_portfolios, legsize)

        # concatenate
        flags = pf["p_high"].where(pf["p_high"] != 0) \
            .fillna(pf["p_low"].where(pf["p_low"] != 0) * -1)

        res = cls(position_flags=flags, leverage=leverage)

        # meta info for the no of portfolios
        res.n_portfolios = n_portfolios

        return res

    def __add__(self, other):
        """Combine two strategies.
        A + B means strategy A is traded unless strategy B disagrees.

        Not commutative!
        """
        # fill position weights with those of `other`
        new_pos_weights = other.position_weights\
            .replace(0.0, np.nan).fillna(self.position_weights)

        # the new strategy is a strategy with the above position weights and
        #  net leverage
        new_strat = TradingStrategy(position_flags=new_pos_weights,
                                    leverage="net")

        new_strat.position_flags = \
            new_strat.position_weights / new_strat.position_weights * \
            np.sign(new_strat.position_weights)

        return new_strat


class LongShortStrategyGenerator:
    """

    Parameters
    ----------
    signals
    signals_se
    kwargs : dict
        keyword arguments to TradingStrategy.long_short

    Returns
    -------
    generator

    """
    def __init__(self, signals, signals_se, **kwargs):
        """
        """
        self.signals = signals
        self.signals_se = signals_se
        self.strat_constructor = TradingStrategy.long_short
        self.strategy_constructor_kwargs = kwargs

    def generate(self, n_sim=10):
        """

        Parameters
        ----------
        n_sim

        Returns
        -------

        """
        for n in range(n_sim):
            signals_sim = pd.DataFrame(
                np.random.normal(loc=self.signals, scale=self.signals_se),
                index=self.signals.index,
                columns=self.signals.columns
            )

            yield self.strat_constructor(
                sort_values=signals_sim, **self.strategy_constructor_kwargs
            )
