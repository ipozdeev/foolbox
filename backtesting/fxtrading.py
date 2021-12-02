import pandas as pd
import numpy as np
import warnings
from foolbox import portfolio_construction as poco
from foolbox.finance import into_currency


class StrategyFactory:
    """Factory for constructing strategies."""
    def __init__(self):
        """
        """
        pass

    @staticmethod
    def rescale_weights(weights, leverage="net"):
        """Rescale weights of (possibly) leveraged positions.

        Parameters
        ----------
        weights : pandas.DataFrame
            of position flags in form of +1, -1 and 0
        leverage : str
            'zero' to make position weights sum up to zero;
            'net' to make absolute weights of short and long positions sum up
                to one each (sum(abs(negative positions)) = 1);
            'unlimited' for unlimited leverage.
        """
        # weights are understood to be fractions of portfolio value
        if leverage.startswith("zero"):
            # NB: dangerous, because meaningless whenever there are long/sort
            #   positions only
            # make sure that pandas does not sum all nan's to zero
            row_lev = np.abs(weights).apply(axis=1, func=np.nansum)

            # divide by leverage
            pos_weights = weights.divide(row_lev, axis=0)

        elif leverage.startswith("net"):
            # deleverage positive and negative positions separately
            row_lev_pos = weights.where(weights > 0)\
                .apply(axis=1, func=np.nansum)
            row_lev_neg = -1 * weights.where(weights < 0)\
                .apply(axis=1, func=np.nansum)

            # divide by leverage
            pos_weights = \
                weights.where(weights < 0).divide(
                    row_lev_neg, axis=0).fillna(
                        weights.where(weights > 0).divide(
                            row_lev_pos, axis=0))

        elif leverage.startswith("unlim"):
            pos_weights = np.sign(weights)

        else:
            raise NotImplementedError("Leverage not known!")

        return pos_weights

    def position_flags_to_weights(self, position_flags, leverage="net"):
        """
        leverage : str
            "unlimited" for allowing unlimited leverage
            "net" for restricting it to 1
            "zero" for restricting that short and long positions net out to
                no-leverage
        """
        # position flags need be of a suitable type
        position_flags = position_flags.astype(float)

        res = self.rescale_weights(position_flags, leverage=leverage)

        return res

    @staticmethod
    def position_weights_to_actions(weights):
        """Get actions from (deleveraged if needed) position weights."""

        # the very first row must be all zeros: the day before does not
        #   exist to open a position on
        weights.iloc[0, :] *= np.nan

        # reinstall zeros where nan's are
        weights = weights.fillna(value=0.0)

        # actions are changes in positions: to realize a return on day t, a
        #   position on t-1 should be opened
        actions = weights.diff().shift(-1)

        return actions


class FXTradingStrategy:
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
    def actions(self):
        return self._actions

    @actions.getter
    def actions(self):
        if self._actions is None:
            self._actions = StrategyFactory().position_weights_to_actions(
                self.position_weights)

        return self._actions

    @actions.setter
    def actions(self, value):
        self._actions = value

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
                StrategyFactory().position_flags_to_weights(
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
                   leverage="net", **kwargs):
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
        kwargs : dict
            additional arguments to portfolio_construction.rank_sort()

        Returns
        -------
        res : FXTradingStrategy

        """
        # sort
        pf = poco.rank_sort(sort_values, sort_values, n_portfolios,
                            legsize, **kwargs)

        # retain only 'portfolio...' keys
        pf = {k: v for k, v in pf.items() if k.startswith("portfolio")}

        # number of n_portfolios
        n_pf = len(pf)

        # long and short flags
        flags_long = pf["portfolio"+str(n_pf)].notnull().where(
            pf["portfolio" + str(n_pf)].notnull()) * 1

        flags_short = pf["portfolio1"].notnull().where(
            pf["portfolio1"].notnull()) * -1

        # concatenate
        flags = flags_long.fillna(flags_short)

        # need to open and close it somewhere
        flags.iloc[0, :] = np.nan
        flags.iloc[-1, :] = np.nan

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
        new_strat = FXTradingStrategy(position_flags=new_pos_weights,
                                      leverage="net")
        new_strat.position_flags = \
            new_strat.position_weights / new_strat.position_weights * \
            np.sign(new_strat.position_weights)

        return new_strat


class FXTradingEnvironment:
    """
    """
    def __init__(self, spot_prices, swap_points, counter_currency="usd"):
        """
        Parameters
        ----------
        spot_prices : pandas.DataFrame
            in units of a common counter name, which is then becomes the
            denomination name
            indexed by time, columned by a `MultiIndex` of (bid/ask, name)
        swap_points : pandas.DataFrame
            of swap points, such that swap_points + spot = forward;
            indexed by time, columned by a `MultiIndex` of (bid/ask, name)
        counter_currency : str
        """
        self.spot_prices = spot_prices
        self.swap_points = swap_points

        self.spot_prices.columns.names = ["bid_or_ask", "name"]
        self.swap_points.columns.names = ["bid_or_ask", "name"]

        self.counter_currency = counter_currency.lower()

    @property
    def mid_spot_prices(self):
        return (self.spot_prices["bid"] + self.spot_prices["ask"]) / 2

    @property
    def mid_swap_points(self):
        return (self.swap_points["bid"] + self.swap_points["ask"]) / 2

    @property
    def currencies(self):
        return self.spot_prices.columns.get_level_values("name").unique()

    @classmethod
    def from_scratch(cls, spot_prices, swap_points=None, **kwargs):
        """

        Parameters
        ----------
        spot_prices : dict or pandas.DataFrame
            see FXTradingEnvironment
        swap_points : dict or pandas.DataFrame
            see FXTradingEnvironment
        kwargs : Any

        Returns
        -------
        res : FXTradingEnvironment

        """
        if isinstance(spot_prices, dict):
            if not all([p in ["bid", "ask"] for p in spot_prices.keys()]):
                raise ValueError("When providing a dictionary, it must " +
                                 "include 'bid' and 'ask' dataframes!")
            spot_prices = pd.concat(spot_prices, axis=1)
        else:
            raise ValueError(
                "class of spot_prices: " +
                str(spot_prices.__class__) +
                " not implemented!")

        if swap_points is None:
            swap_points = spot_prices * 0.0
        elif isinstance(swap_points, dict):
            if not all([p in ["bid", "ask"] for p in swap_points.keys()]):
                raise ValueError("When providing a dictionary, it must " +
                                 "include 'bid' and 'ask' dataframes!")
            swap_points = pd.concat(swap_points, axis=1)
        else:
            raise ValueError(
                "class of swap_points: " +
                str(swap_points.__class__) +
                " not implemented!")

        res = cls(spot_prices, swap_points, **kwargs)

        return res

    @staticmethod
    def spread_scaler(df_ba, df_mid, scale):
        # calculate spread
        ba_spread = pd.concat(
            {p: df_ba[p] - df_mid for p in ["bid", "ask"]},
            axis=1
        )

        # scale spread
        ba_spread_scaled = ba_spread * scale

        # reinstall
        df_new = df_mid + ba_spread_scaled

        return df_new

    def with_scaled_costs(self, scale=1.0):
        """Get a version of `self` with scaled bid-ask spread.

        Returns
        -------
        res : FXTradingEnvironment

        """
        res = FXTradingEnvironment(
            spot_prices=self.spread_scaler(self.spot_prices,
                                           self.mid_spot_prices,
                                           scale=scale),
            swap_points=self.spread_scaler(self.swap_points,
                                           self.mid_swap_points,
                                           scale=scale)
        )

        return res

    def remove_swap_outliers(self):
        """
        """
        res = self.swap_points.where(
            np.abs(self.swap_points) < self.swap_points.std()*25
        )

        self.swap_points = res

    def remove_bid_ask_violation(self, correct=False):
        """
        """
        violations = pd.concat(
            {c: (self.spot_prices["ask"] - self.spot_prices["bid"]) < 0
             for c in ["bid", "ask"]},
            axis=1, names=self.spot_prices.columns.names
        )

        if correct:
            spot_ba = pd.concat(
                {p: self.spot_prices[p] - self.mid_spot_prices
                 for p in ["bid", "ask"]},
                axis=1
            )
            swap_ba = pd.concat(
                {p: self.swap_points[p] - self.mid_swap_points
                 for p in ["bid", "ask"]},
                axis=1
            )
            self.spot_prices = self.spot_prices.where(~violations).fillna(
                self.mid_spot_prices + spot_ba.rolling(252, min_periods=1)\
                    .mean().shift(1)
            )
            self.swap_points = self.swap_points.where(~violations).fillna(
                self.mid_swap_points + swap_ba.rolling(252, min_periods=1)\
                    .mean().shift(1)
            )
        else:
            self.spot_prices = self.spot_prices.where(~violations)
            self.swap_points = self.swap_points.where(~violations)

    def align_spot_and_swap(self):
        """
        """
        self.swap_points, self.spot_prices = \
            self.swap_points.align(self.spot_prices, axis=0, join="inner")

    def fillna(self, which="swap_points", **kwargs):
        """
        """
        if which == "spot_prices":
            self.spot_prices.fillna(inplace=True, **kwargs)
        elif which == "swap_points":
            self.swap_points.fillna(inplace=True, **kwargs)
        elif which == "both":
            self.swap_points.fillna(inplace=True, **kwargs)
            self.spot_prices.fillna(inplace=True, **kwargs)
        else:
            raise ValueError("Only 'swap_points', 'spot_prices' and 'both' " +
                             "are acceptable as arguments!")

    def reindex_with_freq(self, reindex_freq=None):
        """
        Parameters
        ----------
        reindex_freq : str
            pandas frequency string, e.g. 'B' for business day
        kwargs : dict
            arguments to .fillna()

        """
        new_idx = pd.date_range(self.swap_points.index[0],
                                self.swap_points.index[-1],
                                freq=reindex_freq)

        self.swap_points = self.swap_points.reindex(index=new_idx)

        new_idx = pd.date_range(self.spot_prices.index[0],
                                self.spot_prices.index[-1],
                                freq=reindex_freq)

        self.spot_prices = self.spot_prices.reindex(index=new_idx)

    def drop(self, *args, **kwargs):
        """
        """
        self.spot_prices = self.spot_prices.drop(*args, axis=1,
                                                 level="name",
                                                 **kwargs)
        self.swap_points = self.swap_points.drop(*args, axis=1,
                                                 level="name",
                                                 **kwargs)

    def to_another_counter_currency(self, new_counter_currency):
        """

        Parameters
        ----------
        new_counter_currency

        Returns
        -------

        """
        if new_counter_currency.lower() == self.counter_currency:
            return self

        s = np.log(self.spot_prices)
        f = np.log(self.spot_prices + self.swap_points)

        s_new = pd.concat({
            bid_or_ask: into_currency(grp[bid_or_ask],
                                      new_cur=new_counter_currency,
                                      counter_cur=self.counter_currency)
            for bid_or_ask, grp in s.groupby(axis=1, level=0)
        }, axis=1, names=s.columns.names)
        ask = s_new.loc[:, ("ask", self.counter_currency)].copy()
        bid = s_new.loc[:, ("bid", self.counter_currency)].copy()
        s_new.loc[:, ("ask", self.counter_currency)] = bid
        s_new.loc[:, ("bid", self.counter_currency)] = ask

        f_new = pd.concat({
            bid_or_ask: into_currency(grp[bid_or_ask],
                                      new_cur=new_counter_currency,
                                      counter_cur=self.counter_currency)
            for bid_or_ask, grp in f.groupby(axis=1, level=0)
        }, axis=1, names=f.columns.names)
        ask = f_new.loc[:, ("ask", self.counter_currency)]\
            .rename(columns={"ask": "bid"})
        bid = f_new.loc[:, ("bid", self.counter_currency)] \
            .rename(columns={"bid": "ask"})
        f_new.loc[:, ("ask", self.counter_currency)] = bid
        f_new.loc[:, ("bid", self.counter_currency)] = ask

        res = FXTradingEnvironment(
            spot_prices=np.exp(s_new),
            swap_points=np.exp(f_new) - np.exp(s_new),
            counter_currency=new_counter_currency).with_scaled_costs(0.5)

        return res


class FXTrading:
    """
    """
    def __init__(self, environment, strategy, settings=None):
        """
        """
        # check if prices are not missing where actions are not
        self.check_alignment(prices=environment.spot_prices,
                             actions=strategy.actions)

        # attributes
        self.strategy = strategy
        self.actions = strategy.actions
        self.prices = environment.spot_prices.copy()
        self.swap_points = environment.swap_points.copy()
        self.settings = settings

        # collect positions into portfolio
        positions = [FXPosition(currency=p) for p in self.actions.columns]
        portfolio = FXPortfolio(positions=positions)

        self.portfolio = portfolio

    @staticmethod
    def check_alignment(prices, actions):
        """Check if prices are not missing where actions are not missing.

        Parameters
        ----------
        prices : pandas.DataFrame
        actions : pandas.DataFrame

        Returns
        -------

        """
        # get rid of zeros in actions: these play no role
        act_no_zeros = actions.replace(0.0, np.nan).dropna(how="all")

        # reindex to have prices of same dimensions as actions
        prc_reix = prices.reindex(
            index=act_no_zeros.index,
            columns=pd.MultiIndex.from_product(
                [prices.columns.get_level_values(0).unique(),
                 act_no_zeros.columns]
            )
        )

        # need ask quotes where actions are positive
        ask_prc = prc_reix["ask"].where(act_no_zeros > 0)
        buy_sig = act_no_zeros.where(act_no_zeros > 0)

        # assert
        if not ask_prc.notnull().equals(buy_sig.notnull()):
            raise ValueError("There are buy signals where ask prices are "
                             "missing!")

        # need bid quotes where actions are negative
        bid_prc = prc_reix["bid"].where(act_no_zeros < 0)
        sell_sig = act_no_zeros.where(act_no_zeros < 0)

        # assert
        if not bid_prc.notnull().equals(sell_sig.notnull()):
            raise ValueError("There are sell signals where bid prices are "
                             "missing!")

    def backtest(self, method="balance"):
        """ Backtest a strategy based on self.`signals`.

        Routine, for each date t:
        1) take changes in positions
        2) rebalance accordingly
        3) roll after rebalancing
        4) record balance / liquidation value

        Parameters
        ----------
        method : str
            'balance' (smoother) or 'unrealized_pnl' (more jagged)

        Returns
        -------
        payoff : pandas.Series
            cumulative payoff, starting at 1
        """
        # allocate space for liquidation values
        payoff = pd.Series(index=self.actions.index)

        # loop over time and position weights
        for t, row in self.actions.iterrows():

            # if t < self.prices.major_axis[0]:
            #     continue
            if t > self.prices.index[-1]:
                return payoff

            # if t > pd.Timestamp("2005-06-08"):
            #     print("Stop right there! Criminal scum!")

            # fetch prices and swap points ----------------------------------
            these_prices = self.prices.loc[t, :].unstack(level=0)
            these_swap_points = self.swap_points.loc[t, :].unstack(level=0)

            # rebalance -----------------------------------------------------
            self.portfolio.rebalance_from_dpw(row, these_prices)

            # roll over -----------------------------------------------------
            self.portfolio.roll_over(these_swap_points)

            # get liquidation value -----------------------------------------
            if method == "balance"[:len(method)]:
                res = self.portfolio.balance
            elif method == "unrealized_pnl"[:len(method)]:
                res = self.portfolio.get_margin_closeout_value(these_prices)
            else:
                raise ValueError("Unknown method! Choose 'balance' or "
                                 "'unrealized_pnl' instead.")

            payoff.loc[t] = res

        return payoff

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
        self.strategy.position_flags.to_excel(writer,
                                              sheet_name='position_flags')
        self.strategy.position_weights.to_excel(writer,
                                                sheet_name='position_weights')
        self.strategy.actions.to_excel(writer, sheet_name='actions')

        self.prices.loc["ask", :, :].to_excel(writer, sheet_name='p_ask')
        self.prices.loc["bid", :, :].to_excel(writer, sheet_name='p_bid')
        self.swap_points.loc["ask", :, :].to_excel(writer,
                                                   sheet_name='swap_ask')
        self.swap_points.loc["bid", :, :].to_excel(writer,
                                                   sheet_name='swap_bid')

        # save and close
        writer.save()

        return

    def add_junior_signals(self, signals):
        """Restimate self.position_weights, self.position_flags, and
         self.actions given a dataframe of signals that do not override the
         original ones

        Parameters
        ----------
        signals: pd.DataFrame
            of signals junior (i.e. not overriding) to the original ones

        Returns
        -------

        """
        # Process the new signals similarly to the old ones
        signals = signals.replace(0.0, np.nan)
        position_flags = signals.reindex(index=self.prices.major_axis)
        # shift by blackout
        position_flags = position_flags.shift(-1*self.settings["blackout"])

        # fill into the past
        position_flags = position_flags.fillna(
            method="bfill",
            limit=self.settings["holding_period"]-1)

        # Merge with the old position flags
        self.position_flags = self.position_flags.fillna(position_flags)

        # Adjust the attributes accordingly
        # in case of equally-weighted no-leverage portfolio, position weights
        #   are signals divided by the absolute sum thereof
        # also, make sure that pandas does not sum all nan's to zero
        row_leverage = np.abs(self.position_flags).apply(axis=1,
                                                         func=np.nansum)
        self.position_weights = \
            self.position_flags.divide(row_leverage, axis=0)
        # reinstall zeros where nan's are
        self.position_weights = self.position_weights.fillna(value=0.0)

        # actions are changes in positions: to realize a return on day t, a
        #   position on t-1 should be opened.
        self.actions = self.position_weights.diff().shift(-1)


class FXPortfolio:
    """Portfolio of FX positions.

    Parameters
    ----------
    positions : list-like
        of Position objects
    """
    def __init__(self, positions):
        """
        """
        # start with balance = 1.0
        self.balance = 1.0
        # currencies
        self.currencies = [p.name for p in positions]
        # represent positions as a pandas.Series
        self.positions = pd.Series(
            data=positions,
            index=self.currencies).rename("positions")
        # position weights: start with 0 everywhere
        self.position_weights = pd.Series(0.0, index=self.currencies)

    def get_margin_closeout_value(self, bid_ask_prices):
        """Calculate margin closeout: how much value can be in open positions.

        Parameters
        ----------
        bid_ask_prices : pd.DataFrame
            of quotes, indexed by name names, column'ed by ['bid', 'ask']
        """
        # 1) from each position, get unrealized p&l
        # concat positions and prices (index by name names)
        both = pd.concat((self.positions, bid_ask_prices), axis=1)

        # function to apply over rows
        func = lambda x: x["positions"].get_unrealized_pnl(x.drop("positions"))

        # for all positions, apply the func to get unrealized p&l
        res = both.apply(func=func, axis=1).sum()
        # pf = FXPortfolio(positions=positions)
        # pf.positions.apply(func=func).sum()
        # both = pd.concat((positions, prices), axis=1)

        # 2) add balance
        res += self.balance

        return res

    def rebalance_from_dpw(self, dpw, bid_ask_prices):
        """Rebalance from change in position weights.
        0. skip if changes in position weights are all zero
        i. close positions to be closed, drain realized p&l onto balance
        ii. determine liquidation value: this much can be reallocated
        iii. determine how much is to be bought/sold, in units of base name
        iv. transact in each position accordingly

        Parameters
        ----------
        dpw : pandas.Series
            indexed by currencies
        bid_ask_prices : pandas.DataFrame
            indexed by name names, columned with ['bid', 'ask']

        Returns
        -------

        """
        # if no change is necessary, skip
        if (dpw == 0.0).all():
            return

        # get margin closeout value
        # TODO: mid price is suggested by OANDA
        # prices_mid = pd.concat([bid_ask_prices.mean(axis=0).to_frame().T, ]*2,
        #                        axis=0)
        # prices_mid.index = bid_ask_prices.index

        prices_mid = pd.concat({"bid": bid_ask_prices.mean(axis=1),
                                "ask": bid_ask_prices.mean(axis=1)},
                               axis=1)
        marg_closeout = self.get_margin_closeout_value(prices_mid)

        # new position weights
        new_pw = self.position_weights + dpw

        # determine dp
        # TODO: mid price?
        new_p = (new_pw * marg_closeout).divide(prices_mid.mean(axis=1))
        dp = new_p - self.get_position_quantities()

        dp = dp.mask(dpw == 0, 0)

        # rebalance
        self.rebalance_from_dp(dp=dp, bid_ask_prices=bid_ask_prices)

        # drag position weights
        self.position_weights = new_pw

    def rebalance_from_dp(self, dp, bid_ask_prices):
        """ Reabalance given differences in position quantities.
        Parameters
        ----------
        dp : pandas.Series
            indexed with name names, e.g. 'aud'
        bid_ask_prices : pandas.DataFrame
            indexed with name names
        """
        # loop over names of dp, transact in respective name
        for cur, p in dp.iteritems():
            new_r = self.positions[cur].transact(p, bid_ask_prices.loc[cur])

            if new_r is not None:
                self.balance += new_r

        return

    def roll_over(self, bid_ask_points):
        """Roll over all positions.

        Parameters
        ----------
        bid_ask_points : pandas.DataFrame
            indexed by ['bid', 'ask'], columned by position names
        """
        for cur, p in self.positions.iteritems():
            p.roll_over(bid_ask_points.loc[cur])

    def get_position_quantities(self):
        """
        """
        res = self.positions.map(lambda x: x.quantity)

        return res

    @staticmethod
    def fxmul(main, other):
        """
        """
        masked_ba = other["bid"].mask(main > 0, other["ask"])
        res = main.mul(masked_ba)

        return res

    @staticmethod
    def fxdivide(main, other):
        """
        """
        masked_ba = other["bid"].mask(main > 0, other["ask"])
        res = main.divide(masked_ba)

        return res


class FXPosition:
    """Position in foreign name vs.

    Parameters
    ----------
    currency : str
        a string, most logically a 3-letter ISO such as 'usd' or 'cad'
    """
    def __init__(self, currency="xxx", tolerance=1e-6):
        """
        """
        self._tolerance = tolerance

        self.currency = currency

        # see property below
        self.quantity = 0
        self.avg_price = 0
        self.expiry_dt = None
        self.open_dt = None

    @property
    def is_open(self):
        return self.quantity != 0

    # @is_open.setter
    # def is_open(self, value):
    #     if not value:
    #         self.quantity = 0.0
    #         self.avg_price = 0.0
    #     else:
    #         raise ValueError("Impossible operation; use .transact() instead.")

    def open(self, open_dt, quantity, open_price, expiry_dt):
        """

        Parameters
        ----------
        open_dt
        quantity
        open_price
        expiry_dt

        Returns
        -------

        """
        assert quantity != 0.0

        self.open_dt = open_dt
        self.quantity = quantity
        self.expiry_dt = expiry_dt
        self.avg_price = open_price

    @property
    def position_sign(self):
        if not self.is_open:
            return np.nan
        return np.sign(self.quantity)

    def __str__(self):
        """
        """
        if not self.is_open:
            return self.currency + ' ' + "(closed)"

        qty = np.round(self.quantity, 4)

        s = '{} {:.4f} {}'.format("long" if self.quantity > 0 else "short",
                                  qty,
                                  self.currency)

        return s

    @staticmethod
    def choose_bid_or_ask(qty, bid_ask_prices):
        """Choose bid or ask depending on the sign of `quantity`

        Parameters
        ----------
        qty : float
            quantity
        bid_ask_prices : pandas.Series
            indexed by ['bid', 'ask']

        Returns
        -------
        res : float
            either bid or ask price from `bid_ask_prices`
        """
        if qty == 0:
            return np.nan

        res = bid_ask_prices["ask"] if qty > 0 else bid_ask_prices["bid"]

        return res

    def transact(self, quantity, bid_ask_prices, fee):
        """Transact.

        Discriminates between cases of (partial) closing and refilling, since
        closing is associated with p&l.

        Parameters
        ----------
        quantity : float
        bid_ask_prices : float
        feen : float

        """
        if np.abs(quantity) < self._tolerance:
            return

        if (self.position_sign == np.sign(quantity)) | (not self.is_open):
            # refill if position not open or if signs match
            res = self.refill(quantity, self.choose_bid_or_ask(), fee)

        else:
            # otherwise partially close, or close and reopen
            if (np.abs(self.quantity) - np.abs(quantity)) < self._tolerance:
                rest_qty = quantity + self.quantity
                res = self.close(bid_ask_prices)
                self.transact(rest_qty, bid_ask_prices)
            else:
                res = self.drain(quantity, price)

        return res

    def refill(self, quantity, price, fee=0.0) -> float:
        """
        cannot result in new quantity being 0

        returns the negative of the fee paid

        Parameters
        ----------
        quantity : float
        price : float
        fee : float

        Returns
        -------
        float
        """
        # if np.abs(quantity) < self._tolerance:
        #     return

        assert (self.position_sign == np.sign(quantity)) | (not self.is_open)

        # total quantity
        total_qty = self.quantity + quantity

        self.avg_price = (self.avg_price * self.quantity + price * quantity) / \
            total_qty

        self.quantity += quantity

        return -fee

    def drain(self, quantity, price, fee=0.0):
        """Partially close the position.

        Returns
        -------
        res : float
            p&l, if any
        """
        # if np.abs(quantity) < self._tolerance:
        #     return 0.0
        assert (self.position_sign != np.sign(quantity)) & self.is_open

        if np.abs(self.quantity) < np.abs(quantity):
            raise ValueError("Amount drained exceed position size; use " +
                             ".transact() instead")

        # calculate p&l
        res = np.abs(quantity) * (price - self.avg_price) * self.position_sign

        # change quantity
        self.quantity += quantity

        if np.abs(self.quantity) < self._tolerance:
            self.quantity = 0.0

        return res

    def close(self, bid_ask_prices):
        """
        """
        if not self.is_open:
            return 0.0

        # deterimine transaction price: buy at ask, sell at bid
        price = self.choose_bid_or_ask(-1*self.quantity, bid_ask_prices)

        res = self.drain(-1*self.quantity, price)

        self.is_open = False

        return res

    def roll_over(self, bid_ask_points):
        """Roll over.

        Changes thje average price.

        Parameters
        ----------
        bid_ask_prices : pandas.Series
            indexed by ['bid', 'ask']
        """
        # deterimine transaction points: buy at ask, sell at bid
        points = self.choose_bid_or_ask(self.quantity, bid_ask_points)

        if not self.is_open:
            return

        # add swap points to the average price
        self.avg_price += points

        return

    def get_unrealized_pnl(self, bid_ask_prices):
        """Get the unrealized profit and loss.

        Parameters
        ----------
        bid_ask_prices: pd.Series
            indexed with ['bid', 'ask']

        Returns
        -------
        res: float
            with unrealized profit and loss of the position
        """
        if not self.is_open:
            return 0.0

        # deterimine transaction price: buy at ask, sell at bid
        price = self.choose_bid_or_ask(-1*self.quantity, bid_ask_prices)

        res = self.quantity * (price - self.avg_price)

        return res


if __name__ == "__main__":
    pass
