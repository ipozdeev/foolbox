import pandas as pd
import numpy as np
import warnings
from foolbox import portfolio_construction as poco


class StrategyFactory():
    """Factory for constructing strategies."""
    def __init__(self):
        """
        """
        pass

    @staticmethod
    def deleverage_position_flags(position_flags, method="net"):
        """Deleverage position flags.

        Parameters
        ----------
        method : str
            'zero' to make position weights sum up to zero;
            'net' to make absolute weights of short and long positions sum up
                to one each (sum(abs(negative positions)) = 1);
            'unlimited' for unlimited leverage.
        """
        if method not in ("zero", "net", "unlimited"):
            raise ValueError("leverage not known!")

        # weights are understood to be fractions of portfolio value
        if method == "zero":
            # NB: dangerous, because meaningless whenever there are long/sort
            #   positions only
            # make sure that pandas does not sum all nan's to zero
            row_lev = np.abs(position_flags).apply(axis=1, func=np.nansum)

            # divide by leverage
            pos_weights = position_flags.divide(row_lev, axis=0)

        elif method == "net":
            # deleverage positive and negative positions separately
            row_lev_pos = position_flags.where(position_flags > 0)\
                .apply(axis=1, func=np.nansum)
            row_lev_neg = -1*position_flags.where(position_flags < 0)\
                .apply(axis=1, func=np.nansum)

            # divide by leverage
            pos_weights = \
                position_flags.where(position_flags < 0).divide(
                    row_lev_neg, axis=0).fillna(
                position_flags.where(position_flags > 0).divide(
                    row_lev_pos, axis=0))

        elif method == "unlimited":
            pos_weights = position_flags.copy()

        return pos_weights

    @staticmethod
    def position_weights_to_actions(position_weights):
        """Get actions from (deleveraged if needed) position weights."""

        # the very first row must be all zeros: the day before does not
        #   exist to open a position on
        position_weights.iloc[0, :] *= np.nan

        # reinstall zeros where nan's are
        position_weights = position_weights.fillna(value=0.0)

        # actions are changes in positions: to realize a return on day t, a
        #   position on t-1 should be opened
        actions = position_weights.diff().shift(-1)

        return actions


class FXTradingStrategy():
    """
    """
    def __init__(self, actions, position_flags=None, position_weights=None):
        """
        """
        self.actions = actions
        self._position_flags = position_flags
        self._position_weights = position_weights

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
        if self._position_weights is None:
            raise ValueError("No position weights specified!")
        return self._position_weights

    @position_weights.setter
    def position_weights(self, value):
        self._position_weights = value

    @classmethod
    def from_position_flags(cls, position_flags, leverage="net"):
        """
        leverage : str
            "unlimited" for allowing unlimited leverage
            "net" for restricting it to 1
            "zero" for restricting that short and long positions net out to
                no-leverage
        """
        # position flags need be of a suitable type
        p_flags = position_flags.astype(float)

        pos_weights = StrategyFactory().deleverage_position_flags(
            p_flags, method=leverage)
        actions = StrategyFactory().position_weights_to_actions(
            pos_weights)

        strat = cls(actions=actions)
        strat.position_weights = pos_weights
        strat.position_flags = position_flags

        return strat

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
        # TODO: relevant?
        position_flags.iloc[0, :] = np.nan
        position_flags.iloc[-1, :] = np.nan

        # pos_weights = StrategyFactory().deleverage_position_flags(
        #     position_flags, method=leverage)
        # actions = StrategyFactory().position_weights_to_actions(
        #     pos_weights)
        #
        # strat = cls(actions=actions)
        # strat.position_weights = pos_weights
        # strat.position_flags = position_flags
        #
        # return strat

        return cls.from_position_flags(position_flags, leverage)

    @classmethod
    def monthly_from_daily_signals(cls, signals_d, agg_fun=None, n_portf=3,
        leverage="net"):
        """
        Parameters
        ----------
        log_fdisc : pd.DataFrame
            of approximate interest rate differentials

        Returns
        -------
        strat : FXTradingStrategy
        """
        if agg_fun is None:
            agg_fun = lambda x: x.iloc[-1,:]

        signals_m = signals_d.resample('M').apply(agg_fun)
        pf = poco.rank_sort(signals_m, signals_m.shift(1), n_portf)

        short_leg = -1*pf["portfolio1"].notnull().where(
            pf["portfolio1"]).astype(float)
        long_leg = pf["portfolio"+str(n_portf)].notnull().where(
            pf["portfolio"+str(n_portf)]).astype(float)
        both_legs = short_leg.fillna(long_leg)

        # reindex, bfill
        position_flags = both_legs.reindex(index=signals_d.index,
            method="bfill")

        # need to open and close it somewhere
        position_flags.iloc[0,:] = np.nan
        position_flags.iloc[-1,:] = np.nan

        # # construct strategy
        # pos_weights = StrategyFactory().deleverage_position_flags(
        #     position_flags, method=leverage)
        # actions = StrategyFactory().position_weights_to_actions(
        #     pos_weights)
        #
        # strat = cls(actions=actions)
        # strat.position_weights = pos_weights
        # strat.position_flags = position_flags
        #
        # return strat

        return cls.from_position_flags(position_flags, leverage)

    @classmethod
    def from_long_short(cls, portfolios, leverage="net"):
        """

        Parameters
        ----------
        portfolios : dict
            output of portfolio_construction.rank_sort()
        leverage : str
            'net', 'unlimited' or 'zero'

        Returns
        -------
        res : FXTradingStrategy

        """
        # retain only 'portfolio...' keys
        pf = {k: v for k, v in portfolios.items() if k.startswith("portfolio")}

        # number of portfolios
        n_pf = len(pf)

        # long and short flags
        flags_long = pf["portfolio"+str(n_pf)].notnull().where(
            pf["portfolio" + str(n_pf)].notnull()) * 1

        flags_short = pf["portfolio1"].notnull().where(
            pf["portfolio1"].notnull()) * -1

        # concatenate
        flags = flags_long.fillna(flags_short)

        res = cls.from_position_flags(flags, leverage=leverage)

        return res

    def __add__(self, other):
        """Combine two strategies.
        A + B means strategy A is taken as the base strategy and 'enhanced'
        with strategy B.

        Not commutative!
        """
        # fill position weights with those of `other`
        new_pos_weights = other.position_weights.fillna(self.position_weights)

        # the new strategy is a strategy with the above compbo and unlimited
        #   leverage
        new_strat = FXTradingStrategy.from_position_flags(new_pos_weights,
            leverage="net")
        new_strat.position_flags = \
            new_strat.position_weights / new_strat.position_weights * \
            np.sign(new_strat.position_weights)

        return new_strat


class FXTradingEnvironment():
    """
    """
    def __init__(self, spot_prices, swap_points):
        """
        Parameters
        ----------
        spot_prices : pd.Panel
            with assets along the minor axis, time on major_axis
        swap_points : pd.Panel
            with assets along the minor axis, time on major_axis
        """
        self.spot_prices = spot_prices
        self.swap_points = swap_points

    @property
    def mid_spot_prices(self):
        return (self.spot_prices["bid"] + self.spot_prices["ask"]) / 2

    @property
    def mid_swap_points(self):
        return (self.swap_points["bid"] + self.swap_points["ask"]) / 2

    @property
    def currencies(self):
        return self.spot_prices.minor_axis

    @classmethod
    def from_scratch(cls, spot_prices, swap_points=None):
        """
        """
        if isinstance(spot_prices, dict):
            if not all([p in ["bid", "ask"] for p in spot_prices.keys()]):
                raise ValueError("When providing a dictionary, it must "+
                    "include 'bid' and 'ask' dataframes!")
        elif isinstance(spot_prices, pd.DataFrame):
            spot_prices = {
                "bid": spot_prices,
                "ask": spot_prices}
        else:
            raise ValueError(
                "class of spot_prices: " +
                str(spot_prices.__class__) +
                " not implemented!")

        if swap_points is None:
            swap_points = {
                "bid": spot_prices["bid"]*0.0,
                "ask": spot_prices["bid"]*0.0}
        elif isinstance(swap_points, dict):
            if not all([p in ["bid", "ask"] for p in swap_points.keys()]):
                raise ValueError("When providing a dictionary, it must "+
                    "include 'bid' and 'ask' dataframes!")
        elif isinstance(swap_points, pd.DataFrame):
            swap_points = {
                "bid": swap_points,
                "ask": swap_points}
        else:
            raise ValueError(
                "class of swap_points: " +
                str(swap_points.__class__) +
                " not implemented!")

        spot_prices = pd.Panel.from_dict(spot_prices, orient="items")
        swap_points = pd.Panel.from_dict(swap_points, orient="items")

        res = cls(spot_prices, swap_points)

        return res

    def remove_swap_outliers(self):
        """
        """
        res = {}
        for k, v in dict(self.swap_points).items():
            res[k] = v.where(np.abs(v) < v.std()*25)

        self.swap_points = pd.Panel.from_dict(res, orient="items")

    def align_spot_and_swap(self):
        """
        """
        common_idx = self.swap_points.major_axis.union(
            self.spot_prices.major_axis)

        self.swap_points = self.swap_points.reindex(major_axis=common_idx)
        self.spot_prices = self.spot_prices.reindex(major_axis=common_idx)

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
            raise ValueError("Only 'swap_points', 'spot_prices' and 'both' "+
                "are acceptable as arguments!")

    def reindex_with_freq(self, reindex_freq=None, **kwargs):
        """
        Parameters
        ----------
        reindex_freq : str
            pandas frequency string, e.g. 'B' for business day
        kwargs : dict
            arguments to .fillna()

        """
        swap_points_idx = self.swap_points.major_axis
        new_idx = pd.date_range(swap_points_idx[0], swap_points_idx[-1],
            freq=reindex_freq)

        self.swap_points = self.swap_points.reindex(major_axis=new_idx)

        spot_prices_idx = self.spot_prices.major_axis
        new_idx = pd.date_range(spot_prices_idx[0], spot_prices_idx[-1],
            freq=reindex_freq)

        self.spot_prices = self.spot_prices.reindex(major_axis=new_idx)

    def drop(self, **kwargs):
        """
        """
        self.spot_prices = self.spot_prices.drop(**kwargs)
        self.swap_points = self.swap_points.drop(**kwargs)


class FXTrading():
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
        prices : pandas.Panel
        actions : pandas.DataFrame

        Returns
        -------

        """
        # get rid of zeros in actions: these play no role
        act_no_zeros = actions.replace(0.0, np.nan).dropna(how="all")

        # reindex to have prices of same dimensions as actions
        prc_reix = prices.reindex(major_axis=act_no_zeros.index,
                                  minor_axis=act_no_zeros.columns)

        # need ask quotes where actions are positive
        ask_prc = prc_reix.loc["ask", :, :].where(act_no_zeros > 0)
        buy_sig = act_no_zeros.where(act_no_zeros > 0)

        # assert
        if not ask_prc.notnull().equals(buy_sig.notnull()):
            raise ValueError("There are buy signals where ask prices are "
                             "missing!")

        # need bid quotes where actions are negative
        bid_prc = prc_reix.loc["bid", :, :].where(act_no_zeros < 0)
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
            if t > self.prices.major_axis[-1]:
                return payoff

            # if t > pd.Timestamp("2005-06-08"):
            #     print("Stop right there! Criminal scum!")

            # fetch prices and swap points ----------------------------------
            these_prices = self.prices.loc[:, t, :].T
            these_swap_points = self.swap_points.loc[:, t, :].T

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


class FXPortfolio():
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
        self.currencies = [p.currency for p in positions]
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
            of quotes, indexed by currency names, column'ed by ['bid', 'ask']
        """
        # 1) from each position, get unrealized p&l
        # concat positions and prices (index by currency names)
        both = pd.concat((self.positions.to_frame().T, bid_ask_prices), axis=0)

        # function to apply over rows
        func = lambda x: x["positions"].get_unrealized_pnl(x.drop("positions"))

        # for all positions, apply the func to get unrealized p&l
        res = both.apply(func=func, axis=0).sum()
        # pf = FXPortfolio(positions=positions)
        # pf.positions.apply(func=func).sum()
        # both = pd.concat((positions, prices), axis=1)

        # 2) add balance
        res += self.balance

        return res

    def rebalance_from_dpw(self, dpw, bid_ask_prices):
        """ Rebalance from change in position weights.
        0. skip if changes in position weights are all zero
        i. close positions to be closed, drain realized p&l onto balance
        ii. determine liquidation value: this much can be reallocated
        iii. determine how much is to be bought/sold, in units of base currency
        iv. transact in each position accordingly
        """
        # if no change is necessary, skip
        if (dpw == 0.0).all():
            return

        # get margin closeout value
        # TODO: mid price is suggested by OANDA
        prices_mid = pd.concat([bid_ask_prices.mean(axis=0).to_frame().T, ]*2,
            axis=0)
        prices_mid.index = bid_ask_prices.index
        marg_closeout = self.get_margin_closeout_value(prices_mid)

        # new position weights
        new_pw = self.position_weights + dpw

        # determine dp
        # TODO: mid price?
        new_p = (new_pw * marg_closeout).divide(prices_mid.iloc[0, :], axis=0)
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
            indexed with currency names, e.g. "aud"
        """
        # loop over names of dp, transact in respective currency
        for cur, p in dp.iteritems():
            new_r = self.positions[cur].transact(p, bid_ask_prices.loc[:, cur])

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
            p.roll_over(bid_ask_points.loc[:, cur])

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


class FXPosition():
    """Position in foreign currency vs.

    Parameters
    ----------
    currency : str
        a string, most logically a 3-letter ISO such as 'usd' or 'cad'
    """
    def __init__(self, currency="xxx", tolerance=1e-10):
        """
        """
        self._tolerance = tolerance

        self.currency = currency

        # see property below
        self.is_open = False

    @property
    def is_open(self):
        return self.quantity != 0

    @is_open.setter
    def is_open(self, value):
        if not value:
            self.quantity = 0.0
            self.avg_price = 0.0
        else:
            raise ValueError("Impossible operation; use .transact() instead.")

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
        """Choose bid or ask depending on the sign of `qty`

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

    def transact(self, qty, bid_ask_prices):
        """Transact.

        Discriminates between cases of (partial) closing and refilling, since
        closing is associated with p&l.

        Parameters
        ----------
        bid_ask_prices : pandas.Series
            indexed by ['bid', 'ask']
        """
        if np.abs(qty) < self._tolerance:
            return

        # deterimine transaction price: buy at ask, sell at bid
        transaction_price = self.choose_bid_or_ask(qty, bid_ask_prices)

        if (self.position_sign == np.sign(qty)) | (not self.is_open):
            # refill if position not open or if signs match
            res = self.refill(qty, transaction_price)
        else:
            # otherwise partially close, or close and reopen
            if np.abs(self.quantity) - np.abs(qty) < self._tolerance:
                rest_qty = qty + self.quantity
                res = self.close(bid_ask_prices)
                self.transact(rest_qty, bid_ask_prices)
            else:
                res = self.drain(qty, transaction_price)

        return res

    def refill(self, qty, price):
        """
        cannot result in new quantity being 0

        Returns
        -------
        None
        """
        # if np.abs(qty) < self._tolerance:
        #     return

        assert (self.position_sign == np.sign(qty)) | (not self.is_open)

        # total quantity
        total_qty = self.quantity + qty

        self.avg_price = (self.avg_price*self.quantity + price*qty) / total_qty

        self.quantity += qty

        return

    def drain(self, qty, price):
        """Partially close the position.

        Returns
        -------
        res : float
            p&l, if any
        """
        # if np.abs(qty) < self._tolerance:
        #     return 0.0
        assert (self.position_sign != np.sign(qty)) & self.is_open

        if np.abs(self.quantity) < np.abs(qty):
            raise ValueError("Amount drained exceed position size; use " +
                ".transact() instead")

        # calculate p&l
        res = np.abs(qty) * (price - self.avg_price) * self.position_sign

        # change quantity
        self.quantity += qty

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

    from foolbox.api import *
    from foolbox.wp_tabs_figs.wp_settings import settings
    from foolbox.utils import *

    # Set the output path, input data and sample
    start_date = pd.to_datetime(settings["sample_start"])
    end_date = pd.to_datetime(settings["sample_end"])
    drop_curs = settings["drop_currencies"]
    avg_impl_over = settings["avg_impl_over"]
    avg_refrce_over = settings["avg_refrce_over"]
    base_lag = settings["base_holding_h"] + 2
    base_th = settings["base_threshold"]

    # carry
    data_d = pd.read_pickle(data_path+"fx_by_tz_sp_fixed.p")

    fx_tr_env = FXTradingEnvironment.from_scratch(
        spot_prices={
            "bid": data_d["spot_bid"].loc[:, start_date:, "NYC"],
            "ask": data_d["spot_ask"].loc[:, start_date:, "NYC"]},
        swap_points={
            "bid": data_d["tnswap_bid"].loc[:, start_date:, "NYC"],
            "ask": data_d["tnswap_ask"].loc[:, start_date:, "NYC"]}
            )

    fx_tr_env.drop(labels=["dkk"], axis="minor_axis", errors="ignore")
    fx_tr_env.remove_swap_outliers()
    fx_tr_env.reindex_with_freq('B')
    fx_tr_env.align_spot_and_swap()
    fx_tr_env.fillna(which="both", method="ffill")

    fdisc_d = fx_tr_env.swap_points.mean(axis="items")
    spot_d = fx_tr_env.spot_prices.mean(axis="items")

    # dr = -1*np.log((fdisc_d + spot_d)/spot_d).rolling(22).mean()

    # fx_tr_str_carry = FXTradingStrategy.monthly_from_daily_signals(
    #     signals_d=dr, n_portf=3, leverage="net")

    # ois-implied forecasts
    curs = [p for p in spot_d.columns if p not in ["dkk", "nok", "jpy"]]

    # forecast direction
    signals_fcast = get_pe_signals(curs, base_lag, base_th*100, data_path,
        fomc=False,
        avg_impl_over=avg_impl_over,
        avg_refrce_over=avg_refrce_over,
        bday_reindex=True)

    signals_fomc = get_pe_signals(curs + ["nok", "jpy"],
        base_lag, base_th*100, data_path,
        fomc=True,
        avg_impl_over=avg_impl_over,
        avg_refrce_over=avg_refrce_over,
        bday_reindex=True)

    signals_fcast.loc[:, "nok"] = np.nan
    signals_fcast.loc[:, "jpy"] = np.nan

    signals_fcast, signals_fomc = signals_fcast.align(signals_fomc, axis=0,
        join="outer")

    signals_fcast = signals_fcast.loc[start_date:end_date].reindex(
        index=pd.date_range(start_date, end_date, freq='B'))

    strategy_fcast = FXTradingStrategy.from_events(signals_fcast,
        blackout=1, hold_period=10, leverage="net")

    # combined_strat = fx_tr_str_carry + strategy_fcast

    # fx_tr = FXTrading(environment=fx_tr_env, strategy=fx_tr_str_carry)
    fx_tr = FXTrading(environment=fx_tr_env, strategy=strategy_fcast)
    # fx_tr = FXTrading(environment=fx_tr_env, strategy=combined_strat)

    # car = fx_tr.backtest("unrealiz")
    strat = fx_tr.backtest("unrealiz")

    # car.dropna().plot()
    strat.dropna().plot()
    strat = strat.loc[
        strategy_fcast.position_flags.dropna(how="all").index]

    car_r = np.log(car).diff()
    strat_r = np.log(strat).diff()

    pd.concat((car_r, strat_r), axis=1).cumsum().plot()

    pd.concat((car_r, strat_r), axis=1).corr()

    taf.descriptives(np.log(res_unr.dropna()).diff().to_frame()*100, scale=1)

    res_bal.dropna().plot()

    true_carry = poco.get_carry(pickle_name="data_wmr_dev_m",
        key_name="rx",
        n_portf=3)
    car = true_carry.loc[start_date:, "hml"].cumsum()

    pd.concat((
        np.log(res_bal.rename("retail_balance")),
        np.log(res_unr.rename("retail_unr")),
        car.rename("true_carry")), axis=1).ffill().plot()

    fx_tr_str_carry.actions.cumsum().loc[:, "aud"].plot()

    new_strat = fx_tr_str_carry + strategy_fcast

    # data
    prices, swap_points = fetch_the_data(data_path,
        drop_curs=settings["drop_currencies"],
        align=True,
        add_usd=False)

    # some empty frame to prepend the signals with
    empty_month = pd.DataFrame(
        index=pd.date_range(
            start_date-DateOffset(months=1),
            start_date, freq='B'),
        columns=prices.minor_axis).drop(start_date, axis=0)

    # drop redundand stuff
    prices = prices.loc[:,empty_month.index[0]:,:]
    swap_points = swap_points.loc[:,empty_month.index[0]:,:]

    # space for output
    nav_fcast = pd.Panel(
        items=np.arange(1,16),
        major_axis=prices.major_axis,
        minor_axis=np.arange(1,26))
    nav_perf = pd.DataFrame(index=prices.major_axis, columns=np.arange(1,16))

    # the loop
    for h in range(10,16):
        # h = 10
        for p in range(10,26):
            # p = 10
            print("holding horizon: {}, threshold: {}".format(h,p))
            # Get signals for all countries except for the US
            signals = get_pe_signals(prices.minor_axis,
                lag=h+1+settings["base_blackout"],
                threshold=p,
                data_path=data_path,
                fomc=False,
                avg_impl_over=settings["avg_impl_over"],
                avg_refrce_over=settings["avg_refrce_over"],
                bday_reindex=True)

            if add_usd:
                signals_usd = get_pe_signals(prices.minor_axis,
                    lag=h+1+settings["base_blackout"],
                    threshold=p,
                    data_path=data_path,
                    fomc=True,
                    avg_impl_over=settings["avg_impl_over"],
                    avg_refrce_over=settings["avg_refrce_over"],
                    bday_reindex=True)

                signals, signals_usd = signals.align(signals_usd, axis=0)

                signals = signals.fillna(value=0.0).add(
                    signals_usd.fillna(value=0.0))

            # rm to start date
            signals = signals.loc[start_date:,:]

            # add some nan's at the beginning
            signals = pd.concat((empty_month, signals), axis=0)

            trading_settings = {
                "holding_period": h,
                "blackout": settings["base_blackout"]}

            fxtr = FXTrading(
                prices=prices,
                swap_points=swap_points,
                signals=signals,
                settings=trading_settings)

            this_nav = fxtr.backtest()

            nav_fcast.loc[h,:,p] = this_nav


            # useless comment here
            #
            # and here
