import pandas as pd
import numpy as np
import warnings
# import ipdb

class FXTradingStrategy():
    """
    """
    def __init__(self, actions=None):
        """
        """
        self._actions = actions

    @property
    def actions(self):
        return self._actions

    @actions.getter
    def actions(self):
        if self._actions is None:
            raise ValueError("No actions specified!")
        return self._actions

    @actions.setter
    def actions(self, value):
        self._actions = value

    @classmethod
    def from_events(cls, events, blackout, hold_period, leverage="full"):
        """
        Parameters
        ----------
        events : pd.DataFrame
            of events: -1,0,1
        blackout : int
            the period to close position in: negative numbers indicate closing
            before event
        hold_period : int
            the number of periods to maintain position
        leverage : str
            "full" for allowing unlimited leverage
            "none" for restricting it to 1
            "net" for restricting that short and long positions net out to
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
        if position_flags.iloc[:hold_period,:].notnull().any().any():
            warnings.warn("There are not enough observations at the start; " +
                "will delete the first events")
            position_flags.iloc[:hold_period,:] = np.nan

        # NB: dimi/
        # # Wipe the intruders who have events inside the first holding period
        # position_flags.iloc[:self.settings["holding_period"], :] = np.nan
        # /dimi

        # fill into the past
        position_flags = position_flags.fillna(method="bfill",
            limit=hold_period-1)

        # II. position weights
        # weights are understood to be fractions of portfolio value invested
        # in case of equally-weighted no-leverage portfolio, position weights
        #   are events divided by the absolute sum thereof
        if leverage == "none":
            # also, make sure that pandas does not sum all nan's to zero
            row_leverage = np.abs(position_flags).apply(axis=1, func=np.nansum)
            # divide by leverage
            position_weights = position_flags.divide(row_leverage, axis=0)

        elif leverage == "net":
            row_leverage_pos = position_flags.where(position_flags > 0)\
                .apply(axis=1, func=np.nansum)
            row_leverage_neg = -1*position_flags.where(position_flags < 0)\
                .apply(axis=1, func=np.nansum)
            position_weights = position_flags.where(position_flags < 0)\
                .divide(row_leverage_neg, axis=0).fillna(
                position_flags.where(position_flags > 0)\
                        .divide(row_leverage_pos, axis=0))

        elif leverage == "full":
            row_leverage = 1.0
            position_weights = position_flags.copy()

        else:
            raise ValueError("admissible types of leverage are 'none', "+
                "'net' and 'full'!")

        # reinstall zeros where nan's are
        position_weights = position_weights.fillna(value=0.0)

        # actions are changes in positions: to realize a return on day t, a
        #   position on t-1 should be opened
        actions = position_weights.diff().shift(-1)

        res = cls(actions=actions)

        return res

    @classmethod
    def monthly_from_daily_signals(cls, signals_d, agg_fun=None, n_portf=3,
        leverage="net"):
        """
        Parameters
        ----------
        log_fdisc : pd.DataFrame
            of approximate interest rate differentials
        """
        if agg_fun is None:
            agg_fun = lambda x: x.iloc[-1,:]

        signals_m = signals_d.resample('M').apply(agg_fun)
        pf = poco.rank_sort(signals_m, signals_m.shift(1), n_portf)

        short_leg = -1*pf["portfolio1"].notnull().where(
            pf["portfolio1"])
        long_leg = pf["portfolio"+str(n_portf)].notnull().where(
            pf["portfolio"+str(n_portf)])

        both_legs = short_leg.fillna(long_leg)

        # reindex, bfill
        position_flags = both_legs.reindex(index=signals_d.index,
            method="bfill")

        # need to close it somewhere
        position_flags.iloc[0,:] = np.nan
        position_flags.iloc[-1,:] = np.nan

        # to actions
        if leverage == "none":
            # also, make sure that pandas does not sum all nan's to zero
            row_leverage = np.abs(position_flags).apply(axis=1, func=np.nansum)
            # divide by leverage
            position_weights = position_flags.divide(row_leverage, axis=0)

        elif leverage == "net":
            row_leverage_pos = position_flags.where(position_flags > 0)\
                .apply(axis=1, func=np.nansum)
            row_leverage_neg = -1*position_flags.where(position_flags < 0)\
                .apply(axis=1, func=np.nansum)
            position_weights = position_flags.where(position_flags < 0)\
                .divide(row_leverage_neg, axis=0).fillna(
                position_flags.where(position_flags > 0)\
                        .divide(row_leverage_pos, axis=0))

        elif leverage == "full":
            row_leverage = 1.0
            position_weights = position_flags.copy()

        else:
            raise ValueError("admissible types of leverage are 'none', "+
                "'net' and 'full'!")

        actions = position_weights.replace(np.nan, 0.0).diff().shift(-1)

        res = cls(actions=actions)

        return res


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
        Parameters
        ----------
        prices : pandas.Panel
            with currencies along the items axis, time on major_axis
        swap_points : pandas.Panel
            with currencies along the items axis, time on major_axis
        """
        self.prices = environment.spot_prices.copy()
        self.swap_points = environment.swap_points.copy()
        self.actions = strategy.actions

        self.settings = settings

        # collect positions into portfolio
        positions = [FXPosition(currency=p) for p in self.actions.columns]
        portfolio = FXPortfolio(positions=positions)

        self.portfolio = portfolio

        # self.position_flags = position_flags
        # self.position_weights = position_weights

    def backtest(self):
        """ Backtest a strategy based on self.`signals`.

        Routine, for each date t:
        1) take changes in positions
        2) rebalance accordingly
        3) roll after rebalancing
        4) record liquidation value

        """
        # find rows where some action takes place: self.actions or
        #   self.position_flags are not zero
        # good_idx = self.actions.replace(0.0, np.nan).dropna(how="all").index\
        #     .union(
        #     self.position_flags.replace(0.0, np.nan).dropna(how="all").index)

        # ipdb.set_trace()

        # allocate space for liquidation values
        liquidation_v = pd.Series(index=self.prices.major_axis)

        # loop over time and position weights
        for t, row in self.actions.iterrows():

            # if t > pd.to_datetime("2002-01-29"):
            #     ipdb.set_trace()

            # fetch prices and swap points ----------------------------------
            these_prices = self.prices.loc[:,t,:]
            these_swap_points = self.swap_points.loc[:,t,:]

            # rebalance -----------------------------------------------------
            self.portfolio.rebalance_from_dpw(row, these_prices)

            # roll over -----------------------------------------------------
            self.portfolio.roll_over(these_swap_points)

            # get liquidation value -----------------------------------------
            liquidation_v.loc[t] = self.portfolio.get_liquidation_value(
                prices=these_prices)

        return liquidation_v

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
        # self.signals.to_excel(writer, sheet_name='signals')
        # self.position_flags.to_excel(writer, sheet_name='position_flags')
        # self.position_weights.to_excel(writer, sheet_name='position_weights')
        # self.actions.to_excel(writer, sheet_name='actions')

        self.prices.loc["ask",:,:].to_excel(writer, sheet_name='p_ask')
        self.prices.loc["bid",:,:].to_excel(writer, sheet_name='p_bid')
        self.swap_points.loc["ask",:,:].to_excel(writer, sheet_name='swap_ask')
        self.swap_points.loc["bid",:,:].to_excel(writer, sheet_name='swap_bid')

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

    def get_margin_closeout_value(self, prices):
        """Calculate margin closeout: how much value can be in open positions.

        Parameters
        ----------
        prices : pd.DataFrame
            of quotes, indexed by currency names, column'ed by ['bid', 'ask']
        """
        # 1) from each position, get unrealized p&l
        # concat positions and prices (index by currency names)
        both = pd.concat((self.positions, self.prices), axis=1)

        # function to apply over rows
        func = lambda x: x["positions"].get_unrealized_pnl(x.drop("positions"))

        # for all positions, apply the func to get unrealized p&l
        res = self.positions.apply(axis=1, func=func)

        # 2) add balance
        res += self.balance

        return res

    def get_liquidation_value(self, prices):
        """Calculate the total liquidation value of portfolio.

        Liquidation value is the result of closing all open position
        immediately and draining any balance.

        Parameters
        ----------
        prices : pd.DataFrame
            of quotes, indexed by currency names, column'ed by ['bid', 'ask']
        """
        # init result at 0
        res = 0.0

        # loop over positions, for repective currency fetch price & get p&l
        for p in self.positions.values:
            res += p.get_unrealized_pnl(prices.loc[p.currency])

        # drain the balance
        res += self.balance

        return res

    def rebalance_from_dpw(self, dpw, prices):
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

        # new position weights
        new_pw = self.position_weights + dpw

        # 1) the ones that get closed
        to_close_idx = new_pw.ix[(new_pw == 0) & (dpw != 0)].index
        for idx, p in self.positions.ix[to_close_idx].iteritems():
            # close -> realized p&l here
            p.close(prices.loc[idx,:])

        # for each closed position, replenish the balance
        for p in self.positions.ix[to_close_idx].values:
            self.balance += p.realized_pnl
            p.realized_pnl = 0.0

        # recover liquidation value
        lqd_val = self.get_liquidation_value(prices)

        # 2) for each position, calculate additional amount to be transacted
        # total dollar amount
        pos_dol_vals = new_pw * lqd_val
        # total quantity: divide through ask prices for buys, bid for sells
        new_qty = self.fxdivide(pos_dol_vals, prices)
        # current quantities
        cur_qty = self.positions.map(lambda x: x.end_quantity)
        cur_pos_types = self.positions.map(lambda x: x.position_float())
        cur_qty *= cur_pos_types

        # difference in positions
        dp = new_qty - cur_qty

        # rebalance
        for idx, p in self.positions.drop(to_close_idx).iteritems():
            p.transact(dp[idx], prices.loc[idx,:])

        # drag position weights
        self.position_weights = new_pw

    def rebalance_from_dp(self, dp, prices):
        """ Reabalance given differences in position quantities.
        Parameters
        ----------
        dp : pandas.Series
            indexed with currency names, e.g. "aud"
        """
        # loop over names of dp, transact in respective currency
        for idx, p in dp.iteritems():
            self.positions[idx].transact(p, prices.loc[idx,:])

    # def dpw2quantities(self, dpw, prices):
    #     """ Transform position weights to quantitites to be transacted.
    #     Parameters
    #     ----------
    #     new_pw : pandas.Series
    #         of new position weights, indexed with currency names
    #     """
    #     # 1) calculate liquidation value of portfolio
    #     lqd_val = self.get_liquidation_value(prices)
    #
    #     # 3) dollar value of positions
    #     dpw_dollar_val = dpw * lqd_val
    #
    #     # 4) from dollar value to foreign currency quantity
    #     qty = self.fxdivide(dpw_dollar_val, prices)
    #
    #     return qty

    def roll_over(self, swap_points):
        """ Roll over all positions.
        """
        for idx, p in self.positions.iteritems():
            p.roll_over(swap_points.loc[idx,:])

    def get_position_quantities(self):
        """
        """
        res = self.positions.map(lambda x: x.end_quantity)

        # res = pd.Series(data=position_quantities, index=self.currencies)

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
    def __init__(self, currency="xxx"):
        """
        """
        self.currency = currency

        # see property below
        self.is_open = False

    @property
    def is_open(self):
        return self.quantity != 0

    @is_open.setter:
    def is_open(self, value):
        if not(value):
            self.quantity = 0.0
            self.avg_price = 0.0
        else:
            raise ValueError("Impossible operation; use .transact() instead.")

    def __str__(self):
        """
        """
        if not(self.is_open) is None:
            return self.currency + ' ' + "(closed)"

        qty = np.round(self.end_quantity, 4)

        s = '{} {:.4f} {}'.format(
            "long" if self.quantity > 0 else "short",
            self.quantity,
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
        # deterimine transaction price: buy at ask, sell at bid
        transaction_price = self.choose_bid_or_ask(qty, bid_ask_prices)

        if (np.sign(self.quantity) == np.sign(qty)) || not(self.is_open):
            # refill if position not open or if signs match
            res = self.refill(qty, transaction_price)
        else:
            # otherwise partially close, or close and reopen
            if np.abs(self.quantity) < np.abs(qty):
                res = self.close()
                self.refill(qty + self.quantity, transaction_price)
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
        assert (np.sign(self.quantity) == np.sign(qty)) | not(self.is_open)

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
        assert (np.sign(self.quantity) ! np.sign(qty)) & self.is_open

        if np.abs(self.quantity) < np.abs(qty):
            raise ValueError("Amount drained exceed position size; use " +
                ".transact() instead")

        # calculate p&l
        res = qty * (price - self.avg_price)

        # change quantity
        self.quantity += qty

        return res

    def close(self, bid_ask_prices):
        """
        """
        if not(self.is_open):
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

        if not(self.is_open):
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
        if not(self.is_open):
            return 0.0

        # deterimine transaction price: buy at ask, sell at bid
        price = self.choose_bid_or_ask(-1*self.quantity, bid_ask_prices)

        res = self.quantity * (price - self.avg_price)

        return res

# class FXPosition():
#     """Represent opened and closed positions in a foreign currency.
#
#     Parameters
#     ----------
#     currency: str
#         reperesenting counter currency of an FXPosition
#     """
#
#     def __init__(self, currency):
#         """
#         """
#         self.currency = currency
#
#         # Set the initial values for an empty position
#         self.position_type = None  # Set to str 'short' or 'long'
#         #self.initial_price = 0
#         self.initial_quantity = 0
#         self.avg_price = 0
#
#         self.unrealized_pnl = 0
#         self.realized_pnl = 0
#
#         # By default, no transactions take place
#         self.end_quantity = self.initial_quantity
#
#     def __str__(self):
#         """Handle string representation of the object.
#         """
#         if self.position_type is None:
#             return ' '.join((self.currency, "inactive"))
#
#         qty = np.round(self.end_quantity, 4)
#
#         res = ' '.join((self.position_type, str(qty), self.currency))
#
#         return res
#
#     def position_float(self):
#         """
#         """
#         if self.position_type is None:
#             return 0
#
#         return 1 if self.position_type == "long" else -1
#
#     def close(self, prices):
#         """
#         """
#         if self.position_type is None:
#             return
#
#         if self.position_type == "long":
#             self.transact(-1*self.end_quantity, prices)
#         else:
#             self.transact(self.end_quantity, prices)
#
#         self.end_quantity = 0
#         self.avg_price = 0
#
#     def transact(self, quantity, price):
#         """Wrapper around the 'buy()' and 'sell()' methods, recognizing sells
#         as transactions with negative quantity. Furthermore, updates the
#         intial_quantity to end_quantity, allowing for sequential transactions
#
#         Parameters
#         ----------
#         quantity: float
#             units of currency to transact, negative for sells
#         price: pd.Series
#             indexed with 'bid' and 'ask' containing the corresponding price
#             quotes
#
#         Returns
#         -------
#
#         """
#         if quantity < 0:
#             self.sell(-1*quantity, price["bid"])
#         elif quantity > 0:
#             self.buy(quantity, price["ask"])
#         else:
#             Warning("Transacting zero quantity, nothing happens")
#             return
#
#         self.initial_quantity = self.end_quantity
#
#     def buy(self, quantity, price):
#         """Buys 'quantity' units at 'price' price
#
#         Parameters
#         ----------
#         quantity: float
#             number of units of the currency to buy
#         price: float
#             transaction price
#
#         Returns
#         -------
#
#         """
#         # When attempting to transact zero units do nothing, pop up a warning
#         if quantity == 0:
#             Warning("Transacting zero quantity, nothing happens")
#             return
#
#         # If there is no open position open a long one, set the initial price
#         if self.position_type is None:
#             self.position_type = "long"
#             self.avg_price = price
#
#         # If the initial position is long, then buy MOAR
#         if self.position_type == "long":
#             # Increase the quantity
#             self.end_quantity = self.initial_quantity + quantity
#             # Compute VWAP
#             self.avg_price = \
#                 (self.avg_price * self.initial_quantity +
#                  price * quantity) / self.end_quantity
#
#         # If short -- partial close at ask or flip to long
#         else:
#             # If quantity to buy is leq than that available - partial close
#             if self.initial_quantity >= quantity:
#                 # Reduce the quanity
#                 self.end_quantity = self.initial_quantity - quantity
#                 # Intuition: price > avg_price means loss in short position
#                 # NB: what about: initial_price -> avg_price
#                 self.realized_pnl = \
#                     self.realized_pnl - quantity * (price - self.avg_price)
#                 # Average price remains the same
#
#             # Else the position is closed and opened in the opposite direction
#             else:
#                 self.flip(quantity, price)
#
#         # Check the end quantity, render position type to None if nothing left
#         if self.end_quantity == 0:
#             self.position_type = None
#             self.avg_price = 0
#
#     def sell(self, quantity, price):
#         """Sells 'quantity' units at 'price' price
#
#         Parameters
#         ----------
#         quantity: float
#             number of units of the currency to sell
#         price: float
#             transaction price
#
#         Returns
#         -------
#
#         """
#         # When attempting to transact zero units do nothing, pop up a warning
#         if quantity == 0:
#             Warning("Transacting zero quantity, nothing happens")
#             return
#
#         # If there is no open position, create a short one, set average price
#         if self.position_type is None:
#             self.position_type = "short"
#             self.avg_price = price
#
#         # If the initial position is long, partial close or flip to short
#         if self.position_type == "long":
#             # If quantity to sell is leq than that available - partial close
#             if self.initial_quantity >= quantity:
#                 # Reduce the quanity
#                 self.end_quantity = self.initial_quantity - quantity
#                 # Intuition: price > avg_price means gain in long position
#                 #
#                 #  initial_price -> avg_price?
#                 # initial_price -> avg_price?
#                 self.realized_pnl = \
#                     self.realized_pnl + quantity * (price - self.avg_price)
#                 # Average price remains the same
#
#             # Else the position is closed and opened in the opposite direction
#             else:
#                 self.flip(quantity, price)
#
#         # If short, short even, more. It's FX after all
#         else:
#             # Increase the quantity
#             self.end_quantity = self.initial_quantity + quantity
#             # Compute VWAP
#             self.avg_price = \
#                 (self.avg_price * self.initial_quantity +
#                  price * quantity) / self.end_quantity
#
#         # Check the end quantity, render position type to None if nothing left
#         if self.end_quantity == 0:
#             self.position_type = None
#             self.avg_price = 0
#
#     def flip(self, quantity, price):
#         """Utility method wrapping 'buy()' and 'sell()' which induce a change
#         in self.position_type. For example, flipping position from long to
#         short, when sell amount exceeds the current quantity
#
#         Parameters
#         ----------
#         quantity: float
#             number of units of the currency to transact
#         price: float
#             transaction price
#
#         Returns
#         -------
#
#         """
#         # If the intital position was long, sell it out
#         if self.position_type == "long":
#             # First, close the existing position, by selling initial quantity
#             self.sell(self.initial_quantity, price)
#
#             # Set the leftover quantity to trade in opposite direction
#             quantity_flip = quantity - self.initial_quantity
#
#             # Reset the initial quantity, nothing is left on balance
#             self.initial_quantity = 0
#
#             # Swap the position type
#             self.position_type = "short"
#
#             # And sell even more
#             self.sell(quantity_flip, price)
#
#             # Finally, set the new average prive
#             self.avg_price = price
#
#         # Similarly take care of short positions buying moar
#         elif self.position_type == "short":
#             self.buy(self.initial_quantity, price)
#             quantity_flip = quantity - self.initial_quantity
#             self.initial_quantity = 0
#             self.position_type = "long"
#             self.buy(quantity_flip, price)
#             self.avg_price = price
#
#         else:
#             raise ValueError("position_type should be either 'long' or "
#                              "'short'")
#
#     def roll_over(self, swap_points):
#         """Rolls the position overnight accruing swap points to the VWAP
#
#         Parameters
#         ----------
#         swap_points: pd.Series
#             indexed with 'bid' and 'ask' and containing corresponding quotes
#
#         Returns
#         -------
#
#         """
#         if self.end_quantity == 0:
#             return
#
#         swap_points_ask = swap_points["ask"]
#         swap_points_bid = swap_points["bid"]
#         # Accrue swap points to the average price
#         if self.position_type == "long":
#             self.avg_price = self.avg_price + swap_points_ask
#         elif self.position_type == "short":
#             self.avg_price = self.avg_price + swap_points_bid
#         else:
#             raise ValueError("position_type should be either 'long' or "
#                              "'short'")
#
#     def get_market_value(self, market_prices):
#         """Computes liquidation value of the position at given market prices
#
#         Parameters
#         ----------
#         market_prices: pd.Series
#             indexed with 'bid' and 'ask' and containing corresponding
#             exchange rates
#
#         Returns
#         -------
#         market_value: float
#             with liquidation value of the position at the given market prices
#
#         """
#         # Long positions are
#         if self.position_type == "long":
#             liquidation_price = market_prices["bid"]
#         elif self.position_type == "short":
#             liquidation_price = market_prices["ask"]
#         else:
#             raise ValueError("position_type should be either 'long' or "
#                              "'short'")
#
#         market_value = liquidation_price * self.end_quantity
#
#         return market_value
#
#     def get_unrealized_pnl(self, market_prices):
#         """Get the unrealized profit and loss, comparing the average price of
#         the position with given market prices at which the position can be
#         liquidated
#
#         Parameters
#         ----------
#         market_prices: pd.Series
#             indexed with 'bid' and 'ask' and containing corresponding
#             exchange rates
# 
#         Returns
#         -------
#         unrealized_pnl: float
#             with unrealized profit and loss of the position at the given market
#             prices
#
#         """
#         # Shortcut to the price quotes
#         ask = market_prices["ask"]
#         bid = market_prices["bid"]
#
#         # Liquidate long positions at bid
#         if self.position_type == "long":
#             unrealized_pnl = (bid - self.avg_price) * self.end_quantity
#         # And short positions at ask, mind the minus sign
#         elif self.position_type == "short":
#             unrealized_pnl = (self.avg_price - ask) * self.end_quantity
#         else:
#             # raise ValueError("position_type should be either 'long' or "
#             #                  "'short'")
#             return 0.0
#
#         return unrealized_pnl


if __name__ == "__main__":

    from foolbox.api import *
    from foolbox.wp_tabs_figs.wp_settings import settings
    from foolbox.utils import *

    # Set the output path, input data and sample
    start_date = pd.to_datetime(settings["sample_start"])
    end_date = pd.to_datetime(settings["sample_end"])
    # start_date = pd.to_datetime("2002-01-29")

    # data
    with open(data_path+"fx_by_tz_aligned_d.p", mode="rb") as fname:
        data_merged_tz = pickle.load(fname)

    # Import the all fixing times for the dollar index
    with open(data_path+"fx_by_tz_sp_fixed.p", mode="rb") as fname:
        data_all_tz = pickle.load(fname)

    fx_tr_env = FXTradingEnvironment.from_scratch(
        spot_prices={
            "bid": data_merged_tz["spot_bid"],
            "ask": data_merged_tz["spot_ask"]},
        swap_points={
            "bid": data_merged_tz["tnswap_bid"],
            "ask": data_merged_tz["tnswap_ask"]}
            )

    fx_tr_env.drop(labels=settings["drop_currencies"], axis="minor_axis")
    fx_tr_env.remove_swap_outliers()
    fx_tr_env.reindex_with_freq('B')
    fx_tr_env.align_spot_and_swap()
    fx_tr_env.fillna(which="both", method="ffill")

    signals = get_pe_signals(fx_tr_env.spot_prices.minor_axis,
        lag=10,
        threshold=0.10,
        data_path=data_path,
        fomc=False,
        avg_impl_over=settings["avg_impl_over"],
        avg_refrce_over=settings["avg_refrce_over"],
        bday_reindex=True)

    events = signals.reindex(index=fx_tr_env.spot_prices.major_axis)
    events = events.loc[start_date:end_date,:]
    # events.loc["2002-01":"2002-02"].dropna(how="all")

    # with open(data_path+"events.p", mode="rb") as fname:
    #     events = pickle.load(fname)
    # perfect_sig = np.sign(events["joint_cbs"])
    # perfect_sig = perfect_sig.loc[:,fx_tr_env.spot_prices.minor_axis]
    #
    # events = perfect_sig.reindex(index=fx_tr_env.spot_prices.major_axis)
    # events = events.loc[start_date:]

    fx_tr_str = FXTradingStrategy.from_events(events,
        blackout=-1, hold_period=5, leverage="none")

    fx_tr = FXTrading(environment=fx_tr_env, strategy=fx_tr_str)

    res = fx_tr.backtest()
    res.dropna().plot()

    # carry
    with open(data_path+"data_wmr_dev_d.p", mode="rb") as hangar:
        data_d = pickle.load(hangar)

    fdisc_d = data_d["fwd_disc"]
    fdisc_d = fdisc_d.reindex(index=fx_tr_env.spot_prices.major_axis,
        method="ffill").drop(settings["drop_currencies"], axis=1)
    fdisc_d = fdisc_d.loc[start_date:]

    fx_tr_str_carry = FXTradingStrategy.monthly_from_daily_signals(
        signals_d=fdisc_d, n_portf=3, leverage="net")

    fx_tr = FXTrading(environment=fx_tr_env, strategy=fx_tr_str_carry)

    fx_tr_str_carry.actions
    fx_tr_env.spot_prices.minor_axis
    res = fx_tr.backtest()
    res.dropna().plot()

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
            and here
