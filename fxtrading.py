import pandas as pd
import numpy as np
import ipdb

class FXTrading():
    """
    """
    def __init__(self, prices, swap_points, signals, settings):
        """
        """
        self.prices = prices
        self.swap_points = swap_points
        self.signals = signals.replace(0.0, np.nan)
        self.settings = settings

        # position flags are continuous indicators
        self.position_flags = self.signals.\
            reindex(index=prices.major_axis).\
            fillna(method="bfill", limit=self.settings["h"])
        # position weights are signals through absolute sum thereof
        self.position_weights = self.position_flags.divide(
            np.abs(self.position_flags).sum(axis=1), axis=0).fillna(0.0)

        # actions
        self.actions = self.position_weights.diff().shift(-1)

        # portfolio
        positions = [FXPosition(currency=p) for p in self.signals.columns]
        self.portfolio = FXPortfolio(positions=positions)

    def backtest(self):
        """
        """
        # allocate space for P&L
        lqd_val = pd.Series(index=self.position_flags.index)

        # loop over time
        for t, row in self.actions.iterrows():
            # fetch prices and swap points
            these_prices = self.prices.loc[:,t,:]
            these_swap_points = self.swap_points.loc[:,t,:]

            # rebalance
            self.portfolio.rebalance_from_position_weights(row, these_prices)

            # roll
            self.portfolio.roll_over(these_swap_points)

            # get liquidation value
            lqd_val.loc[t] = self.portfolio.get_liquidation_value(these_prices)

        return lqd_val


class FXPortfolio():
    """
    """
    def __init__(self, positions):
        """
        """
        self.currencies = [p.currency for p in positions]
        self.positions = pd.Series(data=positions, index=self.currencies)

        self.balance = 1.0
        self.pw = pd.Series(0.0, index=[p.currency for p in positions])

    # @property
    # def prices(self):
    #     if self._prices is not None:
    #         return self._prices
    #
    #     return pd.Series(np.nan, index=["bid","ask"])
    #
    # @prices.setter
    # def prices(self, value):
    #     self._prices = value
    #
    # @property
    # def swap_points(self):
    #     if self._swap_points is not None:
    #         return self._swap_points
    #
    #     return pd.Series(np.nan, index=["bid","ask"])
    #
    # @swap_points.setter
    # def swap_points(self, value):
    #     self._swap_points = value

    # def get_market_value(self, prices):
    #     """
    #     """
    #     # init result at 0
    #     res = 0.0
    #
    #     # loop over positions, for repective currency fetch price & apply the
    #     #   same method
    #     for p in self.positions:
    #         res += p.get_market_value(prices.loc[p.currency])
    #
    #     # do not forget about balance
    #     res += self.balance
    #
    #     return res

    def get_liquidation_value(self, prices):
        """ Calculate total liquidation value of portfolio.
        """
        # init result at 0
        res = 0.0

        # loop over positions, for repective currency fetch price & apply the
        #   same method
        for p in self.positions.values:
            res += p.get_unrealized_pnl(prices.loc[p.currency])

        # do not forget about balance
        res += self.balance

        return res

    def rebalance_from_position_weights(self, dpw, prices):
        """
        """
        # previos positions weights
        if (dpw == 0.0).all():
            return

        new_pw = self.pw + dpw

        # 1) the ones that get closed
        to_close_idx = new_pw.ix[new_pw == 0].index
        for idx, p in self.positions.ix[to_close_idx].iteritems():
            # close -> realized p&l here
            p.close(prices.loc[idx,:])

        # replenish balance
        for p in self.positions.ix[to_close_idx].values:
            self.balance += p.realized_pnl
            p.realized_pnl = 0.0

        # recover liquidation value
        lqd_val = self.get_liquidation_value(prices)

        # 2) for each position, calculate additional amount to be transacted
        # total dollar amount
        pos_dol_vals = new_pw * lqd_val
        # total quantity
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
        self.pw = new_pw

    def rebalance_from_dpw(self, dpw, prices):
        """
        """
        dp = self.dpw2quantities(dpw=dpw, prices=prices)

        self.rebalance_from_dp(dp, prices)

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

    def dpw2quantities(self, dpw, prices):
        """ Transform position weights to quantitites to be transacted.
        Parameters
        ----------
        new_pw : pandas.Series
            of new position weights, indexed with currency names
        """
        # 1) calculate liquidation value of portfolio
        lqd_val = self.get_liquidation_value(prices)

        # 3) dollar value of positions
        dpw_dollar_val = dpw * lqd_val

        # 4) from dollar value to foreign currency quantity
        qty = self.fxdivide(dpw_dollar_val, prices)

        return qty

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

class FXPosition(object):
    """Handles transaction and roll overs of positions in a foreign currency

    """

    def __init__(self, currency):
        """

        Parameters
        ----------
        currency: str
            reperesenting counter currency of an FXPosition


        Returns
        -------

        """
        self.currency = currency

        # Set the initial values for an empty position
        self.position_type = None  # Set to str 'short' or 'long'
        self.initial_price = 0
        self.initial_quantity = 0
        self.avg_price = 0

        self.unrealized_pnl = 0
        self.realized_pnl = 0

        # By default, no transactions take place
        self.end_quantity = self.initial_quantity

    def __str__(self):
        """ Handles string representation of the object
        """
        if self.position_type is None:
            return ' '.join((self.currency, "inactive"))

        qty = np.round(self.end_quantity, 4)

        res = ' '.join((self.position_type, str(qty), self.currency))

        return res

    def position_float(self):
        """
        """
        if self.position_type is None:
            return 0

        return 1 if self.position_type == "long" else -1

    def close(self, prices):
        """
        """
        if self.position_type is None:
            return

        if self.position_type == "long":
            self.sell(self.end_quantity, prices["bid"])
        else:
            self.buy(self.end_quantity, prices["ask"])

    def transact(self, quantity, price):
        """Wrapper around the 'buy()' and 'sell()' methods, recognizing sells
        as transactions with negative quantity. Furthermore, updates the
        intial_quantity to end_quantity, allowing for sequential transactions

        Parameters
        ----------
        quantity: float
            units of currency to transact, negative for sells
        price: pd.Series
            indexed with 'bid' and 'ask' containing the corresponding price
            quotes

        Returns
        -------

        """
        if quantity < 0:
            self.sell(-1*quantity, price["bid"])
        elif quantity > 0:
            self.buy(quantity, price["ask"])
        else:
            Warning("Transacting zero quantity, nothing happens")
            return

        self.initial_quantity = self.end_quantity

    def buy(self, quantity, price):
        """Buys 'quantity' units at 'price' price

        Parameters
        ----------
        quantity: float
            number of units of the currency to buy
        price: float
            transaction price

        Returns
        -------

        """
        # When attempting to transact zero units do nothing, pop up a warning
        if quantity == 0:
            Warning("Transacting zero quantity, nothing happens")
            return

        # If there is no open position open a long one, set the initial price
        if self.position_type is None:
            self.position_type = "long"
            self.initial_price = price

        # If the initial position is long, then buy MOAR
        if self.position_type == "long":
            # Increase the quantity
            self.end_quantity = self.initial_quantity + quantity
            # Compute VWAP
            self.avg_price = \
                (self.initial_price * self.initial_quantity +
                 price * quantity) / self.end_quantity

        # If short -- partial close at ask or flip to long
        else:
            # If quantity to buy is leq than that available - partial close
            if self.initial_quantity >= quantity:
                # Reduce the quanity
                self.end_quantity = self.initial_quantity - quantity
                # Intuition: price > init_price means loss in short position
                # NB: what about: initial_price -> avg_price
                self.realized_pnl = \
                    self.realized_pnl - quantity * (price - self.avg_price)
                # Average price remains the same

            # Else the position is closed and opened in the opposite direction
            else:
                self.flip(quantity, price)

        # Check the end quantity, render position type to None if nothing left
        if self.end_quantity == 0:
            self.position_type = None

    def sell(self, quantity, price):
        """Sells 'quantity' units at 'price' price

        Parameters
        ----------
        quantity: float
            number of units of the currency to sell
        price: float
            transaction price

        Returns
        -------

        """
        # When attempting to transact zero units do nothing, pop up a warning
        if quantity == 0:
            Warning("Transacting zero quantity, nothing happens")
            return

        # If there is no open position, create a short one, set initial price
        if self.position_type is None:
            self.position_type = "short"
            self.initial_price = price

        # If the initial position is long, partial close or flip to short
        if self.position_type == "long":
            # If quantity to sell is leq than that available - partial close
            if self.initial_quantity >= quantity:
                # Reduce the quanity
                self.end_quantity = self.initial_quantity - quantity
                # Intuition: price > init_price means gain in long position
                #
                § initial_price -> avg_price?
                self.realized_pnl = \
                    self.realized_pnl + quantity * (price - self.avg_price)
                # Average price remains the same

            # Else the position is closed and opened in the opposite direction
            else:
                self.flip(quantity, price)

        # If short, short even, more. It's FX after all
        else:
            # Increase the quantity
            self.end_quantity = self.initial_quantity + quantity
            # Compute VWAP
            self.avg_price = \
                (self.initial_price * self.initial_quantity +
                 price * quantity) / self.end_quantity

        # Check the end quantity, render position type to None if nothing left
        if self.end_quantity == 0:
            self.position_type = None

    def flip(self, quantity, price):
        """Utility method wrapping 'buy()' and 'sell()' which induce a change
        in self.position_type. For example, flipping position from long to
        short, when sell amount exceeds the current quantity

        Parameters
        ----------
        quantity: float
            number of units of the currency to transact
        price: float
            transaction price

        Returns
        -------

        """
        # If the intital position was long, sell it out
        if self.position_type == "long":
            # First, close the existing position, by selling initial quantity
            self.sell(self.initial_quantity, price)

            # Set the leftover quantity to trade in opposite direction
            quantity_flip = quantity - self.initial_quantity

            # Reset the initial quantity, nothing is left on balance
            self.initial_quantity = 0

            # Swap the position type
            self.position_type = "short"

            # And sell even more
            self.sell(quantity_flip, price)

            # Finally, set the new average prive
            self.avg_price = price

        # Similarly take care of short positions buying moar
        elif self.position_type == "short":
            self.buy(self.initial_quantity, price)
            quantity_flip = quantity - self.initial_quantity
            self.initial_quantity = 0
            self.position_type = "long"
            self.buy(quantity_flip, price)
            self.avg_price = price

        else:
            raise ValueError("position_type should be either 'long' or "
                             "'short'")

    def roll_over(self, swap_points):
        """Rolls the position overnight accruing swap points to the VWAP

        Parameters
        ----------
        swap_points: pd.Series
            indexed with 'bid' and 'ask' and containing corresponding quotes

        Returns
        -------

        """
        if self.end_quantity == 0:
            return

        swap_points_ask = swap_points["ask"]
        swap_points_bid = swap_points["bid"]
        # Accrue swap points to the average price
        if self.position_type == "long":
            self.avg_price = self.avg_price + swap_points_ask
        elif self.position_type == "short":
            self.avg_price = self.avg_price + swap_points_bid
        else:
            raise ValueError("position_type should be either 'long' or "
                             "'short'")

    def get_market_value(self, market_prices):
        """Computes liquidation value of the position at given market prices

        Parameters
        ----------
        market_prices: pd.Series
            indexed with 'bid' and 'ask' and containing corresponding
            exchange rates

        Returns
        -------
        market_value: float
            with liquidation value of the position at the given market prices

        """
        # Long positions are
        if self.position_type == "long":
            liquidation_price = market_prices["bid"]
        elif self.position_type == "short":
            liquidation_price = market_prices["ask"]
        else:
            raise ValueError("position_type should be either 'long' or "
                             "'short'")

        market_value = liquidation_price * self.end_quantity

        return market_value

    def get_unrealized_pnl(self, market_prices):
        """Get the unrealized profit and loss, comparing the average price of
        the position with given market prices at which the position can be
        liquidated

        Parameters
        ----------
        market_prices: pd.Series
            indexed with 'bid' and 'ask' and containing corresponding
            exchange rates

        Returns
        -------
        unrealized_pnl: float
            with unrealized profit and loss of the position at the given market
            prices

        """
        # Shortcut to the price quotes
        ask = market_prices["ask"]
        bid = market_prices["bid"]

        # Liquidate long positions at bid
        if self.position_type == "long":
            unrealized_pnl = (bid - self.avg_price) * self.end_quantity
        # And short positions at ask, mind the minus sign
        elif self.position_type == "short":
            unrealized_pnl = (self.avg_price - ask) * self.end_quantity
        else:
            # raise ValueError("position_type should be either 'long' or "
            #                  "'short'")
            return 0.0

        return unrealized_pnl

# class FXPosition():
#     """
#     """
#     def __init__(self, base_cur, counter_cur):
#         """
#         """
#         self.base_cur = base_cur
#         self.counter_cur = counter_cur
#
#         # closed by default
#         self.is_open = False
#
#     def open(self, price, direction="long", size=0.0):
#         """
#         Parameters
#         ----------
#         price : float
#             ask price for long position, bid price for short ones, expressed
#             in units of counter currency
#         size : float
#             units of base currency lent out (`bought`)
#         """
#         if size == 0.0:
#             return
#
#         self.is_open = True
#         self.size = size
#         self.direction = 1 if direction == "long" else -1
#         self.accum_roll = 0.0
#         self.price_open = price
#
#     def close(self, price, partial=None):
#         """
#         price : float
#             bid price for long position, ask price for short ones, expressed
#             in units of counter currency
#         partial : float
#             share of position size to be closed
#         """
#         if partial is None:
#             partial = 1.0
#
#         payoff = self.get_unrealized_pl(price=price) * partial
#
#         # reduce size
#         self.size *= (1.0-partial)
#
#         if partial == 1.0:
#             self.is_open = False
#
#         return payoff
#
#     def get_spot_ret(self, price):
#         """ Calculate change in the spot price.
#         price : float
#             bid price for long position, ask price for short ones, expressed
#             in units of counter currency
#         """
#         if not self.is_open:
#             return np.nan
#
#         # spot return on one unit of base currency
#         unit_spot_ret = (price - self.price_open)*self.direction
#
#         # spot ret on the whole position
#         spot_ret = unit_spot_ret * self.size
#
#         return spot_ret
#
#     def get_unrealized_pl(self, price):
#         """ Calculate total return: spot return plus total roll.
#         """
#         # payoff as difference between open and close prices
#         payoff = self.get_spot_ret(price) + self.accum_roll
#
#         return payoff
#
#     def roll_over(self, swap_points):
#         """
#         Parameters
#         ----------
#         swap_points : float
#             in units of counter currency per unit of base currency
#         """
#         if not self.is_open:
#             return
#
#         self.accum_roll += (self.size * swap_points) * self.direction

# class FXSeries(pd.Series):
#     """
#     """
#     def mask_bid_ask(self, bid, ask):
#         """ Match negative entries in `self` with `bid`, long with `ask`.
#         """
#         res = bid.mask(self > 0, ask)
#
#         return res
#
#     def fxdivide(self, other):
#         """
#         """
#         masked_ba = self.mask_bid_ask(other["bid"], other["ask"])
#         res = self.divide(masked_ba)
#
#         return res
#
#     def fxmul(self, other):
#         """
#         """
#         masked_ba = self.mask_bid_ask(other["bid"], other["ask"])
#         res = self.mul(masked_ba)
#
#         return res


if __name__ == "__main__":

    curs = ["nzd",]
    dt = pd.date_range("2001-01-03", periods=5, freq='D')

    bid = pd.DataFrame(
        data=np.array([[1.0],[1.1],[1.21],[1.21],[1.05]]),
        index=dt,
        columns=curs)
    ask = bid + 0.05
    prices = pd.Panel.from_dict({"bid": bid, "ask": ask}, orient="items")

    swap_points = -prices/25

    signals = bid*0.0
    signals.iloc[3,0] = 1

    settings = {"h": 1}

    fxtr = FXTrading(prices, swap_points, signals, settings)

    ipdb.set_trace()
    fxtr.backtest()


    # fxtr.position_flags
    # fxtr.position_weights
    #
    # fxtr.portfolio.positions
    # fxtr.portfolio.get_position_quantities()
    #
    # dp = pd.Series(data=[100, -10], index=fxtr.portfolio.currencies)
    #
    # fxtr.portfolio.rebalance_from_dp(dp=dp, prices=prices.iloc[:,0,:])
    #
    # fxtr.portfolio.get_position_quantities()

    # fx_pos = FXPosition(currency="nzd")
    #
    # # buy some
    # ipdb.set_trace()
    # fx_pos.buy(quantity=0.25, price=0.5)
    #
    # # sell same some
    # fx_pos.sell(quantity=0.25, price=0.66)
