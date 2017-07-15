import pandas as pd
import numpy as np
# import ipdb

class FXTrading():
    """
    """
    def __init__(self, prices, swap_points, signals, settings):
        """
        Parameters
        ----------
        prices : pandas.Panel
            with currencies along the items axis, time on major_axis
        swap_points : pandas.Panel
            with currencies along the items axis, time on major_axis
        signals : pd.DataFrame
            of +1,0,-1, indexed by currency names (same as the items axis
            above). These will be shifted around according to settings.
        settings : dict
            'holding_period' - holding period, in periods
            'blackout' - periods before a +1/-1 signal when position is closed
        """
        self.prices = prices
        self.swap_points = swap_points
        self.settings = settings

        # put signals into the required form: continuous frame etc. ---------
        # position flags are a continuous panel of subsequent signals
        #   indicating where a position is held from dusk till dawn (such that
        #   a return is realized)
        # to be able to fill na backwards, replace zeros with nan's
        self.signals = signals.replace(0.0, np.nan)

        # shift by balckout, extend from that `holding_period` into the past
        #   to arrive at position flags
        # align signals with prices: it is hereby assumed that the latter are
        #   indexed comme il faut, e.g. with business days
        # ipdb.set_trace()
        position_flags = self.signals.reindex(index=prices.major_axis)
        # shift by blackout
        position_flags = position_flags.shift(-1*settings["blackout"])
        # fill into the past
        position_flags = position_flags.fillna(
            method="bfill",
            limit=self.settings["holding_period"]-1)

        # in case of equally-weighted no-leverage portfolio, position weights
        #   are signals divided by the absolute sum thereof
        # also, make sure that pandas does not sum all nan's to zero
        row_leverage = np.abs(position_flags).apply(axis=1, func=np.nansum)
        position_weights = position_flags.divide(row_leverage, axis=0)
        # reinstall zeros where nan's are
        position_weights = position_weights.fillna(value=0.0)

        # actions are changes in positions: to realize a return on day t, a
        #   position on t-1 should be opened.
        actions = position_weights.diff().shift(-1)

        # collect positions into portfolio
        positions = [FXPosition(currency=p) for p in self.signals.columns]
        portfolio = FXPortfolio(positions=positions)

        self.position_flags = position_flags
        self.positions_weights = position_weights
        self.actions = actions
        self.portfolio = portfolio

    def backtest(self):
        """ Backtest a strategy based on self.`signals`.

        Routine, for each date t:
        1) take changes in positions
        2) rebalance accordingly
        3) roll after rebalancing
        4) record liquidation value

        """
        # allocate space for liquidation values
        liquidation_v = pd.Series(index=self.position_flags.index)

        quantity = self.position_flags.copy()*np.nan

        # loop over time and position weights
        # ipdb.set_trace()
        for t, row in self.actions.iterrows():
            # if t > pd.to_datetime("2006-04-19"):
            #     ipdb.set_trace()

            # fetch prices and swap points ----------------------------------
            these_prices = self.prices.loc[:,t,:]
            these_swap_points = self.swap_points.loc[:,t,:]

            # rebalance -----------------------------------------------------
            self.portfolio.rebalance_from_dpw(row, these_prices)

            # get quantity --------------------------------------------------
            quantity.loc[t,:] = self.portfolio.positions.map(
                lambda x: x.end_quantity)

            # roll over -----------------------------------------------------
            self.portfolio.roll_over(these_swap_points)

            # get liquidation value -----------------------------------------
            liquidation_v.loc[t] = self.portfolio.get_liquidation_value(
                prices=these_prices)
            liquidation_v.loc[t] = self.portfolio.balance

        return liquidation_v, quantity

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
        self.position_weights.to_excel(writer, sheet_name='position_flags')
        self.actions.to_excel(writer, sheet_name='actions')

        self.prices.loc[:,:,"ask"].to_excel(writer, sheet_name='p_ask')
        self.prices.loc[:,:,"bid"].to_excel(writer, sheet_name='p_bid')
        self.swap_points.loc[:,:,"ask"].to_excel(writer, sheet_name='_ask')
        self.swap_points.loc[:,:,"bid"].to_excel(writer, sheet_name='_bid')

        # save and close
        writer.save()

        return


class FXPortfolio():
    """
    """
    def __init__(self, positions):
        """ Portfolio of FX positions.
        """
        # start with balance = 1.0
        self.balance = 1.0
        # currencies
        self.currencies = [p.currency for p in positions]
        # represent positions as a pandas.Series
        self.positions = pd.Series(
            data=positions,
            index=self.currencies)
        # position weights: start with 0 everywhere
        self.position_weights = pd.Series(0.0, index=self.currencies)

    def get_liquidation_value(self, prices):
        """ Calculate total liquidation value of portfolio.

        Liquidation value is the result of closing all open position
        immediately and draining any balance.

        Parameters
        ----------
        prices : pd.DataFrame
            of quotes, indexed with currency names, column'ed by ["bid", "ask"]
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
        to_close_idx = new_pw.ix[new_pw == 0].index
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
        #self.initial_price = 0
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
            self.transact(-1*self.end_quantity, prices)
        else:
            self.transact(self.end_quantity, prices)

        self.end_quantity = 0.0

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
            self.avg_price = price

        # If the initial position is long, then buy MOAR
        if self.position_type == "long":
            # Increase the quantity
            self.end_quantity = self.initial_quantity + quantity
            # Compute VWAP
            self.avg_price = \
                (self.avg_price * self.initial_quantity +
                 price * quantity) / self.end_quantity

        # If short -- partial close at ask or flip to long
        else:
            # If quantity to buy is leq than that available - partial close
            if self.initial_quantity >= quantity:
                # Reduce the quanity
                self.end_quantity = self.initial_quantity - quantity
                # Intuition: price > avg_price means loss in short position
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

        # If there is no open position, create a short one, set average price
        if self.position_type is None:
            self.position_type = "short"
            self.avg_price = price

        # If the initial position is long, partial close or flip to short
        if self.position_type == "long":
            # If quantity to sell is leq than that available - partial close
            if self.initial_quantity >= quantity:
                # Reduce the quanity
                self.end_quantity = self.initial_quantity - quantity
                # Intuition: price > avg_price means gain in long position
                #
                #  initial_price -> avg_price?
                # initial_price -> avg_price?
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
                (self.avg_price * self.initial_quantity +
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


class FXPosition_deprecated():
    """
    """
    def __init__(self, base_cur, counter_cur):
        """
        """
        self.base_cur = base_cur
        self.counter_cur = counter_cur

        # closed by default
        self.is_open = False

    def open(self, price, direction="long", size=0.0):
        """
        Parameters
        ----------
        price : float
            ask price for long position, bid price for short ones, expressed
            in units of counter currency
        size : float
            units of base currency lent out (`bought`)
        """
        if size == 0.0:
            return

        self.is_open = True
        self.size = size
        self.direction = 1 if direction == "long" else -1
        self.accum_roll = 0.0
        self.price_open = price

    def close(self, price, partial=None):
        """
        price : float
            bid price for long position, ask price for short ones, expressed
            in units of counter currency
        partial : float
            share of position size to be closed
        """
        if partial is None:
            partial = 1.0

        payoff = self.get_unrealized_pl(price=price) * partial

        # reduce size
        self.size *= (1.0-partial)

        if partial == 1.0:
            self.is_open = False

        return payoff

    def get_spot_ret(self, price):
        """ Calculate change in the spot price.
        price : float
            bid price for long position, ask price for short ones, expressed
            in units of counter currency
        """
        if not self.is_open:
            return np.nan

        # spot return on one unit of base currency
        unit_spot_ret = (price - self.price_open)*self.direction

        # spot ret on the whole position
        spot_ret = unit_spot_ret * self.size

        return spot_ret

    def get_unrealized_pl(self, price):
        """ Calculate total return: spot return plus total roll.
        """
        # payoff as difference between open and close prices
        payoff = self.get_spot_ret(price) + self.accum_roll

        return payoff

    def roll_over(self, swap_points):
        """
        Parameters
        ----------
        swap_points : float
            in units of counter currency per unit of base currency
        """
        if not self.is_open:
            return

        self.accum_roll += (self.size * swap_points) * self.direction


if __name__ == "__main__":

    from foolbox.api import *
    from foolbox.wp_tabs_figs.wp_settings import settings
    from foolbox.utils import *

    # Set the output path, input data and sample
    out_path = data_path + settings["fig_folder"]
    input_dataset = settings["fx_data"]
    start_date = settings["sample_start"]
    end_date = settings["sample_end"]

    # Import the FX data
    with open(data_path+input_dataset, mode="rb") as fname:
        data_merged_tz = pickle.load(fname)

    # Import the all fixing times for the dollar index
    with open(data_path+"fx_by_tz_sp_fixed.p", mode="rb") as fname:
        data_all_tz = pickle.load(fname)

    # Get the individual currenices, spot rates:
    spot_bid = data_merged_tz["spot_bid"]
    spot_ask = data_merged_tz["spot_ask"]
    swap_ask = data_merged_tz["tnswap_ask"]
    swap_bid = data_merged_tz["tnswap_bid"]
    # swap_ask = data_all_tz["tnswap_ask"].loc[:,start_date:end_date,"NYC"]
    # swap_bid = data_all_tz["tnswap_bid"].loc[:,start_date:end_date,"NYC"]

    # spot_ask_us = data_all_tz["spot_ask"].loc[:,start_date:end_date,"NYC"]
    # spot_bid_us = data_all_tz["spot_bid"].loc[:,start_date:end_date,"NYC"]
    # swap_ask_us = data_all_tz["tnswap_ask"].loc[:,start_date:end_date,"NYC"]
    # swap_bid_us = data_all_tz["tnswap_bid"].loc[:,start_date:end_date,"NYC"]

    # Align and ffill the data, first for tz-aligned countries
    (spot_bid, spot_ask, swap_bid, swap_ask) = align_and_fillna(
        (spot_bid, spot_ask, swap_bid, swap_ask),
        "B", method="ffill")

    # # Now for the dollar index
    # (spot_bid_us, spot_ask_us, swap_bid_us, swap_ask_us) =\
    #     align_and_fillna((spot_bid_us, spot_ask_us,
    #                       swap_bid_us, swap_ask_us),
    #                      "B", method="ffill")
    # spot_bid.loc[:,"usd"] = spot_bid_us.drop(settings["drop_currencies"],
    #     axis=1).mean(axis=1)
    # spot_ask.loc[:,"usd"] = spot_ask_us.drop(settings["drop_currencies"],
    #     axis=1).mean(axis=1)
    # swap_bid.loc[:,"usd"] = swap_bid_us.drop(settings["drop_currencies"],
    #     axis=1).mean(axis=1)
    # swap_ask.loc[:,"usd"] = swap_ask_us.drop(settings["drop_currencies"],
    #     axis=1).mean(axis=1)

    prices = pd.Panel.from_dict(
        {"bid": spot_bid, "ask": spot_ask},
        orient="items").drop(settings["drop_currencies"], axis="minor_axis")
    swap_points = pd.Panel.from_dict(
        {"bid": swap_bid, "ask": swap_ask},
        orient="items").drop(settings["drop_currencies"], axis="minor_axis")

    nav_fcast = pd.Panel(
        items=np.arange(1,16),
        major_axis=prices.major_axis,
        minor_axis=np.arange(1,26))
    nav_perf = pd.DataFrame(index=prices.major_axis, columns=np.arange(1,16))

    # the loop
    for h in range(3,4):
        # h = 10
        for p in range(9,10):
            # p = 9
            # Get signals for all countries except for the US
            policy_fcasts = list()
            for curr in prices.minor_axis:
                # Get the predicted change in policy rate
                tmp_pe = PolicyExpectation.from_pickles(data_path, curr)
                policy_fcasts.append(
                    tmp_pe.forecast_policy_change(
                        lag=h+1+settings["base_blackout"],
                        threshold=p/100,
                        avg_impl_over=settings["avg_impl_over"],
                        avg_refrce_over=settings["avg_refrce_over"],
                        bday_reindex=True))

            # Put individual predictions into a single dataframe
            signals = pd.concat(policy_fcasts, join="outer", axis=1)\
                .loc[start_date:end_date,:]
            signals = signals.dropna(how="all")

            # add NA in front of signals
            add_sig = pd.DataFrame(
                index=pd.date_range(
                    signals.index[0]-DateOffset(months=1),
                    signals.index[0], freq='B'),
                columns=signals.columns).drop(signals.index[0], axis=0)
            signals = pd.concat((add_sig, signals), axis=0)

            trading_settings = {
                "holding_period": h,
                "blackout": settings["base_blackout"]}

            fxtr = FXTrading(
                prices=prices.loc[:,signals.index[0]:,:],
                swap_points=swap_points.loc[:,signals.index[0]:,:]*0.0,
                signals=signals,
                settings=trading_settings)

            this_nav, qty = fxtr.backtest()
            # fxtr.actions.head(25)
            # this_nav.plot()
            # qty.loc["2006-04"].where(qty.loc["2006-04"] > 0.0).count()
            # this_nav.loc["2006-03":"2006-04"]
            # fxtr.actions
            # qty.loc["2006-03"]

            nav_fcast.loc[h,:,p] = this_nav

        # perfect -----------------------------------------------------------
        with open(data_path+"events.p", mode="rb") as fname:
            events = pickle.load(fname)
        perfect_sig = np.sign(events["joint_cbs"])
        perfect_sig = perfect_sig.loc[:,prices.minor_axis].dropna(how="all")
        perfect_sig = perfect_sig.loc[start_date:]

        perfect_sig = perfect_sig.where(signals).dropna(how="all")

        fxtr = FXTrading(
            prices=prices,
            swap_points=swap_points*0.0,
            signals=perfect_sig,
            settings=trading_settings)

        this_nav_perf = fxtr.backtest()
        this_nav_perf.plot()

        # perfect foresight
        nav_perf.loc[:,h] = this_nav_perf

    # with open(out_path + "output.p", mode="wb") as hangar:
    #     pickle.dump({"perf": nav_perf, "fcast": nav_fcast}, hangar)
    #
    # with open(out_path + "output.p", mode="rb") as hangar:
    #     new = pickle.load(hangar)

    # %matplotlib inline
    # this_nav.plot()
    # new = new["fcast"]
    # new.loc[2,"2003-07",1]
    # prices.loc[:,:,"aud"].plot()

    # swap_points.loc[:,"2004-04":,"aud"].rolling(5).mean().plot()

    #
    # %matplotlib inline
    # fig, ax = plt.subplots()
    # nav_perfect.plot(ax=ax, color='k')
    # nav_perf.loc[:,trading_settings["holding_period"]].plot(ax=ax,
    #     color=my_red)
    # fig.tight_layout()
    # fig.savefig(out_path + "nav_perfect.pdf", transparent=True)
