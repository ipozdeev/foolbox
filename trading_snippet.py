import pandas as pd

class FXPosition(object):
    """

    """

    def __init__(self, currency):
        """

        Parameters
        ----------
        currency: str


        Returns
        -------

        """
        self.currency = currency

        self.position_type = None
        self.initial_price = 0
        self.initial_quantity = 0
        self.avg_price = 0

        self.unrealized_pnl = 0
        self.realized_pnl = 0

        self.end_quantity = self.initial_quantity

    def buy(self, quantity, price):
        """

        Parameters
        ----------
        quantity
        price

        Returns
        -------

        """
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
                self.realized_pnl = \
                    self.realized_pnl - quantity * (price - self.initial_price)
                # Average price remains the same

            # Else the position is closed and opened in the opposite direction
            else:
                self.flip(quantity, price)

        # Check the end quantity, render position type to None if nothing left
        if self.end_quantity == 0:
            self.position_type = None

    def sell(self, quantity, price):
        """

        Parameters
        ----------
        quantity
        price

        Returns
        -------

        """
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
                self.realized_pnl = \
                    self.realized_pnl + quantity * (price - self.initial_price)
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
        """

        Parameters
        ----------
        quantity
        price

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
        else:
            self.buy(self.initial_quantity, price)
            quantity_flip = quantity - self.initial_quantity
            self.initial_quantity = 0
            self.position_type = "long"
            self.buy(quantity_flip, price)
            self.avg_price = price

    def roll_over(self, swap_points):
        """

        Parameters
        ----------
        swap_points: pd.Series
            indexed with 'bid' and 'ask' and containing corresponding quotes

        Returns
        -------

        """
        swap_points_ask = swap_points["ask"]
        swap_points_bid = swap_points["bid"]
        # Accrue swap points to the average price
        if self.position_type == "long":
            self.avg_price = self.avg_price + swap_points_ask
        else:
            self.avg_price = self.avg_price + swap_points_bid

    def get_market_value(self, market_prices):
        """

        Parameters
        ----------
        market_prices: pd.Series
            indexed with 'bid' and 'ask' and containing corresponding
            exchange rates

        Returns
        -------

        """
        # Long positions are
        if self.position_type == "long":
            liquidation_price = market_prices["bid"]
        else:
            liquidation_price = market_prices["ask"]

        market_value = liquidation_price * self.end_quantity

        return market_value

    def get_unrealized_pnl(self, market_prices):
        """

        Parameters
        ----------
        market_prices: pd.Series
            indexed with 'bid' and 'ask' and containing corresponding
            exchange rates

        Returns
        -------

        """
        # Shortcut to the price quotes
        ask = market_prices["ask"]
        bid = market_prices["bid"]

        # Liquidate long positions at bid
        if self.position_type == "long":
            unrealized_pnl = (bid - self.avg_price) * self.end_quantity
        # And short positions at ask, mind the minus sign
        else:
            unrealized_pnl = (self.avg_price - ask) * self.end_quantity

        return unrealized_pnl



def main():
    pass

if __name__ == '__main__':

    currency = "gbp"
    position_type = "long"

    initial_price = 1.25
    initial_quantity = 1

    pos = FXPosition(currency=currency)
    pos.initial_price = 1.25


    pos.sell(1.5, 1.23)


    prices = pd.Series([1.27, 1.23], index=["ask", "bid"])
    swap_points = pd.Series([0.02, 0.01], index=["ask", "bid"])

    pos.get_market_value(prices)
    pos.roll_over(swap_points)
    pos.get_market_value(prices)

    prices = pd.DataFrame({"ask": [1.27, 1.28],
                           "bid": [1.23, 1.24]})
    swap_points = pd.DataFrame({"ask": [0.02, 0.025],
                                "bid": [0.01, 0.015]})

    pos = FXPosition(currency=currency)
    pos.buy(1, prices.loc[0, "ask"])
    print(pos.get_unrealized_pnl(prices.loc[0, :]))
    print(pos.get_market_value(prices.loc[0, :]))

    # Do a barrel roll
    pos.roll_over(swap_points.loc[0, :])
    print(pos.get_unrealized_pnl(prices.loc[0, :]))
    print(pos.get_market_value(prices.loc[0, :]))

    # Assign the initial price to the next date
    pos.initial_price = pos.avg_price
    pos.sell(0.5, prices.loc[1, "bid"])
    print(pos.get_unrealized_pnl(prices.loc[1, :]))
    print(pos.get_market_value(prices.loc[1, :]))

    pos.roll_over(swap_points.loc[1, :])
    print(pos.get_unrealized_pnl(prices.loc[1, :]))
    print(pos.get_market_value(prices.loc[1, :]))



