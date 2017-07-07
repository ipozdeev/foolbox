import pandas as pd
import numpy as np

class Trading():
    """
    """
    def __init__(self):
        """
        """
        pass


class FXPortfolio():
    """
    """
    def __init__(self):
        """
        """
        pass


class FXPosition():
    """
    """
    def __init__(self, base_cur, counter_cur):
        """
        """
        self.base_cur = base_cur
        self.counter_cur = counter_cur

        # closed by default
        self.is_open = False

        pass

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
    my_fx = FXPosition(base_cur="xxx", counter_cur="yyy")
    my_fx.open(price=100.0, direction="long", size=10)
    print(my_fx.accum_roll)
    my_fx.roll_over(2.2)
    print(my_fx.accum_roll)
    res = my_fx.close(price=105)
    print(res)
