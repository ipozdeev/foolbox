import unittest
# from numpy.testing import assert_almost_equal

import pandas as pd
import numpy as np
# import ipdb

from foolbox.fxtrading import *

class TestPosition(unittest.TestCase):
    """
    """
    def setUp(self):
        """
        """
        currency = "nzd"
        prices = pd.Series(data=[0.5, 0.75], index=["bid","ask"])
        swap_points = pd.Series(data=[-0.1, -0.05], index=["bid","ask"])

        self.currency = currency
        self.prices = prices
        self.swap_points = swap_points

    def test_init(self):
        """
        """
        fx_pos = FXPosition(currency=self.currency)

        self.assertFalse(fx_pos.is_open)
        self.assertEqual(fx_pos.quantity, 0.0)
        self.assertEqual(fx_pos.avg_price, 0.0)

    # @unittest.skip
    def test_transact(self):
        """
        """
        fx_pos = FXPosition(currency=self.currency)

        # buy nothing
        fx_pos.transact(qty=-0.0, bid_ask_prices=self.prices)
        self.assertEqual(fx_pos.quantity, 0.0)
        self.assertEqual(fx_pos.avg_price, 0.0)
        self.assertFalse(fx_pos.is_open)

        # buy some
        res_buy = fx_pos.transact(qty=0.25, bid_ask_prices=self.prices)
        self.assertIsNone(res_buy)
        self.assertEqual(fx_pos.avg_price, self.prices["ask"])
        self.assertEqual(fx_pos.quantity, 0.25)
        self.assertTrue(fx_pos.is_open)

        # sell some
        res_sell = fx_pos.transact(qty=-0.25, bid_ask_prices=self.prices)
        self.assertIsNotNone(res_sell)
        self.assertEqual(fx_pos.quantity, 0.0)
        self.assertFalse(fx_pos.is_open)

        # sell some more
        res_buy_another = fx_pos.transact(qty=-0.5, bid_ask_prices=self.prices)
        self.assertIsNone(res_buy_another)
        self.assertEqual(fx_pos.quantity, -0.5)

    # @unittest.skip
    def test_flip(self):
        """
        """
        fx_pos = FXPosition(currency=self.currency)
        res_buy = fx_pos.transact(+0.33, self.prices)
        res_sell = fx_pos.transact(-1.0, self.prices)

        self.assertEqual(fx_pos.avg_price, self.prices["bid"])
        self.assertAlmostEqual(fx_pos.quantity, -0.67, places=6)
        self.assertAlmostEqual(res_sell,
            -0.33*(self.prices.loc["ask"] - self.prices.loc["bid"]),
            places=6)

        res_buy_another = fx_pos.transact(+0.67, self.prices)
        self.assertFalse(fx_pos.is_open)
        self.assertAlmostEqual(res_buy_another,
            -1*0.67*(self.prices.loc["ask"] - self.prices.loc["bid"]),
            places=6)

    # @unittest.skip
    def test_roll(self):
        """
        """
        fx_pos = FXPosition(currency=self.currency)

        # empty roll
        fx_pos.roll_over(bid_ask_points=self.swap_points)
        self.assertEqual(fx_pos.avg_price, 0.0)

        # non-empty roll
        qty = 10
        fx_pos.transact(qty, bid_ask_prices=self.prices)
        self.assertEqual(fx_pos.avg_price, self.prices["ask"])
        fx_pos.roll_over(bid_ask_points=self.swap_points)
        self.assertEqual(fx_pos.avg_price,
            self.prices["ask"]+self.swap_points["ask"])

        # sell everything, roll_over
        fx_pos = FXPosition(currency=self.currency)
        fx_pos.transact(-qty-1, bid_ask_prices=self.prices)
        fx_pos.roll_over(bid_ask_points=self.swap_points)
        self.assertEqual(fx_pos.avg_price,
            self.prices["bid"]+self.swap_points["bid"])

    # @unittest.skip
    def test_unrealized_pnl(self):
        """
        """
        fx_pos = FXPosition(currency=self.currency)
        self.assertEqual(fx_pos.get_unrealized_pnl(self.prices), 0.0)

        fx_pos.transact(-2.0, bid_ask_prices=self.prices)
        self.assertEqual(fx_pos.get_unrealized_pnl(self.prices+1),
            -2.0*(self.prices["ask"]+1 - self.prices["bid"]))

        fx_pos.roll_over(self.swap_points)
        self.assertEqual(fx_pos.get_unrealized_pnl(self.prices+1),
            -2.0*(self.prices["ask"]+1 - self.prices["bid"] + 0.1))

if __name__ == "__main__":
    unittest.main()
