import unittest
# from numpy.testing import assert_almost_equal

import pandas as pd
import numpy as np
# import ipdb

from foolbox.fxtrading import *

class TestPortfolio(unittest.TestCase):
    """
    """
    def setUp(self):
        """
        """
        currencies = ["nzd","chf"]
        dt = pd.date_range("2001-01-03", periods=2, freq='D')

        prices = pd.DataFrame({
            "nzd": [0.75, 1.33],
            "chf": [10.0, 11.0]}, index=["bid", "ask"])
        swap_points = pd.DataFrame({
            "nzd": [-0.1, -0.05],
            "chf": [2.5, 3.0]}, index=["bid", "ask"])

        positions = [FXPosition(currency=p) for p in currencies]

        self.currencies = currencies
        self.prices = prices
        self.swap_points = swap_points
        self.positions = positions

    # @unittest.skip('')
    def test_init(self):
        """
        """
        fx_pf = FXPortfolio(positions=self.positions)

        self.assertEqual(np.abs(fx_pf.position_weights).mean(), 0.0)

    # @unittest.skip('')
    def test_roll_over(self):
        """
        """
        fx_pf = FXPortfolio(positions=self.positions)

        fx_pf.roll_over(bid_ask_points=self.swap_points)

        for _, p in fx_pf.positions.iteritems():
            self.assertEqual(p.avg_price, 0.0)

        fx_pf.positions["nzd"].transact(-1, self.prices.loc[:, "nzd"])

        fx_pf.roll_over(bid_ask_points=self.swap_points)

        self.assertEqual(fx_pf.positions["nzd"].avg_price,
            self.prices.loc["bid", "nzd"] + self.swap_points.loc["bid", "nzd"])

    # @unittest.skip('')
    def test_rebalance(self):
        """
        """
        fx_pf = FXPortfolio(positions=self.positions)

        dp_1 = pd.Series([2.0, -2], index=self.currencies)
        fx_pf.rebalance_from_dp(dp_1, bid_ask_prices=self.prices)

        self.assertTrue(fx_pf.get_position_quantities().equals(dp_1))

        # rebalance again
        dp_2 = pd.Series([1.0, 1], index=self.currencies)
        fx_pf.rebalance_from_dp(dp_2, bid_ask_prices=self.prices)

        self.assertTrue(fx_pf.get_position_quantities().equals(dp_1+dp_2))

    # @unittest.skip('')
    def test_rebalance_from_dpw(self):
        """
        """
        fx_pf = FXPortfolio(positions=self.positions)

        pw_1 = pd.Series([0.0, 1.0], index=self.currencies)
        fx_pf.rebalance_from_dpw(pw_1, bid_ask_prices=self.prices)
        print(fx_pf.get_position_quantities())

        pw_2 = pd.Series([0.5, -1.5], index=self.currencies)
        fx_pf.rebalance_from_dpw(pw_2, bid_ask_prices=self.prices)
        print(fx_pf.get_position_quantities())

    # @unittest.skip('')
    def test_fetch_info(self):
        """
        """
        # equate bid and ask -> liquidation value = initial capital
        prc = self.prices.copy()
        prc.loc["ask", :] = prc.loc["bid", :]

        fx_pf = FXPortfolio(positions=self.positions)
        pw_1 = pd.Series([0.5, 0.5], index=self.currencies)
        fx_pf.rebalance_from_dpw(pw_1, bid_ask_prices=prc)

        self.assertEqual(fx_pf.get_margin_closeout_value(self.prices), 1.0)

        print(fx_pf.balance)

if __name__ == "__main__":
    unittest.main()
