import unittest
# from numpy.testing import assert_almost_equal

import pandas as pd
import numpy as np
# import ipdb

from foolbox.fxtrading import *

# class TestPosition(unittest.TestCase):
#     """
#     """
#     def setUp(self):
#         """
#         """
#         currency = "nzd"
#         prices = pd.Series(data=[0.5, 0.75], index=["bid","ask"])
#         swap_points = pd.Series(data=[-0.1, -0.05], index=["bid","ask"])
#
#         self.currency = currency
#         self.prices = prices
#         self.swap_points = swap_points
#
#     def test_init(self):
#         """
#         """
#         fx_pos = FXPosition(currency=self.currency)
#
#         self.assertIsNone(fx_pos.position_type)
#         self.assertEqual(fx_pos.end_quantity, 0.0)
#         self.assertEqual(fx_pos.avg_price, 0.0)
#
#     def test_transact(self):
#         """
#         """
#         fx_pos = FXPosition(currency=self.currency)
#
#         # buy nothing
#         fx_pos.transact(quantity=-0.0, price=self.prices)
#         self.assertEqual(fx_pos.end_quantity, 0.0)
#         self.assertEqual(fx_pos.avg_price, 0.0)
#         self.assertIsNone(fx_pos.position_type)
#
#         # buy some
#         fx_pos.transact(quantity=0.25, price=self.prices)
#         self.assertEqual(fx_pos.avg_price, self.prices["ask"])
#         self.assertEqual(fx_pos.end_quantity, 0.25)
#         self.assertEqual(fx_pos.position_type, "long")
#
#         # sell some
#         fx_pos.transact(quantity=-0.25, price=self.prices)
#         self.assertEqual(fx_pos.end_quantity, 0.0)
#         self.assertIsNone(fx_pos.position_type)
#
#         # sell some more
#         fx_pos.transact(quantity=-0.5, price=self.prices)
#         self.assertEqual(fx_pos.end_quantity, 0.5)
#         self.assertEqual(fx_pos.position_type, "short")
#
#         # flip
#         fx_pos.transact(quantity=1.5, price=self.prices)
#         self.assertEqual(fx_pos.end_quantity, 1)
#         self.assertEqual(fx_pos.position_type, "long")
#
#     def test_flip(self):
#         """
#         """
#         fx_pos = FXPosition(currency=self.currency)
#         fx_pos.transact(+0.33, self.prices)
#         fx_pos.transact(-1.0, self.prices)
#
#         self.assertEqual(fx_pos.avg_price, self.prices["bid"])
#         self.assertEqual(fx_pos.realized_pnl,
#             -1*self.prices.diff().loc["ask"]*0.33)
#
#     def test_roll(self):
#         """
#         """
#         fx_pos = FXPosition(currency=self.currency)
#
#         # empty roll
#         fx_pos.roll_over(swap_points=self.swap_points)
#         self.assertEqual(fx_pos.avg_price, 0.0)
#
#         # non-empty roll
#         qty = 10
#         fx_pos.transact(quantity=qty, price=self.prices)
#         self.assertEqual(fx_pos.avg_price, self.prices["ask"])
#         fx_pos.roll_over(swap_points=self.swap_points)
#         self.assertEqual(fx_pos.avg_price,
#             self.prices["ask"]+self.swap_points["ask"])
#
#         # sell everything, roll_over
#         fx_pos = FXPosition(currency=self.currency)
#         fx_pos.transact(-11, price=self.prices)
#         fx_pos.roll_over(swap_points=self.swap_points)
#         self.assertEqual(fx_pos.avg_price,
#             self.prices["bid"]+self.swap_points["bid"])
#
#     def test_unrealized_pnl(self):
#         """
#         """
#         fx_pos = FXPosition(currency=self.currency)
#         self.assertEqual(fx_pos.get_unrealized_pnl(self.prices), 0.0)
#
#         fx_pos.transact(-2.0, price=self.prices)
#         self.assertEqual(fx_pos.get_unrealized_pnl(self.prices+1),
#             -2.0*(self.prices["ask"]+1 - self.prices["bid"]))
#
#         fx_pos.roll_over(self.swap_points)
#         self.assertEqual(fx_pos.get_unrealized_pnl(self.prices+1),
#             -2.0*(self.prices["ask"]+1 - self.prices["bid"] + 0.1))
#
#
# class TestPortfolio(unittest.TestCase):
#     """
#     """
#     def setUp(self):
#         """
#         """
#         currencies = ["nzd","chf"]
#         dt = pd.date_range("2001-01-03", periods=2, freq='D')
#
#         prices = pd.DataFrame({
#             "nzd": [0.75, 1.33],
#             "chf": [10.0, 11.0]}, index=["bid", "ask"]).T
#         swap_points = pd.DataFrame({
#             "nzd": [-0.1, -0.05],
#             "chf": [2.5, 3.0]}, index=["bid", "ask"]).T
#
#         positions = [FXPosition(currency=p) for p in currencies]
#
#         self.currencies = currencies
#         self.prices = prices
#         self.swap_points = swap_points
#         self.positions = positions
#
#     def test_init(self):
#         """
#         """
#         fx_pf = FXPortfolio(positions=self.positions)
#
#         self.assertEqual(np.abs(fx_pf.pw).mean(), 0.0)
#
#     def test_roll_over(self):
#         """
#         """
#         fx_pf = FXPortfolio(positions=self.positions)
#
#         fx_pf.roll_over(swap_points=self.swap_points)
#
#         for _, p in fx_pf.positions.iteritems():
#             self.assertEqual(p.avg_price, 0.0)
#
#     def test_rebalance(self):
#         """
#         """
#         fx_pf = FXPortfolio(positions=self.positions)
#
#         dp_1 = pd.Series([2,-2], index=self.currencies)
#         fx_pf.rebalance_from_dp(dp_1, prices=self.prices)
#
#         # print(dp)
#         # print(fx_pf.get_position_quantities())
#
#         self.assertTrue(
#             fx_pf.get_position_quantities().equals(np.abs(dp_1)))
#
#         # rebalance again
#         dp_2 = pd.Series([1,1], index=self.currencies)
#         fx_pf.rebalance_from_dp(dp_2, prices=self.prices)
#         self.assertTrue(
#             fx_pf.get_position_quantities().equals(np.abs(dp_1+dp_2)))
#
#     def test_rebalance_from_pw(self):
#         """
#         """
#         fx_pf = FXPortfolio(positions=self.positions)
#
#         pw_1 = pd.Series([0.0,1.0], index=self.currencies)
#         fx_pf.rebalance_from_dpw(pw_1, prices=self.prices)
#         print(fx_pf.positions)
#
#         pw_2 = pd.Series([0.5,-1.5], index=self.currencies)
#         fx_pf.rebalance_from_dpw(pw_2, prices=self.prices)
#         print(fx_pf.positions)
#
#     def test_fetch_info(self):
#         """
#         """
#         # equate bid and ask -> liquidation value = initial capital
#         prc = self.prices.copy()
#         prc.loc[:,"ask"] = prc.loc[:,"bid"]
#
#         fx_pf = FXPortfolio(positions=self.positions)
#         pw_1 = pd.Series([0.5,0.5], index=self.currencies)
#         fx_pf.rebalance_from_dpw(pw_1, prices=prc)
#
#         self.assertEqual(fx_pf.get_liquidation_value(prc), 1.0)

class TestFXTrading(unittest.TestCase):
    """
    """
    def setUp(self):
        """
        """
        curs = ["gbp", "chf"]
        dt = pd.date_range("2001-01-03", periods=5, freq='D')

        bid = pd.DataFrame(
            data=np.abs(np.random.normal(size=(5,2))),
            index=dt,
            columns=curs)
        ask = bid + 1.0
        prices = pd.Panel.from_dict({"bid": bid, "ask": ask}, orient="items")

        swap_points = prices/100

        signals = bid*0.0
        signals.iloc[3,0] = 1
        signals.iloc[2,1] = -1

        settings = {"h": 2}

        fxtr = FXTrading(prices, swap_points, signals, settings)

        self.prices = prices
        self.swap_points = swap_points
        self.signals = signals
        self.settings = settings

    def test_fxtrading(self):
        """
        """
        fx_tr = FXTrading(
            self.prices,
            self.swap_points,
            self.signals,
            self.settings)

        res = fx_tr.backtest()

        print(res)

if __name__ == "__main__":
    unittest.main()
