import unittest
# from numpy.testing import assert_almost_equal

import pandas as pd
import numpy as np
# import ipdb

from foolbox.fxtrading import *

class TestFXTrading(unittest.TestCase):
    """
    """
    def setUp(self):
        """
        """
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

        settings = {"holding_period": 1, "blackout": 1}

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
