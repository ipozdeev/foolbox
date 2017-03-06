import unittest
from numpy.testing import assert_almost_equal

import pandas as pd
import numpy as np
import random

from foolbox.EventStudy import EventStudy

class TestEventStudy(unittest.TestCase):
    """
    """
    @classmethod
    def setUpClass(cls):
        """
        """
        # parameters of distribution
        sigma = 1.5
        mu = 1.0
        # number of observations
        T = 10000
        # number pf assets in 2D case
        N = 5
        # number of events
        K = 10

        # time index
        idx = pd.date_range("1990-01-01", periods=T, frequency='D')

        # simulate data
        data_1d = pd.Series(
            data=np.random.normal(size=(T,))*sigma+mu,
            index=idx)
        data_1d.name = "haha"

        # simulate events
        events = sorted(random.sample(idx.tolist()[T//3:T//2], K))
        events = pd.Series(index=events, data=np.arange(len(events)))

        # store
        cls.data_1d = data_1d
        cls.sigma = sigma
        cls.mu = mu
        cls.events = events
        cls.K = K
        cls.N = N

class TestEventStudyInit(TestEventStudy):
    """
    """
    def test_init_1d(self):
        """
        """
        randint = random.randint(3,10)
        evt_study = EventStudy(
            data=self.data_1d,
            events=self.events,
            window=[-randint,-1,0,randint])

        self.assertTupleEqual(evt_study.before.shape, (randint, self.K))

class TestEventStudyStuff(TestEventStudy):
    """
    """
    def setUp(self):
        """
        """
        randint = random.randint(3,10)
        evt_study = EventStudy(
            data=self.data_1d,
            events=self.events,
            window=[-randint,-1,0,randint])

        # confidence interval, upper bound
        ls = np.hstack((np.arange(randint,0,-1), np.arange(1,randint+2)))
        true_ci = self.sigma*1.65*np.sqrt(ls)/np.sqrt(self.K)+self.mu*ls

        self.true_ci = true_ci
        self.evt_study = evt_study
        self.randint = randint

    def test_get_ts_cumsum(self):
        """
        """
        ts_mu = self.evt_study.get_ts_cumsum(
            self.evt_study.before, self.evt_study.after)
        self.assertTupleEqual(ts_mu.shape,
            (self.randint*2+1, self.K))

    def test_get_cs_mean(self):
        """
        """
        cs_mu = self.evt_study.get_cs_mean(
            self.evt_study.before, self.evt_study.after)
        self.assertTupleEqual(cs_mu.shape,
            (self.randint*2+1,))

    def test_ci_simple(self):
        """
        """
        ci = self.evt_study.get_ci(ps=(0.05,0.95), method="simple")
        assert_almost_equal(ci.iloc[:,-1].values, self.true_ci, decimal=0)

    # def test_ci_boot(self):
    #     """
    #     """
    #     ci = self.evt_study.get_ci(ps=0.9, method="boot", M=500)
    #     assert_almost_equal(
    #         ci.iloc[:,-1].values,
    #         self.true_ci,
    #         decimal=0)

    def test_plot(self):
        """
        """
        ci = self.evt_study.get_ci(ps=0.9, method="simple")
        fig = self.evt_study.plot()
        fig.show()
        # fig.savefig("c:/users/hsg-spezial/desktop/fig_evt.png")

if __name__ == "__main__":
    unittest.main()
