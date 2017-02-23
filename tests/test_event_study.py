import unittest
from numpy.testing import assert_almost_equal
import scipy.stats as st

import pandas as pd
import numpy as np
import string
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
        data_1d = pd.DataFrame(
            data=np.random.normal(size=(T,1))*sigma+mu,
            index=idx,
            columns=random.sample(string.ascii_lowercase, 1))

        data_2d = pd.DataFrame(
            data=np.random.normal(size=(T,N))*sigma+mu,
            index=idx,
            columns=random.sample(string.ascii_lowercase, N))

        # simulate events
        events = sorted(random.sample(idx.tolist()[T//3:T//2], K))

        # store
        cls.data_1d = data_1d
        cls.data_2d = data_2d
        cls.sigma = sigma
        cls.mu = mu
        cls.events = events
        cls.K = K
        cls.N = N

# class TestEventStudyInit(TestEventStudy):
#     """
#     """
#     def test_init_1d(self):
#         """
#         """
#         randint = random.randint(3,10)
#         evt_study = EventStudy(
#             data=self.data_1d,
#             events=self.events,
#             window=[-randint,-1,0,randint])
#
#         self.assertTupleEqual(evt_study.before.shape, (randint, self.K))
#
#     def test_init_2d(self):
#         """
#         """
#         randint = random.randint(3,10)
#         evt_study = EventStudy(
#             data=self.data_2d,
#             events=self.events,
#             window=[-randint,-1,0,randint])
#
#         self.assertTupleEqual(evt_study.before.shape,
#             (self.N, randint, self.K))

class TestEventStudyStuff(TestEventStudy):
    """
    """
    def setUp(self):
        """
        """
        randint = random.randint(3,10)
        evt_study = EventStudy(
            data=self.data_2d,
            events=self.events,
            window=[-randint,-1,0,randint])

        # confidence interval, upper bound
        ls = np.hstack((np.arange(randint,0,-1), np.arange(1,randint+2)))
        true_ci = self.sigma*1.65*np.sqrt(ls)/np.sqrt(self.K)+self.mu*ls

        self.true_ci = true_ci
        self.evt_study = evt_study
        self.randint = randint

    # def test_get_ts_cumsum(self):
    #     """
    #     """
    #     ts_mu = self.evt_study.get_ts_cumsum()
    #     self.assertTupleEqual(ts_mu.shape,
    #         (self.N, self.randint*2+1, self.K))
    #
    # def test_get_cs_mean(self):
    #     """
    #     """
    #     cs_mu = self.evt_study.get_cs_mean()
    #     self.assertTupleEqual(cs_mu.shape,
    #         (self.randint*2+1, self.N))
    #
    # def test_ci_simple(self):
    #     """
    #     """
    #     ci = self.evt_study.get_ci(ps=(0.05,0.95), method="simple")
    #     assert_almost_equal(ci.iloc[0,:,-1].values, self.true_ci, decimal=0)
    #
    # def test_ci_boot(self):
    #     """
    #     """
    #     ci = self.evt_study.get_ci(ps=0.9, method="boot", M=5000)
    #     assert_almost_equal(
    #         ci.iloc[0,:,-1].values,
    #         self.true_ci,
    #         decimal=0)

    def test_plot(self):
        """
        """
        fig = self.evt_study.plot()
        fig.savefig("c:/users/hsg-spezial/desktop/fig_evt.png")

if __name__ == "__main__":
    unittest.main()
# lol.cumsum().ix[:,3,:]
# lol
# wut = cumulants*lol.mean(axis=2).mean().values[np.newaxis,:]
# waf = -10*cumulants*lol.mean(axis=2).mean().values[np.newaxis,:]
# np.dstack((wut,waf)).shape
# pd.Panel(data=np.swapaxes(np.dstack((wut,waf)), 0, 1))
# cumulants*lol.mean(axis=2).mean().values[np.newaxis,:][0,0]
# np.arange(6)[3::-1]
# lol.apply(lambda x: x.quantile(0.95), axis="minor")
# lol = pd.Panel(np.empty(shape=(2,3,4)))
# lol
# lol.ix[:,:,0]
# lol.iloc[[0,],:,:].squeeze()
# lol.iloc[:,:,1]
# 1.5/np.sqrt(10)*1.95+1.
# lol.iloc[:,:,1].values.mean(axis=0)
# lol = pd.Panel(
#     data=np.random.normal(size=(5,1000,10)))
# wut = lol.cumsum(axis="items")
# wut = wut.mean(axis="minor_axis")
# 1/np.sqrt(10)*1.64*np.sqrt(np.arange(1,6))
# wut.quantile(0.95)
