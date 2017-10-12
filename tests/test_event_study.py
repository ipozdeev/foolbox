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
        mu = 0.0
        # number of observations
        T = 10000
        # number pf assets in 2D case
        N = 5
        # number of events
        K = 1

        # time index
        idx = pd.date_range("1990-01-01", periods=T, frequency='D')

        # simulate data 1D
        data_1d = pd.Series(
            data=np.random.normal(size=(T,))*sigma+mu,
            index=idx)
        data_1d.name = "haha"

        # simulate data 2D
        data_2d = pd.DataFrame(
            data=np.random.normal(size=(T,N))*sigma,
            index=idx,
            columns=range(N))

        # simulate events
        events = sorted(random.sample(idx.tolist()[T//3:T//2], K))
        events = pd.Series(index=events, data=np.arange(len(events)))

        # events 2D
        events_2d = pd.concat(
            [pd.Series(
                index=sorted(random.sample(idx.tolist()[T//5:T//2], K)),
                data=np.arange(K)) for p in range(N)],
            axis=1)

        # store
        cls.data_1d = data_1d
        cls.data_2d = data_2d
        cls.sigma = sigma
        cls.mu = mu
        cls.events = events
        cls.events_2d = events_2d
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

# class TestEventStudy1D(TestEventStudy):
#     """
#     """
#     def setUp(self):
#         """
#         """
#         randint = random.randint(3,10)
#         evt_study = EventStudy(
#             data=self.data_1d,
#             events=self.events,
#             normal_data=self.mu,
#             window=[-randint,-1,0,randint],
#             x_overlaps=True)
#
#         # confidence interval, upper bound
#         ls = np.hstack((np.arange(randint,0,-1), np.arange(1,randint+2)))
#         true_ci = self.sigma*1.95*np.sqrt(ls)/np.sqrt(self.K)+self.mu*ls
#
#         self.true_ci = true_ci
#         self.evt_study = evt_study
#         self.randint = randint
#
#     def test_ci_boot(self):
#         """
#         """
#         ci = self.evt_study.get_ci(ps=(0.025, 0.975), method="boot", M=50)
#
#         print(ci)
#         print(self.true_ci)
#
#     def test_ci_simple(self):
#         """
#         """
#         ci = self.evt_study.get_ci(ps=(0.025, 0.975), method="simple")
#
#         print(ci)
#         print(self.true_ci)


class TestEventStudy2D(TestEventStudy):
    """
    """
    def setUp(self):
        """
        """
        randint = random.randint(3,10)
        evt_study = EventStudy(
            data=self.data_2d,
            events=self.events_2d,
            normal_data=0.0,
            mean_type="count_weighted",
            window=[-randint,-1,0,randint],
            x_overlaps=True)

        # confidence interval, upper bound
        ls = np.hstack((np.arange(randint,0,-1), np.arange(1,randint+2)))
        true_ci = self.sigma*1.95*np.sqrt(ls)/np.sqrt(self.K)/np.sqrt(self.N)

        self.true_ci = true_ci
        self.evt_study = evt_study
        self.randint = randint

    # def test_ci_simple(self):
    #     """
    #     """
    #     ci = self.evt_study.get_ci(ps=(0.025, 0.975), method="simple")
    #
    #     print(ci)
    #     print(self.true_ci)

    def test_ci_boot(self):
        """
        """
        ci_1 = self.evt_study.get_ci(ps=(0.025, 0.975), method="boot", M=150)
        ci_2 = self.evt_study.get_ci(ps=(0.05, 0.95), method="boot", M=150)

        print(ci_1)
        print(ci_2)
        print(self.true_ci)


    # def test_get_ts_cumsum(self):
    #     """
    #     """
    #     ts_mu = self.evt_study.get_ts_cumsum(
    #         self.evt_study.before, self.evt_study.after)
    #     self.assertTupleEqual(ts_mu.shape,
    #         (self.randint*2+1, self.K))
    #
    # def test_get_cs_mean(self):
    #     """
    #     """
    #     cs_mu = self.evt_study.get_cs_mean(
    #         self.evt_study.before, self.evt_study.after)
    #     self.assertTupleEqual(cs_mu.shape,
    #         (self.randint*2+1,))
    #
    # def test_ci_simple(self):
    #     """
    #     """
    #     ci = self.evt_study.get_ci(ps=(0.05,0.95), method="simple")
    #     assert_almost_equal(ci.iloc[:,-1].values, self.true_ci, decimal=0)

    # def test_ci_boot(self):
    #     """
    #     """
    #     ci = self.evt_study.get_ci(ps=0.9, method="boot", M=500)
    #     assert_almost_equal(
    #         ci.iloc[:,-1].values,
    #         self.true_ci,
    #         decimal=0)

    # def test_plot(self):
    #     """
    #     """
    #     ci = self.evt_study.get_ci(ps=0.9, method="simple")
    #     fig = self.evt_study.plot()
    #     fig.show()
    #     # fig.savefig("c:/users/hsg-spezial/desktop/fig_evt.png")

# class TestSignalFromEvents(unittest.TestCase):
#     """
#     """
#     def setUp(self):
#         """
#         """
#         data = pd.DataFrame(
#             data=np.random.normal(size=(20,2)),
#             index=pd.date_range("2011-01-01",periods=20,freq='D'),
#             columns=["first","second"])
#         events = pd.Series(
#             index=["2011-01-06", "2011-01-15"],
#             data=np.arange(2))
#         window = (-3,-1)
#
#         self.data = data
#         self.events = events
#         self.window = window
#
#     def test_signal_from_events_sum(self):
#         """
#         """
#         res = signal_from_events(self.data, self.events, self.window)
#
#         self.assertTupleEqual(res.shape, (2,2))
#         self.assertEqual(
#             res.ix[0,0],
#             self.data.loc["2011-01-03":"2011-01-05","first"].sum())
#
#     def test_signal_from_events_max_cumsum(self):
#         """
#         """
#         func = lambda x: max(x.cumsum())
#         res = signal_from_events(self.data, self.events, self.window, func)
#
#         self.assertTupleEqual(res.shape, (2,2))
#         self.assertEqual(
#             res.ix[0,0],
#             self.data.loc["2011-01-03":"2011-01-05","first"].cumsum().max())


if __name__ == "__main__":
    unittest.main()
