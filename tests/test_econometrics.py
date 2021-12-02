from unittest import TestCase
import pandas as pd
import numpy as np
from numpy.testing import assert_almost_equal

from foolbox.econometrics.misc import descriptives
from foolbox.econometrics.estimators import estimate_covmat


class TestMisc(TestCase):

    def setUp(self) -> None:
        self.data = pd.DataFrame(np.random.normal(size=(10000, 3)))

    def test_descriptives(self):
        descriptives(self.data)


class TestEstimators(TestCase):

    def setUp(self) -> None:
        self.data = pd.DataFrame(np.random.normal(size=(10000, 3)))

    def test_estimate_covmat(self):
        vcv = estimate_covmat(self.data, assume_centered=True)
        assert_almost_equal(vcv.values, np.eye(3, dtype=float), decimal=1)
