import pandas as pd
import numpy as np
from foolbox import portfolio_construction as poco
from foolbox.linear_models import PureOls, DynamicOLS


class FactorModelEnvironment:
    """
    """
    def __init__(self, assets, factors):
        """
        """
        self.assets = assets
        self.factors = factors

    def get_betas(self, method="simple", denom=False, **kwargs):
        """Calculate betas.

        Parameters
        ----------
        method : str
            'simple', 'rolling', 'expanding', 'grouped_by';
            example with grouped_by: [lambda x: x.year, lambda x: x.month]

        Returns
        -------
        res : pandas.core.generic.NDFrame
            with betas: DataFrame or Panel, depending on the input
        """
        if method not in ("simple", "rolling", "expanding", "grouped_by"):
            raise ValueError("Method not implemented.")

        if method == "simple":
            mod = PureOls(y0=self.assets, X0=self.factors, add_constant=True)
            res = mod.fit(**kwargs).T
        else:
            mod = DynamicOLS(y0=self.assets, x0=self.factors)
            res = mod.fit(method=method, denom=denom, **kwargs)

        return res

    def get_factor_variance(self, method="simple"):
        """

        Parameters
        ----------
        method

        Returns
        -------

        """
        pass

    @classmethod
    def from_weights(cls, assets, weights=None, exclude_self=False):
        """
        """
        if weights is None:
            # rowsums
            weights = assets.notnull().astype(float)

        # if exclude_self
        if exclude_self:
            factors = [
                construct_factor(
                    assets=assets,
                    weights=weights.drop(c, axis=1)).rename(c)
                for c in assets.columns]
            factors = pd.concat(factors, axis=1)

            # one environment for each asset
            res = {
                k: cls(assets=assets[k], factors=factors[k].rename("factor")) \
                for k in assets.columns}

        else:
            # else as usual
            factors = construct_factor(assets, weights).to_frame("factor")
            res = cls(assets=assets, factors=factors)

        return res


def construct_factor(assets, weights):
    """
    """
    w = poco.normalize_weights(weights)
    factor = assets.mul(w, axis=0).sum(axis=1)

    return factor


if __name__ == "__main__":

    pass
