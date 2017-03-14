import pandas as pd
import numpy as np
import statsmodels.api as sm

class Regression():
    """
    Parameters
    ----------
    y0: pandas.Series
        dependent variable/response vector (required)
    X0: pandas.DataFrame/Series
        DataFrame, independent variables/design matrix (required)
    add_constant: bool
        if a constant should be added
    lag_x: int/list
        if some lags of X should be added
    """

    def __init__(self, y0, X0, add_constant=False, lag_x=0):
        """ Prepare data for regression
        """
        # copy
        y, X = y0.copy(), X0.copy()

        # everything better be frames
        if isinstance(X, pd.Series):
            X = X.to_frame()
        if isinstance(y, pd.Series):
            y = y.to_frame()

        # lag regressors
        if not hasattr(lag_x, "__iter__"):
            lag_x = range(lag_x)

        X = pd.concat([X,]+[X.shift(p) for p in lag_x], axis=1)

        # save originally-shaped inputs
        y_orig = y.copy()
        X_orig = sm.add_constant(X) if add_constant else X.copy()

        # align data if needed
        X, y = X.dropna(how="any").align(y.dropna(), join="inner", axis=0)

        # add constant
        if add_constant:
            X = sm.add_constant(X)

        self.X = X
        self.y = y
        self.X_names = X.columns
        self.Y_names = y.columns
        self.X_orig = X_orig
        self.y_orig = y_orig

class PureOls(Regression):
    """
    Performs the purest old-school no-nonsence OLS, inv(X'X)(X'Y), just like
    in the good old times.
    """
    def __init__(self, y0, X0, add_constant, **kwargs):

        super().__init__(y0=y0, X0=X0, add_constant=add_constant, **kwargs)

    def fit(self):
        """
        Evaluate inv(X'X)(X'Y) (in a smart way). Y can have rank > 1 (oh yeah).

        Returns
        -------
        coef: {(N,), (N, K)} ndarray
            coefficients of N regressors for K columns in Y

        """
        coef, _, _, _ = np.linalg.lstsq(self.X.values, self.y.values)

        coef = pd.Series(data=coef.squeeze(), index=self.X_names)

        self.coef = coef

        return coef

    def get_diagnostics(self):
        """
        """
        if not hasattr(self, "eps"):
            _ = self.get_residuals()

        eps_var = self.eps.var().values
        XX = self.X.T.dot(self.X)
        vcv = eps_var*np.linalg.inv(XX)
        se = np.diag(vcv).squeeze()
        tstat = self.coef/se

        res = pd.DataFrame(
            data=np.vstack((self.coef.values, se, tstat.values)),
            columns=self.X_names,
            index=["coef", "se", "tstat"])

        return res

    def get_yhat(self, original=True):
        """
        """
        if not hasattr(self, "coef"):
            _ = self.fit()

        if original:
            yhat = self.X_orig.dot(self.coef).reindex(index=self.y_orig.index)
        else:
            yhat = self.X.dot(self.coef)

        # to frame + rename
        yhat = yhat.to_frame()
        yhat.columns = self.Y_names

        self.yhat = yhat

        return yhat.squeeze()

    def get_residuals(self, original=True):
        """
        Retrieves residuals from regression
        """
        if not hasattr(self, "yhat"):
            _ = self.get_yhat(original=original)

        if original:
            eps = self.y_orig - self.yhat
        else:
            eps = self.y - self.yhat

        # rename
        eps.columns = self.Y_names

        self.eps = eps

        return eps.squeeze()

if __name__ == '__main__':
    pass
    # X = pd.DataFrame(data=np.random.normal(size=(100,2)))
    # y = X.dot(pd.Series(data=[2,5], index=X.columns))
    # mod = PureOls(y, X, add_constant=False)
    # print(mod.get_diagnostics())
