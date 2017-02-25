import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm
import warnings

# Regression
class RegressionModel():
    """
    """
    def __init__(self, y0, X0):
        """
        """
        if not isinstance(y0, np.ndarray):
            y, X = y0.align(X0, join="inner", axis=0, copy=True)
            y, X = y.values, X.values
        else:
            y, X = y0.copy(), X0.copy()

        self.y = y
        self.X = X if X.ndim > 1 else X[:,np.newaxis]

        self.y = self.y.astype(np.float)
        self.X = self.X.astype(np.float)

        # save mu and std anyway as harmless values for it makes life easier
        self.mu = np.zeros(shape=(X.shape[1]+1 if X.ndim > 1 else 2))
        self.sigma = self.mu + 1.0

    def bleach(self, z_score=False, add_constant=True, dropna=True):
        """ Prepare data for regression.
        """
        yX = np.hstack((self.y[:,np.newaxis], self.X))

        # prewhiten
        if z_score:
            # calculate mean and std
            mu = np.nanmean(yX, axis=0)
            sigma = np.nanstd(yX, axis=0)

            # demean and rescale
            yX = (yX-mu)/sigma

            # store mean and std
            self.mu = mu
            self.sigma = sigma

        # drop na from both y and X
        if dropna:
            yX = yX[np.isfinite(yX).all(axis=1),:]

        # reassemble: y is 1D, X is *always* 2D
        self.y = yX[:,0]
        self.X = yX[:,1:]

        # include column of ones if asked
        if add_constant:
            self.X = sm.add_constant(self.X)

class PureOls(RegressionModel):
    """
    >''<
    """
    def fit(self):
        """ Perform the simplest OLS possible.

        b = inv(X'X)(X'Y),
        var(b) = s^2*inv(X'X),
        s^2 = e'e/(T-K)

        At this point there should be no nans in self.X and self.y.

        Parameters
        ----------

        Returns
        -------
        b: (N,) numpy.ndarray
            OLS betas
        s2: float
            error variance (unbiased)
        """
        T,K = self.X.shape

        # b = inv(X'X)(X'Y)
        b, s2, _, _ = np.linalg.lstsq(self.X, self.y)

        # error variance
        s2 = s2.squeeze()/(T-K)
        self.s2 = s2

        # R^2
        R2 = 1-s2/np.var(self.y)
        R2_adj = 1-(1-R2)*(T-1)/(T-K-1)

        return b, R2_adj

class KernelRegression(RegressionModel):
    """
    """
    def cross_validation(self, k=10, h0_lo=1/3, h0_hi=2):
        """ k-fold kross-validation.
        """
        T = len(self.y)

        # need data to be prewhitened
        self.bleach(z_score=True, add_constant=False, dropna=True)

        # prelim regression -------------------------------------------------
        # X and X^2
        aux_X = np.stack(
            (self.X.mean(axis=1), (self.X**2).mean(axis=1)),
            axis=1)

        # add constant
        aux_X = sm.add_constant(aux_X)

        # regress
        coef_0, s2e, _, _ = np.linalg.lstsq(aux_X, self.y)

        # Silverman's rule of thumb
        h0 = T**(-1/5)*\
            abs(coef_0[2])**(-2/5)*\
            (s2e/T)**(1/5)*\
            np.diff(np.percentile(self.X.mean(axis=1), (0.1, 0.9)))**(1/5)*0.6

        # grid search -------------------------------------------------------
        # need y and X stacked together for yield_chunk
        yX = np.hstack((self.y[:,np.newaxis], self.X))
        # grid of h: [1/3*h0, 3*h0]
        h_grid = np.linspace(h0*h0_lo, h0*h0_hi, 10)
        # space for errors
        fit_errs = []
        # loop over h's
        for h in h_grid:
            fit_err = 0.0
            # loop over chunks of np.hstack((y,X))
            for p, q in self.yield_chunk(yX, k):
                # fit
                inplace_fit = p.fit(q.X, h)
                # record mean squared error
                fit_err += \
                    np.linalg.norm(inplace_fit-q.y)**2/len(inplace_fit)
            fit_errs += [fit_err,]

        # find minimum of all fit errors and corresponding h
        amin = np.argmin(fit_errs)
        best_h = h_grid[amin]
        # if optimal h is 1/3*h0 or 3*h0, raise warning
        if amin in [0, 9]:
            warnings.warn("Optimal h found at the {} boundary".format(
                "lower" if amin == 0 else "upper"))

        return best_h

    @staticmethod
    def yield_chunk(array, k=10):
        """ Generate ~equally-sized chunks of `array` and their complements.
        """
        # number of rows
        T = len(array)
        step = int(np.floor(T/k))

        # until out of rows, yield chunk
        p = 0
        while p < T-1:
            p += step
            idx = np.s_[(p-step):p]
            arr_est = np.delete(array, idx, axis=0)
            arr_pred = array[idx,:]
            yield (KernelRegression(arr_est[:,0], X0=arr_est[:,1:]),
                KernelRegression(arr_pred[:,0], X0=arr_pred[:,1:]))

    def fit(self, X_pred, h):
        """ Estimate E[Y|X] nonparametrically.

        Parameters
        ----------
        X_pred : (K,M) numpy.ndarray
            that very X in E[Y|X]

        Returns
        -------
        res : (K,) numpy.ndarray
            of fitted values
        """
        # X_pred must be of two dimensions
        if X_pred.ndim < 2:
            X_pred = X_pred.reshape((-1, 1))

        # bleach X_pred
        X_pred = (X_pred - self.mu[1:])/self.sigma[1:]

        # calculate weights: (K,T); last two args are to aux_wght_calc
        wght = np.apply_along_axis(func1d=self.aux_wght_calc, axis=1,
            arr=X_pred, X_orig=self.X, h=h)

        # weighted y's
        res = wght.dot(self.y)

        # add back mean and sigma (will multiply with 1 and add zero if no
        #   prewhitening took place)
        res = res*self.sigma[0] + self.mu[0]

        return res

    @staticmethod
    def aux_wght_calc(x_pred, X_orig, h):
        """ Calculate weight of each y at x_pred.

        Parameters
        ----------
        x_pred : (M,) numpy.ndarray
            one coordinate point
        X_orig : (T,M) numpy.ndarray
            of independent variables in `KernelRegression()`

        Returns
        -------
        wght : (T,) numpy.ndarray
            of weights for each y to calculated E[Y|X]
        """
        # deviations of `x_pred` from each point in X
        dev = np.prod(norm.pdf(X_orig-x_pred, scale=h), axis=1)

        # ensure summation to 1
        wght = dev/dev.sum()

        return wght

def light_ols(y, X, add_constant=False, ts=False):
    """
    """
    # init
    reg_mod = PureOls(y,X)
    # drop nans, add const if necessary
    reg_mod.bleach(add_constant=add_constant)
    # fit
    b, R2 = reg_mod.fit()
    # t-stats
    if ts:
        covmat = reg_mod.s2*np.linalg.inv(reg_mod.X.T.dot(reg_mod.X))
        se = np.sqrt(np.diag(covmat))
        t = b/se
        res = (b, R2, t)
    else:
        res = (b, R2)

    return res

class DynamicOLS():
    """ One-factor (+constant) OLS setting.
    """
    def __init__(self, method, y0, x0, **kwargs):
        """
        """
        self.method = method
        self.kwargs = kwargs

        self.y, self.x = y0.align(x0, join="inner", axis=0)

        # add name in case `y` does not have one
        if self.y.name is None:
            self.y.name = "response"
        if self.x.name is None:
            self.x.name = "regressor"

    def fit(self):
        """ Calculate rolling one-factor (+constant) beta of `y` wrt `x`.
        """
        x = self.x
        y = self.y

        # align y and x, keep names (important because of name of `y` mostly)
        yx = pd.concat((y, x), axis=1, ignore_index=False)

        # calculate covariance:
        #   items is time, major_axis and minor_axis are y.name + x.columns
        if self.method == "rolling":
            roll_cov_yx = yx.rolling(**self.kwargs).cov()
            # calculate beta
            b = roll_cov_yx.loc[:,y.name,x.name]/\
                roll_cov_yx.loc[:,x.name,x.name]
            # add name to `b`
            b.name = y.name
            # calculate alpha
            a = self.y.rolling(**self.kwargs).mean()-\
                b*self.x.rolling(**self.kwargs).mean()

        elif self.method == "expanding":
            roll_cov_yx = yx.expanding(**self.kwargs).cov()
            # calculate beta
            b = roll_cov_yx.loc[:,y.name,x.name]/\
                roll_cov_yx.loc[:,x.name,x.name]
            # add name to `b`
            b.name = y.name
            # calculate alpha
            a = self.y.expanding(**self.kwargs).mean()-\
                b*self.x.expanding(**self.kwargs).mean()

        elif self.method == "grouped_by":
            roll_cov_yx = yx.groupby(**self.kwargs).cov()
            # TODO: make this generic
            b = roll_cov_yx[y.name][:,:,x.name]/\
                roll_cov_yx[x.name][:,:,x.name]
            # add name to `b`
            b.name = y.name
            # calculate alpha
            a = self.y.groupby(**self.kwargs).mean()-\
                b*self.x.groupby(**self.kwargs).mean()

        return a, b

# if __name__ == "__main__":
#     x = pd.Series(data=np.random.normal(size=(100,)))
#     y = x*2 + np.random.normal(size=(100,))*0.5
#     rm = DynamicOLS(method="rolling", y0=y, x0=x, window=22)
#     b = rm.fit()
#     print(b)
    # yx = pd.concat((y,x), axis=1)
    # yx = pd.DataFrame(data=yx.values,
    #     index=pd.date_range("2011-01-01", periods=100, freq='D'))
    # lol = yx.groupby([lambda x: x.year, lambda x: x.month])\
    #     .cov()
    # lol[1][:,:,0]
    # y = yx[0]
    # x = yx[1]
