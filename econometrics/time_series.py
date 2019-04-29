import numpy as np
import pandas as pd
from statsmodels.tsa import arima_model
from scipy import signal

from rpy2.robjects.packages import importr
from rpy2.robjects import Formula
from rpy2.robjects import pandas2ri
from rpy2.robjects import numpy2ri

pandas2ri.activate()
numpy2ri.activate()


class ARIMA:
    """ARIMA model.

    Parameters
    ----------
    data
    order
    kwargs

    """

    def __init__(self, data, order, **kwargs):
        """
        """
        self.order = order
        self.data = data.copy()

        self.model = arima_model.ARIMA(endog=data, order=order, **kwargs)

        self._fitted = None

    def fit(self, **kwargs):
        """

        Parameters
        ----------
        kwargs : dict

        Returns
        -------

        """
        fitted = self.model.fit(**kwargs)

        self._fitted = fitted

        return fitted

    def predict(self, fitted, out_of_sample=False, is_exp=False, **kwargs):
        """

        Parameters
        ----------
        out_of_sample
        kwargs

        Returns
        -------

        """
        if not out_of_sample:
            pass
        else:
            fcast = fitted.forecast(steps=1)[0][0]

        if is_exp:
            # jensen's term
            var_uhat = fitted.sigma2
            fcast = np.exp(fcast + 0.5 * var_uhat)

        return fcast


class GARCH:
    """ARMA-GARCH model.

    Model of the form:

        y_t = mu + ar_1*y_{t-1} + ... + ma_1*eps_{t-1} + eps_t (1)
        eps_t ~ d(0,sigma_t)
        sigma^2_t = omega + alpha_1*eps_{t-1}^2 + ... + beta_1*sigma^2_{t-1} (2)

    Such that the mean equation (1) is an ARMA process, and the variance
    equation (2) is a GARCH() process.

    Parameters
    ----------
    garch_lags: int
        number of ARCH lags in variance equation
    garch_lags: int
        number of GARCH lags in variance equation
    ar_lags: int
        autoregressive component of mean equation
    ma_lags: int
        moving average component of mean equation
    """

    def __init__(self, arch_lags, garch_lags, ar_lags=0, ma_lags=0):
        """
        """
        # import R packages
        self.r_base = importr("base")
        self.r_fGarch = importr("fGarch")
        self.r_stats = importr("stats")

        # model specs
        self.arch_lags = arch_lags
        self.garch_lags = garch_lags
        self.ar_lags = ar_lags
        self.ma_lags = ma_lags

        # assign coefficient names: omega, ar1,...,ma1,...,alpha1,...,beta1,...
        self.ar_names = ["ar" + str(n + 1) for n in range(ar_lags)]
        self.ma_names = ["ma" + str(n + 1) for n in range(ma_lags)]
        self.arch_names = ["alpha" + str(n + 1) for n in range(arch_lags)]
        self.garch_names = ["beta" + str(n + 1) for n in range(garch_lags)]

        # R formula of the equation
        self.formula = Formula(
            "~arma({ar:1d},{ma:1d})+garch({p:1d},{q:1d})".
                format(ar=self.ar_lags, ma=self.ma_lags,
                       p=self.arch_lags, q=self.garch_lags)
        )

    def fit(self, data, include_mean=True, **kwargs):
        """Estimate model parameters.

        Parameters
        ----------
        data : pandas.Series
            series of values
        include_mean : bool
            whether the unsonditional mean will be estimated or not

        Returns
        -------
        coef : dict
            coefficients
        h : pandas.Series
            conditional variances, with the same index as *data*

        """
        # clean up data
        data_clean = data.dropna()
        # data to R dataframe
        rdf = pandas2ri.py2ri(data_clean)

        # fit GARCH
        fitted = self.r_fGarch.garchFit(formula=self.formula, data=rdf,
                                        trace=False, include_mean=include_mean,
                                        **kwargs)

        fitted.data_orig = data
        fitted.data_clean = data_clean

        return fitted

    def get_h(self, fitted):
        """

        Parameters
        ----------
        fitted

        Returns
        -------

        """
        coef = self.get_coef(fitted)

        # conditional variances
        h = pandas2ri.ri2py(fitted.slots['h.t'])
        h = pd.Series(data=h, index=fitted.data_clean.index)

        return h

    def predict(self, fitted, n_ahead=1, s2_hat_init=None):
        """

        Parameters
        ----------
        fitted
        n_ahead : int
        s2_hat_init : float

        Returns
        -------
        res : float or numpy.ndarray

        """
        coef = self.get_coef(fitted)

        if s2_hat_init is None:
            s2_hat_init = self._predict_one(fitted)

        if n_ahead < 2:
            return s2_hat_init

        else:
            res = pd.Series(index=range(1, n_ahead + 1))
            res.loc[1] = s2_hat_init

            # for t in range(2, n_ahead+1):
            #     res.loc[t] = coef.loc["omega"] +\
            #                  (coef.loc["alpha1"] + coef.loc["beta1"]) *\
            #                  res.loc[t-1]

            x = np.ones((n_ahead - 1,)) * coef["omega"]
            b = [1, ]
            a = [1, -coef.loc[["alpha1", "beta1"]].sum()]
            zi = signal.lfiltic(b, a, [s2_hat_init])

            res.loc[2:] = signal.lfilter(b, a, x, zi=zi)[0]

        return res

    def _predict_one(self, fitted):
        """

        Parameters
        ----------
        fitted

        Returns
        -------

        """
        # lagged conditional variances
        h = pandas2ri.ri2py(fitted.slots['h.t'])[-self.garch_lags:]
        h = pd.Series(data=h, index=self.garch_names)

        # lagged squared residuals
        eps = pandas2ri.ri2py(fitted.slots['residuals'])[-self.arch_lags:] ** 2
        eps = pd.Series(data=eps, index=self.arch_names)

        # 1 to be multiplied with omega
        omega = pd.Series(data=[1], index=["omega"])

        # all together
        data = pd.concat((h, eps, omega))

        res = self.get_coef(fitted).drop("mu", errors="ignore").dot(data)

        return res

    def get_coef(self, fitted):
        """

        Parameters
        ----------
        fitted

        Returns
        -------

        """
        # fetch coefficients from model output
        coef = pandas2ri.ri2py(fitted.slots['fit'][0])

        # assign coefficient names: omega, ar1,...,ma1,...,alpha1,...,beta1,...
        all_names = self.ar_names + ['omega'] + self.ma_names + \
                    self.arch_names + self.garch_names

        # if mean is there, prepend it
        n_coef = self.ar_lags + self.ma_lags + self.arch_lags + self.garch_lags
        if len(coef) > n_coef + 1:
            all_names = ["mu", ] + all_names

        # store coefficients as a dict with the above names
        coef = pd.Series(dict(zip(all_names, coef)))

        return coef

    def __str__(self):
        """
        String representation
        """
        res = "ARMA({ar:<1d},{ar:<1d})+GARCH({p:<1d},{q:<1d}) model" \
            .format(ar=self.ar_lags, ma=self.ma_lags,
                    p=self.arch_lags, q=self.garch_lags)
        return res

