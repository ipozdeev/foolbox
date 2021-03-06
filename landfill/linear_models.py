import pandas as pd
idx = pd.IndexSlice

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn.apionly as sns
from foolbox.data_mgmt.set_credentials import *

import rpy2.robjects as robj
from rpy2.robjects.packages import importr
from rpy2.robjects import Formula

from rpy2.robjects import pandas2ri
pandas2ri.activate()

from rpy2.robjects import numpy2ri
numpy2ri.activate()
from scipy.stats import chi2

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
            X = X.to_frame(name=X.name)
        if isinstance(y, pd.Series):
            y = y.to_frame(name=y.name)

        # lag regressors
        if not hasattr(lag_x, "__iter__"):
            lag_x = range(lag_x)

        X = pd.concat([X, ] + [X.shift(p) for p in lag_x], axis=1)

        # save originally-shaped inputs
        y_orig = y.copy()
        X_orig = sm.add_constant(X) if add_constant else X.copy()

        # align data if needed
        X, y = X.dropna(how="any").align(y.dropna(how="any"),
            join="inner",
            axis=0)

        # add constant
        if add_constant:
            X = sm.add_constant(X)

        self.add_constant = add_constant
        self.X = X
        self.y = y
        self.X_names = X.columns
        self.Y_names = y.columns
        self.X_orig = X_orig
        self.y_orig = y_orig

    def fit(self, **kwargs):
        """
        """
        pass

    def get_yhat(self, newdata=None, original=True):
        """
        """
        if not hasattr(self, "coef"):
            _ = self.fit()

        if newdata is not None:
            if self.add_constant:
                newdata = sm.add_constant(newdata.copy())
            yhat = newdata.dot(self.coef)
        elif original is True:
            yhat = self.X_orig.dot(self.coef).reindex(index=self.y_orig.index)
        else:
            yhat = self.X.dot(self.coef)

        # to frame + rename
        if isinstance(yhat, pd.Series):
            yhat = yhat.to_frame()

        yhat.columns = self.Y_names

        self.yhat = yhat

        return yhat.squeeze()

    def get_residuals(self, original=True):
        """Retrieve residuals from regression."""
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


class PureOls(Regression):
    """
    Performs the purest old-school no-nonsence OLS, inv(X'X)(X'Y), just like
    in the good old times.
    """
    def fit(self):
        """
        Evaluate inv(X'X)(X'Y) (in a smart way). Y can have rank > 1 (oh yeah).

        Returns
        -------
        coef: pandas.DataFrame
            where (K, N)-th value is coef of asset N w.r.t. factor K
        """
        coef_arr, _, _, _ = np.linalg.lstsq(self.X.values, self.y.values)

        coef = pd.DataFrame(data=coef_arr,
            index=self.X.columns,
            columns=self.y.columns)

        self.coef = coef

        return coef

    def get_diagnostics(self, HAC=True):
        """
        """
        if HAC:
            rdata = pandas2ri.py2ri(pd.concat((self.y, self.X), axis=1))

            # import lm and base
            lm = robj.r['lm']
            base = importr('base')

            # write down formula: y ~ x
            fmla = Formula(self.Y_names[0] + " ~ . - 1")

            env = fmla.environment
            env["rdata"] = rdata

            # fit model
            f = lm(fmla, data = rdata)

            # extract coefficients
            coef = pd.Series(f.rx2('coefficients'), index=self.X_names)

            # implementation of Newey-West (1997)
            nw = importr("sandwich")

            # errors
            vcv = nw.NeweyWest(f)
            #vcv = nw.vcovHAC(f)

            # fetch coefficients from model output
            se = pd.Series(np.sqrt(np.diag(vcv)), index=self.X_names)

            # tstat
            tstat = coef / se
            # R-squared
            rsq = sm.OLS(endog=self.y, exog=self.X).fit().rsquared_adj

            # concat and transpose
            res = pd.DataFrame.from_dict({
                "coef": coef,
                "se": se,
                "tstat": tstat}).T
            # add r-squared
            res.loc["adj r2", res.columns[0]] = rsq

            # add nobs
            res.loc["nobs", res.columns[0]] = len(self.y)

            return res

        if not hasattr(self, "eps"):
            _ = self.get_residuals()

        eps_var = self.eps.var().values
        XX = self.X.T.dot(self.X)
        vcv = eps_var*np.linalg.inv(XX)
        se = (np.diag(vcv) ** 0.5).squeeze()
        se = pd.DataFrame(se, index=self.coef.index, columns=self.coef.columns)
        tstat = self.coef/se

        res = pd.concat([self.coef, se, tstat], axis=1).T
        res.columns = self.X_names
        res.index = ["coef", "se", "tstat"]

        return res

    def linear_restrictions_test(self, R, r, HAC=True):
        """Tests linear restrictions on the coeficients in the following form:

                            R * theta_hat = r

        Parameters
        ----------
        R: pd.DataFrame
            with columns named according parameters combinations of which are
            to be tested (K) less or equal the total number of estimated
            parameters, and number of rows equal to the number of linear
            restrictions (q)
        r: pd.Series
            of size q, containing the null hypothesis value for each linear
            restriction. Index is same as in R
        HAC: bool
            whether a robust VCV should be used. Default is True

        Returns
        -------
        res: pd.Series
            with chi-squared - statistic with dof = q, and the corresponding
            p-value

        """
        # Only HAC, only hardcore
        # TODO: refactor, so the estimation doesn't duplicate get_diagnostics()
        if HAC:
            # R-part
            rdata = pandas2ri.py2ri(pd.concat((self.y, self.X), axis=1))

            # import lm and base
            lm = robj.r['lm']
            base = importr('base')

            # write down formula: y ~ x
            fmla = Formula(self.Y_names[0] + " ~ . - 1")

            env = fmla.environment
            env["rdata"] = rdata

            # fit model
            f = lm(fmla, data = rdata)

            # extract coefficients
            coef = pd.Series(f.rx2('coefficients'), index=self.X_names)

            # implementation of Newey-West (1997)
            nw = importr("sandwich")

            # errors
            vcv = nw.NeweyWest(f)

            # vcv
            vcv = pd.DataFrame(np.array(vcv), index=self.X_names,
                               columns=self.X_names)

            # Select the subset with parameters involved in test
            vcv = vcv.loc[R.columns, R.columns]
            coef = coef.loc[R.columns]

            # Compute the Wald statistic
            W = (R.dot(coef) - r).T \
                .dot(np.linalg.inv(R.dot(vcv).dot(R.T))) \
                .dot(R.dot(coef) - r)

            # Degrees of freedom equals the number of restrictions
            q = R.shape[0]

            # Compute p-value
            p_val = 1 - chi2.cdf(W, q)

            # Organize output
            out = pd.Series([W, p_val], index=["chi_sq", "p_val"])

        return out

    def hodrick_vcv(self, forecast_horizon):
        """Computes Hodrick (1992) variance-covariance matrix uder the null
        hypothesis of no predictability for a regression of the following type:

                    y(t -> t+H) = X(t) * beta + epsilon(t -> t+H),

        where t -> t+H, means that y is forecast H-steps ahead

        Parameters
        ----------
        forecast_horizon: int
            specifying forecast horizon H in the example above

        Returns
        -------
        hodrick_vcv: pd.DataFrame
            with Hodrick (1992) VCV estimate

        """
        # Dimensionality of regressors' matrix
        (T, N) = self.X.shape

        # Get demeaned residuals from OLS on a constant
        resids = self.y - self.y.mean()

        # Compute Hodrick's spectral density
        spectral_density = pd.DataFrame(np.zeros((N, N)), index=self.X.columns,
                                        columns=self.X.columns)

        # Spectral density for each point in time
        sd = self.X.rolling(forecast_horizon).sum().shift(1)\
            .mul(resids.squeeze(), axis=0).dropna()

        # Get the time average
        spectral_density = (1 / T) * sd.T.dot(sd)

        # Get the 'sandwich' part of vcv, i.e. E[xx']
        Z = (1 / T) * self.X.T.dot(self.X)

        # Get the output
        hodrick_vcv = \
            np.linalg.inv(Z).dot(spectral_density).dot(np.linalg.inv(Z)) / T

        hodrick_vcv = pd.DataFrame(hodrick_vcv, index=self.X.columns,
                                   columns=self.X.columns)

        return hodrick_vcv

    def to_latex(self, inference="se", fmt_coef="{:.4f}",
                 fmt_inference="({:.2f})", **kwargs):
        """

        Parameters
        ----------
        inference : str
            report errors of the coefficients 'se' or t-stats 't-stat'

        Returns
        -------

        """
        est = self.get_diagnostics(**kwargs)

        coef = est.loc["coef", :]
        infce = est.loc[("se" if inference.startswith("s") else "tstat"), :]

        x_coef = r"\underset{{" + fmt_inference + "}}{{" + fmt_coef + "}}"
        x_name = "X_{{{:s}}}"

        res = " + ".join(
            (x_coef + ' ' + x_name).format(infce[k], coef[k], str(k))
            for k in coef.index if k != "const")

        if self.add_constant:
            res = x_coef.format(infce["const"], coef["const"], "const") + \
                  " + " + res

        res = "y = " + res

        return res


class DynamicOLS:
    """One-factor (+constant) OLS setting."""
    def __init__(self, y0, x0):
        """
        """
        # assert isinstance(x0.squeeze(), pd.Series)

        y, x = y0.align(x0, axis=0, join="inner")

        if isinstance(y, pd.Series):
            if y.name is None:
                y.name = "response"
            y = y.to_frame()

        # add name in case `y` does not have one
        if isinstance(x, pd.Series):
            x = x.to_frame(x.name if x.name is not None else "regressor")
            self.x_colnames = x.columns

        self.x = x
        self.y = y

    def fit(self, method, denom=False, **kwargs):
        """
        Parameters
        ----------
        method : str
            'simple', 'rolling', 'expanding', 'grouped_by';
            example with grouped_by: by=TimeGrouper(freq='M')
        denom : bool

        Returns
        -------
        res : pandas.core.generic.NDFrame
            with betas: DataFrame or Panel, depending on the input
        """
        if self.y.shape[1] < 2:
            res = self.fit_to_series(self.y.squeeze(), self.x,
                                     method=method, denom=denom, **kwargs)
        else:
            res = dict()
            for c, col in self.y.iteritems():
                res[c] = self.fit_to_series(col, self.x, method=method,
                                            denom=denom, **kwargs)

            res = pd.concat(res, axis=1)

        return res

    @staticmethod
    def fit_to_series(y, x, method, denom=False, **kwargs):
        """Calculate rolling one-factor (+constant) beta of a series."""

        # assert isinstance(y, pd.Series) & isinstance(x, pd.Series)
        assert isinstance(y, pd.Series)

        # helper to compute inv(X'X)
        def invert_pandas(df):
            aux_res = pd.DataFrame(np.linalg.inv(df), index=df.index,
                                   columns=df.columns)
            return aux_res

        # align y and x, keep names (important because of name of `y` mostly)
        y_concat_x = pd.concat((y, x), axis=1, ignore_index=False)

        # # dropna, demean
        # yx = yx.dropna() - yx.dropna().mean()
        y_concat_x = y_concat_x.dropna()

        # calculate covariance:
        #   items is time, major_axis and minor_axis are y.name + x.columns
        if method in ("rolling", "expanding"):
            # rolling covariance
            atr = getattr(y_concat_x, method)
            roll_cov_yx = atr(**kwargs).cov()

            # roll_cov_yx = y_concat_x.rolling(**kwargs).cov()

            # fetch X'X
            xx = roll_cov_yx.drop(y.name, axis=1).drop(y.name, axis=0, level=1)

            # fetch X'Y
            xy = roll_cov_yx[y.name].drop(y.name, axis=0, level=1)

            # calculate beta: inv(X'X)(X'Y)
            xx_inv = xx.groupby(axis=0, level=0).apply(invert_pandas)
            b = xx_inv.mul(xy, axis=0).groupby(axis=0, level=0).sum()

            # b_old = roll_cov_yx.xs(y.name, level=1, axis=0, drop_level=True)\
            #     .loc[:, x.name] /\
            #     roll_cov_yx.xs(x.name, level=1, axis=0, drop_level=True)\
            #     .loc[:, x.name]

            # calculate alpha: a = mean(y) - mean(y_hat)
            # a = y.rolling(**kwargs).mean() - \
            #     x.rolling(**kwargs).mean().mul(b).sum(axis=1)
            a = getattr(y, method)(**kwargs).mean() - \
                getattr(x, method)(**kwargs).mean().mul(b).sum(axis=1)

            if denom:
                # d = roll_cov_yx.loc[:, x.name, x.name]
                d = np.nan

        # elif method == "expanding":
        #     # expanding covariance
        #     roll_cov_yx = y_concat_x.expanding(**kwargs).cov()
        #
        #     # calculate beta
        #     b = roll_cov_yx.xs(y.name, level=1, axis=0, drop_level=True)\
        #         .loc[:, x.name] /\
        #         roll_cov_yx.xs(x.name, level=1, axis=0, drop_level=True)\
        #             .loc[:, x.name]
        #     # b = roll_cov_yx.loc[idx[:, y.name], x.name] /\
        #     #     roll_cov_yx.loc[idx[:, x.name], x.name]
        #
        #     # calculate alpha
        #     a = y.expanding(**kwargs).mean() - b*x.expanding(**kwargs).mean()
        #
        #     if denom:
        #         d = roll_cov_yx.loc[:, x.name, x.name]

        elif method == "grouped_by":
            # grouped_by covariance
            roll_cov_yx = y_concat_x.groupby(**kwargs).cov()
            if roll_cov_yx.index.nlevels > 2:
                raise ValueError("Ensure the grouper returns two levels!")

            b = roll_cov_yx.loc[idx[:, y.name], x.name].xs(
                y.name, level=1, axis=0) /\
                roll_cov_yx.loc[idx[:, x.name], x.name].xs(
                    x.name, level=1, axis=0)

            # calculate alpha
            a = y.groupby(**kwargs).mean() - b*x.groupby(**kwargs).mean()

            if denom:
                d = roll_cov_yx.loc[idx[:, x.name], x.name].xs(
                    x.name, level=1, axis=0)

        res = pd.concat((a.rename("const"), b), axis=1)

        if denom:
            res.loc[:, "denominator"] = d

        res = res.reindex(index=y.index)
        res.loc[y.first_valid_index():y.last_valid_index()] = \
            res.loc[y.first_valid_index():y.last_valid_index()].ffill()

        return res

    @staticmethod
    def get_dynamic_cov(y, x, method="expanding", **kwargs):
        """

        Parameters
        ----------
        y : pandas.DataFrame or pandas.Series
        x : pandas.DataFrame or pandas.Series
        method : str
        kwargs : dict

        Returns
        -------

        """
        # do not allow for Series
        if isinstance(y, pd.Series):
            y = y.to_frame("y")
        if isinstance(x, pd.Series):
            x = x.to_frame("x")

        # merge
        yx = y.join(x, how="outer", lsuffix='_y', rsuffix='_x')

        if method == "rolling":
            # rolling covariance
            yx_dyn_covmat = yx.rolling(**kwargs).cov()

        elif method == "expanding":
            # expanding covariance
            yx_dyn_covmat = yx.expanding(**kwargs).cov()

        elif method == "grouped_by":
            # grouped_by covariance
            yx_dyn_covmat = yx.groupby(**kwargs).cov()
            if yx_dyn_covmat.index.nlevels > 2:
                raise ValueError("Ensure the grouper returns two levels!")

        return yx_dyn_covmat


def get_dynamic_betas(Y, x, method, **kwargs):
    """
    """
    if isinstance(Y, pd.Series):
        Y = Y.to_frame()

    res = pd.DataFrame()
    for c in Y.columns:
        mod = DynamicOLS(method=method, y0=Y[c], x0=x, **kwargs)
        _, b = mod.fit()
        res.loc[:, c] = b

    return res


class PrincipalComponents:
    """
    """
    def __init__(self, X, n_comps):
        """
        """
        self.X_orig = X.copy()
        self.X = X.dropna(how="any")
        self.n_comps = n_comps

    @property
    def loadings(self):
        """
        """
        return self._loadings

    @loadings.getter
    def loadings(self):
        """
        """
        if not hasattr(self, "_loadings"):
            return None

        return self._loadings.loc[:, :("pc_"+str(self.n_comps))]

    def estimate(self):
        """
        """
        # principal components: model
        # vcv
        # ipdb.set_trace()
        vcv = self.X.cov()

        # eigenvalues + -vectors
        eigval, eigvec = np.linalg.eig(vcv)

        # sort eigenvalues (real is there just in case)
        arg_sort = np.argsort(-1*np.real(eigval))

        # construct matrix of loadings
        loadings = pd.DataFrame(
            data=eigvec[:, arg_sort],
            index=self.X_orig.columns,
            columns=["pc_"+str(p) for p in range(1, len(eigval)+1)])

        self._loadings = loadings

    def fit(self, newdata=None):
        """

        # variance ratio
        var_ratio = mod.explained_variance_ratio_

        """
        if not hasattr(self, "_loadings"):
            self.estimate()

        if newdata is None:
            newdata = self.X_orig

        # principal components: fit
        fitted = newdata.dot(self.loadings)

        # X_d.dot(loadings)
        # mod.fit_transform(X)

        return fitted

    def rotate_components(self, criterion="varimax", **kwargs):
        """
        """
        if not hasattr(self, "_loadings"):
            self.estimate()

        # compute the new matrix of loadings
        if criterion == "varimax":
            new_loadings = self.varimax(self.loadings.values, **kwargs)
        else:
            raise NotImplementedError("only varimax currently implemented!")

        # to dataframe
        new_loadings = pd.DataFrame(
            data=new_loadings,
            index=self.X_orig.columns,
            columns=["pc_"+str(p) for p in range(1,new_loadings.shape[1]+1)])

        self._loadings = new_loadings

    def plot(self, cumsum=False, newdata=None):
        """
        """
        pcs = self.fit(newdata=newdata)
        lds = self.loadings

        if lds.shape[1] < 2:
            lds.loc[:,"pc_2"] = 0.0

        # plot
        f, ax = plt.subplots(3, figsize=(8.4*0.9,11.7*0.9), sharex=False)

        # plot dynamics -----------------------------------------------------
        if cumsum:
            pcs.cumsum().plot(ax=ax[0])
        else:
            pcs.plot(ax=ax[0])

        ax[0].set_xlabel('', visible=False)

        # ax[0].set_title(k, fontsize=14)
        # ax[0].legend_.remove()

        # plot structure ----------------------------------------------------
        # for two components only!

        # plot x-axis
        ax[1].plot((-1.0,1.0), (0.0,0.0), linewidth=1.5,
            alpha=0.25, label="pc_1")
        # plot y-axis
        ax[1].plot((0.0,0.0), (-1.0,1.0), linewidth=1.5,
            alpha=0.25, label="pc_2")

        ax[1].set_xlim((-1.0,1.0))
        ax[1].set_ylim((-1.0,1.0))

        for idx, row in lds.loc[:,:"pc_2"].iterrows():
            ax[1].scatter(row["pc_1"], row["pc_2"],
                marker='.', s=10, color='k')
            ax[1].annotate(r"{}".format(idx),
                xy=(row["pc_1"], row["pc_2"]+0.025),
                rotation=45, ha="left", va="bottom")

        # ax[1].set_title("structure")
        ax[1].legend()

        # correlation structure of the raw data
        plt.setp(ax[2].get_xticklabels(), rotation=90, fontsize=10)
        plt.setp(ax[2].get_yticklabels(), rotation=0, fontsize=10)

        sns.heatmap(self.X.corr(),
            ax=ax[2], center=0, annot=True, linewidths=.25,
            cmap=sns.cubehelix_palette(light=1, as_cmap=True))

        # ax[2].set_title("correlation between factors")

        return ax, f

    @staticmethod
    def varimax(Phi, gamma = 1.0, q = 10, tol = 1e-6):
        """
        """
        p,k = Phi.shape
        R = np.eye(k)
        d=0
        for i in range(q):
            d_old = d
            Lambda = np.dot(Phi, R)
            u,s,vh = np.linalg.svd(np.dot(Phi.T,np.asarray(Lambda)**3 - \
                (gamma/p)*np.dot(Lambda,
                    np.diag(np.diag(np.dot(Lambda.T,Lambda))))))
            R = np.dot(u,vh)
            d = np.sum(s)
            if d_old!=0 and d/d_old < 1 + tol: break

        return np.dot(Phi, R)

if __name__ == '__main__':
    pass
    X = pd.DataFrame(data=np.random.normal(size=(1000, 2)), columns=["x", "e"])
    y = X.dot(pd.Series(data=[0.5, 0.5], index=X.columns))
    y.name = "y"
    mod = PureOls(y, X[["x"]], add_constant=True)
    vcv = mod.hodrick_vcv(1)
    print(np.diag(vcv)**0.5)
    mod.fit()
    print(mod.get_diagnostics())

    mu = 0.08
    theta = 0.70
    sigma = 0.16
    dt = 1/12
    T = 500
    M = int(T/dt)
    r0 = (theta * mu * dt) / (1-theta * dt) # start from unconditional mean

    N = 100
    out = pd.DataFrame(np.nan, index=range(N), columns=["simple", "hodrick"])
    for n in np.arange(0, N):
        ret = pd.Series([r0]*M)
        epsilon = np.random.standard_normal((M,))
        for t in np.arange(1, M):
            ret[t] = mu * (theta - ret[t-1]) * dt + \
                     np.sqrt(dt) * sigma * epsilon[t]

        y = ret.rolling(6).sum()
        y.name = "y"

        z = pd.DataFrame(np.random.standard_normal((M, 1)), columns=["x"])
        mod = PureOls(y, z[["x"]], add_constant=True)

        vcv = mod.hodrick_vcv(6)
        se_h = vcv.loc["x", "x"] ** 0.5
        mod.fit()
        se_s = mod.get_diagnostics(HAC=False).loc["se", "x"]

        out.loc[n, "hodrick"] = se_h
        out.loc[n, "simple"] = se_s
        print(n)


    print(mod.linear_restrictions_test(
        pd.DataFrame([[1, 0], [0, 1]], columns=["const", "x"]),
        pd.Series([0, 2])))

    Y = pd.concat((y,y), axis=1)
    np.linalg.lstsq(X.values, y.values)

    X.index = pd.date_range("2000-01-01", periods=1000, freq="B")
    y.index = pd.date_range("2000-01-01", periods=1000, freq="B")
    mod = DynamicOLS(y0=y, x0=X.loc[:, "x"])
    from pandas.tseries.resample import TimeGrouper
    lol=mod.fit(method="grouped_by", by=TimeGrouper(freq='M'))
    idx = pd.IndexSlice
    lol["y", :]
    lol.loc[idx[:, :, 'x'], :]
    lol[y.name][:, 'x']
    lol.index.get_level_values(0)

    lol.loc[idx[:, :, "x"], :]

    lol.index.droplevel()

    lol.loc[idx[:, "y"], "x"].index.droplevel(1)

    lol.loc[idx[:, "y"], "x"].xs("y", level=1, axis=0)

    lol.loc[(slice(None), 'x'), 'x']
