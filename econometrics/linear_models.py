import pandas as pd
import numpy as np
import statsmodels.api as sm

import rpy2.robjects as robj
from rpy2.robjects.packages import importr
from rpy2.robjects import Formula

from rpy2.robjects import pandas2ri
pandas2ri.activate()

from rpy2.robjects import numpy2ri
numpy2ri.activate()
from scipy.stats import chi2


class Regression:
    """
    Parameters
    ----------
    y : pandas.Series
        dependent variable/response vector (required)
    x : pandas.DataFrame/Series
        DataFrame, independent variables/design matrix (required)
    add_constant: bool
        if a constant should be added
    lag_x: int/list
        if some lags of x should be added
    """
    def __init__(self, y, x, add_constant=False, lag_x=0):
        # copy
        y, x = y.copy(), x.copy()

        #
        if isinstance(x, pd.Series):
            x = x.to_frame()
        if isinstance(y, pd.DataFrame):
            y = y.squeeze()

        assert isinstance(y, pd.Series)

        # lag regressors
        if not hasattr(lag_x, "__iter__"):
            lag_x = range(lag_x)

        def rename_lags(c, l):
            return "{}_{}".format(c, l)

        x = pd.concat([x, ] +
                      [x.shift(p).rename(columns=rename_lags) for p in lag_x],
                      axis=1)

        # save originally-shaped inputs
        y_orig = y.copy()
        x_orig = x.assign(const=1.0) if add_constant else x.copy()

        # align data if needed
        x, y = x.dropna().align(y.dropna(), join="inner", axis=0)

        # add constant
        if add_constant:
            x = x.assign(const=1.0)

        self.add_constant = add_constant
        self.lag_x = lag_x
        self.x = x
        self.y = y
        self.x_names = x.columns
        self.y_name = y.name
        self.x_orig = x_orig
        self.y_orig = y_orig

        self.coef = None

    def estimate(self, **kwargs):
        """
        """
        pass

    def get_yhat(self, newdata=None, use_original_x=True):
        """

        Parameters
        ----------
        newdata : pandas.DataFrame
        use_original_x

        Returns
        -------

        """
        if not hasattr(self, "coef"):
            _ = self.estimate()

        if newdata is not None:
            if self.add_constant:
                newdata = newdata.assign(const=1.0)
        else:
            if use_original_x:
                newdata = self.x_orig
            else:
                newdata = self.x

        yhat = newdata.mul(self.coef, axis=1).sum(axis=1, skipna=False)

        yhat.name = self.y_name

        return yhat

    def get_residuals(self, use_original_x=True):
        """Retrieve residuals from regression."""
        if use_original_x:
            eps = self.y_orig - self.get_yhat(use_original_x=use_original_x)
        else:
            eps = self.y - - self.get_yhat(use_original_x=use_original_x)

        # rename
        eps.name = self.y_name

        return eps


class OLS(Regression):
    """Purest old-school no-nonsence OLS, inv(x'x)(x'y).
    """
    def estimate(self):
        """Evaluate inv(x'x)(x'y) (in a smart way).

        Y can have rank > 1.

        Returns
        -------
        coef: pandas.DataFrame
            where (K, N)-th value is coef of asset N w.r.t. factor K
        """
        coef_arr, _, _, _ = np.linalg.lstsq(self.x.values, self.y.values)

        coef = pd.DataFrame(data=coef_arr,
                            index=self.x.columns,
                            columns=self.y.columns)

        self.coef = coef

        return coef

    def get_diagnostics(self, hac=True):
        """
        """
        if hac:
            rdata = pandas2ri.py2ri(pd.concat((self.y, self.x), axis=1))

            # import lm and base
            lm = robj.r['lm']

            # write down formula: y ~ x
            fmla = Formula(self.y_name[0] + " ~ . - 1")

            env = fmla.environment
            env["rdata"] = rdata

            # fit model
            f = lm(fmla, data = rdata)

            # extract coefficients
            coef = pd.Series(f.rx2('coefficients'), index=self.x_names)

            # implementation of Newey-West (1997)
            nw = importr("sandwich")

            # errors
            vcv = nw.NeweyWest(f)

            # fetch coefficients from model output
            se = pd.Series(np.sqrt(np.diag(vcv)), index=self.x_names)

            # tstat
            tstat = coef / se

            # R-squared
            rsq = sm.OLS(endog=self.y, exog=self.x).fit().rsquared_adj

            # concat and transpose
            res = pd.DataFrame.from_dict({"coef": coef,
                                          "se": se,
                                          "tstat": tstat})
            res = res.T

            # add r-squared
            res.loc["adj r2", res.columns[0]] = rsq

            # add nobs
            res.loc["nobs", res.columns[0]] = len(self.y)

            return res

        if not hasattr(self, "eps"):
            eps = self.get_residuals()

        eps_var = eps.var().values
        xx = self.x.T.dot(self.x)
        vcv = eps_var*np.linalg.inv(xx)
        se = (np.diag(vcv) ** 0.5).squeeze()
        se = pd.DataFrame(se, index=self.coef.index, columns=self.coef.columns)
        tstat = self.coef/se

        res = pd.concat([self.coef, se, tstat], axis=1).T
        res.columns = self.x_names
        res.index = ["coef", "se", "tstat"]

        return res
    
    def linear_restrictions_test(self, R, r, HAC=True):
        """Tests linear constraints on the coeficients in the following form:

                            R * theta_hat = r

        Parameters
        ----------
        R: pd.DataFrame
            with columns named according parameters combinations of which are
            to be tested (K) less or equal the total number of estimated
            parameters, and number of rows equal to the number of linear
            constraints (q)
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
            rdata = pandas2ri.py2ri(pd.concat((self.y, self.x), axis=1))

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
            coef = pd.Series(f.rx2('coefficients'), index=self.x_names)

            # implementation of Newey-West (1997)
            nw = importr("sandwich")

            # errors
            vcv = nw.NeweyWest(f)

            # vcv
            vcv = pd.DataFrame(np.array(vcv), index=self.x_names,
                               columns=self.x_names)

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
        (T, N) = self.x.shape

        # Get demeaned residuals from OLS on a constant
        resids = self.y - self.y.mean()

        # Compute Hodrick's spectral density
        spectral_density = pd.DataFrame(np.zeros((N, N)), index=self.x.columns,
                                        columns=self.x.columns)

        # Spectral density for each point in time
        sd = self.x.rolling(forecast_horizon).sum().shift(1)\
            .mul(resids.squeeze(), axis=0).dropna()

        # Get the time average
        spectral_density = (1 / T) * sd.T.dot(sd)

        # Get the 'sandwich' part of vcv, i.e. E[xx']
        Z = (1 / T) * self.x.T.dot(self.x)

        # Get the output
        hodrick_vcv = \
            np.linalg.inv(Z).dot(spectral_density).dot(np.linalg.inv(Z)) / T

        hodrick_vcv = pd.DataFrame(hodrick_vcv, index=self.x.columns,
                                   columns=self.x.columns)

        return hodrick_vcv

    def to_latex(self, inference="se", fmt_coef="{:.4f}",
                 fmt_inference="({:.2f})", **kwargs):
        """Write as LaTeX equation.

        Parameters
        ----------
        inference : str
            report errors of the coefficients 'se' or t-stats 't-stat'
        fmt_coef : str
        fmt_inference : str
        kwargs : any

        Returns
        -------
        res : str

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

            idx = pd.IndexSlice

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
