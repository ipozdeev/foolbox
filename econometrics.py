#
import itertools
import logging
import string
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

logging.basicConfig(level=logging.INFO)

import foolbox.data_mgmt.set_credentials as set_credentials
set_credentials.set_r_environment()

import rpy2.robjects as robj
from rpy2.robjects.packages import importr
from rpy2.robjects import Formula

from rpy2.robjects import pandas2ri
pandas2ri.activate()

from rpy2.robjects import numpy2ri
numpy2ri.activate()

class Garch():
    """
    Model of the form:
    y_t = mu + ar_1*y_{t-1} + ... + ma_1*eps_{t-1} + eps_t (1)
    eps_t ~ d(0,sigma_t)
    sigma^2_t = omega + alpha_1*eps_{t-1}^2 + ... + beta_1*sigma^2_{t-1} (2)
    Such that the mean equation (1) is an ARMA process, and the variance
    equation (2) is a GARCH() process.

    Parameters
    ----------
    P: int
        number of ARCH lags in variance equation
    Q: int
        number of GARCH lags in variance equation
    AR: int
        autoregressive component of mean equation
    MA: int
        moving average component of mean equation
    """
    def __init__(self, P, Q, AR = 0, MA = 0):
        # import some R packages
        self.r_base   = importr("base")
        self.r_fGarch = importr("fGarch")
        self.r_stats  = importr("stats")

        # model specs
        self.P        = P
        self.Q        = Q
        self.AR       = AR
        self.MA       = MA

        # R formula of the equation
        self.formula  = Formula(
            "~arma({ar:1d},{ma:1d})+garch({p:1d},{q:1d})".\
            format(ar=self.AR,ma=self.MA,p=self.P,q=self.Q)
            )


    def fit(self, data, include_mean = True):
        """
        Estimates parameters of the model.

        Parameters
        ----------
        data: pd.Series
            series of values
        include_mean: Boolean
            whether the mean will be estimated or not

        Returns
        -------
        coef: dict
            coefficients
        h: pd.Series
            conditional variances, with the same index as *data*

        """
        # data to R dataframe
        self.rdf = pandas2ri.py2ri(data.dropna()) # TODO: work around dropna?

        # fit GARCH
        self.g = self.r_fGarch.garchFit(
            formula = self.formula,
            data    = self.rdf,
            trace   = False,
            include_mean = include_mean)

        # fetch coefficients from model output
        coef = pandas2ri.ri2py(self.g.slots['fit'][0])

        # assign coefficient names: omega, ar1,...,ma1,...,alpha1,...,beta1,...
        ar_names  = ["ar"+str(n) for n in range(1, self.AR+1)]
        ma_names  = ["ma"+str(n) for n in range(1, self.MA+1)]
        p_names   = ["alpha"+str(n) for n in range(1, self.P+1)]
        q_names   = ["beta"+str(n) for n in range(1, self.Q+1)]
        all_names = ar_names + ['omega'] + ma_names + p_names + q_names

        # if mean is there, prepend it
        if include_mean:
            all_names = ["mu",] + all_names

        # store coefficients as a dict with the above names
        self.coef = dict(zip(all_names, coef))

        # conditional variances
        h = pandas2ri.ri2py(self.g.slots['h.t'])
        self.h = pd.Series(data = h, index = data.dropna().index)

    def __str__(self):
        """
        String representation
        """
        return("ARMA({ar:<1d},{ar:<1d})+GARCH({p:<1d},{q:<1d}) model".\
            format(ar=self.AR, ma=self.MA, p=self.P,q=self.Q))


def nyan(data):
    """
    Randomly inserts a kitten into *data*. Try to find it!!!

    Parameters
    data: pd.DataFrame
        some data

    ----------
    """
    data.dtype = str
    T,N = data.shape

    newT = int(np.random.uniform(size=1)*(T))
    newN = int(np.random.uniform(size=1)*(N))

    data.ix[newT,newN] = ">''<"

    return data


def rOls(Y, X, const = False, HAC = False):
    """
    Performs the simplest OLS possible: b = inv(X'X)(X'Y)

    Parameters
    ----------
    TODO: rewrite with pandas
    X: NumPy 2D-array
        exogenous variables
    Y: NumPy 1D- or 2D-array
        response
    cosnt: if the const should be added to X

    Returns
    -------
    coef: NumPy 1D-array
        OLS betas

    >''<
    """
    # concatenate the two with an inner join
    data = pd.concat([Y, X], axis = 1, join = "inner")

    # drop NaNs
    data.dropna(inplace = True)

    # rename columns (needed for R)
    data.columns = ["v"+str(p) for p in range(data.shape[1])]

    # # reassemble Y and X
    # Y = data.ix[:,0]
    # X = data.ix[:,1:]

    # # add constant if needed
    # if const:
    #     X = np.hstack((np.ones((T,1), X))
    #
    # # inv(X'X)(X'Y)
    # coef = np.linalg.inv(np.dot(X.T,X)).\
    #     dot(np.dot(X.T,Y))

    # to RDataframe
    rdata = pandas2ri.py2ri(data)

    # import lm and base
    lm = robj.r['lm']
    base = importr('base')

    # write down formula: y ~ x
    if const:
        fmla = Formula("v0 ~ .")
    else:
        fmla = Formula("v0 ~ . - 1")

    env = fmla.environment
    env["rdata"] = rdata

    # fit model
    f = lm(fmla, data = rdata)

    # extract coefficients
    coef = np.array(f.rx2('coefficients'))

    # implementation of Newey-West (1997)
    if HAC:
        nw = importr("sandwich")

        # errors
        vcv = nw.NeweyWest(f)
        #vcv = nw.vcovHAC(f)

        # fetch coefficients from model output
        se = np.sqrt(np.diag(vcv))

    else:
        se = np.array(base.summary(f).rx2('coefficients'))[:,1]

    # Get the adjusted R-squared
    adj_r_sq = np.array(base.summary(f).rx2('adj.r.squared'))[0]

    return coef, se, adj_r_sq


def pureOls(Y, X, const = False):
    """
    Performs the simplest OLS possible:
    b = inv(X'X)(X'Y),
    var(b) = s^2*inv(X'X),
    s^2 = e'e/(T-N)

    Parameters
    ----------
    Y: pandas DataFrame/Series
        response
    Y: pandas DataFrame/Series
        exogenous variables
    cosnt: if the const should be added to X

    Returns
    -------
    coef: NumPy 1D-array
        OLS betas
    se: NumPy 1D-array
        standard errors

    >''<
    """

    # # concatenate Y, X with an inner join, ignore_index to preclude duplicates
    # data = pd.concat([Y, X], axis = 1, join = "inner", ignore_index = True)
    #
    # # drop NaNs
    # data.dropna(inplace = True)
    #
    # # reassemble Y and X
    # Y = data.ix[:,0]
    # X = data.ix[:,1:]

    # if dataframes
    if (type(X) == pd.DataFrame) | (type(X) == pd.Series):
        data = pd.concat((Y,X), axis=1, ignore_index=True)
        data.dropna(inplace = True)

        Y = data[0].values.reshape(-1,1)
        X = data.drop(0, axis = 1).values

    else:
        # 1d to 2d
        if Y.ndim == 1:
            Y = Y.reshape(-1,1)

        data = np.hstack((Y,X))
        data = data[~np.isnan(data).any(axis = 1)]

        # reassemble Y and X
        Y = data[:, 0]
        X = data[:, 1:]

    if np.diff(data.shape) > 0:
        return np.empty(X.shape[1]+int(const)) * np.nan

    # add constant if needed
    if const:
        X = np.hstack((np.ones((X.shape[0],1)), X))

    # dimensions
    T,N = X.shape

    # b = inv(X'X)(X'Y)
    coef = np.linalg.inv(np.dot(X.T, X)).dot(np.dot(X.T, Y))

    # # var(b) = s^2*inv(X'X)
    # vcv = ee/(T-N)*np.linalg.inv(X.T.dot(X))
    #
    # # se
    # se = np.sqrt(np.diag(vcv))

    return coef.squeeze()

def powerSet(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(1, len(s)+1))


def crossValidation(Y, X, ps, const = False):
    """
    """
    # deal with NAs: retain complete cases
    data = pd.concat((Y,X), axis = 1, ignore_index = True).dropna(how = "all")

    # to NumPy array
    Y = data.ix[:, 0].values.reshape(-1,1)  # reshape to 2d array
    X = data.ix[:, 1:].values

    # parameters
    T = X.shape[0]
    M = len(ps)

    # space for errors
    e = np.empty(shape = (T,M))

    for p in range(M):
        logging.info("doing combination {} out of {}".format(p+1, M))

        # select columns in X
        XX = X[:, ps[p]]

        # add constant if needed
        if const:
            XX = sm.add_constant(XX)

        # loop over rows
        for q in range(T):
            thisY = np.delete(Y, q, axis = 0)    # drop q'th row
            thisX = np.delete(XX, q, axis = 0)   # same

            b = pureOls(thisY, thisX, const = False)

            # forecast the value not seen by the model
            # yHat = np.dot(fit.params, XX[q,:])
            yHat = np.dot(b, XX[q,:])

            # record the error
            e[q,p] = np.squeeze(Y[q] - yHat)

        for q in range(T):
            thisY = np.delete(Y, q, axis = 0)    # drop q'th row
            thisX = np.delete(XX, q, axis = 0)   # same

            b = pureOls(thisY, thisX, const = False)

            # forecast the value not seen by the model
            # yHat = np.dot(fit.params, XX[q,:])
            yHat = np.dot(b, XX[q,:])

            # record the error
            e[q,p] = np.squeeze(Y[q] - yHat)

    # root mean squared error
    rmse = np.nanmean(e*e, axis = 0)

    return(rmse)


def interpolateQ(data_q, data_m):
    """
    Interpolates

    Parameters
    ----------
    data_q: pd.DataFrame
        dates are at quarter end
    data_m: pd.DataFrame
        dates are at quarter end

    Returns
    -------


    """
    # merge monthly-aggregated-to-quarterly data and truly quarterly data
    mySum = lambda x: pd.DataFrame.sum(x, skipna = False)   # fun to sum w/o NA
    data_q = pd.concat((data_m.resample('Q').apply(mySum), data_q), axis = 1)

    # rename columns: a, b etc., then inf
    data_q.columns = \
        [n for n in string.ascii_lowercase[:data_q.shape[1]-1]]+["inf"]

    # regress inflation onto explanatory variables
    # Y: inflation
    Y = data_q["inf"]

    # X: factors + their lags
    X = pd.concat((data_q.ix[:,:-1], data_q.ix[:,:-1].shift()), axis = 1)

    # cross-validation
    ps = list(powerSet(range(X.shape[1])))              # power set
    rmse = crossValidation(Y, X, ps, const = True)      # rmse across ps el-ts
    minIdx = np.argmin(rmse)                            # where is the min rmse?

    # re-calculate the model to get the coefficients
    mod = sm.OLS(Y, sm.add_constant(X.ix[:, ps[minIdx]]), missing = "drop")
    res = mod.fit()
    coef = res.params

    # interpolate using monthly variables
    data_m = pd.concat((data_m, data_m.shift()), axis = 1)
    inf_hat_m = sm.add_constant(data_m.ix[:, ps[minIdx]]).values.dot(coef)
    inf_hat_m = pd.DataFrame(data    = inf_hat_m,       # to DataFrame
                             index   = data_m.index,
                             columns = ["inf"])

    # aggregate at quarterly frequency
    inf_hat_q = inf_hat_m.resample('Q').apply(mySum)

    # calculate errors between true quarterly and fitted-m-resampled-to-q
    err = data_q["inf"].to_frame().subtract(inf_hat_q, axis = 0)

    # divide errors evenly between months of corresponding quarter
    err = err.resample('M')                          # resample
    err = err.fillna(method = "backfill")/3          # distribute evenly

    # add the distributed error back
    inf_hat_m = inf_hat_m.add(err)

    # check if resampled quarterly are indeed quarterly
    check = inf_hat_m.resample('Q').apply(mySum)
    if not round(check.tail(), 5).equals(
        round(data_q["inf"].to_frame().tail(), 5)):
        warnings.warn("The resampled monthly interpolated data is not the " +
            "same as the quarterly data to 5 digits")

    return(inf_hat_m)


def rolling_ols(y, X, window, min_periods=None, min_obs=None, custom_idx=None,
                constant=False):
    """
    Estimates rolling OLS regression for each column in dataframe y on
    regressors in X, returning coefficient estimates and the end of the
    estimation window's residual.

    Parameters
    ----------
    y: pandas.DataFrame
        containing columns of dependent variables
    X: pandas.DataFrame
        of regressors
    window: int
        length of the estimation window
    min_periods: int
        minimum number if datapoints required to produce first estimate, equals
        window by default
    min_obs: int
        minimum number of valid observations required to produce an estimate
        equals half of the window (rounded down to nearest integer) by default
    custom_idx: pandas.tseries.index.DatetimeIndex
        for time series data, custom_idx is a subindex of the input data's
        index, specifying the dates for which estimates should be reported. For
        example, if the input data has daily frequency, but the parameters
        should be estimated for the last day of the month only, custom_idx
        includes last days of each month only
    constant: boolean
        include intercept in regression if True, default is False

    Returns
    -------
    params: dictionary
        of dataframes where each dataframe named beta0, beta1, ..., betaN
        contains rolling estimates of coefficients corresponding to column
        index number of X (if constant == False, otherwise, beta0 is intercept)
    resids: pandas.DataFrame
        of the last residuals in each estimation run

    """

    # Set minimum number of periods equal window if not specified
    if min_periods == None:
        min_periods = window
    if min_obs == None:
        min_obs = int(window/2)

    # Create output data structures
    params = {}


    # Get number of parameters, depending on inclusion of the intercept
    if constant==True:
        n_params = X.shape[1] + 1
        X = sm.add_constant(X)
    else:
        n_params = X.shape[1]

    # Get the first valid index where all regressors are available
    first_x_obs = X.dropna().first_valid_index()
    last_x_obs = X.dropna().last_valid_index()

    # Without custom index provided, roll over all observations, in this case,
    # time-series index for the input data is not required
    if custom_idx is None:
        # Fill the dictionary with dataframes of parameter estimates
        for p in range(n_params):
            params['beta'+str(p)] = pd.DataFrame(index=y.index,
                                                 columns=y.columns,
                                                 dtype="float")
        resids = pd.DataFrame(index=y.index, columns=y.columns, dtype="float")

        # Iterate over columns in y:
        for name in y.columns:
            temp_series = y[name]  # get a series

            # Locate first and last non-NAN datapoints, compare with those of X
            first_obs = max(temp_series.first_valid_index(), first_x_obs)
            last_obs = min(temp_series.last_valid_index(), last_x_obs)
            # Get the corresponding numeric indices
            first_obs_idx = temp_series.index.get_loc(first_obs)
            last_obs_idx = temp_series.index.get_loc(last_obs)

            # Iterate through rows:
            for t in np.arange(first_obs_idx+min_periods-1, last_obs_idx+1):
                                            # -1 as indexing starts from zero
                                            # +1 as the right interval is open
                # Select data within the window
                temp_y = temp_series.ix[t-window+1:t+1]
                temp_X = X.ix[t-window+1:t+1, :]
                if temp_y.count() < min_obs:
                    # Check if there's enough observations
                    pass
                else:
                    # Estimate OLS
                    model = sm.OLS(temp_y, temp_X, missing="drop")
                    estimates = model.fit().params

                    # Save coefficient estimates
                    for p in range(n_params):
                        params['beta'+str(p)][name].ix[t] = estimates[p]

                    # Save the last residual
                    resids[name].ix[t] = model.fit().resid[-1]
    else:
        # Fill the dictionary with dataframes of parameter estimates
        for p in range(n_params):
            params['beta'+str(p)] = pd.DataFrame(index=custom_idx,
                                                 columns=y.columns,
                                                 dtype="float")
        resids = pd.DataFrame(index=custom_idx, columns=y.columns,
                              dtype="float")

        # Iterate over columns in y:
        for name in y.columns:
            temp_series = y[name]  # get a series

            # Locate first and last non-NAN datapoints, compare with those of X
            first_obs = max(temp_series.first_valid_index(), first_x_obs)
            last_obs = min(temp_series.last_valid_index(), last_x_obs)
            # Add minimum periods, some clunky shenanigans involved
            first_obs = temp_series.index.get_loc(first_obs)+min_periods-1
            first_obs = temp_series.index[first_obs]
            # Get the corresponding numeric indices, from custom index
            first_obs_idx = custom_idx.get_loc(first_obs, method="bfill")
            last_obs_idx = custom_idx.get_loc(last_obs, method="ffill")

            # Iterate over the custom index:
            for stamp in custom_idx[first_obs_idx:last_obs_idx+1]:
                # Locate the custom stamp in the input data, use the previous
                # value if there is no exact match
                t = temp_series.index.get_loc(stamp, method="ffill")
                # Select data within the window
                temp_y = temp_series.ix[t-window+1:t+1]
                temp_X = X.ix[t-window+1:t+1, :]

                if temp_y.count() < min_obs:
                    # Check if there's enough observations
                    pass
                else:
                    # Estimate OLS
                    model = sm.OLS(temp_y, temp_X, missing="drop")
                    estimates = model.fit().params

                    # Save coefficient estimates
                    for p in range(n_params):
                        params['beta'+str(p)][name].ix[stamp] = estimates[p]

                    # Save the last residual
                    resids[name].ix[stamp] = model.fit().resid[-1]

    return params, resids


def rolling_ols2(y, X, window, min_periods=None, min_obs=None, custom_idx=None,
                 constant=False):
    """
    Estimates rolling OLS regression for each column in dataframe y on
    regressors in X, returning coefficient estimates and the end of the
    estimation window's residual.

    Parameters
    ----------
    y: pandas.DataFrame
        containing columns of dependent variables
    X: pandas.DataFrame
        of regressors
    window: int
        length of the estimation window
    min_periods: int
        minimum number if datapoints required to produce first estimate, equals
        window by default
    min_obs: int
        minimum number of valid observations required to produce an estimate
        equals half of the window (rounded down to nearest integer) by default
    custom_idx: pandas.tseries.index.DatetimeIndex
        for time series data, custom_idx is a subindex of the input data's
        index, specifying the dates for which estimates should be reported. For
        example, if the input data has daily frequency, but the parameters
        should be estimated for the last day of the month only, custom_idx
        includes last days of each month only
    constant: boolean
        include intercept in regression if True, default is False

    Returns
    -------
    params: dictionary
        of dataframes where each dataframe named beta0, beta1, ..., betaN
        contains rolling estimates of coefficients corresponding to column
        index number of X (if constant == False, otherwise, beta0 is intercept)
    opt_out: dictionary
        of additional outputs, see description by key below
    resids: pandas.DataFrame
        of the last residuals in each estimation run
    mse: pandas.DataFrame
        of residual variancies

    """

    # Set minimum number of periods equal window if not specified
    if min_periods == None:
        min_periods = window
    if min_obs == None:
        min_obs = int(window/2)

    # Create output data structures
    params = {}
    opt_out = {}

    # Get number of parameters, depending on inclusion of the intercept
    if constant==True:
        n_params = X.shape[1] + 1
        X = sm.add_constant(X)
    else:
        n_params = X.shape[1]

    # Get the first valid index where all regressors are available
    first_x_obs = X.dropna().first_valid_index()
    last_x_obs = X.dropna().last_valid_index()

    # Without custom index provided, roll over all observations, in this case,
    # time-series index for the input data is not required
    if custom_idx is None:
        # Fill the dictionary with dataframes of parameter estimates
        for p in range(n_params):
            params['beta'+str(p)] = pd.DataFrame(index=y.index,
                                                 columns=y.columns,
                                                 dtype="float")
        opt_out["resids"] = pd.DataFrame(index=y.index,
                                         columns=y.columns,
                                         dtype="float")
        opt_out["mse"] = pd.DataFrame(index=y.index,
                                      columns=y.columns,
                                      dtype="float")

        # Iterate over columns in y:
        for name in y.columns:
            temp_series = y[name]  # get a series

            # Locate first and last non-NAN datapoints, compare with those of X
            first_obs = max(temp_series.first_valid_index(), first_x_obs)
            last_obs = min(temp_series.last_valid_index(), last_x_obs)
            # Get the corresponding numeric indices
            first_obs_idx = temp_series.index.get_loc(first_obs)
            last_obs_idx = temp_series.index.get_loc(last_obs)

            # Iterate through rows:
            for t in np.arange(first_obs_idx+min_periods-1, last_obs_idx+1):
                                            # -1 as indexing starts from zero
                                            # +1 as the right interval is open
                # Select data within the window
                temp_y = temp_series.ix[t-window+1:t+1]
                temp_X = X.ix[t-window+1:t+1, :]
                if temp_y.count() < min_obs:
                    # Check if there's enough observations
                    pass
                else:
                    # Estimate OLS
                    model = sm.OLS(temp_y, temp_X, missing="drop")
                    estimates = model.fit().params

                    # Save coefficient estimates
                    for p in range(n_params):
                        params['beta'+str(p)][name].ix[t] = estimates[p]

                    # Save the last residual
                    opt_out["resids"][name].ix[t] = model.fit().resid[-1]

                    # Save MSE
                    opt_out["mse"][name].ix[t] = model.fit().mse_resid
    else:
        # Fill the dictionary with dataframes of parameter estimates
        for p in range(n_params):
            params['beta'+str(p)] = pd.DataFrame(index=custom_idx,
                                                 columns=y.columns,
                                                 dtype="float")

        opt_out["resids"] = pd.DataFrame(index=custom_idx,
                                         columns=y.columns,
                                         dtype="float")
        opt_out["mse"] = pd.DataFrame(index=custom_idx,
                                      columns=y.columns,
                                      dtype="float")

        # Iterate over columns in y:
        for name in y.columns:
            temp_series = y[name]  # get a series

            # Locate first and last non-NAN datapoints, compare with those of X
            first_obs = max(temp_series.first_valid_index(), first_x_obs)
            last_obs = min(temp_series.last_valid_index(), last_x_obs)
            # Add minimum periods, some clunky shenanigans involved
            first_obs = temp_series.index.get_loc(first_obs)+min_periods-1
            first_obs = temp_series.index[first_obs]
            # Get the corresponding numeric indices, from custom index
            first_obs_idx = custom_idx.get_loc(first_obs, method="bfill")
            last_obs_idx = custom_idx.get_loc(last_obs, method="ffill")

            # Iterate over the custom index:
            for stamp in custom_idx[first_obs_idx:last_obs_idx+1]:
                # Locate the custom stamp in the input data, use the previous
                # value if there is no exact match
                t = temp_series.index.get_loc(stamp, method="ffill")
                # Select data within the window
                temp_y = temp_series.ix[t-window+1:t+1]
                temp_X = X.ix[t-window+1:t+1, :]

                if temp_y.count() < min_obs:
                    # Check if there's enough observations
                    pass
                else:
                    # Estimate OLS
                    model = sm.OLS(temp_y, temp_X, missing="drop")
                    estimates = model.fit().params

                    # Save coefficient estimates
                    for p in range(n_params):
                        params['beta'+str(p)][name].ix[stamp] = estimates[p]

                    # Save the last residual
                    opt_out["resids"][name].ix[stamp] = model.fit().resid[-1]

                    # Save the MSE
                    opt_out["mse"][name].ix[stamp] = model.fit().mse_resid

    return params, opt_out



def expanding_ols(y, X, min_periods, constant=False):
    """
    Estimates expanding window OLS regression for each column in dataframe y on
    regressors in X, returning coefficient estimates and the end of the
    estimation window's residual.

    Parameters
    ----------
    y: pandas.DataFrame
        containing columns of dependent variables
    X: pandas.DataFrame
        of regressors
    min_periods: int
        minimum number if datapoints required to produce first estimate
    constant: boolean
        include intercept in regression if True, default is False

    Returns
    -------
    params: dictionary
        of dataframes where each dataframe named beta0, beta1, ..., betaN
        contains rolling estimates of coefficients corresponding to column
        index number of X (if constant == False, otherwise, beta0 is intercept)
    resids: pandas.DataFrame
        of the last residuals in each estimation run

    """

    # Create output data structures
    params = {}
    resids = pd.DataFrame(index=y.index, columns=y.columns, dtype="float")

    # Get number of parameters, depending on inclusion of the intercept
    if constant==True:
        n_params = X.shape[1] + 1
        X = sm.add_constant(X)
    else:
        n_params = X.shape[1]

    # Get the first valid index where all regressors are available
    first_x_obs = X.dropna().first_valid_index()
    last_x_obs = X.dropna().last_valid_index()

    # Fill the dictionary with dataframes of parameter estimates
    for p in range(n_params):
        params['beta'+str(p)] = pd.DataFrame(index=y.index, columns=y.columns,
                                             dtype="float")

    # Iterate over columns in y:
    for name in y.columns:
        temp_series = y[name]  # get a series

        # Locate first and last non-NAN datapoints, compare with those of X's
        first_obs = max(temp_series.first_valid_index(), first_x_obs)
        last_obs = min(temp_series.last_valid_index(), last_x_obs)
        # Get the corresponding numeric indices
        first_obs_idx = temp_series.index.get_loc(first_obs)
        last_obs_idx = temp_series.index.get_loc(last_obs)

        # Iterate through rows:
        for t in np.arange(first_obs_idx+min_periods-1, last_obs_idx+1):
                                        # -1 since indexing starts from zero
                                        # +1 since the right interval is open
            # Select data within the window
            temp_y = temp_series.ix[first_obs_idx:t+1]
            temp_X = X.ix[first_obs_idx:t+1, :]

            # Estimate OLS
            model = sm.OLS(temp_y, temp_X, missing="drop")
            estimates = model.fit().params

            # Save coefficient estimates
            for p in range(n_params):
                params['beta'+str(p)][name].ix[t] = estimates[p]

            # Save the last residual
            resids[name].ix[t] = model.fit().resid[-1]

    return params, resids


def rolling_garch(y, window, min_periods=None):
    """
    Rolling GARCH

    """
    params = dict()
    h      = pd.DataFrame(index=y.index, columns=y.columns, dtype="float")

    # Fill the dictionary with dataframes of parameter estimates
    par_names = ["mu", "ar1", "omega", "alpha1", "beta1"]
    for p in par_names:
        params[p] = pd.DataFrame(index=y.index,
            columns=y.columns,
            dtype="float")

    # Iterate over columns in y:
    for name in y.columns:
        temp_series = y[name]

        # Iterate through rows:
        first_obs = temp_series.first_valid_index()
        last_obs  = temp_series.last_valid_index()

        first_obs_idx = temp_series.index.get_loc(first_obs)
        last_obs_idx  = temp_series.index.get_loc(last_obs)

        for t in np.arange(first_obs_idx+min_periods, last_obs_idx+1):
            temp_y = temp_series.ix[t-window+1:t]
            model  = Garch(1,1,1,0)
            try:
                model.fit(temp_y, include_mean = True)
                h[name].ix[t] = model.h[-1]
                for p in par_names:
                    params[p][name].ix[t] = model.coef[p]
            except:
                h[name].ix[t] = np.NaN
                for p in par_names:
                    params[p][name].ix[t] = np.NaN




    return params, h

def nw_cov(data):
    """Computes Newey-West variance-covariance matrix for a dataframe, with
    number of lags selected according to Newey-West (1994)

    Parameters
    ----------
    data: pandas.DataFrame
        for which variance-covariance matrix is to be estimated

    Returns
    -------
    vcv: pandas.DataFrame
        containing Newey-West variance-covariance matrix

    """
    # Frist, convert input data into R matrix
    array = data.dropna().values  # get numpy array
    n_rows, n_cols = array.shape  # get dimensions
    # Create R matrix
    rdata = robj.r.matrix(data.values, nrow=n_rows, ncol=n_cols)

    # Estimate Newey-West variance-covariance matrix
    nw = importr("sandwich")
    vcv = nw.lrvar(rdata, type="Newey-West", prewhite=False, adjust=False)

    # Conver output to data frame
    vcv = pd.DataFrame(np.array(vcv), index = data.columns,
                       columns = data.columns, dtype="float")

    return vcv
