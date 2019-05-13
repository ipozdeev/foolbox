import numpy as np
# from scipy.optimize import minimize
# import pandas as pd

# import rpy2.robjects as robj
# from rpy2.robjects.packages import importr
# from numba import jit


# @jit
def ols(y, x):
    """

    Parameters
    ----------
    y
    x

    Returns
    -------

    """
    coef, _, _, _ = np.linalg.lstsq(x, y)

    return coef


# # @jit
# def bond_model_ls(y, x):
#     """
#
#     Parameters
#     ----------
#     y : numpy.ndarray
#         (n,m) of dependent variables
#     x : numpy.ndarray
#         (n, mk) of regressors,  exactly k for each x in sequential order
#
#     Returns
#     -------
#
#     """
#     # no of obs, no of equations
#     n_obs, m = y.shape
#
#     # no of regressors
#     k = x.shape[1] // m
#
#     # stacked, column by column
#     y_stacked = y.flatten(order='F')
#     x_stacked = np.vstack((x[:, (p*k):(p*k+k)] for p in range(m)))
#
#     # start with ols
#     ols_coef = ols(y_stacked, x_stacked)
#
#     # residuals -> to covariance
#     eps = get_unstacked_eps(y_stacked, x_stacked, ols_coef, n_obs)
#
#     s2 = np.cov(eps.T)
#     sigma_inv = np.linalg.inv(s2)
#
#     def obj_fun(b):
#         # uhat = (y_stacked - x_stacked.dot(b)).reshape(n, -1, order='F')
#         eps = get_unstacked_eps(y_stacked, x_stacked, b, n_obs)
#         ssr = np.sum([np.dot(np.dot(r, sigma_inv), r.T) for r in eps])
#         return ssr
#
#     minz = minimize(obj_fun, ols_coef)
#
#     res = minz.x
#
#     return res
#
#
# # @jit
# def sur_fgls(y_stacked, x_stacked, n_obs, n_x, iterate=False, tol=1e-03):
#     """Feasible GLS for a SUR system.
#
#     Parameters
#     ----------
#     y_stacked : numpy.ndarray
#         (n,m)
#     x_stacked : numpy.ndarray
#         (nm,p) block-diagonal, with p equal to the total no of regressors
#     n_obs : int
#         no of observations; must be the same for all equations
#     n_x : numpy.ndarray
#         (m,) nos of regressors for each equation
#     iterate : bool
#     tol : float
#
#     Returns
#     -------
#     b_new : numpy.ndarray
#         (p,) of coefficients
#
#     """
#     # OLS first ---------------------------------------------------------
#     # estimate betas
#     b_old = ols(y_stacked, x_stacked)
#     b_new = b_old.copy()
#
#     # use sample covmat of residuals to do gls --------------------------
#     # init difference of betas
#     db = 1.0 + tol
#
#     # start iteration (will finish after one round if iter==False)
#     while db > tol:
#         # residuals using the currently estimated beta
#         eps = get_unstacked_eps(y_stacked, x_stacked, b_old, n_obs)
#
#         # vcv of residuals -> as the 'spread' in the GLS sandwich
#         eps_vcv = get_eps_covmat(eps, n_obs, n_x)
#
#         # estimate
#         b_new = sur_gls(y_stacked, x_stacked, eps_vcv)
#
#         # stop iterating if not asked otherwise
#         if not iterate:
#             return b_new
#
#         # if asked, calculate mean absolute diff across parameters
#         db = np.abs(b_new - b_old).mean() / np.abs(b_old).mean()
#
#         # reassign beta to be able to iterate further
#         b_old = b_new.copy()
#
#     return b_new
#
#
# # @jit
# def get_unstacked_eps(y_stacked, x_stacked, b, n_obs):
#     """Get fitted unstacked residuals.
#
#     Parameters
#     ----------
#     y_stacked : numpy.ndarray
#         (nm,) of dependent variables
#     x_stacked : numpy.ndarray
#         (nm,p) of regressors, block-diagonal
#     b : numpy.ndarray
#         (p,) of coefficients
#     n_obs : int
#
#     Returns
#     -------
#     eps : numpy.ndarray
#         (n,m) of residuals
#
#     """
#     # dot x_stacked and coef to obtain (NM,1) array of fitted y
#     yhat_stacked = x_stacked.dot(b)
#
#     # residuals
#     eps_stacked = y_stacked - yhat_stacked
#
#     # reshape order='F' needed to reshape by stacking column by column
#     eps = eps_stacked.reshape(n_obs, -1, order='F')
#
#     return eps
#
#
# # @jit
# def sur_gls(y_stacked, x_stacked, omega):
#     """Estimate a (stacked) system with GLS.
#
#     Parameters
#     ----------
#     y_stacked : numpy.ndarray
#         (nm,) of dependent variables
#     x_stacked : numpy.ndarray
#         (nm,p) of regressors
#     omega : numpy.ndarray
#         (m,m) covariance matrix of residuals
#
#     Returns
#     -------
#     b_gls : numpy.ndarray
#         (p,) of coefficients
#
#     """
#     # determine the no of observations (known that stacked matrices are
#     # (NM)x(KM) where N is the no of obs, K - of regressors, M - of
#     # equations
#     n_obs = x_stacked.shape[0] // omega.shape[0]
#
#     # beta = inv(part_a)*part_b, see Greene, SUR chapter for details ----
#     # Sigma-inverse, np.array
#     sigma_inv_np = np.kron(np.linalg.inv(omega), np.eye(n_obs))
#
#     # first fart of the equation, np.array
#     # need part_a_tmp as .inv will kill index and columns
#     part_a_tmp = x_stacked.T.dot(sigma_inv_np).dot(x_stacked)
#     part_a_np = np.linalg.inv(part_a_tmp)
#
#     # second part
#     part_b_np = x_stacked.T.dot(sigma_inv_np).dot(y_stacked)
#
#     b_gls = part_a_np.dot(part_b_np)
#
#     return b_gls
#
#
# # @jit
# def get_eps_covmat(eps, n_obs, n_x):
#     """Estimate (unbiased) covmat of residuals for GLS.
#
#     Unbiased estimator of sigma_{pq} involves (T-K_p)(T-K_q)^(1/2);
#     details are in Greene (2006), p. 258
#
#     Parameters
#     ----------
#     eps : numpy.ndarray
#         (n,m) of residuals
#     n_obs : int
#         no of obs in each equation
#     n_x : numpy.ndarray
#         (m,) of the no. of regressors in each equation; can be different
#         for each one
#
#     Returns
#     -------
#     eps_vcv : numpy.ndarray
#         (m,m) square matrix of covariances
#
#     """
#     numerator = eps.T.dot(eps)
#     denominator = np.sqrt(np.outer(n_obs-n_x, n_obs-n_x))
#
#     eps_vcv = numerator / denominator
#
#     return eps_vcv
#
#
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
    # # Frist, convert input data into R matrix
    # array = data.dropna().values  # get numpy array
    # n_rows, n_cols = array.shape  # get dimensions
    #
    # # Create R matrix
    # rdata = robj.r.matrix(data.values, nrow=n_rows, ncol=n_cols)
    #
    # # Estimate Newey-West variance-covariance matrix
    # nw = importr("sandwich")
    # vcv = nw.lrvar(rdata, type="Newey-West", prewhite=False, adjust=False)
    #
    # # Conver output to data frame
    # vcv = pd.DataFrame(np.array(vcv), index = data.columns,
    #                    columns = data.columns, dtype="float")
    #
    # return vcv

    pass
