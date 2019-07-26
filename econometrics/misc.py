import numpy as np
import pandas as pd
from statsmodels import api as sm


def descriptives(data, ann=1, scl=1, cov_lags=1):
    """Estimates frequency-adjusted descriptive statistics for each series in
    the input DataFrame, by default assumes monthly inputs and annualized
    output.

    The statistics include: mean, standard error of mean, median, standard
    deviation, skewness, kurtosis, Sharpe ratio, first order autocorrelation
    coefficient and it's standard error

    Parameters
    ----------
    data: pandas.DataFrame
        of series to describe
    ann: int or float
        annualization factor, e.g. to annualize monthly data, set ann=12
    scl : int or float
        scale, most usually 100
    cov_lags : int
        max lags for robust covariance estimation

    Returns
    -------
    out: pandas.DataFrame
        with frequency-adjusted descriptive statistics for each column in data.

    """
    # Ibo nehuy
    if isinstance(data, pd.Series):
        data = data.to_frame("xyz")

    # Names for rows in the output DataFrame
    rows = ["mean", "se_mean", "tstat", "median", "std", "q95",
            "q05", "skewness", "kurtosis", "sharpe", "ac1", "count",
            "first_obs"]

    # Generate the output DataFrame
    out = pd.DataFrame(index=rows, columns=data.columns, dtype=float)

    data = data * scl

    # compute what can be computed in vectorized form
    out.loc["mean", :] = data.mean() * ann
    out.loc["median", :] = data.median() * ann
    out.loc["std", :] = data.std() * np.sqrt(ann)
    out.loc["q95", :] = data.quantile(0.95) * ann
    out.loc["q05", :] = data.quantile(0.05) * ann
    out.loc["skewness", :] = data.skew()
    out.loc["kurtosis", :] = data.kurtosis()
    out.loc["sharpe", :] = out.loc["mean"].div(out.loc["std"])
    out.loc["count", :] = data.count()

    # Iterate over input's columns, computing statistics
    for c, c_col in data.iteritems():
        endog = c_col.dropna().values
        exog = np.ones(shape=(len(endog),))
        mod = sm.OLS(endog, exog)
        mod_fit = mod.fit(cov_type='HAC', cov_kwds={"maxlags": cov_lags})
        se_mean = mod_fit.bse

        out.loc["se_mean", c] = se_mean[0] * ann
        out.loc["tstat", c] = out.loc["mean", c] / out.loc["se_mean", c]

        ac = c_col.corr(c_col.shift(1))
        out.loc["ac1", c] = ac

        out.loc["first_obs", c] = c_col.first_valid_index()

    return out
