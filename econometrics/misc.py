import numpy as np
import pandas as pd
from statsmodels import api as sm


def descriptives(data, ann=1, scl=1, cov_lags=1):
    """Estimate frequency-adjusted descriptive statistics for each series in
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
        scale such as 100
    cov_lags : int
        max lags for robust covariance estimation using statsmodels.OLS

    Returns
    -------
    res: pandas.DataFrame
        with frequency-adjusted descriptive statistics for each column in data.
    """
    if isinstance(data, pd.Series):
        data = data.to_frame("data" if data.name is None else data.name)

    # Names for rows in the output DataFrame
    rownames = [
        "mean", "sterr", "tstat", "median", "std", "dstd",
        "q95", "q05", "skew", "kurt", "sharpe", "ac1", "count",
        "maxdd"
    ]

    # Generate the output DataFrame
    res = pd.DataFrame(index=rownames, columns=data.columns, dtype=float)

    data = data * scl

    # compute what can be computed in vectorized form
    res.loc["mean", :] = data.mean() * ann
    res.loc["median", :] = data.median() * ann
    res.loc["std", :] = data.std() * np.sqrt(ann)
    res.loc["dstd", :] = data.where(data < 0).std() * np.sqrt(ann)
    res.loc["q95", :] = data.quantile(0.95) * ann
    res.loc["q05", :] = data.quantile(0.05) * ann
    res.loc["skew", :] = data.skew()
    res.loc["kurt", :] = data.kurtosis()
    res.loc["sharpe", :] = res.loc["mean"].div(res.loc["std"])
    res.loc["count", :] = data.count()
    res.loc["maxdd", :] = (
        data.cumsum().expanding(min_periods=1).max() - data.cumsum()
    ).max()

    # Iterate over input's columns, computing statistics
    for c, c_col in data.iteritems():
        if c_col.dropna().empty:
            continue

        endog = c_col.dropna().values
        exog = np.ones(shape=(len(endog),))
        mod = sm.OLS(endog, exog)
        mod_fit = mod.fit(cov_type='HAC', cov_kwds={"maxlags": cov_lags})
        se_mean = mod_fit.bse[0]

        res.loc["sterr", c] = se_mean * ann
        res.loc["tstat", c] = res.loc["mean", c] / res.loc["sterr", c]

        ac = c_col.corr(c_col.shift(1))
        res.loc["ac1", c] = ac

    return res
