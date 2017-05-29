import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from foolbox import econometrics as ec
import matplotlib
import matplotlib.pyplot as plt
plt.rc("font", family="serif", size=12)

def fama_macbeth_first(Y, X):
    """ First stage of Fama-MacBeth
    """
    # Y = pd.DataFrame(np.random.normal(size=(100,4)))
    # X = pd.DataFrame(np.random.normal(size=(100,1)))
    # assert `X` and `Y` have columns
    if isinstance(Y, pd.Series):
        Y = Y.to_frame()
    if isinstance(X, pd.Series):
        X = X.to_frame()

    # align
    y, x = Y.align(sm.add_constant(X), axis=0, join="left", copy=True)

    # dimensions
    T, N = y.shape
    _, K = x.shape

    # init array for betas - will be broadcast later
    betas = np.empty(shape=(K, N))

    # loop over assets in `y`
    for c in range(len(y.columns)):
        B, _, _, _ = \
            np.linalg.lstsq(y.ix[:,[c]].dropna().values,x.dropna().values)
        betas[:,c] = B

    # store alphas separately
    a = betas[0,:]
    b = betas[1:,:]

    # construct panel whose slice along time axis (`y`.index) is `betas`
    b = pd.Panel(
        data=np.tile(b.transpose()[:,np.newaxis,:], (1,T,1)),
        items=y.columns,
        major_axis=y.index,
        minor_axis=x.columns[1:])

    return b, a

def fama_macbeth_second(Y, betas):
    """ Perform second stage of Fama-MacBeth.

    Parameters
    ----------
    Y : pd.DataFrame
        of assets
    betas : pandas.Panel
        of betas items=assets, with major_axis=time, minor_axis=factors

    Returns
    -------
    lambdas : pandas.DataFrame
        time-series of risk premia estimates for each factor
    alphas : pandas.DataFrame
        time-series of pricing errors for each asset

    >''<
    """
    Y = Y.astype(np.float)
    betas = betas.astype(np.float)

    if not isinstance(betas, pd.Panel):
        betas = pd.Panel.from_dict({"f1":betas}, orient="minor")

    # purge Y of nans and align with betas
    y = Y.dropna(inplace=False, axis=0, thresh=betas.shape[2]+1)
    common_idx = y.index.intersection(betas.major_axis)
    b = betas.loc[:,common_idx,:]
    y = y.loc[common_idx,:]

    # storage for risk premia: DataFrame of nan's
    lambdas = pd.DataFrame(data=np.empty(shape=b.iloc[0,:,:].shape)*np.nan,
        index=y.index,
        columns=b.minor_axis,
        dtype="float")

    # storage for residuals from second stage regressions: DataFrame of nan's
    alphas = pd.DataFrame(data=np.empty(shape=b.iloc[:,:,0].shape)*np.nan,
        index=y.index,
        columns=Y.columns,
        dtype="float")

    # cross-sectional regression for each time period
    for idx, row in y.iterrows():
        # idx, row = list(y.iterrows())[10]
        # betas are now regressors
        x2 = b.loc[:,idx,:].values.transpose()
        # responses are values of `row`
        y2 = row.values[:,np.newaxis]
        # record index of nan's to use later when substituting betas
        nan_idx = np.isfinite(np.hstack((y2,x2))).any(axis=1)
        # there goes ols on values without nan's
        try:
            B, _, _, _ = np.linalg.lstsq(x2[nan_idx,:], y2[nan_idx,:])
        except:
            B = np.nan*np.arange(x2.shape[1])
        # save premia and pricing errors
        lambdas.loc[idx,:] = B.squeeze()
        alphas.loc[idx,:] = row-(x2.dot(B)).squeeze()

    return lambdas, alphas


def descriptives(data, scale=12):
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
    scale: float
        controls frequency conversion, e.g. to annualize monthly data, set
        scale=12 (default)

    Returns
    -------
    out: pandas.DataFrame
        with frequency-adjusted descriptive statistics for each column in data.

    """
    # Names for rows in the output DataFrame
    rows = ["mean", "se_mean", "median", "std", "skew", "kurt", "sharpe",
            "ac1", "se_ac1"]
    # Generate the output DataFrame
    out = pd.DataFrame(index=rows, columns=data.columns, dtype="float")

    # Iterate over input's columns, computing statistics
    for column in data.columns:

        # Compute mean with HAC-adjusted standard error
        mean, se_mean, _ = ec.rOls(data[column], None, const=True, HAC=True)
        out[column]["mean"] = mean[0] * scale
        out[column]["se_mean"] = se_mean[0] * scale

        # Compute autocorrelation with HAC-adjusted standard error, suppress
        # inntercept
        ac, se_ac, _ = ec.rOls(data[column], data[column].shift(1),
                               const=False, HAC=True)
        out[column]["ac1"] = ac[0]
        out[column]["se_ac1"] = se_ac[0]

        # Compute the remaining statistics
        out[column]["median"] = data[column].median() * scale
        out[column]["std"] = data[column].std() * np.sqrt(scale)
        out[column]["skew"] = data[column].skew()
        out[column]["kurt"] = data[column].kurt()
        out[column]["sharpe"] = data[column].mean() / data[column].std() *\
                                np.sqrt(scale)

    return out


def ts_ap_tests(y, X, scale=12):
    """Runs time-series asset pricing tests by regressing returns of test
    assets on factor returns. That is, given a single factor model with factor
    F, and two test assets y1, y2, and estimates two regressions:

                            y1 = a1 + b1 * F
                            y2 = a2 + b2 * F

    then reporting coefficient  estimates and corresponding standard errors.
    Intercept is always assumed. Standard errors are HAC-adjusted.

    Parameters
    ----------
    y: pandas.DataFrame
        of returns on test assets (dependen variables)
    X: pandas.DataFrame
        factor returns (regressors)
    scale: float
        number to multiply intercepts from time-series regressions, e.g. 12,
        so that intercepts are annualized. Default is 12

    Returns
    -------
    out: pandas.DataFrame
        of time-series asset pricing test results

    """
    # Generate row names for the output
    out_index = list()                  # start with an empty list
    for column in y.columns:            # iterate over returns of test assets
        out_index.append(column)        # row names are names of test assets
        out_index.append("se_"+str(column))
                                        # the following row contains SE's

    # Generate column names
    out_columns = ["alpha"]             # the first element is the intercept
    for column in X.columns:            # iterate over regressors
        out_columns.append(column)      # fill column names of the output
    out_columns.append("adj_r_sq")      # the last column is the adjusted R-sq

    # Generate the output DataFrame
    out = pd.DataFrame(index=out_index, columns=out_columns, dtype="float")

    # Run OLS regressing each test asset in y on a set of factors in X
    for column in y.columns:
        coef, se, adj_r_sq = ec.rOls(y[column], X, const=True, HAC=True)

        # Get the output in desired format, iterating over a list of tuples
        # (column_name, column_index)
        for out_col, i in list(zip(out.columns, range(len(out.columns)))):
            # Put R-squared in the corresponding column
            if out_col == "adj_r_sq":
                out[out_col][column] = adj_r_sq
                out[out_col]["se_"+str(column)] = np.nan
            # Fill in the regressors' coefficients
            else:
                out[out_col][column] = coef[i]
                out[out_col]["se_"+str(column)] = se[i]

    # Finally, multiply the first column by the desired scale
    out.alpha = out.alpha * scale

    return out


def convert_to_latex(data, decimals=2, final_conversion=True):
    """A utility function to convert DataFrames created by the functions:

    descriptives
    ts_ap_tests

    to strings with LaTeX tables, rounding values and enclosing standard errors
    into parentheses.

    Parameters
    ----------
    data: pandas.DataFrame
        to convert to LaTeX
    decimals: int
        report descriptives up to a certain number of decimals, default is 2
    final_conversion: bool
        whether to convert final output to string (True), or return a
        processed dataframe (False). Default is True.

    Returns
    -------
    table: str
        with code to generate a LaTeX table

    """
    # Round the data
    data = data.round(decimals)

    # Encose standard erros in parentheses, mind the warnings
    for row_name, row in data.iterrows():  # iterate over rows
        if row_name[:3] == "se_":          # enclose standard errors in () ...
            for col_name in data.columns:  # ... column by column
                data[col_name][row_name] = "(" +\
                                            str(data[col_name][row_name]) +\
                                            ")"
    # Handle the multiindex case for FMB tests
    if type(data.index) == pd.indexes.multi.MultiIndex:
        # Start iterating over rows in multiindex
        for (level, row_name), row in data.iterrows():
            if row_name[:3] == "se_":           # enclose se's into () ...
                for col_name in data.columns:   # ... column by column ...
                                                # ... skipping nans
                    if pd.notnull(data[col_name][level][row_name]):
                        data.loc[(level, row_name), col_name] = "(" +\
                                        str(data[col_name][level][row_name]) +\
                                                                 ")"

    # Output is a string with LaTeX table
    if final_conversion:
        out = data.to_latex(na_rep="")
    else:
        out = data

    return out


def fmb_in_sample(y, X, stage2_intercept=False):
    """Given arrays of test assets y and factors X, performs Fama-MacBeth
    procedure, with factor loadings estimated over the whole sample in the
    first step. During the first step, intercepts are auto-included

    Parameters
    ----------
    y: pandas.DataFrame
        of returns on test assets (dependent variables)
    X: pandas.DataFrame
        factor returns (regressors)
    stage2_intercept: bool
        intercept in the second stage of the procedure, default is False

    Returns
    -------
    out: pandas.DataFrame
        of FMB estimation results

    """
    # Step 1: time-series regresions: run OLS regressing each test asset in y
    #on a set of factors in X, intercept is always included in the first stage
    betas = list()          # list to collect first-stage estimates
    for column in y.columns:
        #coef, se, adj_r_sq = ec.rOls(y[column], X, const=True, HAC=False)
        model = sm.OLS(y[column], sm.add_constant(X), missing="drop")
        estimates = model.fit().params
        betas.append(estimates[1:].values)

    # Create a dataframe of loadings of each test asset to each factor
    betas = pd.DataFrame(betas, index=y.columns, columns=X.columns,
                         dtype="float")

    # Step 2: corss-sectional regressions of returns on factor loadings for
    # each time period, getting risk premia
    # DataFrame for risk premia (lambdas), and if needed intercept (gamma)
    if stage2_intercept:

        lambdas_columns = ["const"]  # the first column is intercept
        for column in betas.columns:
            lambdas_columns.append(column)
        lambdas = pd.DataFrame(index=y.index, columns=lambdas_columns,
                               dtype="float")

    else:
        lambdas = pd.DataFrame(index=y.index, columns=betas.columns,
                               dtype="float")

    if stage2_intercept:
        stage2_X = sm.add_constant(betas)
    else:
        stage2_X = betas

    # Alphas are residuals from second stage regressions
    alphas = pd.DataFrame(index=y.index, columns=y.columns, dtype="float")
                                                            # pricing errors

    # Run cross-sectional regression for each time period
    for row, values in y.iterrows():
        # Only consider time periods where all returns are present
        if pd.isnull(y.ix[row]).sum() == 0:
            # Run OLS
            model = sm.OLS(y.ix[row], stage2_X)
            estimates = model.fit().params
            #coefs, _, _, = ec.rOls(y.ix[row], betas,
            #                       const=stage2_intercept, HAC=False)
            # Get premia and pricing errors
            lambdas.ix[row] = estimates
            alphas.ix[row] = model.fit().resid
            #alphas.ix[row] = y.ix[row] - betas.dot(coefs)




    out = betas, lambdas, alphas
    return out


def fmb_rolling(y, X, window, stage2_intercept=False):
    """Given arrays of test assets y and factors X, performs Fama-MacBeth
    procedure, with factor loadings in the first step estimated over the
    rolling windows

    Parameters
    ----------
    y: pandas.DataFrame
        of returns on test assets (dependent variables)
    X: pandas.DataFrame
        factor returns (regressors)
    window: int
        width of the rolling window for the first step regressions
    stage2_intercept: bool
        intercept in the second stage of the procedure, default is False.
        Currently is the only supported behavior

    Returns
    -------
    out: pandas.DataFrame
        of FMB estimation results

    """
    # First stage estimates
    betas, _ = ec.rolling_ols(y, X, window=window, min_periods=window,
                              constant=True)

    # Step 2: corss-sectional regressions of returns on factor loadings for
    # each time period, getting risk premia
    # DataFrame for risk premia (lambdas)
    lambdas = pd.DataFrame(index=y.index, columns=X.columns, dtype="float")
    # Alphas are residuals from second stage regressions
    alphas = pd.DataFrame(index=y.index, columns=y.columns, dtype="float")
                                                            # pricing errors

    # Run cross-sectional regression for each time period
    for row, values in y.iterrows():
        # Only consider time periods where all returns are present
        if pd.isnull(y.ix[row]).sum() == 0:
            stage2_y = y.ix[row]
            # Get the stage 2 regressors (betas)
            stage2_X = pd.DataFrame(index=y.columns, columns=X.columns,
                                    dtype="float")
            # Map columns in X to corresponding betas, beta0 is constant => +1
            X_beta_map = list(zip(X.columns, np.arange(1, len(X.columns)+1)))
            # Iterate over X and corresponding ts-betas!
            for x_col, beta_num in X_beta_map:
                # Fill the columns in X with corresponding betas
                stage2_X[x_col] = betas["beta"+str(beta_num)].ix[row]

            if pd.isnull(stage2_X).sum().sum() == 0:

                # Run OLS
                model = sm.OLS(stage2_y, stage2_X)
                estimates = model.fit().params

                # Get premia and pricing errors
                lambdas.ix[row] = estimates
                alphas.ix[row] = model.fit().resid
                #alphas.ix[row] = y.ix[row] - betas.dot(coefs)


    return betas, lambdas, alphas


def fmb_expanding(y, X, min_periods, stage2_intercept=False):
    """Given arrays of test assets y and factors X, performs Fama-MacBeth
    procedure, with factor loadings in the first step estimated over the
    rolling windows

    Parameters
    ----------
    y: pandas.DataFrame
        of returns on test assets (dependent variables)
    X: pandas.DataFrame
        factor returns (regressors)
    min_periods: int
        number of periods required to produce first first-stage estimate
    stage2_intercept: bool
        intercept in the second stage of the procedure, default is False.
        Currently is the only supported behavior

    Returns
    -------
    out: pandas.DataFrame
        of FMB estimation results

    """
    # First stage estimates
    betas, _ = ec.expanding_ols(y, X, min_periods=min_periods,
                              constant=True)

    # Step 2: corss-sectional regressions of returns on factor loadings for
    # each time period, getting risk premia
    # DataFrame for risk premia (lambdas)
    lambdas = pd.DataFrame(index=y.index, columns=X.columns, dtype="float")
    # Alphas are residuals from second stage regressions
    alphas = pd.DataFrame(index=y.index, columns=y.columns, dtype="float")
                                                            # pricing errors

    # Run cross-sectional regression for each time period
    for row, values in y.iterrows():
        # Only consider time periods where all returns are present
        if pd.isnull(y.ix[row]).sum() == 0:
            stage2_y = y.ix[row]
            # Get the stage 2 regressors (betas)
            stage2_X = pd.DataFrame(index=y.columns, columns=X.columns,
                                    dtype="float")
            # Map columns in X to corresponding betas, beta0 is constant => +1
            X_beta_map = list(zip(X.columns, np.arange(1, len(X.columns)+1)))
            # Iterate over X and corresponding ts-betas!
            for x_col, beta_num in X_beta_map:
                # Fill the columns in X with corresponding betas
                stage2_X[x_col] = betas["beta"+str(beta_num)].ix[row]

            if pd.isnull(stage2_X).sum().sum() == 0:

                # Run OLS
                model = sm.OLS(stage2_y, stage2_X)
                estimates = model.fit().params

                # Get premia and pricing errors
                lambdas.ix[row] = estimates
                alphas.ix[row] = model.fit().resid
                #alphas.ix[row] = y.ix[row] - betas.dot(coefs)


    return betas, lambdas, alphas


def fmb_results(factors, lambdas, alphas, scale=12):
    """Computes average risk-premia, corresponding standard errors, and
    performs joint significance tests for pricing errors for Fama-MacBeth
    procedure estimates

    Parameters
    ----------
    factors: pandas.DataFrame
        of factors used in the FMB estimation (needed for Shanken  error-in-
        variables correction)
    lambdas: pandas.DataFrame
        of factor risk premia
    alphas: pandas.DataFrame
        of pricing errors from second stage of FMB estimation
    scale: float
        number to multiply mean risk premia and corresponding standard errors,
        e.g. 12, so that these values are annualized if inputs are monthly.
        Default is 12

    Returns
    -------
    out: pandas.DataFrame
        with results of FMB asset pricing tests, index contains risk premia
        (lambdas), Shanken (Sh) and Newey-West (NW), standard errors (for
        lambdas) and p-values (for chi-squared tests of joint significance of
        pricing errors). Columns contain factors, and two chi-squared tests
        (Shanken and Newey-West)

    """
    # Create dataframe for output
    index = ["lambdas", "se_Sh", "se_NW"]
    columns = factors.columns.tolist()
    columns.extend(["chi2_Sh","chi2_NW"])
    out = pd.DataFrame(index=index, columns=columns, dtype="float")

    # Compute Shanken standard errors of risk-premia
    n_obs = lambdas.count().min()     # get the number of observations
    vcv_sh = 1/n_obs * (\
    (1 + lambdas.mean().dot(np.linalg.inv(factors.cov())).dot(lambdas.mean()))\
    *(lambdas.cov() - factors.cov()) + factors.cov())
                                           # get VCV with Shanken correction
    se_sh = np.sqrt(np.diag(vcv_sh))       # get the Shanken se's
    se_sh = pd.DataFrame(se_sh, index=lambdas.columns, columns=["se_Sh"],
                         dtype="float").T  # convert into a dataframe

    # Compute Newey-West standard errors of risk-premia
    # TODO: check consistency with NW in ec.rOls, there are some discrepancies
    se_nw = np.sqrt(np.diag(ec.nw_cov(lambdas.dropna())))  # get the Newey-West se's
    se_nw = pd.DataFrame(se_nw, index=lambdas.columns, columns=["se_NW"],
                         dtype="float").T         # convert into a dataframe


    # Write standard errors of risk premia into output
    out.ix["lambdas"] = lambdas.mean() * scale  # average risk premia
    out.ix["se_Sh"] = se_sh.ix["se_Sh"] * scale
    out.ix["se_NW"] = se_nw.ix["se_NW"] * scale

    # Chi-squared tests of joint significance of pricing errors:
    # First, compute VCV matrix of alphas with Shanken correction
    n_obs_alphas = alphas.count().min()     # get the number of observations
    vcv_alphas_sh =  1/n_obs_alphas * \
    (1 + lambdas.mean().dot(np.linalg.inv(factors.cov())).dot(lambdas.mean()))\
    * alphas.cov()

    # Second, compute Newey-West VCV matrix of alphas
    vcv_alphas_nw = ec.nw_cov(alphas.dropna())

    # Third, compute test statistics along with corresponding p-values
    chi2_stat_sh = alphas.mean().dot(np.linalg.inv(vcv_alphas_sh))\
    .dot(alphas.mean())
    chi2_stat_nw = alphas.mean().dot(np.linalg.inv(vcv_alphas_nw))\
    .dot(alphas.mean())

    # Degrees of freedom = # test assets - # of estimated risk premia
    # TODO: think about handling constant in the second stage of FMB
    df = alphas.shape[1] - lambdas.shape[1]

    # Compute p-values
    p_val_sh = 1 - stats.chi2.cdf(chi2_stat_sh, df)
    p_val_nw = 1 - stats.chi2.cdf(chi2_stat_nw, df)

    # Record test results into output
    out["chi2_Sh"].ix["lambdas"] = chi2_stat_sh
    out["chi2_NW"].ix["lambdas"] = chi2_stat_nw

    out["chi2_Sh"].ix["se_Sh"] = p_val_sh
    out["chi2_NW"].ix["se_Sh"] = p_val_nw

    return out


def error_plot(realized, predicted, scale=12, **kwargs):
    """Produces a standard pricing errors plot, given average returns on test
    assets and average fitted returns of these assets

    Parameters
    ----------
    realized: pandas.DataFrame
        of realized mean excess returns on test assets
    predicted: pandas.DataFrame
        of fitted mean excess returnson test assets

    Returns
    -------
    fig: matplotlib.figure.Figure
        with pricing errors plot
    """
    y = predicted.values * scale
    x = realized.values * scale
    labels = predicted.index.tolist()

    fig, ax = plt.subplots(**kwargs)
    ax.scatter(x, y)

    # Set axis limits to lowest and higest values of current instance
    lower_lim = min([ax.get_xlim()[0],  ax.get_ylim()[0]])
    upper_lim = max([ax.get_xlim()[1],  ax.get_ylim()[1]])
    ax.set(xlim=(lower_lim, upper_lim), ylim=(lower_lim, upper_lim))

    # Set axis labels
    plt.xlabel('Realized (in % p.a.)')
    plt.ylabel('Fitted (in % p.a.)')
    plt.title("Mean excess return")

    # Draw a 45-degree line
    ax.plot(ax.get_xlim(), ax.get_ylim(), color="black", lw=1, ls="--")

    # Annotate portfolos
    for i, label in enumerate(labels):
        ax.annotate(label, (x[i],y[i]), xycoords='data',
                    xytext=(5, -10), textcoords="offset points")

    # Additional settings
    ax.grid(False)

    return fig

def se_plot(estimates, se, scale=1):
    """Given dataframes of estimates and corresponding standard errors, plots
    estimate +/- 2SE

    Parameters
    ----------
    estimates: pandas.DataFrame
        containing estimates, e.g. means
    se: pandas.DataFrame
        containing standard errors of estimates
    scale: int
        controls scale of estimates, e.g. 12 for conversion from monthly to
        annual. Default is 1

    Returns
    -------
    se_plot: matplotlib.figure.Figure
        plotting estimate +/- 2se

    """
    # Rescale inputs
    estimates = estimates * scale
    se = se * scale

    # Create dataframes containing estimates +/- 2se
    plus_2se = pd.DataFrame(estimates.values+2*se.values, index=se.index,
                            columns=["+2SE"])
    minus_2se = pd.DataFrame(estimates.values-2*se.values, index=se.index,
                            columns=["-2SE"])

    # Draw the figure
    fig, ax = plt.subplots(figsize=(8,8))
    ax.plot(estimates, color="red", lw=1, ls="-")
    ax.plot(plus_2se, color="black", lw=1, ls="--")
    ax.plot(minus_2se, color="black", lw=1, ls="--")

    return fig, ax
