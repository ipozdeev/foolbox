import pandas as pd
import warnings
import numpy as np
from itertools import permutations
from distutils.version import LooseVersion
from functools import reduce
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.tsatools import freq_to_period

from foolbox.econometrics.equations_system import Equations


def get_period_getter(freqstr):
    """Return the function to extract the period specified by a string.

    Parameters
    ----------
    freqstr : str

    Returns
    -------
    func : callable

    """
    # periodicity function
    if freqstr.startswith("W"):
        def freqfun(x):
            return x.week
    elif freqstr.startswith("M"):
        def freqfun(x):
            return x.month
    elif freqstr.startswith("Q"):
        def freqfun(x):
            return x.quarter
    elif freqstr.startswith("A"):
        def freqfun(x):
            return x.year
    else:
        raise NotImplementedError

    return freqfun


def to_spdf(func, cls=None):
    """Transform generic output of `func` to a child of SPDF.

    Acts on methods having `self` as the first argument; static methods
    won't work.

    Returns
    -------
    wrapper : callable
    """
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        return cls(res)

    return wrapper


class SPDF:
    """Special-purpose dataframe.

    Parameters
    ----------
    df : pandas.DataFrame

    """
    def __init__(self, df, **kwargs):
        """
        """
        self.df = df

        # assign attributes
        for k, v in kwargs.items():
            self.__setattr__(k, v)

    def __repr__(self):
        res = str(self.__class__).split('.')[-1][:-2] + ":\n" + repr(self.df)
        return res

    def __getattr__(self, item):
        res = getattr(self.df, item)

        if callable(res):
            res = to_spdf(res, self.__class__)

        return res

    def __add__(self, other):
        res = self.__class__(self.df + other)

        return res

    def __mul__(self, other):
        res = self.__class__(self.df * other)

        return res

    def __sub__(self, other):
        res = self.__class__(self.df - other)

        return res

    def __radd__(self, other):
        res = self.__class__(other + self.df)

        return res

    def __rmul__(self, other):
        res = self.__class__(other * self.df)

        return res

    def __rsub__(self, other):
        res = self.__class__(other - self.df)

        return res

    def __div__(self, other):
        res = self.__class__(self.df / other)

        return res

    def __pow__(self, other):
        """
        """
        res = self.__class__(self.df ** other)

        return res

    @property
    def loc(self):
        return self.df.loc

    @property
    def iloc(self):
        return self.df.iloc

    def _interlace(self, func, columns=None, drop_alike=False, **kwargs):
        """Create all possible cross-transformations of the columns.

        Similar in spirit to the broadcasted version of func(x, x') if x is
        a 2D column vector. With `drop_alike` set to `False`, outputs a
        dataframe of size Tx(NN), otherwise Tx(NN-N). A straightforward
        application is to calculate all possible exchange cross-rates given
        a dataframe of exchange rates against a common currency.

        Parameters
        ----------
        func : callable
            function of 2 args operating on 2D arrays, such as __div__
        columns : list-like
            columns of the resulting dataframe
        drop_alike : bool
            whether to drop columns resulting from func(a,a) rather than
            func(a,b)
        kwargs : dict
            arguments to pandas.MultiIndex

        Returns
        -------
        res : pandas.DataFrame

        """
        # NB that self.columns redirects .columns to self.df
        if columns is None:
            pairs = [(p, q) for p in self.columns for q in self.columns]
            columns = pd.MultiIndex.from_tuples(pairs, **kwargs)

        # apply func w/broadcasting along the 3rd ax
        fx_3d = self.values[:, :, np.newaxis]

        # apply function, swapaxes acts as transposition in 3D space
        fx_cross_3d = func(fx_3d, fx_3d.swapaxes(1, 2))

        # reshape back to 2d; order ='C' to get the labels right
        fx_cross = fx_cross_3d.reshape((fx_3d.shape[0], -1, 1),
                                       order='C').squeeze(axis=2)

        # assemble
        res = pd.DataFrame(fx_cross, index=self.index, columns=columns)

        # drop xxxxxx-like columns
        if drop_alike:
            res = res.drop(res.columns[[p == q for p, q in res.columns]],
                           axis=1)

        return res

    def to_dataframe(self, drop_level=False):
        """Convert to a simple `pandas.DataFrame` squeezing column names.

        Returns
        -------
        res : pandas.DataFrame

        """
        res = self.df.copy()

        if isinstance(res.columns, pd.MultiIndex):
            if drop_level:
                res.columns = res.columns.droplevel(level=1)
            else:
                res.columns = res.columns.map('_'.join)

        return res


class FXDataFrame(SPDF):
    """DataFrame tailored to representing time-series of exchange rates.

    Parameters
    ----------
    df : pandas.DataFrame or FXDataFrame
    counter : str
        (optional) common counter currency

    """
    def __init__(self, df, counter=None):
        """
        """
        if isinstance(df, SPDF):
            # construct from self-like structure
            df_out = df.df
            # TODO: dirty hacks all around
            if counter is not None:
                if isinstance(df_out.columns, pd.MultiIndex):
                    df_out.columns = df_out.columns.droplevel("counter")
        elif isinstance(df, pd.Series):
            # construct from a pandas.Series
            df_out = df.to_frame()
        else:
            df_out = df

        # proper multiindex
        try:
            tuples_index = self.proper_multiindex(df_out.columns, counter)
            df_out.columns = tuples_index
        except AttributeError:
            pass

        super(FXDataFrame, self).__init__(df=df_out)

    @property
    def currencies(self):
        res = list(self.columns.get_level_values("base").unique()) + \
              list(self.columns.get_level_values("counter").unique())

        return list(set(res))

    def interlace(self, *args, **kwargs):
        """
        """
        res = self._interlace(*args, **kwargs)

        return self.__class__(res)

    def exhaust(self, drop_inverse=False):
        """Calculate all possible cross-rates of the exchange rates.

        Returns
        -------
        res : FXDataFrame

        """
        # for the sake of code brevity
        cols_0 = self.columns.get_level_values("base").unique()
        cols_1 = self.columns.get_level_values("counter").unique()

        # shortcuts
        if any([len(p) == 1 for p in [cols_0, cols_1]]):
            pairs = [
                ((p, q) if len(cols_1) == 1 else (q, p))
                for p in (cols_0 if len(cols_1) == 1 else cols_1)
                for q in (cols_0 if len(cols_1) == 1 else cols_1)
            ]

            # new columns names
            idx = pd.MultiIndex.from_tuples(pairs, names=["base", "counter"])

            fx_cross = self.interlace(func=lambda a, b: a / b,
                                      columns=idx, drop_alike=True)

            res = pd.concat(
                (fx_cross.df, self.df, self.fx_invert().df),
                axis=1)

            res = FXDataFrame(res)

        else:
            # iterate through all currencies, using the previous
            #   single-counter-currency code
            pairs = list(permutations(self.currencies, 2))
            idx = pd.MultiIndex.from_tuples(pairs, names=["base", "counter"])

            # kind of space for all possible cross-rates
            res = self.reindex(columns=idx)

            for c in self.currencies:
                this_portion_c = res.select_currency(
                    counter=c, simplify=False).exhaust().df
                this_portion_b = res.select_currency(
                    base=c, simplify=False).exhaust().df

                res = FXDataFrame(
                    res.df.fillna(this_portion_c).fillna(this_portion_b))

        if drop_inverse:
            res = res.drop_inverse()

        return res

    def drop_inverse(self):
        """Trim dataframe to retain only one of (xxxyyy, yyyxxx).

        Returns
        -------
        res : pandas.DataFrame

        """
        res = dict()

        for c, c_col in self.df.iteritems():
            if (c not in res) and (c[::-1] not in res):
                res[c] = c_col

        res = pd.concat(res, axis=1)

        return self.__class__(res)

    def calculate_rx(self, rf, horizon=1, log=True):
        """Calculate excess returns.

        Parameters
        ----------
        rf : RFDataFrame
            of risk-free rate, in (frac of 1) per period
        horizon : int
        log : bool

        Returns
        -------
        res : pandas.DataFrame

        """
        # cross-rf
        rf = rf + 1

        rf_x = rf.interlace(func=lambda a, b: a / b, drop_alike=True)
        rf_x.columns.names = self.columns.names

        # reindex to retain only rf with matching columns in ds
        rf_x = rf_x.reindex(columns=self.df.columns)

        # spot returns
        if log:
            ds = np.log(self.df).diff(horizon)
            rf_x = np.log(rf_x.df)
            res = ds + rf_x.shift(horizon)
        else:
            raise NotImplementedError

        return self.__class__(res)

    @staticmethod
    def fx_mul_series(first, second):
        """Multiply two series that are exchange rates.

        Extracts info from series' names (so the series must be named as
        ('xxx', 'yyy')): multiplication is only possible if one is the base
        currency of one matches the counter currency of the other.

        Parameters
        ----------
        first : pandas.Series
        second : pandas.Series

        Returns
        -------
        res : pandas.Series
            or None if multiplication makes no sense

        """
        def check_currency_alignment(ab, cd):
            """Determine if the base of one and counter of the other match."""
            if ab[0] == cd[1]:
                # case audusd * nzdaud
                aux_res = (cd[0], ab[1])
            elif ab[1] == cd[0]:
                # case audusd * usdnzd
                aux_res = (ab[0], cd[1])
            else:
                # some weird insensible case
                aux_res = None

            return aux_res

        # series must be named with ('xxx', 'yyy')
        assert isinstance(first.name, tuple)
        assert isinstance(second.name, tuple)

        xxxyyy = first.name
        pppqqq = second.name

        # check if multiplication makes sense
        res_name = check_currency_alignment(xxxyyy, pppqqq)

        # multiply if it does
        if res_name is not None:
            res = first.mul(second).rename(res_name)
        else:
            res = None

        return res

    def fx_mul(self, other):
        """Multiply.

        Wrapper around self.fx_mul_series, and essentially a loop over
        columns of `self`.

        Parameters
        ----------
        other : pandas.DataFrame

        Returns
        -------

        """
        if self.df.shape[1] < 2:
            res = self.fx_mul_series(self.df.squeeze(), other.df.squeeze())

            return self.__class__(res)

        # loop over columns
        res = list()

        for c, c_col in self.df.iteritems():
            this_prod = self.fx_mul_series(c_col, other.df.squeeze())
            if this_prod is not None:
                res.append(this_prod)

        # concat all series
        if len(res) > 0:
            res = pd.concat(res, axis=1)
        else:
            res = None

        return self.__class__(res)

    def fx_invert(self):
        """Compute exchange rate of yyy in units of xxx given the inverse.

        Basically, inverts the df and swaps the columns levels without
        changing level names.

        Returns
        -------
        res : pandas.DataFrame

        """
        # invert
        self_df_inv = self.df.pow(-1)

        # swap currencies: what used to be base, is no counter and vice versa
        self_df_inv = self_df_inv.swaplevel(axis=1)

        # ...but do not change names!
        self_df_inv.columns.names = ["base", "counter"]

        return self.__class__(self_df_inv)

    def fx_div(self, other):
        """Divide.

        Parameters
        ----------
        other : FXDataFrame
            with one column only

        Returns
        -------
        res : pandas.DataFrame

        """
        res = self.fx_mul(other.fx_invert())

        return res

    @staticmethod
    def proper_multiindex(idx, counter=None):
        """Construct a proper two-level `pandas.MultiIndex` for the dataframe.

        Parameters
        ----------
        idx : list-like
        counter : any

        Returns
        -------
        tuples_index : pandas.MultiIndex

        """
        if idx.nlevels > 1:
            idx.names = ["base", "counter"]
            return idx

        if counter is not None:
            tpl = [(c, counter) for c in idx]

        else:

            list_idx = list(idx)

            if isinstance(list_idx[0], str):
                tpl = [(p[:3], p[-3:]) for p in idx]
            elif isinstance(list_idx[0], tuple):
                tpl = list_idx
            else:
                raise ValueError("Not supported.")

        # construct MultiIndex
        tuples_index = pd.MultiIndex.from_tuples(tuples=tpl)

        tuples_index.names = ["base", "counter"]

        return tuples_index

    def select_currency(self, base=None, counter=None, simplify=False):
        """Select columns corresponding to a specific base or counter currency.

        Parameters
        ----------
        base : str
            a valid base currency name, must be present in the columns
        counter : str
            a valid counter currency name, must be present in the columns
        simplify : bool
            True for retaining one level only in the columns and convert to
            `pandas.DataFrame`

        Returns
        -------
        res

        """
        if (counter is not None) and (base is not None):
            res = self.df.loc[:, (base, counter)]

        else:
            res = self.df.xs((counter if base is None else base),
                             axis=1,
                             level=("counter" if base is None else "base"),
                             drop_level=simplify)
        if not simplify:
            res = FXDataFrame(res)

        return res


class EconomicsDataFrame(SPDF):
    """Representation of economics data.

    Parameters
    ----------
    df : pandas.DataFrame
    freq : str or pandas.frequency
    stationary : bool
    seasonal : bool

    """
    def __init__(self, df, freq=None, stationary=False, seasonal=False):
        """
        """
        if isinstance(df, pd.Series):
            df = df.to_frame()

        if isinstance(df, SPDF):
            self.__init__(df.df, freq, stationary)

        if freq is None:
            try:
                freq = df.index.freq
                freqstr = df.index.freqstr
            except AttributeError:
                freqstr = None
        else:
            aux_period = pd.Period("2000-01-01", freq=freq)
            freq = aux_period.freq
            freqstr = aux_period.freqstr

        super(EconomicsDataFrame, self).__init__(df, stationary=stationary,
                                                 freq=freq, freqstr=freqstr,
                                                 seasonal=seasonal)

    def interpolate(self, **kwargs):
        """Interpolate values (no extrapolation).

        Parameters
        ----------
        kwargs : dict
            arguments to `pandas.DataFrame.interpolate()`

        Returns
        -------
        res : EconomicsDataFrame

        """
        if self.stationary:
            y = self.df.copy()
        else:
            y = np.log(self.df)

        inplace = kwargs.pop("inplace", None)

        # due to pandas bug #6424, need workaround --------------------------
        #   time interpolation does not work, so transform to equally-spaced
        #   index first, reindex back later
        old_idx = self.index
        new_idx = pd.period_range(old_idx[0], old_idx[-1], freq=self.freq)
        y = y.reindex(index=new_idx)
        kwargs.update({"method": "linear"})

        # newer pandas can do this comme il faut
        if LooseVersion(pd.__version__) < LooseVersion("0.23.0"):
            idx_beyond = y.bfill().notnull() & y.ffill().notnull()
            res = y.interpolate(**kwargs).where(idx_beyond)
        else:
            kwargs.update({"limit_area": "inside"})
            res = y.interpolate(**kwargs)

        if not self.stationary:
            res = np.exp(res)

        # pandas bug #6424
        res = res.reindex(index=old_idx)

        if inplace:
            self.df = res
        else:
            return self.__class__(res)

    def extrapolate(self, *args, **kwargs):
        """

        Parameters
        ----------
        kwargs

        Returns
        -------

        """
        pass

    def construct_aggregate(self, aggr_map, weights=None):
        """Aggregate columns.

        If an aggregate already exists in `self.df`, NAs therein are filled
        with the synthetically constructed value.

        Handles nested dependencies.

        Parameters
        ----------
        aggr_map : dict
        weights : pandas.DataFrame
            of values used to calculate weights, same shape as `self.df`

        Returns
        -------

        """
        # handle nested dependencies, '_x' for '_extended'
        aggr_map_x = dict()

        for k, v in aggr_map.items():
            if not isinstance(v, (list, tuple)):
                raise ValueError("Aggregated values must be a listor tuple; "
                                 "occured at {}:{}".format(k, v))
            v_x = list()
            for vv in v:
                v_x += aggr_map.get(vv, [vv, ])

            aggr_map_x[k] = v_x

        df = self.df.copy()

        # weights are equal if not provided
        if weights is None:
            weights = (df * np.nan).fillna(1.0)

        # reindex with intersection of all values to exclude cases when
        #   constituents are missing
        constits = list(set(reduce(lambda x, y: x + y, aggr_map_x.values())))
        df_x = df.reindex(columns=constits)
        weights_x = weights.reindex(columns=constits).where(df_x.notnull())

        for k, v in aggr_map_x.items():
            # weights
            this_w = weights_x.loc[:, v].div(weights_x.loc[:, v].sum(axis=1),
                                             axis=0)

            # dot product with weights
            df.loc[:, k] = (df_x.loc[:, v] * this_w).sum(axis=1, skipna=True,
                                                         min_count=1)

        return self.__class__(df)

    @staticmethod
    def from_change_to_level(level_orig, change=None, logchange=None):
        """Turn (log-) changes back into levels.

        Given initial levels of the variable and a new dataframe of either
        simple (log-) changes thereof, produces the new levels by adding the
        cumulative increments (multiplying with the cumulative product).

        Parameters
        ----------
        level_orig : pandas.DataFrame
            of 'original' level values
        change : pandas.DataFrame
            of simple changes
        logchange : pandas.DataFrame
            of log-changes

        Returns
        -------
        res : pandas.DataFrame

        """
        if (change is None) and (logchange is None):
            raise ValueError("Specify one of 'change' and 'logchange'!")

        # fetch the first values in each column to base the cumulated on
        first_values = level_orig.bfill().iloc[0]

        # broadcast them row-wise
        df_init = (level_orig * np.nan).fillna(1.0).mul(first_values, axis=1)

        if change is not None:
            # add to cumsum
            c_ret = change.fillna(0.0).cumsum()\
                .where(level_orig.fillna(change).notnull())
            res = df_init.add(c_ret)

        else:
            # multiply with cumsum
            c_ret = np.exp(logchange.fillna(0.0).cumsum())\
                .where(level_orig.fillna(logchange).notnull())
            res = df_init.mul(c_ret)

        return res

    def fillna(self, level=None, diff=None, logdiff=None, **kwargs):
        """Fill missing values, possibly using (log-) changes.

        Parameters
        ----------
        level : pandas.DataFrame
        diff : pandas.DataFrame
        logdiff : pandas.DataFrame
        kwargs : dict
            arguments to `pandas.fillna`

        Returns
        -------
        res : EconomicsDataFrame

        """
        if (diff is None) and (logdiff is None):
            res = self.df.fillna(level, **kwargs)

        else:
            res = self.from_change_to_level(self.df, diff, logdiff)

        return self.__class__(res)

    def splice(self, other, direction="bfill", method="simple",
               transform=None, **kwargs):
        """Splice dataframe with another.

        Fills NA in `self` with (possibly transformed) values from `other`.
        Fills either trailing NAs (direction='bfill') or leading ones
        (direction='ffill'). Note that no outer join on the index axis is
        performed.

        Parameters
        ----------
        other : EconomicsDataFrame
            of data with possibly fewer NAs
        direction : str
            'bfill' or 'ffill'
        method : str
            estimator applied to the transformed (see below) values in `other`:
            # 'simple' to use no other transformations;
            # 'ols' to use OLS to proxy values in `other` with those in `self`
        transform : str
            transformation to apply to values before proceeding with 'method'
            (see above):
            # 'diff' to difference the series;
            # 'logdiff' to log-difference the series.
        kwargs : dict
            arguments to Equations, e.g. add_constant

        Returns
        -------
        res : EconomicsDataFrame
            with spliced data

        TODO: allow any of the df to have empty columns

        """
        if direction == "bfill":
            aux_res = self.__class__(self.df.iloc[::-1]).splice(
                other=other.__class__(other.df.iloc[::-1]),
                direction="ffill",
                method=method,
                transform=transform,
                **kwargs)

            return self.__class__(aux_res.df.iloc[::-1])

        # if there is nothing to add
        if other.dropna(how="all").empty:
            return self

        # function to transform before and after the actual splice
        if transform not in (None, "diff", "logdiff"):
            raise ValueError("Valid transforms are 'diff' and 'logdiff.")

        if transform is None:
            transform = "None"

        transform_map = {
            "diff": lambda x: x.diff(),
            "logdiff": lambda x: np.log(x).diff(),
            "None": lambda x: x
        }

        transform_to = transform_map[transform]

        self_trans = transform_to(self.df)
        other_trans = transform_to(other.df)

        # # align the two
        # self_trans, other_trans = self_trans.align(other_trans, axis=0,
        #                                            join="outer")

        # drop columns that do not have a match in `other`
        not_in_other = self_trans.reindex(
            columns=self_trans.columns.difference(
                other_trans.dropna(axis=1, how="all").columns))
        self_trans, other_trans = self_trans.align(
            other_trans.dropna(axis=1, how="all"), axis=1, join="inner")

        # fit values --------------------------------------------------------
        if method == "simple":
            yhat = other_trans

        elif method == "rescale":
            yhat = other_trans\
                .sub(other_trans.mean())\
                .div(other_trans.std())\
                .mul(self_trans.std())\
                .add(self_trans.mean())

        elif method == "ols":
            # make `other` conform to Equations arglist
            x = other_trans.rename(columns=lambda c: (c, 'x'))
            x.columns = pd.MultiIndex.from_tuples(x.columns)
            # model
            mod = Equations(y=self_trans, x=x, **kwargs)
            # estimate
            est = mod.one_by_one()
            # fitted values
            yhat = est.get_yhat(original_x=True)

        else:
            raise ValueError("Valid methods are 'simple', 'rescale', 'ols'.")

        # splice ------------------------------------------------------------
        ff = self_trans.fillna(yhat).where(self_trans.ffill().notnull())

        # reinstall columns that were not in `other`
        ff = pd.concat((ff, not_in_other), axis=1)\
            .reindex(columns=self.columns)

        if transform == "diff":
            res = self.from_change_to_level(self.df, change=ff)
        elif transform == "logdiff":
            res = self.from_change_to_level(self.df, logchange=ff)
        else:
            res = ff

        return self.__class__(res)

    def deseasonalize(self, method, freq=None):
        """

        Parameters
        ----------
        method : str
            'ols' or 'mean'
        freq : str

        Returns
        -------

        """
        if not self.seasonal:
            warnings.warn("The series is already not seasonal, as indicated "
                          "by `self.seasonal=False`.")

        if method.lower().startswith("ols"):
            raise NotImplementedError("There is currently an error in ols.")
            res = self._deseasonalize_by_ols(freq)
        elif method.lower().startswith("mean"):
            res = self._deseasonalize_by_demeaning(freq)
        elif method.lower() == "stl":
            res = self._deseasonalize_by_stl(freq)
        else:
            raise NotImplementedError("Method must be 'ols', 'means' or "
                                      "'stl'.")

        self.seasonal = False

        return self.__class__(res)

    def _deseasonalize_by_ols(self, freq=None):
        """Deseasonalize using dummies.

        Parameters
        ----------
        freq : str
            frequency to get dummies for
        """
        if freq is None:
            freq = self.freqstr

        # get dummies -------------------------------------------------------
        d = self.df.index.map(get_period_getter(freq))
        d = pd.Series(d, index=self.df.index)

        dummies = pd.concat(
            {c: pd.get_dummies(d, drop_first=True) for c in self.df.columns},
            axis=1)

        # set up regression system ------------------------------------------
        y = self.df.copy()

        if not self.stationary:
            y = np.log(y).diff()

        mod = Equations(y=y, x=dummies, add_constant=False)

        # estimate ----------------------------------------------------------
        mod_est = mod.one_by_one()

        # predict -----------------------------------------------------------
        yhat = mod_est.get_residuals(original_x=True, stack=False)

        if not self.stationary:
            first_values = pd.Series({c: v.dropna().iloc[0]
                                      for c, v in self.df.iteritems()})
            df_init = pd.DataFrame(1.0, index=self.df.index,
                                   columns=self.df.columns).mul(first_values)
            yhat = df_init.where(self.df.notnull()).mul(np.exp(yhat.cumsum()))

        return yhat

    def _deseasonalize_by_demeaning(self, freq=None):
        """Deseasonalize using means.

        Parameters
        ----------
        freq : str or list-like
            frequency to get dummies for

        Returns
        -------
        res : pandas.DataFrame
        """
        if freq is None:
            freq = self.freqstr

        # set up regression system ------------------------------------------
        y = self.df.copy()

        if not self.stationary:
            y = np.log(y).diff()

        # demean ------------------------------------------------------------
        # function to fetch the period
        freqfun = get_period_getter(freq)

        # locally demean each group
        y_desn = list()
        for f, grp in y.groupby(freqfun):
            y_desn.append(grp - grp.mean())

        # concat and add back the total mean
        y_desn = pd.concat(y_desn, axis=0).sort_index(axis=0) + y.mean()

        # ether return as is or cumprod conditional on being stationary -----
        if not self.stationary:
            # fetch the first values in each column to base the cumprod off
            res = self.from_change_to_level(level_orig=self.df,
                                            logchange=y_desn)

        else:
            res = y_desn

        return res

    def _deseasonalize_by_stl(self, freq=None):
        """

        Returns
        -------

        """
        if freq is None:
            freq = freq_to_period(self.freq)

        y = self.df.copy()

        if not self.stationary:
            y = np.log(y)

        def deseasonalize_one(x):
            stl = seasonal_decompose(x, model='additive', freq=freq)
            aux_res = x - stl.seasonal
            return aux_res

        y_ds = pd.concat(
            {cname: deseasonalize_one(col.dropna())
             for cname, col in y.iteritems()},
            axis=1
        )

        y_ds = y_ds.reindex(index=y.index)

        if not self.stationary:
            res = self.from_change_to_level(level_orig=self.df,
                                            logchange=y_ds.diff())
        else:
            res = y_ds

        return res

    def concat(self, other, **kwargs):
        """

        Parameters
        ----------
        other : pandas.DataFrame or EconomicsDataFrame
        kwargs : dict
            arguments to `pandas.concat`

        Returns
        -------
        res : EconomicsDataFrame

        """
        if other is None:
            return self

        if not isinstance(other, EconomicsDataFrame):
            other = EconomicsDataFrame(other)

        # indices to the common frequency
        freq_len_self = pd.to_datetime("2000-01-01") + self.freq
        freq_len_other = pd.to_datetime("2000-01-01") + other.freq

        if freq_len_self <= freq_len_other:
            the_freq = self.freq
        else:
            the_freq = other.freq

        res = pd.concat((
            self.df.resample(the_freq, convention='e').asfreq(),
            other.df.resample(the_freq, convention='e').asfreq()), **kwargs)

        return self.__class__(res)

    def to_dataframe(self, datetimeindex=False, **kwargs):
        """

        Parameters
        ----------
        datetimeindex : bool
            True to convert PeriodIndex to DatetimeIndex (using endpoints)
        kwargs : any
            e.g. drop_level

        Returns
        -------

        """
        res = super().to_dataframe(**kwargs)

        if datetimeindex:
            res.index = res.index.to_timestamp(how='e')

        return res


class CPIDataFrame(EconomicsDataFrame):
    """
    """
    # def __init__(self, df, *args, **kwargs):
    #     # do not allow zeros in CPI
    #     super().__init__(df.replace(0.0, np.nan), *args, **kwargs)

    # def extrapolate(self, **kwargs):
    #     """
    #
    #     Parameters
    #     ----------
    #     kwargs
    #
    #     Returns
    #     -------
    #
    #     """
    #     # to inflation
    #     inflation = np.log(self.df).diff()
    #
    #     # extrapolate column-by-column
    #     inflation_x = dict()
    #
    #     for c, c_col in inflation.iteritems():
    #         inflation_x[c] = extrapolate(c_col, **kwargs)
    #
    #     inflation_x = pd.concat(inflation_x, axis=1)
    #
    #     # from inflation back to cpi
    #     res = self.fillna(logdiff=inflation_x)
    #
    #     return res

    def interlace(self):
        """

        Returns
        -------

        """
        def func(a, b):
            return a / b

        res = self._interlace(func=func, columns=None, drop_alike=True,
                              names=["base", "counter"])

        return res

    def to_inflation(self, log=True, yoy=False):
        """

        Parameters
        ----------
        log : bool
            True to compute log-differences
        yoy : bool
            True to compute year-on-year changes

        Returns
        -------
        res : EconomicsDataFrame
            with inflation

        """
        cpi_lag = self.df.shift(12 if yoy else 1)

        res = np.log(self.df / cpi_lag)

        if not log:
            res = np.exp(res) - 1

        return EconomicsDataFrame(res, stationary=True)


class CAToGDPDataFrame(EconomicsDataFrame):
    """
    """
    def __init__(self, df, freq=None, **kwargs):
        """
        """
        super().__init__(df, freq, stationary=True, **kwargs)

    def interlace(self):
        """

        Returns
        -------

        """
        def func(a, b):
            return a - b

        res = self._interlace(func=func, columns=None, drop_alike=True,
                              names=["base", "counter"])

        return self.__class__(res)


class RFDataFrame(EconomicsDataFrame):
    """
    """
    def interlace(self, *args, **kwargs):
        """

        Parameters
        ----------
        args
        kwargs

        Returns
        -------

        """
        res = self._interlace(*args, **kwargs)

        return self.__class__(res)

    def per_period(self, period):
        """

        Parameters
        ----------
        period : str
            frequency string (pandas-like) such as '1M' or '2Y', can be
            composed of integer of any length and only one letter at the end

        Returns
        -------

        """
        assert isinstance(period, str)
        if len(period) < 2:
            period = "1" + period

        mapping = {
            "M": 1/12,
            "Y": 1.0,
            "D": 1/365,
            "B": 1/252,
            "Q": 1/4
        }

        ratio = int(period[:-1]) * mapping[period[-1]]

        return self.__class__(self.df * ratio)


class REERDataFrame(EconomicsDataFrame):
    """
    """
    def interlace(self):
        """

        Returns
        -------

        """
        def func(a, b):
            return a / b

        res = self._interlace(func=func, columns=None, drop_alike=True,
                              names=["base", "counter"])

        return res

    @classmethod
    def synthetic(cls, spot, cpi):
        """

        Parameters
        ----------
        spot : FXDataFrame
            of exchange rates against a common currency (for the cross-rates)
        cpi : pandas.DataFrame
            of CPI indexes, indexed with the 3-letter ISOs of `spot`

        Returns
        -------
        res : FXDataFrame

        """
        # calculate synthetic REER of aaa against bbb:
        #   REER = S*CPI_base/CPI_counter
        # 1) S & CPI: cross-rates
        fx_x_for_est = spot.exhaust()

        cpi_x_for_est = FXDataFrame(cpi, counter="xxx") \
            .exhaust() \
            .drop("xxx", axis=1, level="counter") \
            .drop("xxx", axis=1, level="base")

        # 2) multiply with cpi
        reer_x_synt = fx_x_for_est.df.mul(cpi_x_for_est.df)

        res = FXDataFrame(reer_x_synt)

        return res
