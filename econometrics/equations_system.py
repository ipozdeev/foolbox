import pandas as pd
import numpy as np
from scipy import linalg
from collections import OrderedDict

from foolbox.econometrics import _estimators


class Equations:
    """Equations system.

    The system has N equations, each featuring 1 dependent variable and
    several (possibly a different number for each equation) regressors.

    Dependent variables are collected into a `pandas.DataFrame` with column
    names corresponding to equation names; regressors are collected into a
    `pandas.DataFrame` columned by a `pandas.MultiIndex` of (equation_name,
    regressor), such that each individual equation can be written as:
        y.loc[:, c] ~ x.loc[:, (c, slice(None))]

    Parameters
    ----------
    y : pandas.DataFrame
        with equation names for columns
    x : pandas.DataFrame
        with multiindex (equation_name, regressor_name) for columns
    add_constant : bool
        True to insert constant into each equation's regressors
    dropna : bool
        True to drop rows with at least one NA in either equation; necessary
        for SUR, not necessary for pooled or one-by-one estimation
    add_jitter : bool
        True to add to x some uniformly distributed rv in (-1e06, 1e06);
        necessary sometimes to avoid matrix_rank(x[i]) < x[i].shape[1]
    """
    def __init__(self, y, x, add_constant=False, dropna=True,
                 add_jitter=False):
        # makes sure all ys have regressors ---------------------------------
        assert all(y.columns.isin(x.columns.get_level_values(0)))

        # store original values ---------------------------------------------
        y_orig = y.copy()

        if add_constant:
            x_orig = self.insert_regressor(x, "const").copy()
        else:
            x_orig = x.copy()

        # set names on levels -----------------------------------------------
        y_orig, x_orig = self.set_index_names(y_orig, x_orig)

        # align on the index axis -------------------------------------------
        if dropna:
            y_aln, x_aln = y_orig.dropna() \
                .align(x_orig.dropna(), axis=0, join="inner")
        else:
            y_aln, x_aln = y_orig.align(x_orig, axis=0, join="inner")

        # sort indices to ease conversion to numpy later
        y_aln = y_aln.sort_index(axis=1)
        x_aln = x_aln.sort_index(axis=1)

        if add_jitter:
            x_aln += np.random.random(size=x_aln.shape) * 1e-06

        self.constraints = None

        self.add_constant = add_constant
        self.y_orig = y_orig
        self.x_orig = x_orig
        self.y = y_aln
        self.x = x_aln
        self.eq_level_name = y_orig.columns.name
        self.reg_level_name = x_orig.columns.names[1]
        self.equations = y_aln.columns
        self.regressors = x_aln.columns.get_level_values(1).unique()

    @staticmethod
    def set_index_names(y, x):
        """Set level names.

        Parameters
        ----------
        y : pandas.DataFrame
        x : pandas.DataFrame

        Returns
        -------
        y : pandas.DataFrame
        x : pandas.DataFrame

        """
        if y.columns.name is None:
            if x.columns.names[0] is None:
                y.columns.name = "equation"
            else:
                y.columns.name = x.columns.names[0]

        if x.columns.names[1] is None:
            x.columns.names = [y.columns.name, "regressor"]
        else:
            x.columns.names = [y.columns.name, x.columns.names[1]]

        return y, x

    def broadcast_over_equations(self, df, axis):
        """Clone dataframe for each equation and stack result horizontally.

        Parameters
        ----------
        df : pandas.DataFrame
        axis : int
            axis to concat along

        Returns
        -------
        res : pandas.DataFrame
            with MultiiIndex for columns

        """
        # repeat and concat
        res = pd.concat({eq_name: df for eq_name in self.equations},
                        axis=axis)

        # rename index/columns levels
        if isinstance(res, pd.Series):
            res.index.names = [self.eq_level_name, self.reg_level_name]
        else:
            res.columns.names = [self.eq_level_name, self.reg_level_name]

        return res

    def insert_regressor(self, x, new_x="const"):
        """Add a new column to all 0-levels of the dataframe.

        Parameters
        ----------
        x : pandas.DataFrame
        new_x : pandas.Series or str
            'const' to add a columns of ones

        Returns
        -------
        res : pandas.DataFrame
            with the new column (equation_name, <colname>)

        """
        if isinstance(new_x, str):
            new_x = pd.Series(1.0, index=x.index).rename("const")
            return self.insert_regressor(x, new_x)

        if new_x.name is None:
            raise ValueError("`new_x` must have a name!")

        # multiindex of type (equation_name, <colname>)
        colnames = x.columns.get_level_values(0).unique()

        # dataframe of new columns with the above multiindex
        new_xs = pd.concat({(c, new_x.name): new_x for c in colnames}, axis=1)
        new_xs.columns.names = x.columns.names

        # concat df and constants and sort columns
        res = pd.concat((x, new_xs), axis=1)

        return res

    def add_constraints(self, r=None, q=None):
        """Add constraints of the form rB = q

        Parameters
        ----------
        r : pandas.DataFrame
        q : pandas.Series

        Returns
        -------

        """
        r = r.reindex(columns=self.x.columns).fillna(0.0)

        if q is None:
            q = pd.Series(0.0, index=r.index)

        self.constraints = {"r": r, "q": q}

    def make_equal_coef_constraints(self, column_mapper=None, grouper=None):
        """Make DataFrame of constraints imposing equality of coefficients.

        Those regressors which belong to the same group, as dictated by
        `grouper` of `column_mapper` will have their coefficients imposed to
        be equal.

        Parameters
        ----------
        column_mapper : callable (optional)
            many-to-fewer transformation to arrive at a grouper
        grouper : pandas.Index (optional)
            grouper

        Returns
        -------
        None

        """
        if grouper is None:
            if not isinstance(column_mapper, (list, tuple)):
                column_mapper = [column_mapper, ]

            reg_level_grouped = self.x.columns.get_level_values("regressor")

            for func in column_mapper:
                reg_level_grouped = reg_level_grouped.map(func)

            grouper = pd.Series(index=self.x.columns, data=reg_level_grouped)

        # make constraints
        constraints = list()

        # the idea is to fill the columns corresponding to the coefficients
        #   which are imposed to be equal with +1 and -1, and to fill the
        #   rest with zeros
        for _, grp in self.x.groupby(axis=1, by=grouper):
            n = grp.shape[1]
            constraints_tmp = pd.DataFrame(
                data=np.hstack((np.ones((n - 1, 1)), -1 * np.eye(n - 1))),
                columns=grp.columns)
            constraints.append(constraints_tmp)

        # concat
        constraints = pd.concat(constraints, axis=0) \
            .reindex(columns=self.x.columns) \
            .fillna(0.0) \
            .reset_index(drop=True)

        # add to self
        self.add_constraints(r=constraints)

    def transform_for_sheppard(self, which_estimator="system"):
        """Represent in a form suitable for Sheppard's linearmodels.

        Parameters
        ----------
        which_estimator : str
            'system' for SUR, IVSystemGMM or 'panel' for PanelOLS

        Returns
        -------

        """
        if which_estimator.lower() == "system":
            equations = OrderedDict()
            for eq in self.equations:
                equations[eq] = {"dependent": self.y[eq], "exog": self.x[eq]}

            res = equations

        elif which_estimator.lower() == "panel":
            y = self.y.stack(dropna=False).swaplevel()
            if not isinstance(y.name, str):
                y.name = "y"
            x = self.x.stack(level=0, dropna=False).swaplevel(axis=0)

            res = (y, x)

        else:
            raise NotImplementedError

        return res

    def to_sheppard(self, estimator, **kwargs):
        """Represent in a form suitable for Sheppard's linearmodels.

        Parameters
        ----------
        estimator : callable
            one of linearmodels' estimators, e.g. PanelOLS, IVSystemGMM or SUR
        kwargs : dict
            arguments to `estimator`, e.g. weights, entity_effects etc.

        Returns
        -------
        model

        """
        assert callable(estimator)

        # differentiate between system (e.g. SUR) and panel (e.g. PanelOLS)
        #   estimators
        if estimator.__module__ == "linearmodels.system.model":
            equations = OrderedDict()
            for eq in self.equations:
                equations[eq] = {"dependent": self.y[eq], "exog": self.x[eq]}

            model = estimator(equations=equations, **kwargs)

        elif estimator.__module__ == "linearmodels.panel.model":
            y = self.y.stack(dropna=False).swaplevel()
            if not isinstance(y.name, str):
                y.name = "y"
            x = self.x.stack(level=0, dropna=False).swaplevel(axis=0)

            model = estimator(dependent=y, exog=x, **kwargs)

        else:
            raise NotImplementedError

        # add constraints
        if self.constraints is not None:
            model.add_constraints(r=self.constraints["r"],
                                  q=self.constraints["q"])

        return model

    def to_gretl(self, path2data):
        """

        Returns
        -------

        """
        # prepare data for export and import to gretl
        self.y.to_csv(path2data + "y.csv", index_label="date")
        x = self.x.copy()
        x.columns = ["_".join(cc) for cc in self.x.columns]
        x.to_csv(path2data + "x.csv", index_label="date")

        # data import statements
        data_cmd = "\n".join((
            "open " + path2data + "y.csv",
            "append " + path2data + "x.csv"
        ))

        # function to write one equation
        def equation_maker(eq_name):
            aux_res = "equation {} ".format(eq_name) + " ".join(
                [eq_name + "_" + x_name for x_name in self.x[eq_name].columns]
            )
            return aux_res

        # create command for system estimation
        system_cmd = "\n".join(
            ["new_sys <- system", ] +
            ["\t{}".format(equation_maker(eq)) for eq in self.equations] +
            ["end system"]
        )

        # constraints
        # eq_name_to_num = {v: k for k, v in enumerate(self.equations)}
        # x_name_to_num = {k: k for k, v in enumerate(self.equations)}
        # restr_enum = pd.concat({
        #     eq_name_to_num[eq_name]: grp.T.reset_index(drop=True).T
        #     for eq_name, grp in self.constraints["r"].groupby(
        #         axis=1, level="equation")
        # }, axis=1)
        #
        # def restriction_maker(restr_row):
        #     # row_idx is pythonic, i.e. 0 indexes the first item
        #     row_no_zeros = restr_row.where(restr_row != 0).dropna()
        #     aux_res = \
        #         " + ".join([
        #             "{mult} * b[{idx_1},{idx_2}]".format(mult=v,
        #                                                  idx_1=eq_num + 1,
        #                                                  idx_2=coef_num + 1)
        #             for (eq_num, coef_num), v in row_no_zeros.iteritems()
        #         ]) +\
        #         " = {}".format(self.constraints["q"].iloc[restr_row.name])
        #     return aux_res
        #
        # restr_list = list()
        # for _, row in restr_enum.iterrows():
        #     restr_list.append("\t{}".format(restriction_maker(row)))

        # for eq_num, eq_name in enumerate(self.equations):
        #     this_grp = self.constraints["r"][eq_name]\
        #         .where(self.constraints["r"][eq_name] != 0.0)\
        #         .dropna(how="all")
        #     for r_idx, row in this_grp.iterrows():
        #         restr_list.append(restriction_maker(eq_num, row))

        # constraints: alternative
        r = ";\\\n\t".join(
            [", ".join(row.astype(str))
             for row in self.constraints["r"].values]
        )
        q = ";\\\n\t".join([row.astype(str) for row in
                            self.constraints["q"].values])

        rmat_str = "matrix Rmat = {{\\\n\t{}\\\n}}"
        qvec_str = "matrix Qvec = {{\\\n\t{}\\\n}}"
        r_mat_str = "\n".join((rmat_str, qvec_str)).format(r, q)

        restr_list = ["\tR = Rmat\n\tq = Qvec", ]

        if self.constraints is None:
            restriction_cmd = ""
        else:
            restriction_cmd = "\n".join(
                [r_mat_str, ] +
                ["restrict new_sys", ] +
                restr_list +
                ["end restrict"]
            )

        # estimate
        estimate_cmd = "estimate new_sys method=sur --iterate"

        # commands together
        commands = "\n\n".join((
            data_cmd, system_cmd, restriction_cmd, estimate_cmd
        ))

        with open(path2data + "script.inp", mode="w") as f:
            f.write(commands)

    def one_by_one(self):
        """Estimate equation-by-equation."""
        coef = dict()

        # loop over equations:
        for eq_name, this_y in self.y.iteritems():
            this_coef = _estimators.ols(this_y.values, self.x[eq_name].values)
            coef[eq_name] = pd.Series(this_coef, index=self.regressors)

        coef = pd.concat(coef, axis=0)

        return coef

    def pooled_ols(self):
        """Estimation via pooled OLS.

        Each equation has to feature the same number of regressors. The
        dependent and independent variables are stacked column-wise to move
        from (n,m) and (n,mk) to (nm,) and (nm,k) respectively, than OLS is
        applied.

        Returns
        -------
        res : SimEqEstimationResults

        """
        y_stacked = self.y.stack()
        x_stacked = self.x.stack(level=self.eq_level_name)

        # estimate
        coef = _estimators.ols(y_stacked.values, x_stacked.values)

        # put structure on the estimated coef
        coef = pd.Series(coef, index=x_stacked.columns)

        return coef

    def sur(self, method="fgls", **kwargs):
        """Seemingly unrelated regressions.

        See Greene for reference.

        Parameters
        ----------
        method : str
            'ols', 'gls' or 'fgls'
        kwargs : dict
            for 'ols': nothing,
            for 'gls': 'omega', the weighting matrix
            for 'fgls': 'iterate' and 'tol'

        Returns
        -------

        """
        # prepare data
        y_stacked, x_stacked, n_x, n_obs = self.prepare_for_sur(self.y, self.x)

        if method == "ols":
            coef = _estimators.ols(y_stacked, x_stacked)
        elif method == "fgls":
            coef = _estimators.sur_fgls(y_stacked, x_stacked, n_obs, n_x,
                                        **kwargs)
        elif method == "gls":
            coef = _estimators.sur_gls(y_stacked, x_stacked, **kwargs)
        else:
            raise ValueError("Method unknown; choose 'ols', 'gls' of 'fgls'.")

        coef = pd.Series(data=coef, index=self.x.columns)

        return coef

    @staticmethod
    def prepare_for_sur(y, x):
        """Prepare variables for SUR estimation.

        Stack variables, count the number of regressors and observations.

        Parameters
        ----------
        y : pandas.DataFrame
            (n,m) of dependent variables, each one corresponsing to an equation
        x : pandas.DataFrame
            (n,p) of regressors, a total of p (can be diff for diff equations)

        Returns
        -------
        y_stacked : numpy.ndarray
            (nm,)
        x_stacked : numpy.ndarray
            (nm,p) block-diagonal
        n_x : pandas.Series
            (m,) the no of regressors for each of m equations
        n_obs : int
            the no of observations; must be the same for all equations

        """
        equations = y.columns

        # stack y, stack x: y becomes (NxM,1) vector with countries on the
        #   outer level
        y_stacked = y.stack().swaplevel(0, 1).sort_index().values
        x_stacked = linalg.block_diag(*[x[eq].values for eq in equations])

        # number of regressors (K) and observations (N)
        # need eqs because of pandas issue #2770 on github
        n_x = x.count(axis=1, level=0).loc[:, equations].mean().values
        n_obs = int(y.count().mean())

        return y_stacked, x_stacked, n_x, n_obs

    def expanding_sur(self, min_periods, **kwargs):
        """Implement expanding windows for the SUR estimation.

        Loop over time is used; (unfortunately) no cool `pandas.expanding()`
        stuff.

        Parameters
        ----------
        min_periods : int
            similar to `min_periods` in `pandas.expanding()`
        kwargs : dict
            additional arguments to `SimultaneousEquations.sur()`

        Returns
        -------
        coef : pandas.Series

        """
        # loop over time ----------------------------------------------------
        coef = list()

        for t in self.y.index[min_periods:]:
            # subsample and prepare data
            y_stacked, x_stacked, n_x, n_obs = \
                self.prepare_for_sur(self.y.loc[:t, :], self.x.loc[:t, :])

            this_coef = _estimators.sur_fgls(y_stacked, x_stacked, n_obs, n_x,
                                             **kwargs)
            coef.append(this_coef)

            print(t)

        # transpose to have dates for index, (country, duration) for columns
        coef = pd.DataFrame(data=coef, index=self.y.index[min_periods:],
                            columns=self.x.columns)

        return coef
