from foolbox.api import *
from wp_tabs_figs.wp_settings import settings

"""Generate table with descriptives for a selection of trading strategies (rx
bas-adjusted returns)
"""


def strat_selector(returns, selection_map, percentiles):
    """Given a selection criterion (e.g. mean return), get a subsample of
    of returns for this criterions' percentiles

    Parameters
    ----------
    returns: pd.DataFrame
        of returns to strategies
    selection_map: callable
        specifying selection criterion, e.g. lambda x: x.sum() for selecting
        strategies according to total return, lambda x: x.mean()/x.std()
        according to Sharpe ratios and so on
    percentiles: iterable
        with values between 0 and 1, specifying percentiles of 'selection_map'
        to be reported

    Returns
    -------
    sample: pd.DataFrame
        containing returns to strategies according percentiles of criterion
        from the selection map

    """
    # Apply the selection map to the returns
    mapped_returns = selection_map(returns)

    # Loop over percentiles, find the corresponding columns
    sample_columns = list()
    for percentile in percentiles:
        # Get return corresponding to the percentile of interest
        this_mapped_return = mapped_returns.quantile(percentile,
                                                     interpolation="lower")
        # Locate it in returns and get the corresponding column id
        sample_columns.extend(
            mapped_returns.where(mapped_returns ==
                                 this_mapped_return).dropna().index.tolist()
            )

    # Locate the requested percentiles in the input data
    sample = returns.loc[:, sample_columns]

    return sample


if __name__ == "__main__":
    out_path = data_path + settings["fig_folder"]

    # Strats file name
    strat_data_file = "broomstick_ret_rx_bas.p"

    percentiles_to_describe = np.arange(0.1, 1, 0.1)
    scale_to = 10

    selection_maps = [
        lambda x: x.sum(),
        lambda x: x.mean() * (scale_to / pd.Series(
            x.columns.get_level_values("holding").tolist(),
            index=x.columns)),
        # lambda x: x.mean() / x.std() * x.count().pow(0.5)
        # lambda x: x.mean() / x.std() * (scale_to / pd.Series(
        #     x.columns.get_level_values("holding").tolist(),
        #     index=x.columns)).pow(0.5),
        # lambda x: taf.descriptives(x, 1).loc["tstat"]
        ]

    # Formatting of numbers and standard errors
    fmt_coef = "{:3.2f}"
    fmt_se = "{:3.2f}"
    fmt_std = "{:3.1f}"

    # Import return to the universe of pre-announcement trading strategies
    with open(data_path + strat_data_file, mode="rb") as halupa:
        all_strats = pickle.load(halupa) * 1e4  # convert to bps
    # all_strats = pd.read_pickle(data_path + "broom_ret_rx_bas.p")

    # Loop over selection maps, get descriptives for each map
    all_descr = list()
    for this_map in selection_maps:
        # Fill this df with descriptives
        this_descr = pd.DataFrame(
            index=["holding", "threshold", "mean", "se_mean", "median", "std",
                   "skew", "kurt", "sharpe", "count"],
            columns=percentiles_to_describe, dtype=str)

        # Get a sample of strategies
        strats = strat_selector(all_strats, this_map, percentiles_to_describe)

        # Get the scaling factor for each, to mimic the 10-days holding period
        scale = pd.Series(
            [scale_to/holding_period for (holding_period, threshold) in
             strats.columns.tolist()],
            index=strats.columns)

        # (De)leverage each strategy to the 10-day holding period
        scaled_strats = strats.mul(scale)

        # Estimate statistics
        tmp_descr = taf.descriptives(scaled_strats, 1)

        # Rescale Sharpe ratio to 10-day holding periods
        # TODO: Discuss 'annualization' vs leverage with Igor, the latter is
        # TODO: meaningless for Sharpe ratios
        tmp_descr.loc[["sharpe"]] *= scale.pow(0.5)

        # Populate the output
        this_descr.loc["holding", :] = \
            tmp_descr.columns.get_level_values("holding").tolist()
        this_descr.loc["threshold", :] = \
            tmp_descr.columns.get_level_values("threshold").tolist()

        tmp_descr.columns = this_descr.columns

        this_descr.loc[["mean", "median", "skew", "kurt", "sharpe"]] = \
            tmp_descr.loc[["mean", "median", "skew", "kurt", "sharpe"]]\
            .applymap(fmt_coef.format)

        this_descr.loc[["std"]] = \
            tmp_descr.loc[["std"]].applymap(fmt_std.format)

        this_descr.loc[["se_mean"]] = tmp_descr.loc[["se_mean"]].applymap(
            ("(" + fmt_se + ")").format)

        this_descr.loc[["count"], :] = scaled_strats.count().values

        # Apply \multicolumn to rows with integers, fuck latex, btw
        this_descr.loc[["holding", "threshold", "count"]] = \
            this_descr.loc[["holding", "threshold", "count"]].applymap(
                lambda x: "\multicolumn{1}{c}{" + str(x) + "}"
            )

        # Append the final output
        all_descr.append(this_descr)

    # Pool and dump the output
    all_descr = pd.concat(all_descr)
    all_descr.to_latex(buf=out_path + "raw_strat_descriptives.tex",
                       column_format="l" + "W" * len(all_descr.columns),
                       escape=False)
