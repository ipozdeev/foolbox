if __name__ == "__main__":

    from foolbox.api import *
    from foolbox.wp_tabs_figs.wp_settings import settings
    from foolbox.utils import *
    from foolbox.fxtrading import *
    from foolbox.visuals import broomstick_plot

    # Set the output path, input data and sample
    start_date = pd.to_datetime(settings["sample_start"])
    end_date = pd.to_datetime(settings["sample_end"])

    # data
    with open(data_path+"fx_by_tz_aligned_d.p", mode="rb") as fname:
        data_merged_tz = pickle.load(fname)

    # import all fixing times for the dollar index
    with open(data_path+"fx_by_tz_sp_fixed.p", mode="rb") as fname:
        data_all_tz = pickle.load(fname)

    fx_tr_env = FXTradingEnvironment.from_scratch(
        spot_prices={
            "bid": data_merged_tz["spot_bid"],
            "ask": data_merged_tz["spot_ask"]},
        swap_points={
            "bid": data_merged_tz["tnswap_bid"],
            "ask": data_merged_tz["tnswap_ask"]}
            )

    fx_tr_env.drop(labels=settings["drop_currencies"], axis="minor_axis")
    fx_tr_env.remove_swap_outliers()
    fx_tr_env.reindex_with_freq('B')
    fx_tr_env.align_spot_and_swap()
    fx_tr_env.fillna(which="both", method="ffill")

    holding_range = np.arange(1, 2, 1)
    threshold_range = np.arange(1, 2, 1)

    combos = list(itools.product(holding_range, threshold_range))
    cols = pd.MultiIndex.from_tuples(combos, names=["holding", "threshold"])

    results = pd.DataFrame(
        index=fx_tr_env.spot_prices.major_axis,
        columns=cols)

    ix = pd.IndexSlice

    for h in holding_range:
        print("h: " + str(h))
        for th in threshold_range:
            print("threshold: " + str(th))
            # signals -------------------------------------------------------
            signals = dict()
            for c in fx_tr_env.spot_prices.minor_axis:
                # c = "nzd"
                pe = PolicyExpectation.from_pickles(data_path, c,
                    impl_rates_pickle="implied_rates_bloomberg.p")
                signals[c] = pe.forecast_policy_change(
                    lag=h+1,
                    threshold=th/100,
                    avg_impl_over=settings["avg_impl_over"],
                    avg_refrce_over=settings["avg_refrce_over"],
                    bday_reindex=True)

            signals = pd.DataFrame.from_dict(signals).loc[start_date:]

            events = signals.reindex(index=fx_tr_env.spot_prices.major_axis)
            events = events.loc[start_date:end_date,:]

            fx_tr_str = FXTradingStrategy.from_events(events,
                blackout=1, hold_period=h, leverage="none")

            fx_tr = FXTrading(environment=fx_tr_env, strategy=fx_tr_str)

            res = fx_tr.backtest(method="unrealized")

            results.loc[:, ix[h, th]] = res

    with open(data_path + "results_deleveraged_1.p", mode='wb') as hangar:
        pickle.dump(results, hangar)

    with open(data_path + "results_deleveraged.p", mode='rb') as hangar:
        data = pickle.load(hangar)


    fig_all = broomstick_plot(data.dropna(how="all").ffill())

    fig_all.tight_layout()

    out_path = data_path + "wp_figures_limbo/"
    fig_all.savefig(out_path + "broomstick_plot_rx_bas_deleveraged.pdf")
