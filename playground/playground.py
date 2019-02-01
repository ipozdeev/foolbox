from foolbox.api import *

path_to_data = set_credentials.set_path("research_data/fx_and_events/")


def main():
    """
    """
    # events
    events_data = pd.read_pickle(path_to_data + "events.p")
    evt = events_data["joint_cbs"]["usd"].dropna().rename("mkt").to_frame()
    # evt = evt.where(evt > 0.0).dropna()

    # market data
    # ff_data = pd.read_csv(path_to_data + "ff_data.csv", index_col=0,
    #                       parse_dates=True)
    # ret = ff_data.loc[:, "Mkt-RF"].rename("mkt").to_frame()
    ff_data = pd.read_pickle(path_to_data + "spy_ret_from_hf_d_2000_2002.p")
    ret = ff_data.to_frame("mkt")

    # EventStudy
    window = (-10, -1, 1, 5)

    # mkt_norm, resid_var = EventStudyFactory().get_normal_data_ewm(
    #     ret, evt, window, alpha=0.1)

    es = EventStudy(ret, evt, window)
    ci = es.get_ci(0.95, "boot", n_iter=125)

    es.plot(plot_ci=True)
    plt.show()

    return es


if __name__ == '__main__':
    main()


