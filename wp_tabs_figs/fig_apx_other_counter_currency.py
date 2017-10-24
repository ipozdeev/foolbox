from foolbox.api import *

if __name__ == "__main__":

    # currency to drop
    drop_curs = ["usd","jpy","dkk"]

    # window
    wa,wb,wc,wd = -10,-1,1,5
    window = (wa,wb,wc,wd)

    s_dt = settings["sample_start"]
    e_dt = settings["sample_end"]

    # data ------------------------------------------------------------------
    data_path = set_credentials.gdrive_path("research_data/fx_and_events/")
    out_path = set_credentials.gdrive_path("opec_meetings/tex/figs/")

    # spot returns + drop currencies ----------------------------------------
    with open(data_path + settings["fx_data"], mode='rb') as fname:
        fx = pickle.load(fname)
    ret = np.log(fx["spot_mid"].drop(drop_curs,axis=1,errors="ignore")).diff()

    # events + drop currencies ----------------------------------------------
    with open(data_path + settings["events_data"], mode='rb') as fname:
        events_data = pickle.load(fname)

    events = events_data["joint_cbs"].drop(drop_curs, axis=1, errors="ignore")
    events = events.loc[s_dt:e_dt]

    # data = ret["nzd"]
    data = ret.copy().loc[s_dt:e_dt]

    # events = events_perf["nzd"].dropna()
    events = events.where(events < 0)

    # normal_data = data.rolling(22).mean().shift(1)
    es = EventStudy(data, events, window, mean_type="count_weighted",
        normal_data=0.0, x_overlaps=True)

    ci_boot_c = es.get_ci(ps=(0.025, 0.975), method="boot", n_blocks=10,
        M=5000)

    es.booted
    qs = (0.01, 0.025, 0.05, 0.95, 0.975, 0.99)
    es.booted.quantile(qs, axis=1).T * 100
