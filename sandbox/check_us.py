import pandas as pd
from foolbox.finance import PolicyExpectation
from foolbox.data_mgmt import set_credentials as set_cred

path_to_data = set_cred.set_path("research_data/fx_and_events/")


if __name__ == "__main__":
    # data
    pkl_roll = "implied_rates_from_1m_ois_roll.p"
    pkl_since = "implied_rates_from_1m_ois_since.p"
    pkl_evt = "events.p"

    ir_roll = pd.read_pickle(path_to_data + pkl_roll)
    ir_since = pd.read_pickle(path_to_data + pkl_since)
    evt = pd.read_pickle(path_to_data + pkl_evt)
    evt = evt["joint_cbs"]

    ir_diff = (ir_roll - ir_since)
    ir_diff = ir_diff.reindex(
        index=pd.date_range(ir_diff.index[0], ir_diff.index[-1], freq='B'))
    ir_diff.shift(10).where(evt.notnull()).dropna(how="all").to_clipboard()

    print(ir_roll.dropna().tail())