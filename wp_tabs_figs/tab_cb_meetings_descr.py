from foolbox.api import *
from wp_tabs_figs.wp_settings import settings

# Get the data
with open(data_path + settings["events_data"], mode="rb") as fname:
    events = pickle.load(fname)

# Set the sample
start_date = settings["sample_start"]
end_date = settings["sample_end"]

# List of CBs to describe and output structure
banks = ["rba", "boc", "ecb", "rbnz", "norges", "riks", "snb", "boe", "fomc"]
tab_meetings_descr =\
    pd.DataFrame(index=banks, columns=["total", "hikes", "cuts"])

# Loop over  banks, count hikes, cuts and meetings over the sample
for bank in banks:
    tmp_policy = events[bank].change[start_date:end_date]
    tab_meetings_descr.loc[bank, "total"] = tmp_policy.count()
    tab_meetings_descr.loc[bank, "hikes"] =\
        tmp_policy.where(tmp_policy > 0).count()
    tab_meetings_descr.loc[bank, "cuts"] =\
        tmp_policy.where(tmp_policy < 0).count()
