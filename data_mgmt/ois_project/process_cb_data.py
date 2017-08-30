import pandas as pd
import numpy as np
import pickle

# Set the user contingent path
from foolbox.data_mgmt import set_credentials as set_cred
path = set_cred.gdrive_path("research_data/fx_and_events/")

# File names for pickling, see sample definitions below
name_all_cbs = path + "ois_project_events.p"  # events

# The current parsing is implementing for reading a single aggregated file
# Currency - central bank dictionary for renaming of joint events
fx_cb_dict = {"fomc": "usd", "boe": "gbp", "boc": "cad", "rba": "aud",
              "rbnz": "nzd", "ecb": "eur", "riks": "sek", "norges": "nok",
              "snb": "chf"}

# Set the sample
#cb_sample_start = "2000-11-01"  # start of the CB sample. Oh Caanadaaa...
cb_sample_start = "1990-01-01"
cb_sample_end = "2017-03-31"


# Parse the summary data on CBs meetings
file_to_parse = path + "central_bank_rates_raw_data/" \
                       "summary_all_central_banks.xlsx"
cb_data = pd.read_excel(file_to_parse, sheetname=None, index_col=0)

# Outputs are separate events, joint events in changes, levels, and unscheduled
# events in rate changes
events = dict()
joint_events = list()
joint_events_lvl = list()
joint_events_plus_unsched = list()
# Loop over the data, dropping comment columns, and unscheduled meetings
for cb, data in cb_data.items():
    if cb == "resources":
        continue
    data = data.drop(["comments"], axis=1)      # drop comments
    data = data[cb_sample_start:cb_sample_end]  # set the sample

    # Copy data with unscheduled events
    data_plus_unsched = data.drop(["scheduled"], axis=1)

    # Choose only scheduled meetings
    data = data[data.scheduled]
    data = data.drop(["scheduled"], axis=1)     # drop the redundant variable

    # Save changes separately for joint events
    change_unsched = data_plus_unsched.change.astype(float).round(4)
    change_unsched.name = cb
    change = data.change.astype(float).round(4)
    change.name = cb
    rate = data.rate.astype(float).round(4)
    rate.name = cb

    # Append the outputs
    events[cb] = data
    joint_events.append(change)
    joint_events_lvl.append(rate)
    joint_events_plus_unsched.append(change_unsched)

# Plug joint events into the final output, using fx names
joint_events = pd.concat(joint_events, join="outer", axis=1)
joint_events = joint_events.rename(columns=fx_cb_dict)
joint_events = joint_events[sorted(joint_events.columns)]
events["joint_cbs"] = joint_events

joint_events_lvl = pd.concat(joint_events_lvl, join="outer", axis=1)
joint_events_lvl = joint_events_lvl.rename(columns=fx_cb_dict)
joint_events_lvl = joint_events_lvl[sorted(joint_events_lvl.columns)]
events["joint_cbs_lvl"] = joint_events_lvl

joint_events_plus_unsched = pd.concat(joint_events_plus_unsched, join="outer",
                                      axis=1)
joint_events_plus_unsched = \
    joint_events_plus_unsched.rename(columns=fx_cb_dict)
joint_events_plus_unsched = \
    joint_events_plus_unsched[sorted(joint_events_plus_unsched.columns)]
events["joint_cbs_plus_unscheduled"] = joint_events_plus_unsched

# Store the bastards
with open(name_all_cbs, "wb") as hunger:
    pickle.dump(events, hunger)
