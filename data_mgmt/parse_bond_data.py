import pandas as pd
import pickle

# Set the user contingent path
from foolbox.data_mgmt import set_credentials as set_cred
path = set_cred.gdrive_path("research_data/fx_and_events/")

# Set the i/o names
file_name = "bond_data.xlsm"
out_name = "bond_data.p"

# Parse the data, throw out the request table sheet
raw_data = pd.read_excel(path + file_name, sheetname=None, index_col=0)
raw_data.pop("REQUEST_TABLE")

# Save the data
with open(path + out_name, mode="wb") as halupa:
    pickle.dump(raw_data, halupa)
