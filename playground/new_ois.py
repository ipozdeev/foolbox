from foolbox.api import *

# Parse the following bugger
file_to_parse = data_path + "ois_data_different_maturities.xlsm"

# Define the file name for pickling
name_ois = "ois_different_maturities.p"

# Load sheets from excel into a dictionary of dataframes
raw_data = pd.read_excel(file_to_parse, sheetname=None, index_col=0)

# Throw away the request table
raw_data.pop("REQUEST_TABLE")
# Clean the data a bit
for key in raw_data.keys():
    raw_data[key] = raw_data[key].dropna(how="all")

# Pickle the bastard
with open(data_path + name_ois, mode="wb") as halupa:
    pickle.dump(raw_data, halupa)
