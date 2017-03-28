from foolbox.api import *

# Parse the following bugger
file_to_parse = data_path + "ois_data.xlsm"

# Define the file name for pickling
name_ois = "ois.p"

# Load sheets from excel into a dictionary of dataframes
raw_data = pd.read_excel(file_to_parse, sheetname=None, index_col=0)

# Populate the output
ois_data = dict()
ois_data["tr_1m"] = raw_data["ois_1m_tr"]
ois_data["tr_3m"] = raw_data["ois_3m_tr"]
ois_data["icap_1m"] = raw_data["ois_1m_icap"]
ois_data["icap_3m"] = raw_data["ois_3m_icap"]

for key in ois_data.keys():
    ois_data[key].columns = [col.lower() for col in ois_data[key].columns]

# Pickle the bastard
with open(data_path + name_ois, mode="wb") as fname:
    pickle.dump(ois_data, fname)
