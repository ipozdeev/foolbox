from foolbox.api import *

file_to_parse = data_path + "msci_ted_vix_2000_2017.xlsm"
out_name = "msci_ted_vix_d.p"

# Parse the data
raw_data = pd.read_excel(file_to_parse, sheetname=None)

# Save the sheets and variables needed
out_data = dict()
out_data["msci_pi"] = raw_data["msci"].astype(float)
out_data["ted_vix"] = raw_data["ted_vix"][["ted", "vix"]].astype(float)

# Dump
with open(data_path + out_name, mode="wb") as halupa:
    pickle.dump(out_data, halupa)
