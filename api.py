# imports a lot of stuff at once
from foolbox.EventStudy import *
import foolbox.portfolio_construction as poco
import foolbox.tables_and_figures as taf
from foolbox.finance import *
import foolbox.data_mgmt.set_credentials as set_credentials
set_credentials.set_r_environment()
data_path = set_credentials.gdrive_path("research_data/fx_and_events/")
import pickle

print("imported pickle, poco, taf, EventStudy, data_path. Set envir for R.")
