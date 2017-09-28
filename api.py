# imports a lot of stuff at once
from foolbox.EventStudy import *
import foolbox.portfolio_construction as poco
import foolbox.tables_and_figures as taf
from foolbox.finance import *
from foolbox.bankcalendars import *
from foolbox.wp_tabs_figs.wp_settings import *
import pickle

import foolbox.data_mgmt.set_credentials as set_credentials
set_credentials.set_r_environment()
data_path = set_credentials.gdrive_path("research_data/fx_and_events/")

print("imported pickle, plt, poco, taf, EventStudy, data_path." +
    "\nSet envir for R and matplotlib settings.")
