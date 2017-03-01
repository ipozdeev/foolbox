"""
Get the current user and set the working directory and paths and user names
for R accordingly
"""

import getpass
import os

def set_path(path_after_gdrive):
    """
    Sets path to FX and CPI data, and R_USER and R_HOME system variables, for
    a given user
    """

    current_user = getpass.getuser()  # get the current user

    # Set path to library and R user and path parameters
    if current_user == "borisenko":
        path = "C:/Users/borisenko/GoogleDrive/"+path_after_gdrive
        os.environ['R_USER'] = "borisenko"
        os.environ['R_HOME'] = "c:/R/R-3.2.5"
    elif current_user == "HSG-Spezial":
        path = "C:/Users/HSG-Spezial/Google Drive/Personal/"+path_after_gdrive
        os.environ['R_USER'] = "HSG-Spezial"
        os.environ['R_HOME'] = "c:/Program Files/R/R-3.2.5"
    elif current_user == "Igor":
        path = "C:/Users/Igor/Google Drive/Personal/"+path_after_gdrive
        os.environ['R_USER'] = "Igor"
        os.environ['R_HOME'] = "c:/Program Files/R/R-3.2.5"

    return path

path = set_path()
