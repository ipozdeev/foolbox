""" Set user-contingent R environmental variables and paths to Google Drive
folders
"""

import getpass
import os

# Default dictionary specifying R and Google Drive paths for each user
user_info =\
    {"borisenko": {"R_USER": "borisenko",
                   "R_HOME": "c:/R/R-3.2.5",
                   "gdrive": "C:/Users/borisenko/GoogleDrive/"
                   },
     "Igor": {"R_USER": "Igor",
              "R_HOME": "c:/Program Files/R/R-3.2.5",
              "gdrive": "C:/Users/Igor/Google Drive/Personal/"
              },
     "HSG-Spezial": {"R_USER": "HSG-Spezial",
                     "R_HOME": "c:/Program Files/R/R-3.2.5",
                     "gdrive": "C:/Users/HSG-Spezial/Google Drive/Personal/"
                     }
     }


def set_r_environment(user_info=user_info):
    """Sets R environmental variables contingently on the current user

    Parameters
    ----------
    user_info: dict
        with key corresponding to users, and a nested dictionary for each key
        contains paths to R. See the default dictionary above

    Returns
    -------
    Nophin'

    """
    # Get the current user
    current_user = getpass.getuser()  # get the current user
    current_user_info = user_info[current_user]
    # Set upt the R environmental variables
    os.environ["R_USER"] = current_user_info["R_USER"]
    os.environ["R_HOME"] = current_user_info["R_HOME"]


def gdrive_path(shared_path, user_info=user_info):
    """Sets path to a shared Google Drive folder, contingent on the current
    user

    Parameters
    ----------
    shared_path: str
        representing a path to or within a shared Google Drive folder
    user_info: dict
        with key corresponding to users, and a nested dictionary for each key
        contains paths to R. See the default dictionary above

    Returns
    -------
    path: str
        user-contingent path to a shared Google Drive folder

    """
    # Get the current user
    current_user = getpass.getuser()

    # Get the relevant information
    current_user_info = user_info[current_user]

    # Set the path
    path = current_user_info["gdrive"] + shared_path

    return path


set_r_environment()
