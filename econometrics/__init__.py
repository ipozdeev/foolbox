import getpass
import os

user_info = {
    "borisenko": {
        "R_USER": "borisenko",
        "R_HOME": "c:/R/R-3.2.5",
        },
    "Igor": {
         "R_USER": "Igor",
         "R_HOME": "c:/Program Files/R/R-3.2.5",
    },
    "HSG-Spezial": {
        "R_USER": "HSG-Spezial",
        "R_HOME": "c:/R/R-3.2.5",
    },
    "pozdeev": {
        "R_USER": "pozdeev",
        "R_HOME": "c:/Program Files/R/R-3.2.5",
    }
}


def set_r_environment():
    """Sets R environmental variables contingently on the current user

    Uses dict with keys corresponding to users, and a nested dictionary for
    each key contains paths to R. See the default dictionary above.

    Returns
    -------
    None

    """
    # Get the current user
    current_user = getpass.getuser()  # get the current user
    current_user_info = user_info[current_user]
    # Set upt the R environmental variables
    os.environ["R_USER"] = current_user_info["R_USER"]
    os.environ["R_HOME"] = current_user_info["R_HOME"]


# ---------------------------------------------------------------------------
set_r_environment()
