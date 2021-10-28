#### ScriptConfig.py
#### Author: Maxie Dion Schmidt (@github/maxieds)
#### Created: 2021.10.07

import os

class ScriptConfig:

    VERSION = "1.0-testing"

    QUIET = False
    VERBOSE = False
    VVERBOSE = False
    DEBUGGING = False
    COLOR_PRINTING = True

    OSDIR_SEPARATOR = os.sep # '/'
    IMAGE_FEXT = '.png'
    CSV_FEXT = '.csv'

    EXIT_SUCCESS = 0
    ERROR_GENERIC = -1
