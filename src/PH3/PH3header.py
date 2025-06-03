# Load project's shell environment variables
import os
import sys
from dotenv import load_dotenv
load_dotenv(dotenv_path="../project.env")
sys.path.append(os.environ["PYTHONPATH"])

# Load project-wide variables
import superheader as sup

###############################################################################
################################## Filesystem #################################

import os

sup.report_dir_if_not_exists(sup.PH1_DATA_ROOT)
sup.report_dir_if_not_exists(sup.PH2_DATA_ROOT)
sup.create_dir_if_not_exists(sup.PH3_DATA_ROOT)

################################## Filesystem #################################
###############################################################################
########################### Dimensionality reduction ##########################

########################### Dimensionality reduction ##########################
###############################################################################