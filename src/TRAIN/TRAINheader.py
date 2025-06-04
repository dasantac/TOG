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

sup.report_dir_if_not_exists(sup.PH3_DATA_ROOT)
sup.create_dir_if_not_exists(sup.TRAIN_BINGEN_ROOT)

################################## Filesystem #################################
###############################################################################
############################### Hyperparameters ###############################



############################### Hyperparameters ###############################
###############################################################################
################################ Architectures ################################

import architecture.archeader as arch

################################ Architectures ################################
###############################################################################