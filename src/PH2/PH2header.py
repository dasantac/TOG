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

sup.report_dir_if_not_exists(sup.PH1_DATA_ROOT)
sup.create_dir_if_not_exists(sup.PH2_DATA_ROOT)

################################## Filesystem #################################
###############################################################################
############################ Feature transformation ###########################

from feature_transformations.CenterOfGravity import CenterOfGravity
from feature_transformations.NormalVector import NormalVector, get_middlepoint_coordinates, v3_handedness
from feature_transformations.ChangeOfBase import ChangeOfBase

import feature_transformations.hand.handheader as hand
import feature_transformations.pose.poseheader as pose

############################ Feature transformation ###########################
###############################################################################