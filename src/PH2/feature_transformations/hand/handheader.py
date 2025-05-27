# Load project's shell environment variables
import os
import sys
from dotenv import load_dotenv
load_dotenv(dotenv_path="../project.env")
sys.path.append(os.environ["PYTHONPATH"])

# Load project-wide and PH2 specific variables
import superheader as sup
import PH2header as ph2
###############################################################################
######################### Step 1: Hand transformations ########################
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # Center of Gravity # # # # # # # # # # # # # # # 

from .CenterOfGravity import CenterOfGravity
from .NormalVector import NormalVector

# # # # # # # # # # # # # # # # Center of Gravity # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
######################### Step 1: Hand transformations ########################
###############################################################################