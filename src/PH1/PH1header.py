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

if os.path.exists(sup.RAW_DATA_ROOT)==False:
  raise Exception(f"Directory {sup.RAW_DATA_ROOT} does not exist. Please investigate") 
else:
  print(f"Directory {sup.RAW_DATA_ROOT} exists. Continuing with execution")


if os.path.exists(sup.PH1_DATA_ROOT)==False:
  print(f"Directory {sup.PH1_DATA_ROOT} does not exist. Creating it and continuing with execution")
  os.makedirs(sup.PH1_DATA_ROOT)
else:
  print(f"Directory {sup.PH1_DATA_ROOT} exists. Continuing with execution")

################################## Filesystem #################################
###############################################################################
####################### nltk corpus to pandas dataframe #######################

from file_reading.fs2df import get_tags as get_tags

####################### nltk corpus to pandas dataframe #######################
###############################################################################
############################## Feature extraction #############################
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # Step 1: Video to landmarks  # # # # # # # # # # # # #

import feature_extraction.ph1step1 as step1

# # # # # # # # # # # # # Step 1: Video to landmarks  # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
############################## Feature extraction #############################
###############################################################################