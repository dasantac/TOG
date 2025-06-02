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

if os.path.exists(sup.PH3_DATA_ROOT)==False:
  raise Exception(f"Directory {sup.PH3_DATA_ROOT} does not exist. Please investigate") 
else:
  print(f"Directory {sup.PH3_DATA_ROOT} exists. Continuing with execution")

################################## Filesystem #################################
###############################################################################
################################ Architectures ################################

import architecture.archeader as arch

################################ Architectures ################################
###############################################################################