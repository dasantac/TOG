# Load project's shell environment variables
import os
import sys
from dotenv import load_dotenv
load_dotenv(dotenv_path="../project.env")
sys.path.append(os.environ["PYTHONPATH"])

###############################################################################
################################### Classes ###################################

#NUM_CLASSES = "all-classes"
NUM_CLASSES = "two-classes"
if NUM_CLASSES == "all-classes":
  big_classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'll', 'm', 'n', 'ñ', 'o', 'p', 'q', 'r', 'rr', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
  small_classes = ['0']
  CLASSES_LIST = big_classes + small_classes
elif NUM_CLASSES == "two-classes":
  CLASSES_LIST = ['a', 'b']

CLASSES_TO_NUMBERS = {CLASSES_LIST[i] : i for i in range(len(CLASSES_LIST))}
NUMBERS_TO_CLASSES = {i : CLASSES_LIST[i] for i in range(len(CLASSES_LIST))}

################################### Classes ###################################
###############################################################################
################################## Filesystem #################################

# Top directory for the project
ROOT = os.environ["TOG_ROOT"]

if os.path.exists(ROOT)==False:
  raise Exception(f"Directory {ROOT} does not exist. Please investigate") 
else:
  print(f"Directory {ROOT} exists. Continuing with execution")

# Directory where all the data used by and created by the project lives
DATA_ROOT = os.path.join(ROOT, 'data')
if os.path.exists(DATA_ROOT)==False:
  raise Exception(f"Directory {DATA_ROOT} does not exist. Please investigate") 
else:
  print(f"Directory {DATA_ROOT} exists. Continuing with execution")

# Directory where all the code used for the project lives
CODE_ROOT = os.path.join(ROOT, 'src')
if os.path.exists(CODE_ROOT)==False:
  raise Exception(f"Directory {CODE_ROOT} does not exist. Please investigate") 
else:
  print(f"Directory {CODE_ROOT} exists. Continuing with execution")

# Directory where all the binaries used by and created by the project live
BIN_ROOT = os.path.join(ROOT, 'bin')
if os.path.exists(BIN_ROOT)==False:
  raise Exception(f"Directory {BIN_ROOT} does not exist. Please investigate") 
else:
  print(f"Directory {BIN_ROOT} exists. Continuing with execution")

################################## Filesystem #################################
###############################################################################