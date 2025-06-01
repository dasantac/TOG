"""
Step 2: Normal Vector
"""

# Load project's shell environment variables
import os
import sys
from dotenv import load_dotenv
load_dotenv(dotenv_path="../project.env")
sys.path.append(os.environ["PYTHONPATH"])

# Load project-wide variables
import superheader as sup
import PH2header as ph2

import numpy as np

###############################################################################
######################### Intermediary transformations ########################

### Get coordinates for A and B ###
def get_middlepoint_coordinates(first, second):
  middle_point = (first + second)/2
  return middle_point

### Get coordinates for v1 and v2 ###
def get_v_coordinates(src, dst):
  v = dst - src
  return v

### Normalize v1 and v2 ###
# Get the norms first
def get_v_norm(v):
  n = np.sqrt(np.dot(v, v.T))
  return n

# Now normalize
def normalize_v(v):
  n = get_v_norm(v)
  nv = v / n
  return nv

### Get coordinates for v3 ###
#/ Normal product of v1 and v2 gives v3
def cross_product(v1, v2):
  v3 = np.cross(v1, v2)
  return v3

#/ To normalize for handedness, we always want v3 to point "out of the palm"
#/ Because of this, we need to invert v3 for one of the hands
def v3_handedness(v3, h):
  # Invert if left hand
  if h == 1:
    v3 = -1 * v3
  return v3

######################### Intermediary transformations ########################
###############################################################################
############################ Driver transformation ############################

def NormalVector(p0, pA, pB):

  v1 = get_v_coordinates(p0, pA)
  v2 = get_v_coordinates(p0, pB)

  nv1 = normalize_v(v1)
  nv2 = normalize_v(v2)

  v3 = cross_product(nv1, nv2)

  return nv1, nv2, v3

############################ Driver transformation ############################
###############################################################################