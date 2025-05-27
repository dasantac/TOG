"""
Step 2: Normal Vector

This file contains the helper functions to get the vector normal to the "plane
formed by the palm of the hand.

We assimilate the "plame formed by the palm of the hand" to the plane defined
by vectors 0A and 0B. See the projects attached media.
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

import pandas as pd


###############################################################################
######################### Intermediary transformations ########################

### Get coordinates for A and B ###
#/ Extract point A (which lies between points 5 and 9) coordinates
def get_A_coordinates():
  pass

#/ Extract point B (which lies between 13 and 17) coordinates
def get_B_coordinates():
  pass

### Get coordinates for v1 and v2 ###
#/ Extract vector v1 (which goes from point 0 to point B) coordinates
def get_v1_coordinates():
  pass

#/ Extract vector v2 (which goes from point 0 to point A) coordinates
def get_v2_coordinates():
  pass

### Normalize v1 and v2 ###
# Get the norms first
def get_v1_norm():
  pass

def get_v2_norm():
  pass

# Now normalize
def normalize_v1():
  pass

def normalize_v2():
  pass

### Get coordinates for v3 ###
#/ Normal product of v1 and v2 gives v3
def get_v3_coordinates():
  pass

#/ To normalize for handedness, we always want v3 to point "out of the palm"
#/ Because of this, we need to invert v3 for one of the hands
def v3_handedness():
  pass

######################### Intermediary transformations ########################
###############################################################################
############################ Driver transformation ############################

def NormalVector(row):
  # Inputs
  handedness = row["handedness"]
  p0 = row.filter(regex="h0")
  p5 = row.filter(regex="h5")
  p9 = row.filter(regex="h9")
  p13 = row.filter(regex="h13")
  p17 = row.filter(regex="h17")

  # Step by step transformations
  pA = get_A_coordinates(p5, p9)
  pB = get_B_coordinates(p13, p17)

  v1 = get_v1_coordinates(p0, pB)
  v2 = get_v2_coordinates(p0, pA)

  n1 = get_v1_norm(v1)
  n2 = get_v2_norm(v2)

  nv1 = normalize_v1(v1, n1)
  nv2 = normalize_v2(v2, n2)

  v3 = get_v3_coordinates(nv1, nv2)
  v3 = v3_handedness(v3, handedness)

  return v3

############################ Driver transformation ############################
###############################################################################