"""
Step 3: Change of Base
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

def move_to_origin(origin, landmarks):
  new = []
  num_points = len(landmarks) // 3

  for i in range(num_points):
    new.append(landmarks[3*i-0] - origin[0]) # x
    new.append(landmarks[3*i-1] - origin[1]) # y
    new.append(landmarks[3*i-2] - origin[2]) # z
  
  return new

def change_base(I, landmarks):
  new = []
  num_points = len(landmarks) // 3

  for i in range(num_points):
    point = np.array([landmarks[3*i-0], landmarks[3*i-1], landmarks[3*i-2]])
    new_point = np.dot(I, point)
    new.extend(new_point)

  return new

######################### Intermediary transformations ########################
###############################################################################
############################ Driver transformation ############################

def ChangeOfBase(origin, base, landmarks):
  # First we move all the landmarks to their new origin
  translated = move_to_origin(origin, landmarks)

  # Then we get the base in matrix form
  B = np.array(base)
  # And it's inverse
  I = np.linalg.inv(B)

  # Then we change of base
  rebased = change_base(I, translated)

  return rebased

############################ Driver transformation ############################
###############################################################################