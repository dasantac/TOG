# Load project's shell environment variables
import os
import sys
from dotenv import load_dotenv
load_dotenv(dotenv_path="../project.env")
sys.path.append(os.environ["PYTHONPATH"])

# Load project-wide and PH2 specific variables
import superheader as sup
import PH2header as ph2

import pandas as pd
import numpy as np

###############################################################################
######################### Step 1: Hand transformations ########################

def CenterOfGravity(hand_landmarks):
  # Inputs
  x_values = hand_landmarks.filter(regex='x$').to_numpy()
  y_values = hand_landmarks.filter(regex='y$').to_numpy()
  z_values = hand_landmarks.filter(regex='z$').to_numpy()

  # MAIN ACTION
  means = ph2.CenterOfGravity([x_values, y_values, z_values])

  # Return as pandas series
  return pd.Series(means)

def NormalVector(row):
  # Inputs
  handedness = row["handedness"]
  p0 = row.filter(regex="h0").to_numpy()
  p5 = row.filter(regex="h5").to_numpy()
  p9 = row.filter(regex="h9").to_numpy()
  p13 = row.filter(regex="h13").to_numpy()
  p17 = row.filter(regex="h17").to_numpy()

  # Necessary pre-transformations
  pA = ph2.get_middlepoint_coordinates(p5, p9)
  pB = ph2.get_middlepoint_coordinates(p13, p17)

  # MAIN ACTION
  v1, v2, v3 = ph2.NormalVector(p0, pB, pA)

  # Necessary post-transformations
  v3 = ph2.v3_handedness(v3, handedness)

  # Return as pandas series
  return pd.Series(np.hstack((v1, v2, v3)))

def ChangeOfBase(row):
  # Inputs
  p0 = row.filter(regex="h0").to_numpy()
  landmarks = row.filter(regex="h(20|1[0-9]|[0-9])([xyz])").to_numpy()
  v1 = row.filter(regex="v1").to_numpy()
  v2 = row.filter(regex="v2").to_numpy()
  v3 = row.filter(regex="v3").to_numpy()

  # MAIN ACTION
  rebased = ph2.ChangeOfBase(p0, np.array([v1, v2, v3]), landmarks)

  # Return as pandas series
  return pd.Series(rebased)


######################### Step 1: Hand transformations ########################
###############################################################################