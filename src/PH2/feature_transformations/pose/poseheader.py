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

def NormalVector(row):
  # Inputs
  p0 = row.filter(regex="p0").to_numpy()
  pA = row.filter(regex="p11").to_numpy()
  pB = row.filter(regex="p12").to_numpy()

  # MAIN ACTION
  v1, v2, v3 = ph2.NormalVector(p0, pA, pB)

  # Return as pandas series
  return pd.Series(np.hstack((v1, v2, v3)))

def ChangeOfBase(row):
  # Inputs
  p0 = row.filter(regex="p0").to_numpy()
  pose_landmarks = row.filter(regex="p(0|11|12)([xyz])").to_numpy()
  mean_hand = row.filter(regex="h_mean").to_numpy()
  v1 = row.filter(regex="p_v1").to_numpy()
  v2 = row.filter(regex="p_v2").to_numpy()
  v3 = row.filter(regex="p_v3").to_numpy()

  # MAIN ACTION
  rebased = ph2.ChangeOfBase(p0, np.array([v1, v2, v3]), np.concatenate((pose_landmarks, mean_hand)))

  # Return as pandas series
  return pd.Series(rebased)

######################### Step 1: Hand transformations ########################
###############################################################################