"""
Step 1: Center of Gravity

This files contains the helper functions to obtain the center of gravity of the
hand landmarks extracted during Phase 1.
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

# Transformation
def CenterOfGravity(hand_landmarks):
  x_values = hand_landmarks.filter(regex='x$')
  y_values = hand_landmarks.filter(regex='y$')
  z_values = hand_landmarks.filter(regex='z$')

  x_mean = x_values.mean()
  y_mean = y_values.mean()
  z_mean = z_values.mean()

  return pd.Series({"h_mean_x":x_mean, "h_mean_y":y_mean, "h_mean_z":z_mean})
