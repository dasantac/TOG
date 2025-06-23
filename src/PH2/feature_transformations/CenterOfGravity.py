"""
Step 1: Center of Gravity
"""

# Load project's shell environment variables
import os
import sys
from dotenv import load_dotenv
load_dotenv(dotenv_path="../project.env")
sys.path.append(os.environ["PYTHONPATH"])

# Load project-wide variables
import superheader as sup
import PH2.PH2header as ph2

# Transformation
def CenterOfGravity(groupings):
  means = []
  for g in groupings:
    means.append(g.mean())

  return means