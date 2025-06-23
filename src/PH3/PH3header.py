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

sup.report_dir_if_not_exists(sup.PH1_DATA_ROOT)
sup.report_dir_if_not_exists(sup.PH2_DATA_ROOT)
sup.create_dir_if_not_exists(sup.PH3_DATA_ROOT)

################################## Filesystem #################################
###############################################################################

import numpy as np

def live(scaler, reducer, hand_landmarks_list, handedness_list, pose_landmarks_list):
  reduced_landmarks_list = []
  for pose_index in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[pose_index]
    for hand_index in range(len(hand_landmarks_list)):
      hand_landmarks = hand_landmarks_list[hand_index]
      handedness = handedness_list[2*hand_index]
      confidence = handedness_list[2*hand_index+1]

      data = np.array(hand_landmarks+pose_landmarks).reshape(1,-1)
      std_landmarks = scaler.transform(data)

      reduced_landmarks = reducer.transform(std_landmarks)

      reduced_landmarks_list.append([
        [pose_index, hand_index, handedness, confidence] + 
        reduced_landmarks.tolist()])
      
  return reduced_landmarks_list

def live_full(scaler, reducer, transformed_landmarks_list):
  reduced_landmarks_list = []
  for pose_index in range(len(transformed_landmarks_list)):
    for hand_index in range(len(transformed_landmarks_list[pose_index])):
      transformed_landmarks = transformed_landmarks_list[pose_index][hand_index][4:]
      handedness = transformed_landmarks_list[pose_index][hand_index][2]
      confidence = transformed_landmarks_list[pose_index][hand_index][3]

      data = np.array(transformed_landmarks).reshape(1,-1)
      std_landmarks = scaler.transform(data)

      reduced_landmarks = reducer.transform(std_landmarks)

      reduced_landmarks_list.append([
        [pose_index, hand_index, handedness, confidence] + 
        reduced_landmarks.tolist()])
      
  return reduced_landmarks_list

