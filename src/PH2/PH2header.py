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

sup.report_dir_if_not_exists(sup.PH1_DATA_ROOT)
sup.create_dir_if_not_exists(sup.PH2_DATA_ROOT)

################################## Filesystem #################################
###############################################################################
############################ Feature transformation ###########################

from PH2.feature_transformations.CenterOfGravity import CenterOfGravity
from PH2.feature_transformations.NormalVector import NormalVector, get_middlepoint_coordinates, v3_handedness
from PH2.feature_transformations.ChangeOfBase import ChangeOfBase

import PH2.feature_transformations.hand.handheader as hand
import PH2.feature_transformations.pose.poseheader as pose

############################ Feature transformation ###########################
###############################################################################

import numpy as np

def live(hand_landmarks_list, handedness_list, pose_landmarks_list):
  transformed_landmarks_list = []
  for pose_index in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[pose_index]
    for hand_index in range(len(hand_landmarks_list)):
      hand_landmarks = hand_landmarks_list[hand_index]
      handedness = handedness_list[2*hand_index]
      confidence = handedness_list[2*hand_index+1]
      
      hand_x = np.array(hand_landmarks[0::3])
      hand_y = np.array(hand_landmarks[1::3])
      hand_z = np.array(hand_landmarks[2::3])

      hp0 = np.array(hand_landmarks[0*3:(0+1)*3:1])
      hp5 = np.array(hand_landmarks[5*3:(5+1)*3:1])
      hp9 = np.array(hand_landmarks[9*3:(9+1)*3:1])
      hp13 = np.array(hand_landmarks[13*3:(13+1)*3:1])
      hp17 = np.array(hand_landmarks[17*3:(17+1)*3:1])

      pp0 = np.array(pose_landmarks[0*3:(0+1)*3:1])
      pp11 = np.array(pose_landmarks[1*3:(1+1)*3:1])
      pp12 = np.array(pose_landmarks[2*3:(2+1)*3:1])

      # Hand
      ## COG
      h_mean = np.array([hand_x.mean(), hand_y.mean(), hand_z.mean()])
      ## Normal Vector
      hpA = get_middlepoint_coordinates(hp5, hp9)
      hpB = get_middlepoint_coordinates(hp13, hp17)
      hv1, hv2, hv3 = NormalVector(hp0, hpA, hpB)
      hv3 = v3_handedness(hv3, handedness)
      ## Change of Base
      w_hand_landmarks = ChangeOfBase(hp0, np.array([hv1, hv2, hv3]), np.array(hand_landmarks))

      # Pose
      ## Normal Vector
      pv1, pv2, pv3 = NormalVector(pp0, pp11, pp12)
      ## Change of Base
      p_h_mean = ChangeOfBase(pp0, np.array([pv1, pv2, pv3]), h_mean)

      transformed_landmarks_list.append([
        [pose_index, hand_index, handedness, confidence] +
        hv1.tolist() +  hv2.tolist() + hv3.tolist() + w_hand_landmarks + p_h_mean])
  
  return transformed_landmarks_list