"""
Step 1 : Video to Landmarks

This file contains helper functions to extract landmarks for each frame in the
videos using Goggle's Mediapipe framework.
"""

# Load project's shell environment variables
import os
import sys
from dotenv import load_dotenv
load_dotenv(dotenv_path="../project.env")
sys.path.append(os.environ["PYTHONPATH"])

# Load project-wide variables
import superheader as sup
import PH1header as ph1

### Setup ###
# Video to frame
import cv2
NUM_FRAMES_EXTRACTED_PER_VIDEO_HALF = 6

# Frame to landmarks
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

mediapipe_hand_landmarker_path=sup.BIN_ROOT+'/load/PH1/mediapipe/hand_landmarker.task'
hands_base_options = python.BaseOptions(model_asset_path=mediapipe_hand_landmarker_path)
hands_options = vision.HandLandmarkerOptions(base_options=hands_base_options, num_hands=2)
hands_detector = vision.HandLandmarker.create_from_options(hands_options)

mediapipe_pose_landmarker_path=sup.BIN_ROOT+'/load/PH1/mediapipe/pose_landmarker_lite.task'
pose_base_options = python.BaseOptions(model_asset_path=mediapipe_pose_landmarker_path)
pose_options = vision.PoseLandmarkerOptions(base_options=pose_base_options, output_segmentation_masks=False)
pose_detector = vision.PoseLandmarker.create_from_options(pose_options)

# to extracted landmakrs as columns in our dataframe
import pandas as pd
original_df_columns = sup.tag_columns + [sup.fileid_col, sup.frame_count_col]

new_column_names = original_df_columns \
  + ["first_frame", "current_frame"] \
  + ["num_candidate_hands", "current_candidate_hand", "detected_handedness"] \
  + ["confidence"] \
  + sup.pf_hand_landmark_columns \
  + ["num_candidate_poses", "current_candidate_pose"] \
  + sup.pf_pose_landmark_columns

### Functions used by data preparation and/or live inference ###
def count_frames(fileid):
  video_path = os.path.join(sup.RAW_DATA_ROOT, fileid)

  # Read the video use OpenCV
  cap = cv2.VideoCapture(video_path)

  if not cap.isOpened():
    print(f"Error: Could not open {video_path}.")
    return
  
  # We only want the 15 frames in the middle of each video
  ## Compute indices of the 15 middle frames
  length = int(cap. get(cv2.CAP_PROP_FRAME_COUNT))
  return length

def extract_hands(image_mp):

  # Hand landmarking
  detection_result = hands_detector.detect(image_mp)

  # work on each hand
  handedness_id_list = list()
  landmark_columns_list = list()
  num_candidates = len(detection_result.handedness)

  for i in range(num_candidates):
    handedness = detection_result.handedness[i][0].display_name
    handedness_id = 0 if handedness == "Right" else 1

    confidence = detection_result.handedness[i][0].score

    hand_landmarks = detection_result.hand_landmarks[i]
    # write to list
    landmark_columns = [0 for _ in range(21*3)]
    for j in range(21):
      landmark_columns[3*j] = hand_landmarks[j].x
      landmark_columns[3*j+1] = hand_landmarks[j].y
      landmark_columns[3*j+2] = hand_landmarks[j].z
    
    handedness_id_list.append(handedness_id)
    handedness_id_list.append(confidence)
    landmark_columns_list.append(landmark_columns)

  return landmark_columns_list, handedness_id_list

def extract_pose(image_mp):
  
  # Pose landmarking
  detection_result = pose_detector.detect(image_mp)

  landmark_columns_list = list()
  num_candidates = len(detection_result.pose_landmarks)

  for i in range(num_candidates):
    pose_landmarks = detection_result.pose_landmarks[i]
    
    # landmarks of interest
    interest = [0, 11, 12]
    # write to list
    landmark_columns = [0 for _ in range(3*3)]

    for k in range(3):
      landmark_columns[3*k] = pose_landmarks[interest[k]].x
      landmark_columns[3*k+1] = pose_landmarks[interest[k]].y
      landmark_columns[3*k+2] = pose_landmarks[interest[k]].z
      
    landmark_columns_list.append(landmark_columns)

  return landmark_columns_list

def extract_landmarks_per_frame(row):
  # table where we will store the data for this video
  results = []

  # Read the video use OpenCV
  video_path = os.path.join(sup.RAW_DATA_ROOT, row[sup.fileid_col])
  cap = cv2.VideoCapture(video_path)

  if not cap.isOpened():
    print(f"Error: Could not open {video_path}.")
    return
  
  # We only want the frames in the middle of each video
  ## Compute indices of the  middle frames
  length = int(cap. get(cv2.CAP_PROP_FRAME_COUNT))
  if length < 2*sup.NUM_FRAMES_EXTRACTED_PER_VIDEO_HALF:
    print(f"Video {video_path} is too short!")
  first_frame = (length // 2) - sup.NUM_FRAMES_EXTRACTED_PER_VIDEO_HALF
  middle_frames = set(range(first_frame, first_frame+2*sup.NUM_FRAMES_EXTRACTED_PER_VIDEO_HALF))
  ## initialize the "current frame" counter to 0
  count = 0
  
  # Get the first frame of the video
  ret, frame = cap.read()

  # We go through the video frame by frame
  while ret:
    if count in middle_frames:
      # Mediapipe setup
      image_mp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      image_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_mp)

      # Mediapipe extraction
      pose_landmark_columns_list = extract_pose(image_mp)
      hand_landmark_columns_list, handedness_id_list = extract_hands(image_mp)
      
      num_candidate_poses = len(pose_landmark_columns_list)
      num_candidate_hands = len(handedness_id_list)

      for candidate_pose in range(len(pose_landmark_columns_list)):
        pose_landmark_columns = pose_landmark_columns_list[candidate_pose]

        for candidate_hand in range(0,len(handedness_id_list), 2):
          handedness_id = handedness_id_list[candidate_hand]
          confidence = handedness_id_list[candidate_hand+1]
          hand_landmark_columns = hand_landmark_columns_list[candidate_hand//2]

          new_row = row.tolist()  \
                  + [first_frame, count-first_frame] \
                  + [num_candidate_hands, candidate_hand, handedness_id] \
                  + [confidence] \
                  + hand_landmark_columns \
                  + [num_candidate_poses, candidate_pose] \
                  + pose_landmark_columns
          
          results.append(new_row)

    # Move on to the next frame
    ret,frame = cap.read()
    count += 1
   
  video_df = pd.DataFrame(results, columns=new_column_names)
  return video_df