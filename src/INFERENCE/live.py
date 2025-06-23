# Load project's shell environment variables
import os
import sys
from dotenv import load_dotenv
load_dotenv(dotenv_path="../project.env")
sys.path.append(os.environ["PYTHONPATH"])

# Load project-wide variables
import superheader as sup
import PH1.feature_extraction.ph1step1 as ph1
import PH2.PH2header as ph2
import PH3.PH3header as ph3
from TRAIN.architecture.BERT import bert

import pickle
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim

def draw_image(frame, original_landmarks, tag):

  height, width, depth = frame.shape
  margin_y, margin_x = int(height/50), int(width/50)

  x_coords = [original_landmarks[3*j] for j in range(21)]
  y_coords = [original_landmarks[3*j+1] for j in range(21)]

  left = int(min(x_coords) * width) - margin_x
  right = int(max(x_coords) * width) + margin_x
  bottom = int(min(y_coords) * height) - margin_y
  top = int(max(y_coords) * height) + margin_y

  blue = (255, 0, 0)
  green = (0,255,0)
  thickness = 2

  frame_w_rect = cv2.rectangle(frame, (left, top), (right, bottom), blue, thickness)
  cv2.putText(frame_w_rect, f'Predicted Class: {tag}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, green, 2, cv2.LINE_AA)
  cv2.imshow('Webcam', frame_w_rect)

def pick_hand_ph1(hand_landmark_columns_list, 
                  handedness_id_list, 
                  pose_landmark_columns_list):
  
  # if only one option, with a handedness prediction and a confidence score
  if len(handedness_id_list) == 2:
    return 0

  # default
  return 0

def pick_hand_ph2(transformed_landmark_list):
  
  # if only one option, with a handedness prediction and a confidence score
  if len(transformed_landmark_list) == 1 and len(transformed_landmark_list[0]) == 1:
    return 0

  # default
  return 0

def pick_hand_ph3(reduced_features_list):
  
  # if only one option, with a handedness prediction and a confidence score
  if len(reduced_features_list) == 1 and len(reduced_features_list[0]) == 1:
    return 0

  # default
  return 0

def predict_sign(model, active_selected_features):
  return model.me(active_selected_features)


def classify_webcam_image_ph1(model):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Extract landmarks
        hand_landmark_columns_list, handedness_id_list, pose_landmark_columns_list = ph1.live(frame)

        if len(pose_landmark_columns_list) == 0:
            cv2.putText(frame, "No person detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
        elif len(hand_landmark_columns_list) == 0:
            cv2.putText(frame, "No hand detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
            cv2.imshow('Webcam', frame) 
        else:

            # Pick active hand
            active_hand = pick_hand_ph1(hand_landmark_columns_list, 
                                        handedness_id_list, 
                                        pose_landmark_columns_list)
            
            active_extracted_features = hand_landmark_columns_list[active_hand]\
                                    + pose_landmark_columns_list[0]
            active_extracted_features = np.array(active_extracted_features).reshape(1, 1, -1)
            active_extracted_features = torch.Tensor(active_extracted_features)
            
            
            if active_hand == -1:
                cv2.putText(frame, "No active hand detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
                cv2.imshow('Webcam', frame)            
            else:
                # Make prediction
                logits = predict_sign(model, active_extracted_features)
                preds = logits.argmax(dim=1).cpu()
                tag = sup.NUMBERS_TO_CLASSES[preds[0].item()]

                # Display the frame with the prediction
                draw_image(frame, hand_landmark_columns_list[active_hand], tag)
                

        
        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

def classify_webcam_image_ph2(model):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Extract landmarks
        hand_landmark_columns_list, handedness_id_list, pose_landmark_columns_list = ph1.live(frame)
        
        if len(pose_landmark_columns_list) == 0:
            cv2.putText(frame, "No person detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
        elif len(hand_landmark_columns_list) == 0:
            cv2.putText(frame, "No hand detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
            cv2.imshow('Webcam', frame) 
        else:
            # Perform geometric transformations
            transformed_landmarks_list = ph2.live(hand_landmark_columns_list, handedness_id_list, pose_landmark_columns_list)

            # Pick active hand
            active_hand = pick_hand_ph2(transformed_landmarks_list)
            
            active_transformed_landmarks = transformed_landmarks_list[0][active_hand][4:]
            active_transformed_landmarks = np.array(active_transformed_landmarks).reshape(1,1,-1)
            active_transformed_landmarks = torch.Tensor(active_transformed_landmarks)
            
            if active_hand == -1:
                cv2.putText(frame, "No active hand detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
                cv2.imshow('Webcam', frame)            
            else:
                # Make prediction
                logits = predict_sign(model, active_transformed_landmarks)
                preds = logits.argmax(dim=1).cpu()
                tag = sup.NUMBERS_TO_CLASSES[preds[0].item()]
                print(tag)

                draw_image(frame, hand_landmark_columns_list[active_hand], tag)
                # Display the framge with the prediction
        
        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

def classify_webcam_image_ph3(model, scaler, reducer):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Extract landmarks
        hand_landmark_columns_list, handedness_id_list, pose_landmark_columns_list = ph1.live(frame)

        if len(pose_landmark_columns_list) == 0:
            cv2.putText(frame, "No person detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
        elif len(hand_landmark_columns_list) == 0:
            cv2.putText(frame, "No hand detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
            cv2.imshow('Webcam', frame) 
        else:
            reduced_features_list = ph3.live(scaler, reducer, hand_landmark_columns_list, 
                                             handedness_id_list, pose_landmark_columns_list)

            # Pick active hand
            active_hand = pick_hand_ph3(reduced_features_list)

            active_reduced_features = reduced_features_list[0][active_hand][4:]
            active_reduced_features = np.array(active_reduced_features).reshape(1,1,-1)
            active_reduced_features = torch.Tensor(active_reduced_features)
            
            if active_hand == -1:
                cv2.putText(frame, "No active hand detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
                cv2.imshow('Webcam', frame)            
            else:
                # Make prediction
                logits = predict_sign(model, active_reduced_features)
                preds = logits.argmax(dim=1).cpu()
                tag = sup.NUMBERS_TO_CLASSES[preds[0].item()]

                # Display the frame with the prediction
                draw_image(frame, hand_landmark_columns_list[active_hand], tag)
                

        
        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

def classify_webcam_image_full(model, scaler, reducer):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Extract landmarks
        hand_landmark_columns_list, handedness_id_list, pose_landmark_columns_list = ph1.live(frame)

        if len(pose_landmark_columns_list) == 0:
            cv2.putText(frame, "No person detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
        elif len(hand_landmark_columns_list) == 0:
            cv2.putText(frame, "No hand detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
            cv2.imshow('Webcam', frame) 
        else:
            # Perform geometric transformations
            transformed_landmarks_list = ph2.live(hand_landmark_columns_list, handedness_id_list, pose_landmark_columns_list)
            
            reduced_features_list = ph3.live_full(scaler, reducer, transformed_landmarks_list)

            # Pick active hand
            active_hand = pick_hand_ph3(reduced_features_list)

            active_reduced_features = reduced_features_list[0][active_hand][4:]
            active_reduced_features = np.array(active_reduced_features).reshape(1,1,-1)
            active_reduced_features = torch.Tensor(active_reduced_features)
            
            if active_hand == -1:
                cv2.putText(frame, "No active hand detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
                cv2.imshow('Webcam', frame)            
            else:
                # Make prediction
                logits = predict_sign(model, active_reduced_features)
                preds = logits.argmax(dim=1).cpu()
                tag = sup.NUMBERS_TO_CLASSES[preds[0].item()]

                # Display the frame with the prediction
                draw_image(frame, hand_landmark_columns_list[active_hand], tag)
                
        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

model_path_root = '/Users/diego/Desktop/iteso/TOG/bin/gen/TRAIN/all-classes/37/BERT/'
bertmini_path = 'gaunernst/bert-mini-uncased/'

model_paths = {
    "all-classes" : {
        sup.DATA_S_PF : {
            True : {
                True : 'True-True-pca--n15-lr1e-05-ep3000s0.8849727651393784.pth',
                False : 'True-False---n75-lr1e-05-ep3000s0.9214995193848126.pth'
            },
            False : {
                True : 'False-True-pca--n15-lr1e-05-ep3000s0.9569582398803802.pth',
                False : 'False-False---n72-lr1e-05-ep6000s0.9690270212538716.pth'
            }
        }
    }
}

def live(inference_config):
    TRAIN_classes = inference_config["TRAIN_classes"]
    data_unit = inference_config["data_unit"]
    PH2 = inference_config["PH2"]
    PH3 = inference_config["PH3"]

    data_config = {
    "data_unit" : data_unit,
    "label_col" : sup.class_numeric_column,
    "class_list" : TRAIN_classes,
    "batch_size" : 1024,
    "PH2" : PH2,
    "PH3" : PH3,
    "kernel" : '',
    }

    if PH3:
        data_config["reducer"] = sup.PH3_REDUCER_NAME_PCA
        data_config["n"] = 15
    else:
        data_config["reducer"] = ''
        if PH2:
            data_config["n"] = 75
        else:
            data_config["n"] = 72


    train_config = {
    "arch" : sup.TRAIN_BERT_CODE,
    "device" : bert.device,
    "loadable" : bert.BERT_MINI,
    "optimizer" : optim.AdamW,
    "lr" : 1e-5,
    "weight_decay" : 0,
    "loss_fn" : nn.CrossEntropyLoss,
    "num_epochs" : 3000
    }




    model = bert.BERT(data_config=data_config, df=None, train_config=train_config)

    model_path = os.path.join(model_path_root, data_unit, bertmini_path, 
                              model_paths["all-classes"][data_unit][PH2][PH3])
    
    model.me.load_state_dict(torch.load(model_path))

    model.me.eval()
    model.me.to('cpu')

    if PH3:
        if PH2:
            PH3_SUB_BINGEN_ROOT = os.path.join(sup.PH3_BINGEN_ROOT, sup.PH3_W2_CODE)
        else:
            PH3_SUB_BINGEN_ROOT = os.path.join(sup.PH3_BINGEN_ROOT, sup.PH3_WO2_CODE)

        PH3_SUB3_BINGEN_ROOT = os.path.join(PH3_SUB_BINGEN_ROOT, sup.PH3_REDUCER_NAME_PCA, '')

        scaler_path = os.path.join(PH3_SUB_BINGEN_ROOT, f"scaler_{sup.DATA_AH_PF}.pkl")
        pca_path = os.path.join(PH3_SUB3_BINGEN_ROOT, f"{sup.DATA_AH_PF}{15}.pkl")

        with open(scaler_path, 'rb') as f:
            ah_pf_scaler = pickle.load(f)

        with open(pca_path, 'rb') as f:
            reducer = pickle.load(f)

    if PH2:
        if PH3:
            classify_webcam_image_full(model, ah_pf_scaler, reducer)
        else:
            classify_webcam_image_ph2(model)
    else:
        if PH3:
            classify_webcam_image_ph3(model, ah_pf_scaler, reducer)
        else:
            classify_webcam_image_ph1(model)

