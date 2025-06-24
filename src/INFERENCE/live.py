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

def get_features_from_frame(frame, model, PH2=False, PH3=False, scaler=None, reducer=None):
    hand_landmark_columns_list, handedness_id_list, pose_landmark_columns_list = ph1.live(frame)

    if len(pose_landmark_columns_list) == 0:
        return -1
    elif len(hand_landmark_columns_list) == 0:
        return -1
    else:
        if PH2:
            transformed = ph2.live(hand_landmark_columns_list, handedness_id_list, pose_landmark_columns_list)
            if PH3:
                features = ph3.live_full(scaler, reducer, transformed)
                active_hand = pick_hand_ph3(features)
            else:
                features = transformed
                active_hand = pick_hand_ph2(features)
        else:
            if PH3:
                features = ph3.live(scaler, reducer, hand_landmark_columns_list,
                                handedness_id_list, pose_landmark_columns_list)
                active_hand = pick_hand_ph3(features)
            else: 
                features = hand_landmark_columns_list + pose_landmark_columns_list
                active_hand = pick_hand_ph1(hand_landmark_columns_list, handedness_id_list, pose_landmark_columns_list)

        if active_hand == -1:
            return -1
        else:
            if PH2 or PH3:
                x = features[0][active_hand][4:]
            else:
                x = features[active_hand] + pose_landmark_columns_list[0]

            x = torch.Tensor(np.array(x).reshape(1, 1, -1))
            return hand_landmark_columns_list[active_hand], x

def _run_webcam_loop(model, PH2=False, PH3=False, scaler=None, reducer=None, data_unit=None):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    use_sequence = (data_unit == sup.DATA_S_PV)
    seq_len = 12
    x_seq = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        features = get_features_from_frame(frame, model, PH2, PH3, scaler, reducer)

        if features == -1:
            cv2.putText(frame, "No detection", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Webcam', frame)
        else:
            hand_landmarks, x = features

            if use_sequence:
                x_seq.append(x)  # shape: [1, 1, D]
                if len(x_seq) < seq_len:
                    cv2.putText(frame, f"Buffering {len(x_seq)}/{seq_len}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow('Webcam', frame)
                    continue
                elif len(x_seq) > seq_len:
                    x_seq.pop(0)

                # Concatenate into [1, 12, D]
                x_cat = torch.cat(x_seq, dim=1)
                logits = predict_sign(model, x_cat)

            else:
                logits = predict_sign(model, x)

            preds = logits.argmax(dim=1).cpu()
            tag = sup.NUMBERS_TO_CLASSES[preds[0].item()]
            draw_image(frame, hand_landmarks, tag)

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
                True : 'True-True-pca--n15-lr1e-05-ep3996s0.8905265406386842.pth',
                False : 'True-False---n75-lr1e-05-ep3000s0.9214995193848126.pth'
            },
            False : {
                True : 'False-True-pca--n15-lr1e-05-ep3000s0.9569582398803802.pth',
                False : 'False-False---n72-lr1e-05-ep6000s0.9690270212538716.pth'
            }
        },
        sup.DATA_S_PV : {
            True : {
                True : 'True-True-pca--n15-lr1e-05-ep15984s0.649167733674776.pth',
                False : 'True-False---n75-lr1e-05-ep15984s0.6594110115236875.pth'
            },
            False : {
                True : 'False-True-pca--n15-lr1e-05-ep15984s0.8284250960307298.pth',
                False : 'False-False---n72-lr1e-05-ep15984s0.8911651728553137.pth'
            }
        }
    }
}

def classify_webcam_image(inference_config):

    TRAIN_classes = inference_config["TRAIN_classes"]
    data_unit = inference_config["data_unit"]
    PH2 = inference_config.get("PH2", False)
    PH3 = inference_config.get("PH3", False)

    # --- Phase-dependent config ---
    if PH3:
        reducer_name = sup.PH3_REDUCER_NAME_PCA
        n_components = 15
    else:
        reducer_name = ''
        n_components = 75 if PH2 else 72

    # --- Build model ---
    data_config = {
        "data_unit" : data_unit,
        "label_col" : sup.class_numeric_column,
        "class_list" : TRAIN_classes,
        "batch_size" : 1024,
        "PH2" : PH2,
        "PH3" : PH3,
        "kernel" : '',
        "reducer" : reducer_name,
        "n" : n_components,
    }

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

    model_path_file = model_paths[TRAIN_classes][data_unit][PH2][PH3]
    full_model_path = os.path.join(model_path_root, data_unit, bertmini_path, model_path_file)
    model.me.load_state_dict(torch.load(full_model_path, map_location='cpu'))
    model.me.eval()
    model.me.to('cpu')

    # --- Load scaler and reducer if needed ---
    scaler = reducer = None
    if PH3:
        sub_root = os.path.join(sup.PH3_BINGEN_ROOT, sup.PH3_W2_CODE if PH2 else sup.PH3_WO2_CODE)
        scaler_path = os.path.join(sub_root, f"scaler_{sup.DATA_AH_PF}.pkl")
        pca_path = os.path.join(sub_root, sup.PH3_REDUCER_NAME_PCA, f"{sup.DATA_AH_PF}{n_components}.pkl")

        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(pca_path, 'rb') as f:
            reducer = pickle.load(f)

    return  _run_webcam_loop(model, PH2=PH2, PH3=PH3, scaler=scaler, reducer=reducer, data_unit=data_unit)
