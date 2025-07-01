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
from TRAIN.architecture.KNN import knn

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

def pick_hand_ph1(features, 
                  handedness_id_list):

    # default
    return 0

def pick_hand_ph2(features, 
                  handedness_id_list):

    # default
    return 0

def pick_hand_ph3(features, 
                  handedness_id_list):

    # default
    return 0

def ph1_scale(model, hand_list, pose_list):
    result = []
    for pose in pose_list:
        for hand in hand_list:
            data = hand+pose
            tostd = np.array(data).reshape(1,-1)
            result.append(model.scaler.transform(tostd))
    return result

def ph2_scale(model, data):
    result = []
    for pose in data:
        for hand in data[pose]:
            data = hand
            tostd = np.array(data).reshape(1,-1)
            result.append(model.scaler.transform(tostd))
    return result

def predict_sign(model, active_selected_features):
    if model.arch == sup.TRAIN_BERT_CODE:
        return model.me(active_selected_features)
    elif model.arch == sup.TRAIN_KNN_CODE:
        print(active_selected_features)
        return model.me.predict(active_selected_features)

def get_features_from_frame(frame, model, data_unit, PH2=False, PH3=False, scaler=None, reducer=None):
    hand_landmark_columns_list, handedness_id_list, pose_landmark_columns_list = ph1.live(frame)

    if len(pose_landmark_columns_list) == 0:
        return -1
    elif len(hand_landmark_columns_list) == 0:
        return -1
    else:
        if data_unit == sup.DATA_S_PF:
            if PH2:
                transformed = ph2.live(hand_landmark_columns_list, handedness_id_list, pose_landmark_columns_list)
                if PH3:
                    reduced = ph3.live_full(scaler, reducer, transformed)
                    active_hand = pick_hand_ph3(reduced, handedness_id_list)
                    features = reduced[0][active_hand][4:]
                else:
                    standardized = ph2_scale(model, transformed)
                    active_hand = pick_hand_ph2(standardized, handedness_id_list)
                    features = standardized[0][active_hand]
            else:
                if PH3:
                    reduced = ph3.live(scaler, reducer, hand_landmark_columns_list,
                                    handedness_id_list, pose_landmark_columns_list)
                    active_hand = pick_hand_ph3(reduced, handedness_id_list)
                    features = reduced[0][active_hand][4:]
                else:
                    standardized = ph1_scale(model, hand_landmark_columns_list, pose_landmark_columns_list)
                    
                    active_hand = pick_hand_ph1(standardized, handedness_id_list)
                    features = standardized[0][active_hand]
        elif data_unit == sup.DATA_S_PV:
            if PH2:
                transformed = ph2.live(hand_landmark_columns_list, handedness_id_list, pose_landmark_columns_list)
                if PH3:
                    reduced = ph3.live_full(scaler, reducer, transformed)
                    active_hand = pick_hand_ph3(reduced, handedness_id_list)
                    features = reduced[0][active_hand][4:]
                else:
                    active_hand = pick_hand_ph2(transformed, handedness_id_list)
                    features = transformed[0][active_hand][4:]
            else:
                if PH3:
                    reduced = ph3.live(scaler, reducer, hand_landmark_columns_list,
                                    handedness_id_list, pose_landmark_columns_list)
                    active_hand = pick_hand_ph3(reduced, handedness_id_list)
                    features = reduced[0][active_hand][4:]
                else:
                    raw = [[hand+pose for hand in  hand_landmark_columns_list] for pose in pose_landmark_columns_list]
                    active_hand = pick_hand_ph1(raw, handedness_id_list)
                    features = raw[0][active_hand]

        if active_hand == -1:
            return -1
        else:
            if model.arch == sup.TRAIN_BERT_CODE and data_unit == sup.DATA_S_PF:
                x = torch.Tensor(np.array(features).reshape(1, 1, -1))
            else:
                x = np.array(features).reshape(1, -1)
            return hand_landmark_columns_list[active_hand], x

def _run_webcam_loop(model, data_unit=None, PH2=False, PH3=False, scaler=None, reducer=None):
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

        features = get_features_from_frame(frame, model, data_unit, PH2, PH3, scaler, reducer)

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
                x_cat = np.concatenate(x_seq, axis=1)
                if not PH3:
                    x_cat = model.scaler.transform(x_cat)
                tomod = torch.Tensor(x_cat.reshape(1, 12, -1))
                logits = predict_sign(model, tomod)

            else:
                logits = predict_sign(model, x)

            if model.arch == sup.TRAIN_BERT_CODE:
                preds = logits.argmax(dim=1).cpu()
                pred = preds[0].item()
            elif model.arch == sup.TRAIN_KNN_CODE:
                pred = logits[0]
            tag = sup.NUMBERS_TO_CLASSES[pred]
            draw_image(frame, hand_landmarks, tag)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)



model_path_root = '/Users/diego/Desktop/iteso/TOG/bin/gen/TRAIN/all-classes/37/'
bertmini_path = 'gaunernst/bert-mini-uncased/'

bert_model_paths = {
    "all-classes" : {
        sup.DATA_S_PF : {
            True : {
                True : 'True-True-pca--n15-lr1e-05-ep2500s0.8883904731389511.pth',
                False : 'True-False---n75-lr1e-05-ep2500s0.9284417387589448.pth'
            },
            False : {
                True : 'False-True-pca--n15-lr1e-05-ep2500s0.9565310263804336.pth',
                False : 'False-False---n72-lr1e-05-ep2500s0.9669977571291253.pth'
            }
        },
        sup.DATA_S_PV : {
            True : {
                True : 'True-True-pca--n15-lr1e-05-ep2000s0.615877080665813.pth',
                False : 'True-False---n75-lr1e-05-ep2000s0.6248399487836107.pth'
            },
            False : {
                True : 'False-True-pca--n15-lr1e-05-ep2000s0.8220230473751601.pth',
                False : 'False-False---n72-lr1e-05-ep2000s0.8898847631241997.pth'
            }
        }
    }
}

knn_model_path = 'False-False---n72-k1-s0.9407241268824095.pkl'

def classify_webcam_image(inference_config):

    TRAIN_classes = inference_config["TRAIN_classes"]
    data_unit = inference_config["data_unit"]
    PH2 = inference_config.get("PH2", False)
    PH3 = inference_config.get("PH3", False)
    arch = inference_config.get("arch", sup.TRAIN_BERT_CODE)

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

    if arch == sup.TRAIN_BERT_CODE:
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

        model_path_file = bert_model_paths[TRAIN_classes][data_unit][PH2][PH3]
        full_model_path = os.path.join(model_path_root, sup.TRAIN_BERT_CODE, data_unit, bertmini_path, model_path_file)
        model.me.load_state_dict(torch.load(full_model_path, map_location='cpu'))
        model.me.eval()
        model.me.to('cpu')

    elif arch == sup.TRAIN_KNN_CODE:
        train_config = {
            "arch" : sup.TRAIN_KNN_CODE,
            "k" : 1
        }

        model = knn.KNN(data_config=data_config, df=None, train_config=train_config)
        model.fit()


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

    return  _run_webcam_loop(model,  data_unit=data_unit, PH2=PH2, PH3=PH3, scaler=scaler, reducer=reducer)
