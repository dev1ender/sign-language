import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
# import nbformat
import tensorflow as tf
import logging
# load model
## variables for openCV

ROWS_PER_FRAME = 543  # number of landmarks per frame


def create_frame_landmark_df(results, frame, xyz):
    xyz_skel = xyz[['type','landmark_index']].drop_duplicates().reset_index(drop=True).copy()
    face = pd.DataFrame()
    pose = pd.DataFrame()
    left_hand = pd.DataFrame()
    right_hand = pd.DataFrame()
    if results.face_landmarks:
        for i, point in enumerate(results.face_landmarks.landmark):
            face.loc[i, ['x','y','z']] = [point.x, point.y, point.z]
    if results.pose_landmarks:
        for i , point in enumerate(results.pose_landmarks.landmark):
            pose.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]
    if results.left_hand_landmarks:
        for i, point in enumerate(results.left_hand_landmarks.landmark):
            left_hand.loc[i, ['x','y','z']] = [point.x, point.y, point.z]
    if results.right_hand_landmarks:
        for i, point in enumerate(results.right_hand_landmarks.landmark):
            right_hand.loc[i, ['x', 'y', 'z']] = [point.x, point.y, point.z]   
    face = face.reset_index() \
        .rename(columns={'index':'landmark_index'}) \
            .assign(type='face')
    pose = pose.reset_index() \
        .rename(columns={'index':'landmark_index'}) \
            .assign(type='pose')
    left_hand = left_hand.reset_index() \
        .rename(columns={'index':'landmark_index'}) \
            .assign(type='left_hand')
    right_hand = right_hand.reset_index() \
        .rename(columns={'index':'landmark_index'}) \
            .assign(type='right_hand')


    landmarks = pd.concat([face,pose,right_hand,left_hand]).reset_index(drop=True)
    landmarks = xyz_skel.merge(landmarks, on=['type','landmark_index'], how='left')
    landmarks = landmarks.assign(frame=frame)
    return landmarks

def load_relevant_data_subset(data):
    data_columns = ['x', 'y', 'z']
    data = data[data_columns]
    # data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)

def prediction_func(data,ORD2SIGN,prediction_fn):
    ## load data from output parquet
    xyz_np = load_relevant_data_subset(data)
    prediction = prediction_fn(inputs=xyz_np)
    pred = prediction['outputs'].argmax()
    prediction_confidence = prediction['outputs'][pred]
    sign = ORD2SIGN[pred]
    return sign, prediction_confidence

    
class GestureModel:
    def __init__(self, model_path, train_csv_path):
        self.interpreter = tf.lite.Interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.found_signatures = list(self.interpreter.get_signature_list().keys())
        self.prediction_fn = self.interpreter.get_signature_runner("serving_default")
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.initialize_holistic()
        self.load_train_data(train_csv_path)
    
    def initialize_holistic(self):
        return self.mp_holistic.Holistic(
            static_image_mode=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

    def read_parquet(self,pq_path):
        data = pd.read_parquet(pq_path)
        return data
        
    def load_train_data(self, train_csv_path):
        self.train_data = pd.read_csv(train_csv_path)
        self.train_data['sign_ord'] = self.train_data['sign'].astype('category').cat.codes
        self.ORD2SIGN = self.train_data[['sign_ord', 'sign']].set_index('sign_ord').squeeze().to_dict()

    def create_landmarks(self, images, parquet_xyz):
        all_landmarks = pd.DataFrame()
        if self.holistic is None:
            print('holistic is Initialized..')
            self.holistic = self.initialize_holistic()
        frame = 0
        for img in images:
            frame += 1
            img.flags.writeable = False
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(image)
            if results.left_hand_landmarks or results.right_hand_landmarks:
                landmarks = create_frame_landmark_df(results,frame,parquet_xyz)
                all_landmarks = pd.concat([all_landmarks,landmarks])
        return all_landmarks
        
    def predict(self, all_landmarks):
        logging.info(f'predicting..')
        logging.info(f'all_landmarks: shape {all_landmarks.shape}')
        if all_landmarks.shape[0] == 0:
            return None
        sign, confidence = prediction_func(all_landmarks,self.ORD2SIGN,self.prediction_fn)
        if confidence > 0.01:
            return sign
        else:
            return None
            