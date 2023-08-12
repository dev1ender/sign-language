import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
# import nbformat
import tensorflow as tf

# load model
## variables for openCV
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

## defining the model 
interpreter = tf.lite.Interpreter("model/model.tflite")
print("GPU is", "available" , tf.config.list_physical_devices('GPU'))

# Create a GPU delegate
# delegate = tf.lite.experimental.load_delegate('libtensorflowlite_gpu_delegate.so')

# # Bind the delegate to the interpreter
# interpreter.add_delegate(delegate)

# # Allocate tensors and invoke inference
# interpreter.allocate_tensors()

found_signatures = list(interpreter.get_signature_list().keys())
prediction_fn = interpreter.get_signature_runner("serving_default")

ROWS_PER_FRAME = 543  # number of landmarks per frame

# load train categories to be used in prediction
train = pd.read_csv('model/train.csv.zip')
train['sign_ord'] = train['sign'].astype('category').cat.codes


def read_parquet(pq_path):
    data = pd.read_parquet(pq_path)
    return data

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

def prediction_func(data):
    # Dictionaries to translate sign <-> ordinal encoded sign
    SIGN2ORD = train[['sign', 'sign_ord']].set_index('sign').squeeze().to_dict()
    ORD2SIGN = train[['sign_ord', 'sign']].set_index('sign_ord').squeeze().to_dict()

    ## load data from output parquet
    xyz_np = load_relevant_data_subset(data)
    prediction = prediction_fn(inputs=xyz_np)
    pred = prediction['outputs'].argmax()
    prediction_confidence = prediction['outputs'][pred]
    sign = ORD2SIGN[pred]
    return sign, prediction_confidence

def create_landmarks(images,parquet_xyz):
    all_landmarks = pd.DataFrame()
    frame = 0
    for image in images:
        with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
            frame += 1
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            if results.left_hand_landmarks or results.right_hand_landmarks:
                landmarks = create_frame_landmark_df(results,frame,parquet_xyz)
                all_landmarks = pd.concat([all_landmarks,landmarks])
    return all_landmarks

def predict(all_landmarks):
    sign, confidence = prediction_func(all_landmarks)
    if confidence > 0.01:
        return sign
    else:
        return None
    
        