import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
# import nbformat
import tensorflow as tf


## variables for openCV
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

## defining the model 
interpreter = tf.lite.Interpreter("src/model.tflite")
found_signatures = list(interpreter.get_signature_list().keys())
prediction_fn = interpreter.get_signature_runner("serving_default")

# Add ordinally Encoded Sign (assign number to each sign name)\
train = pd.read_csv('src/data/train.csv.zip')
train['sign_ord'] = train['sign'].astype('category').cat.codes

## sample parquet
pq_file_sample = "src/data/100015657.parquet"
xyz = pd.read_parquet(pq_file_sample)

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


## load data from parguet 
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



# For webcam input:
def do_capture_loop(xyz,pq_file=None):
    all_landmarks = pd.DataFrame()
    cap = cv2.VideoCapture(1)
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
        frame = 0
        while cap.isOpened():
            frame+=1
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            image = cv2.flip(image,1)

            ## create landmarks dataframe from results
            landmarks = create_frame_landmark_df(results,frame,xyz)

            ## collects 5 frames and then passes for prediction
            for i in range(5):
                all_landmarks = pd.concat([landmarks]).reset_index(drop=True)

            if results.left_hand_landmarks or results.right_hand_landmarks:
                sign, confidence = prediction_func(all_landmarks)
                if confidence > 0.01:
                    draw_predictions(image,text=f"{sign}:{str(confidence)}")
                    # pass
                pass
            
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # Draw landmark annotation on the image.
            
            # mp_drawing.draw_landmarks(
            #     image,
            #     results.face_landmarks,
            #     mp_holistic.FACEMESH_CONTOURS,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=mp_drawing_styles
            #     .get_default_face_mesh_contours_style())
            # mp_drawing.draw_landmarks(
            #     image,
            #     results.pose_landmarks,
            #     mp_holistic.POSE_CONNECTIONS,
            #     landmark_drawing_spec=mp_drawing_styles
            #     .get_default_pose_landmarks_style())
            #Flip the image horizontally for a selfie-view display.
            
            # text = prediction_func(pq_file)
            # print(text)
            # print(text)
            # draw_predictions(image,text)
            # cv2.rectangle(frame, (x, y - text_height - 5), (x + text_width, y), (0, 0, 0), -1)
            # cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # if confidence > 90:
            #     draw_predictions(image,text=sign)
            #     continue

            cv2.imshow('GROOT',image)# cv2.flip(image, 1))

            if cv2.waitKey(1) & 0xFF == 27:
                break
        cap.release()        
    
    
def draw_predictions(image,text):
    # Reading an image in default mode
    # image = cv2.imread(path)
        
    # Window name in which image is displayed
    # window_name = 'Image'
    
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # org
    org = (100, 100)
    
    # fontScale
    fontScale = 2
    
    # Blue color in BGR
    color = (255, 0, 0)
    
    # Line thickness of 2 px
    thickness = 3
    
    # Using cv2.putText() method
    image = cv2.putText(image, text, org, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
    

if __name__ == "__main__":
    pq_file_sample = "src/data/100015657.parquet"
    xyz = pd.read_parquet(pq_file_sample)
    # pq_file = pd.read_parquet('output.parquet')

    do_capture_loop(xyz)
    

    

    




