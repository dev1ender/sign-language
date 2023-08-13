from gevent import monkey
monkey.patch_all()
from flask import Flask, Response, render_template, request, make_response, redirect
import cv2
import base64
import numpy as np
import random
import string
import os
import time
import logging
from ai import GestureModel
from flask_socketio import SocketIO, emit
from utils import generate_unique_number
from auth.auth_blueprint import auth_blueprint
from auth.auth_decorators import login_required
from geventwebsocket.handler import WebSocketHandler

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Path to the folder where processed frames will be saved
PROCESSED_FRAMES_FOLDER = os.path.join(
    os.path.dirname(__file__), "processed_frames")

pq_file_sample = "model/100015657.parquet"

PARQUET_FILE_PATH = os.path.join(os.path.dirname(__file__), pq_file_sample)

# Ensure the folder exists
os.makedirs(PROCESSED_FRAMES_FOLDER, exist_ok=True)

app = Flask(__name__)
app.register_blueprint(auth_blueprint) # registering auth app blueprint
# add eventlet and gevent
socketio = SocketIO(app, async_mode='gevent', handler_class=WebSocketHandler)


# Load the model
model = GestureModel("model/model.tflite", "model/train.csv.zip")
parquet = model.read_parquet(PARQUET_FILE_PATH)
logger.info("Model loaded successfully")


# Processing function
def predict_frame(frames):
    global model
    global parquet

    all_landmarks = model.create_landmarks(frames, parquet)
    sign = model.predict(all_landmarks)

    timestamp = int(time.time())
    if not sign:
        sign = "unknown"
    response = {
        "prediction": sign,
        "timestamp": timestamp,
    }
    return response


@app.route("/")
@login_required
def index():
    logger.info("Rendering index page")
    return render_template("video_stream.html")

user_data = {}


@socketio.on("connect")
def handle_connect():
    print("Client connected started",request.sid)

    # user_sid = request.sid
    # print("Client connected")
    # user_data[user_sid] = {"frames": [], "unique_number": generate_unique_number(10)}
    # print(f"Client with SID {user_sid} connected")


@socketio.on("disconnect")
def handle_disconnect():
    user_sid = request.sid
    if user_sid in user_data:
        del user_data[user_sid]
        print(f"Client with SID {user_sid} disconnected")



@socketio.on("process_frame")
def process_frame(data):
    global frames
    user_sid = request.sid
    print(f"Processing frame for client with SID {user_sid}  {request}")
    if user_sid in user_data:
        user_info = user_data[user_sid]
        frame_data = data.get("frame", "")
        if frame_data:
            image_data = frame_data.split(",")[1]
            decoded_data = base64.b64decode(image_data)
            nparr = np.frombuffer(decoded_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            frames.append(frame)
        if len(user_info["frames"]) == 5:
            logger.info(f"Processing frame with unique number: {user_info}")
            result = predict_frame(user_info["frames"])
            user_info["frames"] = []
            logger.info(
                f"Frame processed successfully with unique frame number: {user_info['unique_number']}, result: {result}")
            # Emit the frame_processed event only to the same client
            emit("frame_processed", result, room=user_sid)  # Emit to the specific client
        logger.info(f"frame count {len(frames)}")



if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    socketio.run(app, host='0.0.0.0', port=8000,log_output=True)
