from flask_socketio import SocketIO, emit, join_room, leave_room
from flask import Flask, Response, render_template, session, request, make_response, redirect
from geventwebsocket.handler import WebSocketHandler
from auth.auth_decorators import login_required
from auth.auth_blueprint import auth_blueprint
from utils import generate_unique_number
from ai import GestureModel
import logging
import time
import os
import string
import random
import numpy as np
import base64
import cv2
from gevent import monkey
monkey.patch_all()

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
app.register_blueprint(auth_blueprint)  # registering auth app blueprint
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
    user_id = session.get("user_id")
    if not user_id:
        session["user_id"] = str(time.time())
    logger.info("Rendering index page")
    return render_template("video_stream.html")


@socketio.on("connect")
def handle_connect():
    user_id = session.get("user_id")
    join_room(user_id)
    print("Client connected started", request.sid)


@socketio.on("disconnect")
def handle_disconnect():
    user_id = session.get("user_id")
    leave_room(user_id)
    print(f"Client with SID {user_id} disconnected")


user_data = {}


@socketio.on("process_frame")
def process_frame(data):
    user_id = session.get("user_id")
    frame_data = data.get("frame", "")
    if frame_data:
        image_data = frame_data.split(",")[1]
        decoded_data = base64.b64decode(image_data)
        nparr = np.frombuffer(decoded_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        user_data[user_id] = frame
    if len(user_data.get(user_id, [])) == 720:
        result = predict_frame(user_data[user_id])
        user_data[user_id] = []
        emit("frame_processed", {"prediction": result}, room=user_id)
    logger.info(f"frame count {len(user_data.get(user_id, []))}")


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    socketio.run(app, host='0.0.0.0', port=8000, log_output=True)
