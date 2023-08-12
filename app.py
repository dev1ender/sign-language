from flask import Flask, Response, render_template, request, make_response
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


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path to the folder where processed frames will be saved
PROCESSED_FRAMES_FOLDER = os.path.join(os.path.dirname(__file__), "processed_frames")

pq_file_sample = "model/100015657.parquet"

PARQUET_FILE_PATH = os.path.join(os.path.dirname(__file__), pq_file_sample)

# Ensure the folder exists
os.makedirs(PROCESSED_FRAMES_FOLDER, exist_ok=True)

app = Flask(__name__)
socketio = SocketIO(app)

# Load the model
model = GestureModel("model/model.tflite", "model/train.csv.zip")
parquet = model.read_parquet(PARQUET_FILE_PATH)
logger.info("Model loaded successfully")


# Processing function
def predict_frame(frames, unique_number):
    global model
    global parquet
    logger.info(f"Processing frame with unique number: {unique_number}")

    all_landmarks = model.create_landmarks(frames,parquet)
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
def index():
    logger.info("Rendering index page")
    return render_template("video_stream.html")

@socketio.on("connect")
def handle_connect():
    print("Client connected")

@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected")

frames = []

@socketio.on("process_frame")
def process_frame(data):
    global frames

    frame_data = data.get("frame", "")
    unique_number = generate_unique_number(10)
    if frame_data:
        image_data = frame_data.split(",")[1]
        decoded_data = base64.b64decode(image_data)
        nparr = np.frombuffer(decoded_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frames.append(frame)
    if len(frames) == 5:
            result = predict_frame(frames, unique_number)
            frames = []
            logger.info(f"Frame processed successfully with unique frame number: {unique_number}, result: {result}")
            emit("frame_processed", result)
    logger.info(f"frame count {len(frames)}")

if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    socketio.run(app, port=5000)
