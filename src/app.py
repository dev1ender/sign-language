from flask import Flask, Response, render_template, request, make_response
import cv2
import base64
import numpy as np
import random
import string
import os
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path to the folder where processed frames will be saved
PROCESSED_FRAMES_FOLDER = os.path.join(os.path.dirname(__file__), "processed_frames")

# Ensure the folder exists
os.makedirs(PROCESSED_FRAMES_FOLDER, exist_ok=True)

app = Flask(__name__)


# Processing function
def process_frame(frame_data, unique_number):
    logger.info(f"Processing frame with unique number: {unique_number}")

    image_data = frame_data.split(",")[1]
    decoded_data = base64.b64decode(image_data)
    nparr = np.frombuffer(decoded_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottom_left_corner = (10, 30)
    font_scale = 1
    font_color = (255, 255, 255)  # White color
    line_type = 2

    cv2.putText(
        frame,
        unique_number,
        bottom_left_corner,
        font,
        font_scale,
        font_color,
        line_type,
    )

    _, buffer = cv2.imencode(".jpg", frame)
    modified_frame_data = buffer.tobytes()

    timestamp = int(time.time())

    response = {
        "unique_no": unique_number,
        "image": f"data:image/jpeg;base64,{modified_frame_data}",
        "timestamp": timestamp,
    }

    json_response = make_response(response)
    json_response.headers["Content-Type"] = "application/json"

    return json_response


@app.route("/")
def index():
    logger.info("Rendering index page")
    return render_template("index.html")


@app.route("/process_frame", methods=["POST"])
def process_frame_route():
    try:
        data = request.json
        frame_data = data.get("frame", "")
        unique_number = "".join(random.choice(string.digits) for _ in range(10))

        if frame_data:
            result = process_frame(frame_data, unique_number)
            logger.info(f"Frame processed successfully with unique frame number: {unique_number}")
            return result

        logger.warning("Frame data not received")
        return "Frame data not received", 400

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return "An error occurred", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
