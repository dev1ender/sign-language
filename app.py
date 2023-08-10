from flask import Flask, Response, render_template, request,make_response
import cv2
import base64
import numpy as np
import random
import string
import os
import time

# Path to the folder where processed frames will be saved
PROCESSED_FRAMES_FOLDER = os.path.join(os.path.dirname(__file__), 'processed_frames')

# Ensure the folder exists
os.makedirs(PROCESSED_FRAMES_FOLDER, exist_ok=True)


app = Flask(__name__)

# Processing function
def process_frame(frame_data,unique_number):
    # Decode base64 image data
    image_data = frame_data.split(',')[1]
    decoded_data = base64.b64decode(image_data)
    nparr = np.frombuffer(decoded_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   
    # Add the unique number to the frame using OpenCV
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottom_left_corner = (10, 30)
    font_scale = 1
    font_color = (255, 255, 255)  # White color
    line_type = 2

    cv2.putText(frame, unique_number, bottom_left_corner, font, font_scale, font_color, line_type)

    # Encode the modified frame back to JPEG format
    _, buffer = cv2.imencode('.jpg', frame)
    # modified_frame_data = base64.b64encode(buffer).decode('utf-8')
    modified_frame_data = buffer.tobytes()
    
    timestamp = int(time.time())

    response = {
        "unique_no": unique_number,
        "image": f"data:image/jpeg;base64,{modified_frame_data}",
        "timestamp": timestamp
    }

    # Create a JSON response
    json_response = make_response(response)
    json_response.headers['Content-Type'] = 'application/json'

    return json_response

def display_base64_image(base64_data, unique_number):
    image_data = base64_data.split(',')[1]
    decoded_data = base64.b64decode(image_data)
    image_path = os.path.join(PROCESSED_FRAMES_FOLDER, f'frame-{unique_number}.jpg')
    with open(image_path, 'wb') as f:
        f.write(decoded_data)
    print(f"Image saved to {image_path}")


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame_route():
    data = request.json
    frame_data = data.get('frame', '')
    # Generate a unique 10-digit number
    unique_number = ''.join(random.choice(string.digits) for _ in range(10))
    # display_base64_image(frame_data, unique_number)
    if frame_data:
        result = process_frame(frame_data,unique_number)
        return result

    return 'Frame data not received', 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
