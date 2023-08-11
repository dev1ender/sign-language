# Hand Sign Detection using TensorFlow Lite and Flask

This project demonstrates hand sign detection using TensorFlow Lite and Flask. It captures video frames, performs hand sign detection using a pre-trained TensorFlow Lite model, adds a unique number to the frame, and displays the processed frames in a web interface using Flask.

## Features

- Capture video frames and perform hand sign detection.
- Display processed frames in a web interface using Flask.
- Add unique identifiers to each processed frame.
- Store processed frames on the server.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Virtualenv (optional but recommended)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/hand-sign-detection.git
   cd hand-sign-detection
   ```

2. (Optional) Create and activate a virtual environment:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required dependencies from requirements.txt:

    ```bash
    Copy code
    pip install -r requirements.txt
    ```

### Running the Application

1. Start the Flask application:

```bash
python app.py
```

2. Open your web browser and navigate to http://127.0.0.1:5000 to access the web interface.

3. Follow the instructions on the web interface to capture and process hand sign frames.

### Usage
- Follow the on-screen instructions to capture video frames.
- The processed frames will be displayed with unique numbers added.
- You can view the stored processed frames in the processed_frames folder.


### Contributing
Contributions are welcome! If you'd like to contribute to this project, please follow the guidelines in CONTRIBUTING.md.

### License
This project is licensed under the MIT License - see the LICENSE file for details.

