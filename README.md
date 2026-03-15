# 🦟 Mosquito Detection System with YOLO11

An end-to-end computer vision API for detecting mosquitoes and other objects using a fine-tuned, exported **YOLO11** model. The model is serialized as TorchScript (`.torchscript`) for optimized CPU/GPU inference without relying on legacy dependencies.

This project was originally built around a custom `YOLOv5` implementation but has been upgraded and cleaned up to fully support the modern `ultralytics` package and `YOLO11` architecture.

---

## ✨ Key Features

- **Upgraded Architecture**: Employs a modernized, clean `.torchscript` model stripped of legacy YOLOv5 dependencies.
- **REST API (Flask / FastAPI)**: Provides a simple REST API endpoint (`/predict`) for easy image submission and base64-encoded response visualization.
- **In-Memory Processing**: The inference pipeline parses base64 image strings directly in-memory, avoiding unnecessary disk I/O and speeding up response times.
- **Production-Ready**: Includes a robust `Dockerfile` for seamless deployment to cloud environments like AWS, Azure, GCP, or Render.
- **CORS Enabled**: The API is configured to allow Cross-Origin Resource Sharing, making it easily consumable by frontend web applications.

---

## 🛠️ Tech Stack

- **Computer Vision**: [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics), OpenCV, PIL
- **Backend Framework**: Flask, Flask-CORS
- **Deployment**: Docker, AWS CLI
- **Language**: Python 3.10

---

## 📁 Project Structure

```text
.
├── app.py                     # Main Flask API server and application entry point
├── mosquito/                  # Core library module
│   └── pipeline/
│       └── predict.py         # Handles loading the YOLO model and running predictions
├── test_api.py                # Script to quickly test the running HTTP API via POST request
├── test_inference.py          # Script to quickly verify local model inference directly
├── yolo11n.torchscript        # Serialized YOLO11 model weights used by the application
├── requirements.txt           # Python dependencies and locked versions
└── Dockerfile                 # Container configuration for production deployment
```

---

## 🚀 Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/sandhya-bdb/object-detection-yolov11.git
cd object-detection-yolov11
```

### 2. Set up a Python Virtual Environment

It is highly recommended to use a virtual environment to avoid dependency conflicts.

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use: `venv\Scripts\activate`
```

### 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## 💻 Usage & Testing

You have two ways to interact with the detection system locally before deploying to production:

### Option A: Local Inference Test (Direct Pipeline)

This script generates a dummy test image in-memory and runs the `predict.py` pipeline directly. This is extremely useful for verifying your TorchScript model loads correctly without standing up the HTTP server.

```bash
python test_inference.py
```

*Expected Output*: You should see pipeline initialization messages followed by a success confirmation indicating the length of the returned base64 image string.

### Option B: REST API Test (End-to-End)

To test the full API workflow as a client would experience it:

1. **Start the API Server** in your primary terminal:
   ```bash
   python app.py
   ```
   *The server will start on `http://0.0.0.0:8080`.*

2. **Run the API Client** in a secondary terminal:
   ```bash
   python test_api.py
   ```
   *This script converts a dummy red test image to Base64, sends an HTTP POST request to `http://127.0.0.1:8080/predict`, and measures both the status code and API response time.*

---

## 🌐 API Endpoints

### 1. `GET /`
- **Description**: Renders the basic HTML index page (if `templates/index.html` is configured).
- **Response**: HTML content.

### 2. `POST /predict`
- **Description**: Accepts a base64 encoded image, runs the YOLO11 inference pipeline, and returns the image with bounding boxes drawn over detected mosquitoes.
- **Payload**:
  ```json
  {
      "image": "<base64_encoded_string>"
  }
  ```
- **Response**:
  ```json
  {
      "image": "<base64_encoded_string_with_predictions>"
  }
  ```

---

## 🐳 Deployment (Docker)

This repository includes a `Dockerfile` built on `python:3.10-slim-buster`, ready for production deployment. It installs necessary system packages (like `ffmpeg`, `libsm6`, `libxext6`) required by OpenCV.

1. **Build the Docker Image**:
   ```bash
   docker build -t mosquito-detection-api .
   ```

2. **Run the Container**:
   ```bash
   docker run -p 8080:8080 mosquito-detection-api
   ```

The API will instantly be exposed on port `8080`. You can test it using the same `test_api.py` script.

---

## 🔄 Model Upgrades & Maintenance

If you need to re-export weights or adjust metadata in the future, the repository contains several utility scripts (e.g., `export_v5_v8.py`, `fix_metadata.py`, `fix_onnx.py`). These were primarily used to transition the older YOLOv5 weights into a clean state compatible with modern `ultralytics` standards. For current operations, the pre-exported `yolo11n.torchscript` is all that is required.
