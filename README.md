# Object Detection with YOLO11

An end-to-end computer vision API for detecting objects using a fine-tuned, exported `YOLO11` model serialized as TorchScript (`.torchscript`). 

This project was originally built around a custom `YOLOv5` implementation but has been upgraded and cleaned up to support the modern `ultralytics` package and `YOLO11` architecture.

## Features

- **Upgraded Architecture**: Uses a modernized, clean `.torchscript` model stripped of legacy YOLOv5 dependencies.
- **FastAPI / Flask**: Provides a REST API endpoint for easy image submission and base64-encoded response visualization.
- **Production Ready**: Includes a `Dockerfile` for easy deployment to cloud services like AWS, Azure, or Render.

## Project Structure

```text
.
├── app.py                # Main Flask API server
├── best.torchscript      # Upgraded, serialized model weights
├── mosquito/             # Core inference and pipeline logic
│   ├── pipeline/
│   │   └── predict.py    # Handles loading the model and running predictions
├── test_api.py           # Script to quickly test the running API
├── test_inference.py     # Script to quickly verify local model inference
├── requirements.txt      # Python dependencies
└── Dockerfile            # Container configuration
```

## Setup Instructions

### 1. Clone the repository and setup environment
Clone the branch and setup a Python virtual environment:
```bash
git clone https://github.com/sandhya-bdb/object-detection-yolov11.git
cd object-detection-yolov11
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage & Testing

You have two ways to interact with the detection system locally before deploying:

### Option A: Local Inference Test
This script generates a dummy test image and runs the `predict.py` pipeline directly. This is useful for verifying your TorchScript model is loading correctly.
```bash
python test_inference.py
```
*If successful, you will see speed metrics and a confirmation message.*

### Option B: REST API Test
To test the full API workflow:

1. **Start the API Server** in one terminal:
   ```bash
   python app.py
   ```
2. **Run the API Client** in a second terminal:
   ```bash
   python test_api.py
   ```
   *This script converts a test image to Base64, sends an HTTP POST request to `localhost:8080/predict`, and measures the API response time.*

---

## Deployment (Docker)

This repository includes a `Dockerfile` ready for production deployment.

1. **Build the Docker Image**:
   ```bash
   docker build -t object-detection-api .
   ```
2. **Run the Container**:
   ```bash
   docker run -p 8080:8080 object-detection-api
   ```
The API is now exposing the `/predict` endpoint on port `8080`.
