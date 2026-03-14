import requests
import base64
import json
import time
import os

from PIL import Image
import io

# We need a dummy image to test
img = Image.new('RGB', (416, 416), color = 'red')
buffered = io.BytesIO()
img.save(buffered, format="JPEG")
img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

print("Starting to send request...")
start = time.time()
response = requests.post("http://127.0.0.1:8080/predict", json={"image": img_b64})
end = time.time()

print(f"Status Code: {response.status_code}")
print(f"Time Taken: {end - start:.2f} seconds")

if response.status_code == 200:
    data = response.json()
    if "image" in data:
        print("Success! Inference pipeline returned an image.")
else:
    print(response.text)
