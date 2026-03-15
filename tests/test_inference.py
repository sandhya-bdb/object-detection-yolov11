import base64
import io
from PIL import Image
from mosquito.pipeline.predict import InferencePipeline
import traceback

print("Initializing pipeline...")
pipeline = InferencePipeline("models/yolo11n.torchscript")

print("Generating dummy image...")
img = Image.new('RGB', (416, 416), color = 'red')
buffered = io.BytesIO()
img.save(buffered, format="JPEG")
img_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

print("Running predict()...")
try:
    res = pipeline.predict(img_b64)
    print("Success! output len:", len(res))
except Exception as e:
    traceback.print_exc()

