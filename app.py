import sys, os 
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS, cross_origin 
from mosquito.pipeline.predict import InferencePipeline

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        # Initialize YOLO model in memory exactly once when the server starts
        self.inference_pipeline = InferencePipeline("models/yolo11n.torchscript")

clApp = ClientApp()

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST','GET'])
@cross_origin()
def predictRoute():
    # Get base64 encoded image string from JSON payload
    data = request.get_json(force=True)
    if 'image' not in data:
        raise KeyError("image key missing in JSON")
    image_b64 = data['image']
    
    # Run inference directly in memory without saving to disk
    predicted_image_b64 = clApp.inference_pipeline.predict(image_b64)
    
    result = {"image": predicted_image_b64}
    return jsonify(result)

@app.route("/live", methods=['GET'])
@cross_origin()
def predictLive():
    return Response(
        "Live webcam inference via OS GUI (cv2.imshow) is deprecated for the HTTP API.", 
        status=501
    )

if __name__ == "__main__":
    clApp = ClientApp()
    app.run(host="0.0.0.0", port=8080, debug=True)
