import onnx
import json

try:
    model = onnx.load("best.onnx")
    metadata = {
        "task": "detect",
        "batch": 1,
        "stride": 32,
        "names": {0: "mosquito"},
        "imgsz": [640, 640]
    }
    meta = model.metadata_props.add()
    meta.key = 'metadata'
    meta.value = json.dumps(metadata)
    
    onnx.save(model, "best_meta.onnx")
    print("Metadata written to best_meta.onnx successfully!")
except Exception as e:
    print("Error:", e)
