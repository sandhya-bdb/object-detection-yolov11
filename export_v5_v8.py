import sys
sys.path.insert(0, './yolov5_internal')

import torch
import warnings
from ultralytics import YOLO

warnings.filterwarnings('ignore')

print("Loading legacy YOLOv5 weights in PyTorch...")
try:
    # Use torch.serialization.safe_globals to bypass PyTorch 2.6 weights_only restrictions
    from models.yolo import Model, DetectionModel
    from models.common import Conv, Bottleneck, SPP, Focus, BottleneckCSP, Concat, SPPF
    import torch.nn as nn
    
    globals_to_allow = [
        Model, DetectionModel, Conv, Bottleneck, SPP, Focus, BottleneckCSP, Concat, SPPF,
        nn.modules.container.Sequential, nn.modules.conv.Conv2d, nn.modules.batchnorm.BatchNorm2d,
        nn.modules.activation.SiLU, nn.modules.pooling.MaxPool2d, nn.modules.upsampling.Upsample
    ]
    
    with torch.serialization.safe_globals(globals_to_allow):
        ckpt = torch.load('best.pt', map_location='cpu', weights_only=True)
except Exception as e:
    print("Falling back to weights_only=False due to:", e)
    ckpt = torch.load('best.pt', map_location='cpu', weights_only=False)

# Export to a clean ultralytics-compatible weights file
model = ckpt['model'] if 'model' in ckpt else ckpt.get('ema') or ckpt

# Instead of passing down the Model object which causes ultralytics checking to fail,
# we strip everything except the raw state dictionary.
print("Saving stripped state_dict...")
state_dict = model.state_dict() if hasattr(model, 'state_dict') else model
torch.save({"model": state_dict}, "best_clean.pt")

print("Validating with ultralytics...")
try:
    y = YOLO('best_clean.pt')
    y.info()
    print("Export successful!")
except Exception as e:
    print("Ultralytics failed:", e)
