import os
from ultralytics import YOLO
import base64
from PIL import Image
import io

class InferencePipeline:
    def __init__(self, model_path="yolo11n.torchscript"):
        """
        Initializes the YOLO model in memory.
        Args:
            model_path (str): The path to the custom exported YOLO weights.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at {model_path}")
        
        # Load the model directly into memory using the ultralytics package
        self.model = YOLO(model_path, task='detect')
        self.model.task = 'detect'
        if hasattr(self.model, 'overrides'):
            self.model.overrides['task'] = 'detect'
    
    def predict(self, base64_image: str) -> str:
        """
        Runs inference on a base64 encoded image and returns a base64 encoded result.
        
        Args:
            base64_image (str): The input image string.
            
        Returns:
            str: The base64 encoded image with drawn YOLO predictions.
        """
        # Decode the incoming base64 string
        img_data = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(img_data)).convert("RGB")
        
        # Run YOLO inference
        results = self.model.predict(source=image, conf=0.5, save=False)
        
        # The results object contains the rendered image as a numpy array
        # Get the first result's image array (plotted with bounding boxes)
        res_plotted = results[0].plot()
        
        # Convert plotted numpy array back to PIL Image
        im_pred = Image.fromarray(res_plotted[..., ::-1]) # Convert BGR to RGB
        
        # Encode back to base64 to return
        buffered = io.BytesIO()
        im_pred.save(buffered, format="JPEG")
        encoded_result = base64.b64encode(buffered.getvalue())
        return encoded_result.decode("utf-8")
