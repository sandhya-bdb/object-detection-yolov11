from ultralytics import YOLO

try:
    # Ultralytics lets you override the task manually if the file is loaded directly with model=...
    model = YOLO("best.torchscript", task="detect")
    model.task = "detect"
    print("Task forced successfully on standard model initialization.")
except Exception as e:
    print(e)
