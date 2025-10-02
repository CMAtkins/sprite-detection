from ultralytics import YOLO

MODEL_PATH = "server/models/best.pt"
model = YOLO(MODEL_PATH)
