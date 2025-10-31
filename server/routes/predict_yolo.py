from fastapi import APIRouter, File, UploadFile
from server.core.model import get_model, ModelType
import cv2, numpy as np

router = APIRouter()

yolo_model = get_model("yolo")


@router.post("/predict/yolo/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    results = yolo_model(img, conf=0.25)

    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": yolo_model.model.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy[0].tolist()
            })

    return {"detections": detections}
