from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import numpy as np
import cv2
from server.routes import predict, predict_annotated

app = FastAPI(title="Sprite Detection API")

app.include_router(predict.router)
app.include_router(predict_annotated.router)

model = YOLO("server/models/best.pt")

@app.get("/")
def read_root():
    return {"message": "Sprite Detection API is running"}


# @app.post("/predict/")
# async def predict(file: UploadFile = File(...)):
#     contents = await file.read()
#     npimg = np.frombuffer(contents, np.uint8)
#     img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
#
#     results = model.predict(img, conf=0.25)
#
#     # Results
#     detections = []
#     for r in results:
#         for box in r.boxes:
#             detections.append({
#                 "class": model.names[int(box.cls)],
#                 "confidence": float(box.conf),
#                 "bbox": box.xyxy[0].tolist()  # [x1, y1, x2, y2]
#             })
#
#     return {"detections": detections}
