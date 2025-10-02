from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from server.routes import predict, predict_annotated

app = FastAPI(title="Sprite Detection API")

app.include_router(predict.router)
app.include_router(predict_annotated.router)

model = YOLO("server/models/best.pt")

@app.get("/")
def read_root():
    return {"message": "Sprite Detection API is running"}