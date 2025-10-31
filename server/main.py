from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from server.routes import batch_detect
from server.routes.predict_yolo import router as yolo_router
from server.routes.predict_keras import router as keras_router
from server.routes.predict_annotated_yolo import router as yolo_annotated_router
# from server.routes.predict_annotated_keras import router as keras_annotated_router
from pathlib import Path
from fastapi.staticfiles import StaticFiles

STATIC_DIR = Path("static")
RESULTS_DIR = STATIC_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Sprite Detection API")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Register routes
app.include_router(yolo_router)
app.include_router(keras_router)
app.include_router(yolo_annotated_router)
# app.include_router(keras_annotated_router)
app.include_router(batch_detect.router)

model = YOLO("server/models/best.pt")

@app.get("/")
def read_root():
    return {"message": "Sprite Detection API is running"}