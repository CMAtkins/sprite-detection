from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from server.routes import predict, predict_annotated, batch_detect
from pathlib import Path
from fastapi.staticfiles import StaticFiles

STATIC_DIR = Path("static")
RESULTS_DIR = STATIC_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Sprite Detection API")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Register routes
app.include_router(predict.router)
app.include_router(predict_annotated.router)
app.include_router(batch_detect.router)

model = YOLO("server/models/best.pt")

@app.get("/")
def read_root():
    return {"message": "Sprite Detection API is running"}