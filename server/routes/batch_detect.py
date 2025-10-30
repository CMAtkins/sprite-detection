from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from pathlib import Path
from datetime import datetime
from typing import List
import zipfile, shutil, os, math, cv2, numpy as np
from ultralytics import YOLO
from server.util.image_grid import make_grid
from server.core.model import model

router = APIRouter()
RESULTS_DIR = Path("static/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Accepts multiple images, runs YOLOv8 inference, saves annotated
# and detected images, makes grids, zips results, returns URLs
# (basically serve integration of client_upload_recursive.py)
@router.post("/batch_detect")
async def batch_detect(files: List[UploadFile] = File(...)):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = RESULTS_DIR / f"batch_{timestamp}"
    detected_dir = batch_dir / "detected"
    grids_dir = batch_dir / "grids"
    batch_dir.mkdir(parents=True)
    detected_dir.mkdir()
    grids_dir.mkdir()

    detections_meta = []

    for file in files:
        # Skip hidden macOS "._" metadata files or empty uploads
        if file.filename.startswith("._") or file.filename.startswith("."):
            print(f"Skipping hidden file: {file.filename}")
            continue

        contents = await file.read()
        img_path = batch_dir / file.filename

        # Skip zero-byte files
        if not contents:
            print(f"Skipping empty file: {file.filename}")
            continue

        with open(img_path, "wb") as f:
            f.write(contents)

        # Perform YOLOv8 inference
        try:
            results = model.predict(source=img_path, save=True, save_txt=False, exist_ok=True, project=batch_dir)
        except Exception as e:
            print(f"⚠️ Skipping {file.filename}: {e}")
            continue

        annotated_path = batch_dir / f"annotated_{file.filename}"

        # YOLO saves annotated images to batch_dir/predict, move or rename if needed
        pred_dir = batch_dir / "predict"
        saved_imgs = list(pred_dir.glob(f"*{Path(file.filename).stem}*.jpg"))
        if saved_imgs:
            saved_imgs[0].rename(annotated_path)
        if pred_dir.exists():
            shutil.rmtree(pred_dir)

        # Count detections and copy detected-only images
        det_count = len(results[0].boxes)
        if det_count > 0 and annotated_path.exists():
            shutil.copy(annotated_path, detected_dir / annotated_path.name)

        detections_meta.append({
            "file": file.filename,
            "detections": det_count
        })

    grid_paths = make_grid(
        list(detected_dir.glob("*.jpg")),  # ← list of image paths
        4, 8, batch_dir, "batch_grid"
    )

    zip_path = RESULTS_DIR / f"{batch_dir.name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(batch_dir):
            for name in files:
                abs_path = Path(root) / name
                rel_path = abs_path.relative_to(RESULTS_DIR)
                zipf.write(abs_path, rel_path)

    zip_url = f"/static/results/{zip_path.name}"
    grid_urls = [f"/static/results/{p.relative_to(RESULTS_DIR)}" for p in grid_paths]

    return JSONResponse({
        "count": len(detections_meta),
        "detections": detections_meta,
        "zip_url": zip_url,
        "grid_urls": grid_urls
    })
