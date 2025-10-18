from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse
from typing import List
from ultralytics import YOLO
import numpy as np
import cv2
import tempfile, os, zipfile
from server.util.image_grid import make_grid

router = APIRouter()
model = YOLO("server/models/best.pt")


@router.post("/predict-annotated/")
async def predict_annotated(files: List[UploadFile] = File(...)):
    """Run YOLO inference on uploaded images and return a ZIP with:
       - annotated images for all files
       - sprites_detected/ folder for those containing detections
       - batch_grid.jpg summarizing all images in batch
    """
    tmpdir = tempfile.mkdtemp()
    batch_dir = os.path.join(tmpdir, "batch_1")
    os.makedirs(batch_dir, exist_ok=True)
    sprites_dir = os.path.join(batch_dir, "sprites_detected")
    os.makedirs(sprites_dir, exist_ok=True)

    zip_path = os.path.join(tmpdir, "results.zip")
    grid_images = []

    for file in files:
        # Decode uploaded image
        contents = await file.read()
        npimg = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        # Run YOLO inference without automatic save
        results = model.predict(img, conf=0.25, save=False)

        # Create annotated image manually
        annotated_img = results[0].plot()
        annotated_name = f"annotated_{file.filename}"
        annotated_path = os.path.join(batch_dir, annotated_name)
        cv2.imwrite(annotated_path, annotated_img)

        # Track image for grid display
        grid_images.append(annotated_img)

        # Check for sprite detections
        found_sprite = any(
            "sprite" in model.names[int(box.cls)].lower()
            for r in results for box in r.boxes
        )

        # Copy to sprites_detected/ if detected
        if found_sprite:
            cv2.imwrite(os.path.join(sprites_dir, annotated_name), annotated_img)

    # Save a grid of all annotated images
    if grid_images:
        grid = make_grid(grid_images, cols=2)
        grid_path = os.path.join(batch_dir, "batch_grid.jpg")
        cv2.imwrite(grid_path, grid)

    # Build the ZIP
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for foldername, _, filenames in os.walk(batch_dir):
            for filename in filenames:
                file_path = os.path.join(foldername, filename)
                arcname = os.path.relpath(file_path, tmpdir)
                zipf.write(file_path, arcname)

    return FileResponse(zip_path, media_type="application/zip", filename="annotated_results.zip")
