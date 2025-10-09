from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse
from typing import List
from server.core.model import model
from server.util.image_grid import make_grid
import numpy as np
import cv2
import tempfile
import os
import zipfile

router = APIRouter()

@router.post("/predict-annotated/")
async def predict_annotated(files: List[UploadFile] = File(...)):
    tmpdir = tempfile.mkdtemp()
    zip_path = os.path.join(tmpdir, "results.zip")

    grid_images = []

    with zipfile.ZipFile(zip_path, "w") as zipf:
        for file in files:
            contents = await file.read()
            npimg = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

            results = model.predict(img, conf=0.25, save=True, project=tmpdir, name="annotated", exist_ok=True)

            # Collect annotated numpy array directly
            annotated_img = results[0].plot()
            grid_images.append(annotated_img)

            # Find YOLO’s saved annotated file
            annotated_dir = os.path.join(tmpdir, "annotated")
            for out_file in os.listdir(annotated_dir):
                full_path = os.path.join(annotated_dir, out_file)
                zipf.write(full_path, arcname=f"annotated_{file.filename}")

    # Build batch grid if we had images
    if grid_images:
        grid = make_grid(grid_images, cols=4)
        grid_path = os.path.join(tmpdir, "batch_grid.jpg")
        cv2.imwrite(grid_path, grid)

        with zipfile.ZipFile(zip_path, "a") as zipf:
            zipf.write(grid_path, arcname="batch_grid.jpg")

    return FileResponse(zip_path, media_type="application/zip", filename="annotated_results.zip")
