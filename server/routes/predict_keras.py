from fastapi import APIRouter, UploadFile, File
from server.core.model import get_model, ModelType
from tensorflow import keras
import numpy as np
from PIL import Image
import io

router = APIRouter()

keras_model = get_model("keras")


@router.post("/predict/keras/")
async def predict_keras(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    target_size = keras_model.input_shape[:2] if getattr(keras_model, "input_shape", None) else (256, 256)
    image = image.resize(target_size)
    x = np.array(image, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)

    preds = keras_model.predict(x)
    conf = float(np.max(preds))
    label_index = int(np.argmax(preds))
    label = "sprite" if label_index == 0 else "non-sprite"

    # standardize output between models, this isn't totally relevant to Keras
    detections = [{
        "class": label,
        "confidence": conf,
        "bbox": [0, 0, image.width, image.height]  # just a placeholder since Keras is a classifier
    }]

    return {"backend": "keras", "detections": detections}