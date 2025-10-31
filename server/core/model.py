import os
from server.core.model_loader import SpriteModel, ModelType

DEFAULT_BACKEND = ModelType.YOLO

YOLO_PATH = "server/models/best2.pt"
KERAS_PATH = "server/models/sprite_fixed_model.keras"


def get_model(backend=DEFAULT_BACKEND):
    if backend == ModelType.YOLO:
        path = YOLO_PATH
    elif backend == ModelType.KERAS:
        path = KERAS_PATH
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    return SpriteModel(path, backend)
