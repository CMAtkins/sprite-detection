import os
import torch
import tensorflow as tf
from ultralytics import YOLO


class ModelType:
    YOLO = "yolo"
    KERAS = "keras"


class SpriteModel:
    def __init__(self, model_path, backend=ModelType.YOLO):
        self.backend = backend.lower()
        self.path = model_path
        self.model = self._load()

        self.input_shape = None
        if self.backend == ModelType.KERAS:     # detect expected input shape
            try:
                # Many Keras models expose .input_shape
                self.input_shape = tuple(self.model.input_shape[1:4])
                print(f"\nLoaded Keras model from: {self.path}")
                print(
                    f"   Expected input shape: (None, {self.input_shape[0]}, {self.input_shape[1]}, {self.input_shape[2]})")
            except Exception as e:
                print(f"\nCould not determine input shape automatically: {e}")
        elif self.backend == ModelType.YOLO:
            # for yolo just show the model summary info
            print(f"\nLoaded YOLO model from: {self.path}")
            try:
                print(f"   → Model names: {self.model.names}")
            except Exception:
                pass

    def _load(self):
        if self.backend == ModelType.YOLO:
            print("Loading YOLOv8 model...")
            return YOLO(self.path)
        elif self.backend == ModelType.KERAS:
            print("Loading Keras model...")
            return tf.keras.models.load_model(self.path)
        else:
            raise ValueError(f"Unknown backend type: {self.backend}")

    def predict(self, image):
        if self.backend == ModelType.YOLO:
            results = self.model(image)
            return results[0].boxes.data.tolist()
        elif self.backend == ModelType.KERAS:
            target_size = (256, 256)
            if self.input_shape is not None:
                target_size = self.input_shape[:2]

            img = tf.image.resize(image, target_size) / 255.0
            pred = self.model.predict(img)
            return float(pred[0][0])

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

