# AI Framework for Detection of Lightning Sprites
This project is created and maintained for educational purposes.
Florida Gulf Coast University Department of Chemistry and Physics, Senior Capstone in Physics, Fall 2025.

## Phase 1: Data Preparation
_Skip and go to phase 2 if image labelling is not necessary_

labelImg: GUI tool to label frames in YOLO format. This was done in a venv using Python 3.10.

`pip install labelImg==1.8.6`

It might be necessary to patch labelImg

```
Traceback (most recent call last):
  File "/Users/chelseaatkins/Desktop/git/sprite-detection/venv/lib/python3.10/site-packages/libs/canvas.py", line 530, in paintEvent
    p.drawLine(self.prev_point.x(), 0, self.prev_point.x(), self.pixmap.height())
TypeError: arguments did not match any overloaded call:
  drawLine(self, l: QLineF): argument 1 has unexpected type 'float'
  drawLine(self, line: QLine): argument 1 has unexpected type 'float'
  drawLine(self, x1: int, y1: int, x2: int, y2: int): argument 1 has unexpected type 'float'
  drawLine(self, p1: QPoint, p2: QPoint): argument 1 has unexpected type 'float'
  drawLine(self, p1: Union[QPointF, QPoint], p2: Union[QPointF, QPoint]): argument 1 has unexpected type 'float'
Abort trap: 6
```

Change line `530` in

`/venv/lib/python3.10/site-packages/libs/canvas.py`

from 

`p.drawLine(self.prev_point.x(), 0, self.prev_point.x(), self.pixmap.height())`

to 

`p.drawLine(int(self.prev_point.x()), 0, int(self.prev_point.x()), self.pixmap.height())`

This may need to be changed in other places, but the error and fix are the same.

## Phase 2: Train Detection Model
**Not tested:** Setup script may be able to expedite this `./setup_training_env.sh`

PyTorch supports Python 3.10, but on macOS there’s a long-standing issue where the PyTorch wheel sometimes fails to link
against NumPy properly, producing `RuntimeError: Numpy is not available` when trying to run the training.
A fresh venv built outside of PyCharm using [MiniConda](https://www.anaconda.com/docs/getting-started/miniconda/main) with Python 3.9 seems to be the most stable solution with dependencies installed in order:

```
conda create -n yolo-venv python=3.9 -y
conda activate yolo-venv
pip install numpy==1.25.2
pip install "opencv-python>=4.6.0,<4.13" --no-deps
pip install ultralytics==8.3.199
```

Run the training

`yolo detect train model=yolov8n.pt data=data/sprite.yaml epochs=50 imgsz=640`

Evaluate on test set

`yolo detect val model=runs/detect/train/weights/best.pt data=data/sprite.yaml split=test`

## Phase 3: Serve the Model
Most dependencies needed to serve the model were installed during Phase 2. **If Phase 2 was skipped, or if working from a fresh
virtual environment**, _all_ dependencies can be downloaded using

`pip install -r server/dependencies.txt`. 

**Otherwise**, just install what is needed for the server

`pip install fastapi uvicorn[standard] python-multipart`

Run server (from repo root)

`uvicorn server.main:app --reload --host 0.0.0.0 --port 8000`

Wait for the "INFO:     Application startup complete." message to display in the console. Ensure the server is running by 
going to `http://0.0.0.0:8000` in a browser. The following message should display {"message":"Sprite Detection API is running"}.

The API docs can be viewed at `http://127.0.0.1:8000/docs`

### Client Upload (Annotated)

In a new terminal from the repo root, while the server is running, test image annotation by pointing to a set of test images using the client upload script.

`python scripts/client_upload.py <img-dir>`

This will save the annotated results (labelled images with bounding boxes and confidence) to `results/annotated`.
The`--outdir` flag can be used to specify a custom output directory if preferred.

For larger image sets or those with nested directories, it is recommended to use the client_upload_recursive script.
By default, the recursive script will batch process into uploads of 100 images. The optional batch flag may be used to
specify a different batching number.

`python scripts/client_upload_recursive.py <image_dir> [--batch N]`

### Client Upload (JSON)

To get classification, confidence, and bounding box information in JSON format for a given image or images, use the following
command:

```
curl -X POST "http://127.0.0.1:8000/predict/" \
    -F "file=@<img_file>"
```
Example output:

```
{
    "detections":[
        {
            "class": "sprite",
            "confidence": 0.5845668315887451,
            "bbox": [
                356.70672607421875,
                395.62652587890625,
                382.57403564453125,
                453.54345703125
            ]
        }
    ]
}
```