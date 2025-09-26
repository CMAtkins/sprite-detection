# AI Framework for Detection of Lightning Sprites
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

`/Users/chelseaatkins/Desktop/git/sprite-detection/venv/lib/python3.10/site-packages/libs/canvas.py`

from 

`p.drawLine(self.prev_point.x(), 0, self.prev_point.x(), self.pixmap.height())`

to 

`p.drawLine(int(self.prev_point.x()), 0, int(self.prev_point.x()), self.pixmap.height())`

This may need to be changed in other places, but the error and fix are the same.

## Phase 2: Train Detection Model
Not tested: Setup script may be able to expedite this `./setup_training_env.sh`

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