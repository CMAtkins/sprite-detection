#!/bin/bash

set -e

VENV_DIR="yolo39"   # or whatever you want it to be named

echo "=== Setting up YOLOv8 environment with Python 3.9 ==="

# Check for python3.9
if ! command -v python3.9 &> /dev/null
then
    echo "Python 3.9 not found. Please install it first:"
    echo "   brew install python@3.9"
    exit 1
fi

# Remove any old venv
if [ -d "$VENV_DIR" ]; then
    echo "Removing old virtual environment: $VENV_DIR"
    rm -rf $VENV_DIR
fi

# Create and activate new venv
python3.9 -m venv $VENV_DIR
source $VENV_DIR/bin/activate

# Upgrade pip/setuptools/wheel
pip install --upgrade pip setuptools wheel

echo "=== Installing dependencies ==="

# 1. NumPy first
pip install numpy==1.25.2

# 2. PyTorch CPU wheels
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. OpenCV (headless not needed here)
pip install "opencv-python>=4.6.0,<4.13" --no-deps

# 4. Ultralytics YOLOv8
pip install ultralytics==8.3.199

# 5. Optional: visualization and utilities, will probably need some of these later
# pip install matplotlib pandas seaborn tqdm Pillow

echo "=== Verifying installation ==="
python - <<'EOF'
import torch, numpy as np, cv2, ultralytics
print("PyTorch:", torch.__version__)
print("NumPy:", np.__version__)
print("OpenCV:", cv2.__version__)
print("Ultralytics:", ultralytics.__version__)
EOF

echo "=== YOLOv8 environment setup complete! ==="
echo "Activate it with: source $VENV_DIR/bin/activate"