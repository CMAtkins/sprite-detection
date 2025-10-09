import cv2
import numpy as np
import math


def make_grid(images, cols):
    if not images:
        return None

    h, w = images[0].shape[:2]
    resized = [cv2.resize(img, (w, h)) for img in images]

    rows = math.ceil(len(resized) / cols)
    while len(resized) < rows * cols:
        resized.append(np.zeros_like(resized[0]))

    grid_rows = [cv2.hconcat(resized[i*cols:(i+1)*cols]) for i in range(rows)]
    grid = cv2.vconcat(grid_rows)
    return grid
