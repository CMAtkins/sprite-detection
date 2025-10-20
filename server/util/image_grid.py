import cv2
import numpy as np
import math
import os


def make_grid(images, cols, padding, save_dir, prefix, bg_color=(255, 255, 255)):
    """
    Create one or more 4x4 grids (or smaller) from a list of images.
    Each grid will be saved to `save_dir` as prefix_1.jpg, prefix_2.jpg, etc.
    """
    if not images:
        return

    total_images = len(images)
    max_per_grid = cols * cols
    n_grids = math.ceil(total_images / max_per_grid)

    for grid_idx in range(n_grids):
        subset = images[grid_idx * max_per_grid : (grid_idx + 1) * max_per_grid]
        rows = math.ceil(len(subset) / cols)

        # Resize all images to the same dimensions
        h, w = subset[0].shape[:2]
        resized = [cv2.resize(img, (w, h)) for img in subset]

        # Compute grid canvas size (with padding)
        grid_h = rows * h + (rows + 1) * padding
        grid_w = cols * w + (cols + 1) * padding
        grid_img = np.full((grid_h, grid_w, 3), bg_color, dtype=np.uint8)

        # Paste images into grid
        for idx, img in enumerate(resized):
            r = idx // cols
            c = idx % cols
            y = padding + r * (h + padding)
            x = padding + c * (w + padding)
            grid_img[y:y + h, x:x + w] = img

        # Save to disk if requested
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            out_name = f"{prefix}_{grid_idx + 1}.jpg" if n_grids > 1 else f"{prefix}.jpg"
            cv2.imwrite(os.path.join(save_dir, out_name), grid_img)

    # Return the number of grids created for logging
    return n_grids