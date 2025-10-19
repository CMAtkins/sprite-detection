#!/usr/bin/env python3
"""
client_upload_recursive.py

Recursively scan a directory for images, send them in batches to the Sprite Detection API,
and extract returned ZIPs (annotated images + sprites_detected folder)
into an organized local results directory. After all batches are done, the entire run
is archived into one timestamped ZIP file inside the results directory.
"""

import requests
import os
import sys
import argparse
import zipfile
import tempfile
from datetime import datetime
from pathlib import Path
import shutil

API_URL = "http://127.0.0.1:8000/predict-annotated/"
DEFAULT_RESULTS_DIR = "results/from_recursive_upload"
VALID_EXTS = (".jpg", ".jpeg", ".png")
BATCH_SIZE = 100


def find_images(root_dir):
    """Recursively find all image files under the given directory."""
    images = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(VALID_EXTS):
                images.append(os.path.join(dirpath, fname))
    return images


def upload_batch(image_batch):
    """Upload a single batch of images to the API."""
    files = [("files", open(img, "rb")) for img in image_batch]
    try:
        response = requests.post(API_URL, files=files)
    finally:
        for _, f in files:
            f.close()

    if response.status_code != 200:
        raise RuntimeError(f"❌ API error {response.status_code}: {response.text}")
    return response.content


def extract_results(zip_bytes, output_dir, batch_idx):
    """Extract one returned ZIP into an output directory.
       Returns (detected, batch_dir, detected_count)
    """
    os.makedirs(output_dir, exist_ok=True)
    tmp_zip = os.path.join(tempfile.gettempdir(), f"sprite_batch{batch_idx}.zip")

    with open(tmp_zip, "wb") as f:
        f.write(zip_bytes)

    batch_dir = os.path.join(output_dir, f"batch_{batch_idx}")
    os.makedirs(batch_dir, exist_ok=True)

    detected = False
    detected_count = 0

    with zipfile.ZipFile(tmp_zip, "r") as zipf:
        # Extract all files first
        zipf.extractall(batch_dir)

        for member in zipf.namelist():
            parts = member.replace("\\", "/").split("/")
            if "sprites_detected" in parts and member.lower().endswith(VALID_EXTS):
                detected = True
                detected_count += 1

    os.remove(tmp_zip)
    return detected, detected_count, batch_dir


def zip_full_run(results_dir):
    """Create a single timestamped ZIP archive containing the full results directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"run_{timestamp}.zip"
    zip_path = os.path.join(results_dir, zip_name)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for foldername, _, filenames in os.walk(results_dir):
            for filename in filenames:
                if filename == zip_name:
                    continue
                file_path = os.path.join(foldername, filename)
                arcname = os.path.relpath(file_path, results_dir)
                zipf.write(file_path, arcname)

    return zip_path


def main():
    parser = argparse.ArgumentParser(description="Recursively upload directories of images for sprite detection.")
    parser.add_argument("dir", help="Path to parent directory to search for images.")
    parser.add_argument("--outdir", help=f"Output directory for extracted results (default={DEFAULT_RESULTS_DIR})")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE, help=f"Number of images per upload batch (default={BATCH_SIZE})")
    args = parser.parse_args()

    root_dir = args.dir
    output_dir = args.outdir if args.outdir else DEFAULT_RESULTS_DIR
    batch_size = args.batch

    if not os.path.isdir(root_dir):
        print(f"❌ Directory not found: {root_dir}")
        sys.exit(1)

    image_paths = find_images(root_dir)
    if not image_paths:
        print("⚠️ No image files found.")
        sys.exit(0)

    print(f"🔎 Found {len(image_paths)} images under {root_dir}")
    total_batches = (len(image_paths) + batch_size - 1) // batch_size

    detected_batches = []

    for i in range(total_batches):
        start = i * batch_size
        end = min(start + batch_size, len(image_paths))
        batch = image_paths[start:end]
        print(f"\n📦 Processing batch {i + 1}/{total_batches} ({len(batch)} images)")
        try:
            zip_bytes = upload_batch(batch)
            detected, detected_count, batch_dir = extract_results(zip_bytes, output_dir, i + 1)
            if detected:
                detected_batches.append(i + 1)
                plural = "sprite" if detected_count == 1 else "sprites"
                print(
                    f"✨ {detected_count} {plural} detected. See: {batch_dir}/sprites_detected/")
            else:
                print(f"⚪ Batch {i + 1}: No sprites detected.")

        except Exception as e:
            print(f"❌ Error processing batch {i + 1}: {e}")

    print("\n---------- SUMMARY ----------")
    print(f"Processed {total_batches} batch(es) from {len(image_paths)} images.")
    if detected_batches:
        print(f"✨ Sprites detected in batch(es): {', '.join(map(str, detected_batches))}")
    else:
        print("⚪ No sprites detected in any batch.")

    print(f"📁 All results extracted to: {output_dir}")

    # Create one ZIP archive for the full run
    zip_path = zip_full_run(output_dir)
    print(f"📦 Full run archived to: {zip_path}")
    print("--------------------------------")


if __name__ == "__main__":
    main()
