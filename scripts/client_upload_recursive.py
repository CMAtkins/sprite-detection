import os
import sys
import requests
import zipfile
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

API_URL = "http://127.0.0.1:8000/predict-annotated/"
DEFAULT_RESULTS_DIR = "results/from_recursive_upload"
VALID_EXTS = (".jpg", ".jpeg", ".png")
BATCH_SIZE = 100


# Recursively find all image files in root_dir
def find_images(root_dir):
    images = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(VALID_EXTS):
                images.append(os.path.join(dirpath, fname))
    return images


# Upload a list of image paths to the FastAPI server and return ZIP
def upload_images(image_paths):
    files = []
    for img in image_paths:
        files.append(("files", open(img, "rb")))
    try:
        response = requests.post(API_URL, files=files)
    finally:
        for _, f in files:
            f.close()

    if response.status_code != 200:
        raise RuntimeError(f"❌ API error {response.status_code}: {response.text}")
    return response.content


# Extract one returned ZIP into an output directory
def extract_results(zip_bytes, output_dir, batch_idx):
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


# Create a single timestamped ZIP archive containing the full results directory
def zip_full_run(output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"run_{timestamp}.zip"
    zip_path = os.path.join(output_dir, zip_name)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for foldername, dirs, filenames in os.walk(output_dir):
            dirs[:] = [d for d in dirs if not d.startswith("run_")]
            for filename in filenames:
                if filename.startswith("run_") and filename.endswith(".zip"):
                    continue
                file_path = os.path.join(foldername, filename)
                arcname = os.path.relpath(file_path, output_dir)
                zipf.write(file_path, arcname)
    return zip_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python client_upload_recursive.py <image_dir> [--batch N]")
        sys.exit(1)

    img_dir = sys.argv[1]
    batch_size = BATCH_SIZE
    if "--batch" in sys.argv:
        batch_index = sys.argv.index("--batch")
        if batch_index + 1 < len(sys.argv):
            batch_size = int(sys.argv[batch_index + 1])

    if not os.path.isdir(img_dir):
        print(f"❌ Directory not found: {img_dir}")
        sys.exit(1)

    image_paths = find_images(img_dir)
    print(f"🔎 Found {len(image_paths)} image(s) under {img_dir}")

    if not image_paths:
        sys.exit(0)

    output_dir = DEFAULT_RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    detected_batches = []
    total_detected = 0
    total_batches = (len(image_paths) + batch_size - 1) // batch_size

    for i in range(total_batches):
        batch_imgs = image_paths[i * batch_size : (i + 1) * batch_size]
        print(f"\n📦 Processing batch {i + 1}/{total_batches} ({len(batch_imgs)} images)")
        try:
            zip_bytes = upload_images(batch_imgs)
            detected, detected_count, batch_dir = extract_results(zip_bytes, output_dir, i + 1)
            if detected:
                detected_batches.append(i + 1)
                total_detected += detected_count
                plural = "sprite" if detected_count == 1 else "sprites"
                print(f"✨ {detected_count} {plural} detected. See: {batch_dir}/sprites_detected/")
            else:
                print(f"⚪ Batch {i + 1}: No sprites detected.")
        except Exception as e:
            print(f"❌ Error processing batch {i + 1}: {e}")

    print("\n---------- SUMMARY ----------")
    print(f"Processed {total_batches} batch(es) from {len(image_paths)} images.")
    if detected_batches:
        print(f"✨ Sprites detected in batch(es): {', '.join(map(str, detected_batches))}")
        print(f"🧠 Total detected images: {total_detected}")
    else:
        print("⚪ No sprites detected in any batch.")

    # Create one ZIP archive for the full run
    zip_path = zip_full_run(output_dir)
    print(f"📁 Full run archived to: {zip_path}")

    # Post run cleanup
    # Remove all batch folders, keep only run_*.zip files.
    removed_dirs = []
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and not item.startswith("run_"):
            shutil.rmtree(item_path)
            removed_dirs.append(item)

    if removed_dirs:
        print(f"🧹 Cleaned up temporary batch folders: {', '.join(removed_dirs)}")
    else:
        print("🧹 No temporary batch folders to clean up.")

    print("--------------------------------")


if __name__ == "__main__":
    main()
