import requests
import os
import sys
import argparse
from datetime import datetime

API_URL = "http://127.0.0.1:8000/predict-annotated/"
DEFAULT_OUTPUT_DIR = "results/annotated"

def upload_dir(img_dir, output_path):
    if not os.path.isdir(img_dir):
        raise ValueError(f"Directory not found: {img_dir}")

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"annotated_results_{timestamp}.zip")

    files = []
    for fname in os.listdir(img_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            files.append(("files", open(os.path.join(img_dir, fname), "rb")))

    if not files:
        raise ValueError(f"No images found in {img_dir}")

    print(f"Uploading {len(files)} images to {API_URL}...")
    response = requests.post(API_URL, files=files)

    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"Annotated results saved to {output_path}")
    else:
        print(f"Something went wrong!! Server returned {response.status_code}: {response.text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload directory of images to Sprite Detection API")
    parser.add_argument("img_dir", help="Path to directory containing images")
    parser.add_argument("--outdir", help="Directory to save results (default=results/annotated)")
    args = parser.parse_args()

    output_dir = args.outdir if args.outdir else DEFAULT_OUTPUT_DIR
    upload_dir(args.img_dir, output_dir)
