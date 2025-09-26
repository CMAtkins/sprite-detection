import cv2
import os

def extract_frames(video_path, output_dir, step=1):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    saved_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            filename = f"frame_{saved_idx:04d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            print(f"Saved {filepath}")
            saved_idx += 1

        frame_idx += 1

    cap.release()
    print(f"Extracted {saved_idx} frames to {output_dir}")

if __name__ == "__main__":
    extract_frames(
        # raw videos are currently in gz
        # extend to unzip and iterate through every video
        # probably more data than needed right now anyway
        video_path="data/raw_videos/sprite_video.mp4",
        output_dir="data/raw_frames",
        step=5  # save every 5th frame to reduce dataset size
    )
