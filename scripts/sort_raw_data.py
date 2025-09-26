import os
import shutil

root_dir = "/Volumes/Memorex USB/interesting_events_Sprites_Only"

# destination directories for copied files
dest_dirs = {
    "raw_images": "/Users/chelseaatkins/Desktop/git/sprite-detection/data/raw/raw_images",
    "raw_videos": "/Users/chelseaatkins/Desktop/git/sprite-detection/data/raw/raw_videos",
    "raw_bmps": "/Users/chelseaatkins/Desktop/git/sprite-detection/data/raw/raw_bmps",
    "raw_xmls": "/Users/chelseaatkins/Desktop/git/sprite-detection/data/raw/raw_xmls"
}

# create destination directories if they don't exist
for d in dest_dirs.values():
    os.makedirs(d, exist_ok=True)

# file extensions mapping
ext_map = {
    ".jpg": "raw_images",
    ".gz": "raw_videos",
    ".bmp": "raw_bmps",
    ".xml": "raw_xmls"
}


def copy_files():
    # go through all subdirectories recursively
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)

        if os.path.isdir(subdir_path):

            for dirpath, dirnames, filenames in os.walk(subdir_path):
                for fname in filenames:

                    ext = os.path.splitext(fname)[1].lower()

                    if ext in ext_map:
                        dest_folder = dest_dirs[ext_map[ext]]
                        src_file = os.path.join(dirpath, fname)
                        dest_file = os.path.join(dest_folder, fname)

                        # avoid overwriting by appending a number if file exists
                        # (this shouldn't matter... they should all have unique names?)
                        counter = 1
                        base_name, extension = os.path.splitext(fname)
                        while os.path.exists(dest_file):
                            dest_file = os.path.join(dest_folder, f"{base_name}_{counter}{extension}")
                            counter += 1

                        shutil.copy2(src_file, dest_file)
                        print(f"Copied {src_file} to {dest_file}")


if __name__ == "__main__":
    copy_files()
