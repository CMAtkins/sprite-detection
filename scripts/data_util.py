import os
import shutil
import random


def move_raw_data(raw_directory, labels_directory, images_directory):
    txt_files = [f for f in os.listdir(raw_directory) if f.endswith(".txt")]
    jpg_files = set(f for f in os.listdir(raw_directory) if f.endswith(".jpg"))

    for txt in txt_files:
        base_name = os.path.splitext(txt)[0]
        jpg = base_name + ".jpg"

        txt_path = os.path.join(raw_directory, txt)
        if jpg in jpg_files:
            jpg_path = os.path.join(raw_directory, jpg)

            txt_dest = os.path.join(labels_directory, txt)
            jpg_dest = os.path.join(images_directory, jpg)

            shutil.copy2(txt_path, txt_dest)
            shutil.copy2(jpg_path, jpg_dest)
            print(f"Copied {txt} and {jpg}")
        else:
            print(f"⚠️ No matching .jpg for {txt}")


def count_txt_files(directory):
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory.")
        return 0

    txt_count = sum(1 for file in os.listdir(directory) if file.endswith(".txt"))
    return txt_count


def split_dataset(images_dir, labels_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    # make sure ratios sum to 1
    if not abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    os.makedirs(output_dir, exist_ok=True)
    for split in ["train", "val", "test"]:
        os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)

    images = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]
    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    train_files = images[:n_train]
    val_files = images[n_train:n_train + n_val]
    test_files = images[n_train + n_val:]

    def copy_files(file_list, split):
        for img in file_list:
            label = img.replace(".jpg", ".txt")
            shutil.copy(os.path.join(images_dir, img), f"{output_dir}/images/{split}/{img}")
            if os.path.exists(os.path.join(labels_dir, label)):
                shutil.copy(os.path.join(labels_dir, label), f"{output_dir}/labels/{split}/{label}")

    copy_files(train_files, "train")
    copy_files(val_files, "val")
    copy_files(test_files, "test")

    print(f"Split complete: {n_train} train, {n_val} val, {n_test} test")


if __name__ == "__main__":

    # PATHS
    raw_images_path = "/Users/chelseaatkins/Desktop/git/sprite-detection/data/raw/raw_images"
    labels_path = "/Users/chelseaatkins/Desktop/git/sprite-detection/data/labels"
    images_path = "/Users/chelseaatkins/Desktop/git/sprite-detection/data/images"
    data_split_path = "/Users/chelseaatkins/Desktop/git/sprite-detection/data/splits"

    # determine how many YOLO files we have available for model training
    count = count_txt_files(raw_images_path)
    print(f"Current number of YOLO files '{raw_images_path}': {count}")

    # DO NOT UNCOMMENT UNLESS MOVING AND SPLITTING NEW DATA FOR THE MODEL TRAINING SET
    # move_raw_data(raw_images_path, labels_path, images_path) # update the training data directories
    # split_dataset(images_path, labels_path, data_split_path, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)



