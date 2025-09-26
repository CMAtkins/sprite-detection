import os
import shutil
import random

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
    split_dataset(
        images_dir="data/images",
        labels_dir="data/labels",
        output_dir="data/splits",
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1
    )
