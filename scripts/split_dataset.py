import os
import random
import shutil

images_path = "data/train/images"
labels_path = "data/train/labels"

val_ratio = 0.2

images = [f for f in os.listdir(images_path) if f.endswith(".jpg") or f.endswith(".png")]
random.shuffle(images)

val_size = int(len(images) * val_ratio)
val_images = images[:val_size]

os.makedirs("data/valid/images", exist_ok=True)
os.makedirs("data/valid/labels", exist_ok=True)

for img in val_images:
    name = os.path.splitext(img)[0]

    img_src = os.path.join(images_path, img)
    lbl_src = os.path.join(labels_path, name + ".txt")

    img_dst = os.path.join("data/valid/images", img)
    lbl_dst = os.path.join("data/valid/labels", name + ".txt")

    shutil.move(img_src, img_dst)

    if os.path.exists(lbl_src):
        shutil.move(lbl_src, lbl_dst)
    else:
        open(lbl_dst, "w").close()