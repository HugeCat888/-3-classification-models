"""
datasets.py
------------
สร้าง tf.data.Dataset จากโฟลเดอร์ data/classification/<split>/<ClassName>/
โดยใช้ preprocessing.resize_with_padding เดียวกับที่ app.py ใช้ตอน inference
เพื่อให้ Train / Validation / Test / Inference สอดคล้องกันตาม Section 10

ฟังก์ชันหลัก:
  - build_dataset()          : สร้าง dataset สำหรับ train/val/test
  - oversample_dataset()     : ทำ oversampling เพื่อแก้ปัญหา class imbalance / ข้อมูลน้อย
"""

import collections
import os

import numpy as np
import tensorflow as tf
from PIL import Image

from config import CLASS_NAMES, get_input_size
from preprocessing import resize_with_padding, get_preprocess_fn


def list_paths_and_labels(directory: str):
    """สแกนโฟลเดอร์ตามชื่อ class ใน CLASS_NAMES คืน (paths, labels) ที่ index ตรงกับ CLASS_NAMES"""
    paths, labels = [], []
    for idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(directory, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                paths.append(os.path.join(class_dir, fname))
                labels.append(idx)
    return paths, labels


def oversample_dataset(
    paths: list,
    labels: list,
    target_per_class: int = None,
    strategy: str = "repeat",
    seed: int = 42,
):
    """
    ทำ Oversampling เพื่อแก้ปัญหา class imbalance และชุดข้อมูลที่น้อยเกินไป

    การทำงาน:
    -----------
    Oversampling คือการ "เพิ่ม" ตัวอย่างของ class ที่มีน้อย โดยการสุ่มซ้ำจาก
    ภาพที่มีอยู่ (ไม่ได้สร้างข้อมูลใหม่จากศูนย์) เพื่อให้แต่ละ class มีจำนวน
    ตัวอย่างเท่ากัน ผลที่ได้คือ:
      - โมเดลได้เห็นทุก class ในสัดส่วนที่ใกล้เคียงกัน → ลด bias
      - เมื่อใช้ร่วมกับ Augmentation (Noise/Blur) ภาพที่ซ้ำจะถูก transform
        ต่างกันในแต่ละ epoch → เพิ่ม diversity ของ training data อย่างมีนัยสำคัญ

    Strategy:
    ----------
    - "repeat"  (default): สุ่มดึงภาพเดิมซ้ำ (with replacement) จนครบ target
                           → เร็ว ใช้ได้เสมอแม้ข้อมูลน้อยมาก
    - "balance" :          เหมือน "repeat" แต่ target_per_class = จำนวน class
                           ที่มากที่สุด (auto-balance โดยไม่ต้องระบุ)

    Args:
        paths            : list ของ path ภาพทั้งหมด
        labels           : list ของ label (int) ที่ตรงกับ paths
        target_per_class : จำนวนภาพต่อ class ที่ต้องการ (None = ใช้ class ที่มากสุด)
        strategy         : 'repeat' หรือ 'balance'
        seed             : random seed เพื่อ reproducibility

    Returns:
        (paths_oversampled, labels_oversampled) — list ขนาดใหญ่ขึ้น ลำดับสุ่มแล้ว

    ตัวอย่าง:
        paths, labels = list_paths_and_labels(train_dir)
        paths, labels = oversample_dataset(paths, labels, target_per_class=300)
    """
    rng = np.random.default_rng(seed)

    # จัดกลุ่ม path ตาม class
    class_to_paths: dict = collections.defaultdict(list)
    for p, l in zip(paths, labels):
        class_to_paths[l].append(p)

    counts = {cls: len(ps) for cls, ps in class_to_paths.items()}
    max_count = max(counts.values())

    if target_per_class is None or strategy == "balance":
        target_per_class = max_count

    print(f"\n[Oversampling] Strategy: '{strategy}' | Target per class: {target_per_class}")
    for idx, class_name in enumerate(CLASS_NAMES):
        n = counts.get(idx, 0)
        added = max(0, target_per_class - n)
        print(f"  {class_name:12s}: {n:4d} images → {target_per_class:4d} (+{added} oversampled)")

    new_paths, new_labels = [], []
    for cls_idx, cls_paths in class_to_paths.items():
        if len(cls_paths) >= target_per_class:
            # มีพอแล้ว — ใช้เท่าที่มี (ไม่ตัดออก)
            new_paths.extend(cls_paths[:target_per_class])
            new_labels.extend([cls_idx] * target_per_class)
        else:
            # oversample: เพิ่มภาพโดยสุ่มซ้ำจากที่มีอยู่
            needed = target_per_class - len(cls_paths)
            extra = rng.choice(cls_paths, size=needed, replace=True).tolist()
            new_paths.extend(cls_paths + extra)
            new_labels.extend([cls_idx] * target_per_class)

    # สุ่มลำดับเพื่อไม่ให้ batch เต็มไปด้วย class เดียวกัน
    indices = rng.permutation(len(new_paths))
    new_paths = [new_paths[i] for i in indices]
    new_labels = [new_labels[i] for i in indices]

    print(f"  Total after oversampling: {len(new_paths)} images\n")
    return new_paths, new_labels


def _load_and_pad_numpy(path_bytes, target_size):
    path = path_bytes.numpy().decode("utf-8")
    img = Image.open(path).convert("RGB")
    img = resize_with_padding(img, target_size)
    return np.array(img).astype("float32")


def _make_load_fn(target_size):
    def load_fn(path, label):
        img = tf.py_function(
            func=lambda p: _load_and_pad_numpy(p, target_size),
            inp=[path],
            Tout=tf.float32,
        )
        img.set_shape([target_size[0], target_size[1], 3])
        return img, label

    return load_fn


def build_dataset(
    directory: str,
    model_name: str,
    batch_size: int,
    shuffle: bool = False,
    augmentation_layer=None,
    seed: int = 42,
    oversample: bool = False,
    oversample_target: int = None,
):
    """
    สร้าง tf.data.Dataset สำหรับ train/validation/test

    Args:
        directory         : path ของโฟลเดอร์ split (train/validation/test)
        model_name        : ชื่อโมเดล เพื่อเลือก input size และ preprocess_fn ที่ถูกต้อง
        batch_size        : ขนาด batch
        shuffle           : สุ่มลำดับข้อมูล (ควรเปิดเฉพาะ train)
        augmentation_layer: keras Sequential layer — ส่งเฉพาะ train (None สำหรับ val/test)
        seed              : random seed
        oversample        : เปิด oversampling ก่อนสร้าง dataset (ใช้กับ train เท่านั้น)
        oversample_target : จำนวนภาพต่อ class ที่ต้องการ (None = auto ตาม class ที่มากสุด)
    """
    target_size = get_input_size(model_name)
    paths, labels = list_paths_and_labels(directory)

    if len(paths) == 0:
        raise ValueError(
            f"ไม่พบภาพในโฟลเดอร์ {directory}\n"
            f"ตรวจสอบว่าวางไฟล์ตามโครงสร้าง data/classification/<split>/<ClassName>/ "
            f"โดยชื่อ ClassName ต้องตรงกับ config.CLASS_NAMES = {CLASS_NAMES}"
        )

    # Oversampling (ก่อน shuffle เพื่อให้ผลสม่ำเสมอ)
    if oversample:
        paths, labels = oversample_dataset(
            paths, labels, target_per_class=oversample_target, seed=seed
        )

    one_hot = tf.keras.utils.to_categorical(labels, num_classes=len(CLASS_NAMES))
    ds = tf.data.Dataset.from_tensor_slices((paths, one_hot))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True)

    ds = ds.map(_make_load_fn(target_size), num_parallel_calls=tf.data.AUTOTUNE)

    if augmentation_layer is not None:
        ds = ds.map(
            lambda x, y: (augmentation_layer(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    preprocess_fn = get_preprocess_fn(model_name)
    ds = ds.map(lambda x, y: (preprocess_fn(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds, len(paths)
