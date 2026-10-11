"""
config.py
---------
ค่ากลางของทั้งโปรเจกต์ สำหรับ Classification Pipeline
(class names, paths, hyperparameters, reproducibility)
"""

import os

# ---------------------------------------------------------------------------
# 1) Durian classes (ตรงกับ Section 3.1)
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "Monthong",     # 0 หมอนทอง
    "Chanee",       # 1 ชะนี
    "Kanyao",       # 2 ก้านยาว
    "BlackThorn",   # 3 หนามดำ
    "MusangKing",   # 4 มูซานคิง
]
NUM_CLASSES = len(CLASS_NAMES)

CLASS_NAMES_TH = {
    "Monthong": "หมอนทอง",
    "Chanee": "ชะนี",
    "Kanyao": "ก้านยาว",
    "BlackThorn": "หนามดำ",
    "MusangKing": "มูซานคิง",
}

# ---------------------------------------------------------------------------
# 2) Paths (ตรงกับโครงสร้างใน Section 15)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CLASSIFICATION_DATA_DIR = os.path.join(DATA_DIR, "classification")

TRAIN_DIR = os.path.join(CLASSIFICATION_DATA_DIR, "train")

# รองรับทั้งชื่อโฟลเดอร์ 'valid' และ 'validation'
_val_candidate = os.path.join(CLASSIFICATION_DATA_DIR, "valid")
if os.path.isdir(_val_candidate):
    VAL_DIR = _val_candidate
else:
    VAL_DIR = os.path.join(CLASSIFICATION_DATA_DIR, "validation")

TEST_DIR = os.path.join(CLASSIFICATION_DATA_DIR, "test")

WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")

# ---------------------------------------------------------------------------
# 3) Input size decision (Section 10.1)
#    - ค่า default: ใช้ขนาดตาม pretrained weight ของแต่ละโมเดล (224x224)
#    - ถ้าโจทย์บังคับ 299x299 จริง ให้ตั้ง FORCE_UNIFORM_INPUT_SIZE = (299, 299)
#      แล้วทุกโมเดลจะถูกบังคับใช้ขนาดเดียวกัน (trade-off ตามที่ระบุใน final_md)
# ---------------------------------------------------------------------------
FORCE_UNIFORM_INPUT_SIZE = None  # เช่น (299, 299) ถ้าต้องการบังคับ

DEFAULT_INPUT_SIZES = {
    "efficientnet_b0": (224, 224),
    "mobilenetv3_large": (224, 224),
    "resnet50": (224, 224),
}


def get_input_size(model_name: str):
    """คืน input size (H, W) ที่จะใช้จริงสำหรับโมเดลนั้น ๆ"""
    if FORCE_UNIFORM_INPUT_SIZE is not None:
        return FORCE_UNIFORM_INPUT_SIZE
    return DEFAULT_INPUT_SIZES.get(model_name, (224, 224))


# ---------------------------------------------------------------------------
# 4) Reproducibility (Section 19)
# ---------------------------------------------------------------------------
RANDOM_SEED = 42


def set_global_seed(seed: int = RANDOM_SEED):
    """ล็อก seed ให้ NumPy / TensorFlow (และ random) เพื่อให้ผล reproduce ได้"""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# 5) Training defaults (Section 17) — จุดเริ่มต้นที่แนะนำ ปรับได้ตามจริง
#    แต่ต้อง log ค่าที่ใช้จริงไว้เสมอ (ดู utils/logging ใน train_classifier.py)
# ---------------------------------------------------------------------------
CLASSIFIER_DEFAULTS = dict(
    batch_size=16,
    epochs_head=15,
    epochs_finetune=15,
    lr_head=1e-3,
    lr_finetune=1e-5,
    early_stopping_patience=8,
    unfreeze_last_n_layers=30,
)


