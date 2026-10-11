"""
train_classifier.py
--------------------
เทรน Classification model ตัวใดตัวหนึ่ง (efficientnet_b0 / mobilenetv3_large /
resnet50) ด้วยกลยุทธ์ Transfer Learning ตาม Section 9:

    Pretrained -> Freeze Base -> Train Head -> Unfreeze -> Fine-tune -> Save Best

ตัวอย่างการใช้งาน:
    python train_classifier.py --model efficientnet_b0 --augmentation v1
    python train_classifier.py --model resnet50 --augmentation none --epochs_head 20

ผลลัพธ์:
    - weights/<model>_<augmentation>_best.keras   (best checkpoint จาก val_loss)
    - weights/<model>_<augmentation>_history.csv  (training log สำหรับ Section 13 / 19)
"""

import argparse
import os
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
os.environ["KERAS_BACKEND"] = "torch"

import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from augmentation import get_augmentation
from config import (
    CLASS_NAMES,
    CLASSIFIER_DEFAULTS,
    RANDOM_SEED,
    TRAIN_DIR,
    VAL_DIR,
    WEIGHTS_DIR,
    get_input_size,
    set_global_seed,
)
from datasets import build_dataset
from models import get_model_module


def parse_args():
    p = argparse.ArgumentParser(description="Train a durian-leaf classifier (Section 9 / 17).")
    p.add_argument("--model", required=True, choices=["efficientnet_b0", "mobilenetv3_large", "resnet50"])
    p.add_argument("--augmentation", default="none", choices=["none", "v1", "v2"],
                   help="Augmentation version: none=Baseline, v1=Mild+Noise/Blur, v2=Moderate+Noise/Blur")
    p.add_argument("--train_dir", default=TRAIN_DIR)
    p.add_argument("--val_dir", default=VAL_DIR)
    p.add_argument("--batch_size", type=int, default=CLASSIFIER_DEFAULTS["batch_size"])
    p.add_argument("--epochs_head", type=int, default=CLASSIFIER_DEFAULTS["epochs_head"])
    p.add_argument("--epochs_finetune", type=int, default=CLASSIFIER_DEFAULTS["epochs_finetune"])
    p.add_argument("--lr_head", type=float, default=CLASSIFIER_DEFAULTS["lr_head"])
    p.add_argument("--lr_finetune", type=float, default=CLASSIFIER_DEFAULTS["lr_finetune"])
    p.add_argument("--unfreeze_layers", type=int, default=CLASSIFIER_DEFAULTS["unfreeze_last_n_layers"])
    p.add_argument("--patience", type=int, default=CLASSIFIER_DEFAULTS["early_stopping_patience"])
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--oversample", action="store_true",
                   help="เปิด oversampling บน train set เพื่อแก้ปัญหาข้อมูลน้อย/imbalance")
    p.add_argument("--oversample_target", type=int, default=None,
                   help="จำนวนภาพต่อ class หลัง oversample (None = ใช้ class ที่มากสุดเป็นเป้าหมาย)")
    return p.parse_args()


def main():
    args = parse_args()
    set_global_seed(args.seed)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    run_name = f"{args.model}_{args.augmentation}"
    print(f"=== Training run: {run_name} ===")
    print(f"Input size (Section 10.1): {get_input_size(args.model)}")
    print(f"Hyperparameters (log these in your report, per Section 19): {vars(args)}")

    # ---- Datasets --------------------------------------------------------
    augmentation_layer = get_augmentation(args.augmentation)
    train_ds, n_train = build_dataset(
        args.train_dir, args.model, args.batch_size,
        shuffle=True, augmentation_layer=None, seed=args.seed,
        oversample=args.oversample, oversample_target=args.oversample_target,
    )
    val_ds, n_val = build_dataset(
        args.val_dir, args.model, args.batch_size,
        shuffle=False, augmentation_layer=None, seed=args.seed,
    )
    print(f"Train images: {n_train} | Validation images: {n_val} | Classes: {CLASS_NAMES}")

    # ---- Build model -------------------------------------------------------
    model_module = get_model_module(args.model)
    input_size = get_input_size(args.model)
    model = model_module.build_model(
        num_classes=len(CLASS_NAMES),
        input_shape=(input_size[0], input_size[1], 3),
        dropout=args.dropout,
        augmentation_layer=augmentation_layer,
    )

    best_ckpt_path = os.path.join(WEIGHTS_DIR, f"{run_name}_best.keras")
    history_csv_path = os.path.join(WEIGHTS_DIR, f"{run_name}_history.csv")

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True),
        ModelCheckpoint(best_ckpt_path, monitor="val_loss", save_best_only=True),
        CSVLogger(history_csv_path, append=True),
    ]

    # ---- Phase 1: Train head (base frozen) --------------------------------
    print("\n--- Phase 1: Train classification head (base frozen) ---")
    model.compile(
        optimizer=Adam(learning_rate=args.lr_head),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_head, callbacks=callbacks)

    # ---- Phase 2: Fine-tune (unfreeze top layers, lower LR) ---------------
    print("\n--- Phase 2: Fine-tune (unfreeze top layers) ---")
    model = model_module.unfreeze_for_finetune(model, num_layers=args.unfreeze_layers)
    model.compile(
        optimizer=Adam(learning_rate=args.lr_finetune),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_finetune, callbacks=callbacks)

    print(f"\nDone. Best checkpoint saved to: {best_ckpt_path}")
    print(f"Training history saved to:    {history_csv_path}")


if __name__ == "__main__":
    main()
