"""
evaluation/evaluate_classifier.py
-----------------------------------
ประเมินผล classifier ที่เทรนแล้วบน Unseen Test Set และวินิจฉัย Learning Curve
อ้างอิงงานวิจัย:
  • "Classification of Durian Leaf Images Using CNN" – JIKO Unkhair
    https://ejournal.unkhair.ac.id/index.php/jiko/article/view/8576/5217
  • "VGG-16 Transfer Learning for Durian Variety Classification" – JUTIF Unsoed
    https://jutif.if.unsoed.ac.id/index.php/jurnal/article/view/5506/1253

กราฟที่สร้าง:
  [1] Learning Curve  – Loss & Accuracy (Train vs Val) + Diagnosis
  [2] Confusion Matrix (Counts เท่านั้น)
  [3] Per-Class Metrics Bar Chart (Precision / Recall / F1 ราย Class)
  [4] Generalization Comparison (Train vs Val vs Unseen Test)

ค่าประสิทธิภาพที่รายงานใน .txt:
  [A] Accuracy, Precision, Recall, F1-Score (per-class + macro + weighted)
  [B] Learning Curve Diagnosis (Fit Status, Gap)
  [C] Generalization Assessment
  [D] Confusion Matrix
  [E] Model Efficiency (Parameters, Size, Latency)

ตัวอย่างการรัน:
    python -m evaluation.evaluate_classifier \\
        --model resnet50 \\
        --checkpoint weights/resnet50_v1_best.keras

ดู evaluation/compare_models.py สำหรับเปรียบเทียบหลายโมเดล
"""

import argparse
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

import augmentation
from config import CLASS_NAMES, TEST_DIR, WEIGHTS_DIR
from datasets import build_dataset
from evaluation.plots import (
    diagnose_and_plot_learning_curve,
    plot_per_class_metrics,
    plot_confusion_matrix_counts,
    plot_generalization_comparison,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a durian-leaf classification model on the Unseen Test Set."
    )
    p.add_argument(
        "--model", required=True,
        choices=["efficientnet_b0", "mobilenetv3_large", "resnet50"],
        help="Model architecture (used to select correct preprocessing/input size)",
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to .keras checkpoint file")
    p.add_argument("--history", default=None,
                   help="Path to _history.csv (auto-detected from weights/ if omitted)")
    p.add_argument("--test_dir", default=TEST_DIR,
                   help="Path to Unseen Test Set directory")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--output_dir", default="evaluation/reports",
                   help="Folder to save all reports and graph images")
    p.add_argument("--no_plots", action="store_true",
                   help="Skip graph generation (compute numbers only)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    run_name     = os.path.splitext(os.path.basename(args.checkpoint))[0]
    version      = run_name.replace(f"{args.model}_", "").replace("_best", "")
    display_name = f"{args.model} ({version})".strip()

    sep = "=" * 68
    print(f"\n{sep}")
    print(f"  Evaluation Pipeline: [{display_name}]")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Test Set   : {args.test_dir}")
    print(f"{sep}")

    # ── 1. Load model & Unseen Test Set ─────────────────────────────────────
    print("\n[1/5] Loading model and building unseen test dataset...")
    model = tf.keras.models.load_model(args.checkpoint)
    test_ds, n_test = build_dataset(args.test_dir, args.model, args.batch_size, shuffle=False)

    # ── 2. Inference ─────────────────────────────────────────────────────────
    print(f"[2/5] Running inference on {n_test} unseen test images...")
    all_probs, all_true_onehot = [], []
    y_true, y_pred = [], []

    start = time.time()
    for batch_x, batch_y in test_ds:
        preds = model.predict(batch_x, verbose=0)
        batch_y_np = batch_y.numpy()
        all_probs.append(preds)
        all_true_onehot.append(batch_y_np)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(batch_y_np, axis=1))
    elapsed = time.time() - start

    y_probs        = np.vstack(all_probs)
    y_true_onehot  = np.vstack(all_true_onehot)

    # ── Core Metrics ─────────────────────────────────────────────────────────
    cce = tf.keras.losses.CategoricalCrossentropy()
    test_loss = float(cce(y_true_onehot, y_probs).numpy())
    test_acc  = float(np.mean(np.array(y_pred) == np.array(y_true)))

    report_text = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    report_dict = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
    cm          = confusion_matrix(y_true, y_pred)

    n_params          = model.count_params()
    model_size_mb     = os.path.getsize(args.checkpoint) / (1024 * 1024)
    avg_inference_ms  = (elapsed / max(n_test, 1)) * 1000

    macro_avg = report_dict.get("macro avg", {})
    macro_precision = macro_avg.get("precision", 0.0)
    macro_recall    = macro_avg.get("recall",    0.0)
    macro_f1        = macro_avg.get("f1-score",  0.0)

    weighted_avg = report_dict.get("weighted avg", {})
    weighted_f1  = weighted_avg.get("f1-score",  0.0)

    # ── 3. Learning Curve Diagnosis ───────────────────────────────────────────
    print("[3/5] Diagnosing training learning curve...")
    history_path = args.history
    if history_path is None:
        candidate1 = args.checkpoint.replace("_best.keras", "_history.csv").replace(".keras", "_history.csv")
        candidate2 = os.path.join(WEIGHTS_DIR, f"{run_name.replace('_best', '')}_history.csv")
        if os.path.exists(candidate1):
            history_path = candidate1
        elif os.path.exists(candidate2):
            history_path = candidate2

    learning_curve_diag = None
    lc_png = os.path.join(args.output_dir, f"{run_name}_learning_curve.png")
    if history_path and os.path.exists(history_path):
        print(f"  -> Found history: {history_path}")
        learning_curve_diag = diagnose_and_plot_learning_curve(
            history_path, lc_png,
            test_loss=test_loss,
            test_acc=test_acc,
            model_label=display_name,
        )
    else:
        print("  -> No training history CSV found (skipping Learning Curve).")

    # ── 4. Generate evaluation graphs ────────────────────────────────────────
    if not args.no_plots:
        print("[4/5] Generating evaluation graphs...")

        # 4.1 Confusion Matrix (Counts เท่านั้น)
        cm_path = os.path.join(args.output_dir, f"{run_name}_confusion_matrix.png")
        plot_confusion_matrix_counts(cm, CLASS_NAMES, cm_path, model_label=display_name)

        # 4.2 Per-Class Metrics Bar Chart
        pcm_path = os.path.join(args.output_dir, f"{run_name}_per_class_metrics.png")
        plot_per_class_metrics(report_dict, CLASS_NAMES, pcm_path, model_label=display_name)

        # 4.3 Generalization Comparison (Train vs Val vs Unseen Test)
        if learning_curve_diag and learning_curve_diag.get("fit_status") != "Unknown":
            gen_path = os.path.join(args.output_dir, f"{run_name}_generalization_comparison.png")
            train_m = {
                "accuracy":  learning_curve_diag.get("train_acc_at_best", 0.0),
                "precision": learning_curve_diag.get("train_acc_at_best", 0.0),
                "recall":    learning_curve_diag.get("train_acc_at_best", 0.0),
                "f1":        learning_curve_diag.get("train_acc_at_best", 0.0),
                "loss":      learning_curve_diag.get("train_loss_at_best", 0.0),
            }
            val_m = {
                "accuracy":  learning_curve_diag.get("best_val_acc", 0.0),
                "precision": learning_curve_diag.get("best_val_acc", 0.0),
                "recall":    learning_curve_diag.get("best_val_acc", 0.0),
                "f1":        learning_curve_diag.get("best_val_acc", 0.0),
                "loss":      learning_curve_diag.get("best_val_loss", 0.0),
            }
            test_m = {
                "accuracy":  test_acc,
                "precision": macro_precision,
                "recall":    macro_recall,
                "f1":        macro_f1,
                "loss":      test_loss,
            }
            plot_generalization_comparison(train_m, val_m, test_m, gen_path,
                                           model_label=display_name)
    else:
        print("[4/5] Skipping graph generation (--no_plots specified).")

    # ── 5. Compile & Save Report ─────────────────────────────────────────────
    print("[5/5] Compiling evaluation report...")

    lines = []
    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("\n" + "=" * 68)
    log(f"  MODEL EVALUATION REPORT: [{display_name}]")
    log(f"  อ้างอิง: JIKO Unkhair (CNN Durian Leaf) & JUTIF Unsoed (VGG-16 Transfer)")
    log("=" * 68)
    log(f"Checkpoint       : {args.checkpoint}")
    log(f"Test Set Images  : {n_test}")
    log(f"Model Parameters : {n_params:,}")
    log(f"Model File Size  : {model_size_mb:.2f} MB")
    log(f"Avg Latency/Img  : {avg_inference_ms:.3f} ms")

    log("\n" + "-" * 68)
    log("[A] ACCURACY & CLASSIFICATION METRICS (Unseen Test Set)")
    log("-" * 68)
    log(f"  Overall Accuracy       : {test_acc:.4f}  ({test_acc:.2%})")
    log(f"  Test Loss (CCE)        : {test_loss:.4f}")
    log(f"  Macro  Precision       : {macro_precision:.4f}")
    log(f"  Macro  Recall          : {macro_recall:.4f}")
    log(f"  Macro  F1-Score        : {macro_f1:.4f}")
    log(f"  Weighted F1-Score      : {weighted_f1:.4f}")
    log("\n  Detailed Per-Class Classification Report:")
    log(report_text)

    if learning_curve_diag and learning_curve_diag.get("fit_status") != "Unknown":
        log("\n" + "-" * 68)
        log("[C] LEARNING CURVE DIAGNOSIS")
        log("-" * 68)
        log(f"  Total Epochs Logged    : {learning_curve_diag['total_epochs']}")
        log(f"  Best Model at Epoch    : {learning_curve_diag['best_epoch_step']}")
        log(f"  Train Loss @ Best      : {learning_curve_diag['train_loss_at_best']:.4f}")
        log(f"  Val   Loss @ Best      : {learning_curve_diag['best_val_loss']:.4f}")
        log(f"  Train Acc  @ Best      : {learning_curve_diag['train_acc_at_best']:.2%}")
        log(f"  Val   Acc  @ Best      : {learning_curve_diag['best_val_acc']:.2%}")
        log(f"  Train-Val Acc Gap      : {learning_curve_diag['acc_gap_at_best']:.2%}")
        log(f"  Fit Status             : {learning_curve_diag['fit_status']}")
        log(f"  Explanation            : {learning_curve_diag['fit_explanation']}")

        if learning_curve_diag.get("unseen_status"):
            log("\n" + "-" * 68)
            log("[E] UNSEEN GENERALIZATION ASSESSMENT")
            log("-" * 68)
            log(f"  Validation Acc         : {learning_curve_diag['best_val_acc']:.2%}")
            log(f"  Unseen Test Acc        : {test_acc:.2%}")
            log(f"  Val -> Unseen Gap      : {learning_curve_diag['unseen_acc_gap']:+.2%}")
            log(f"  Generalization Status  : {learning_curve_diag['unseen_status']}")
            log(f"  Explanation            : {learning_curve_diag['unseen_explanation']}")

    log("\n" + "-" * 68)
    log("[D] CONFUSION MATRIX (Unseen Test Set — Raw Counts)")
    log("-" * 68)
    header = f"  {'':18}" + "".join(f"{c:>12}" for c in CLASS_NAMES)
    log(header)
    for i, row_name in enumerate(CLASS_NAMES):
        row_str = f"  {row_name:<18}" + "".join(f"{cm[i, j]:>12}" for j in range(len(CLASS_NAMES)))
        log(row_str)
    log("\n  (Normalized Confusion Matrix saved as PNG)")
    log("=" * 68)

    # Save report .txt
    report_path = os.path.join(args.output_dir, f"{run_name}_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n[Artifacts saved in '{args.output_dir}']")
    print(f"  Report         : {report_path}")
    if not args.no_plots:
        print(f"  Confusion Mat. : {os.path.join(args.output_dir, f'{run_name}_confusion_matrix.png')}")
        print(f"  Per-Class Bar  : {os.path.join(args.output_dir, f'{run_name}_per_class_metrics.png')}")
        if history_path and os.path.exists(history_path):
            print(f"  Learning Curve : {lc_png}")
            print(f"  Generalization : {os.path.join(args.output_dir, f'{run_name}_generalization_comparison.png')}")


if __name__ == "__main__":
    main()
