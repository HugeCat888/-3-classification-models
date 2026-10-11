"""
evaluation/compare_models.py
------------------------------
เปรียบเทียบประสิทธิภาพของโมเดลหลายตัวพร้อมกันบน Unseen Test Set

อ้างอิงงานวิจัย:
  • "Classification of Durian Leaf Images Using CNN" – JIKO Unkhair
    https://ejournal.unkhair.ac.id/index.php/jiko/article/view/8576/5217
  • "VGG-16 Transfer Learning for Durian Variety Classification" – JUTIF Unsoed
    https://jutif.if.unsoed.ac.id/index.php/jurnal/article/view/5506/1253

ผลลัพธ์ที่ได้:
  1. comparison_bar.png          – Grouped Bar Chart (Acc/Prec/Recall/F1/AUC)
  2. multi_learning_curves.png   – Learning Curve ซ้อนทุกโมเดล
  3. multi_confusion_matrices.png – Confusion Matrix ทุกโมเดลในหน้าเดียว
  4. radar_comparison.png        – Radar (Spider) Chart
  5. model_size_latency.png      – Parameters / Size / Latency Comparison
  6. summary_table.png           – Heatmap Summary Table
  7. comparison_report.txt       – ตารางสรุปตัวเลข

ตัวอย่างการรัน (default: ทุกโมเดลทุก augmentation ที่มี weight):
    python -m evaluation.compare_models

ระบุเฉพาะ checkpoint ที่ต้องการ:
    python -m evaluation.compare_models \\
        --runs "EfficientNetB0/None:weights/efficientnet_b0_none_best.keras" \\
               "MobileNetV3/v1:weights/mobilenetv3_large_v1_best.keras" \\
               "ResNet50/v2:weights/resnet50_v2_best.keras"
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

from config import CLASS_NAMES, TEST_DIR, WEIGHTS_DIR
from datasets import build_dataset
from evaluation.plots import (
    plot_model_comparison_bar,
    plot_multi_learning_curves,
    plot_multi_confusion_matrices,
    plot_radar_comparison,
    plot_model_size_latency,
    plot_summary_table,
    PALETTE,
)


# ---------------------------------------------------------------------------
# Default runs: ทุก checkpoint ที่มีอยู่ใน WEIGHTS_DIR
# ---------------------------------------------------------------------------

DEFAULT_RUNS = [
    # (display_label, model_key, checkpoint_path)
    ("EfficientNetB0 / None",   "efficientnet_b0",    "weights/efficientnet_b0_none_best.keras"),
    ("EfficientNetB0 / Aug-v1", "efficientnet_b0",    "weights/efficientnet_b0_v1_best.keras"),
    ("EfficientNetB0 / Aug-v2", "efficientnet_b0",    "weights/efficientnet_b0_v2_best.keras"),
    ("MobileNetV3 / None",      "mobilenetv3_large",  "weights/mobilenetv3_large_none_best.keras"),
    ("MobileNetV3 / Aug-v1",    "mobilenetv3_large",  "weights/mobilenetv3_large_v1_best.keras"),
    ("MobileNetV3 / Aug-v2",    "mobilenetv3_large",  "weights/mobilenetv3_large_v2_best.keras"),
    ("ResNet50 / None",         "resnet50",           "weights/resnet50_none_best.keras"),
    ("ResNet50 / Aug-v1",       "resnet50",           "weights/resnet50_v1_best.keras"),
    ("ResNet50 / Aug-v2",       "resnet50",           "weights/resnet50_v2_best.keras"),
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare multiple durian-leaf classifiers on the Unseen Test Set."
    )
    p.add_argument(
        "--runs", nargs="+", default=None,
        metavar="LABEL:MODELKEY:CKPT",
        help=(
            'Runs to compare, format "Label:model_key:checkpoint_path". '
            'Leave blank to use all 9 default runs.'
        ),
    )
    p.add_argument("--test_dir",    default=TEST_DIR)
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--output_dir",  default="evaluation/reports")
    p.add_argument("--top_n",       type=int, default=None,
                   help="Keep only top-N models by F1-Score for plots (helps readability)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# INFERENCE HELPER
# ---------------------------------------------------------------------------

def run_inference(model, test_ds, n_test):
    """Returns y_probs, y_true_onehot, elapsed_sec"""
    all_probs, all_true = [], []
    start = time.time()
    for batch_x, batch_y in test_ds:
        preds = model.predict(batch_x, verbose=0)
        all_probs.append(preds)
        all_true.append(batch_y.numpy())
    elapsed = time.time() - start
    return np.vstack(all_probs), np.vstack(all_true), elapsed


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Parse runs ────────────────────────────────────────────────────────────
    if args.runs:
        runs = []
        for token in args.runs:
            parts = token.split(":", 2)
            if len(parts) != 3:
                print(f"[WARN] Skipping malformed run spec: {token!r}  (expected 'Label:model_key:ckpt')")
                continue
            runs.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
    else:
        runs = DEFAULT_RUNS

    # ── Filter to existing checkpoints only ────────────────────────────────────
    valid_runs = [(lbl, mkey, ckpt) for lbl, mkey, ckpt in runs if os.path.exists(ckpt)]
    if not valid_runs:
        print("[ERROR] No valid checkpoint files found. Run train_classifier.py first.")
        sys.exit(1)

    print(f"\n{'=' * 68}")
    print(f"  Multi-Model Comparison  ({len(valid_runs)} checkpoints)")
    print(f"  Test Set: {args.test_dir}")
    print(f"{'=' * 68}\n")

    # ── Evaluate each model ──────────────────────────────────────────────────
    model_results = []
    cm_list       = []
    history_paths = []

    for idx, (label, model_key, ckpt_path) in enumerate(valid_runs):
        print(f"[{idx+1}/{len(valid_runs)}] Evaluating: {label}")
        print(f"  Checkpoint: {ckpt_path}")

        model = tf.keras.models.load_model(ckpt_path, compile=False)
        test_ds, n_test = build_dataset(args.test_dir, model_key, args.batch_size, shuffle=False)

        y_probs, y_true_onehot, elapsed = run_inference(model, test_ds, n_test)
        y_pred = list(np.argmax(y_probs, axis=1))
        y_true = list(np.argmax(y_true_onehot, axis=1))

        cce      = tf.keras.losses.CategoricalCrossentropy()
        test_loss = float(cce(y_true_onehot, y_probs).numpy())
        test_acc  = float(np.mean(np.array(y_pred) == np.array(y_true)))
        avg_latency = (elapsed / max(n_test, 1)) * 1000
        n_params    = model.count_params()
        model_size  = os.path.getsize(ckpt_path) / (1024 * 1024)

        rep_dict = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)
        cm_mat   = confusion_matrix(y_true, y_pred)

        macro   = rep_dict.get("macro avg", {})
        macro_p  = macro.get("precision", 0.0)
        macro_r  = macro.get("recall",    0.0)
        macro_f1 = macro.get("f1-score",  0.0)

        # ROC-AUC macro (quick calculation)
        from sklearn.metrics import roc_auc_score
        try:
            auc_macro = float(roc_auc_score(y_true_onehot, y_probs, average="macro", multi_class="ovr"))
        except Exception:
            auc_macro = 0.0

        # mAP
        from sklearn.metrics import average_precision_score
        try:
            mAP = float(np.mean([
                average_precision_score(y_true_onehot[:, i], y_probs[:, i])
                for i in range(len(CLASS_NAMES))
            ]))
        except Exception:
            mAP = 0.0

        result = {
            "label":         label,
            "model_key":     model_key,
            "checkpoint":    ckpt_path,
            "accuracy":      test_acc,
            "precision":     macro_p,
            "recall":        macro_r,
            "f1":            macro_f1,
            "auc_macro":     auc_macro,
            "mAP":           mAP,
            "test_loss":     test_loss,
            "n_params":      n_params,
            "model_size_mb": model_size,
            "latency_ms":    avg_latency,
            "n_test":        n_test,
            "color":         PALETTE[idx % len(PALETTE)],
        }
        model_results.append(result)
        cm_list.append((cm_mat, label))

        # Collect history CSV for overlay curve
        run_base = os.path.splitext(os.path.basename(ckpt_path))[0].replace("_best", "")
        hist_csv = os.path.join(WEIGHTS_DIR, f"{run_base}_history.csv")
        if os.path.exists(hist_csv):
            history_paths.append((hist_csv, label))

        print(f"  -> Acc={test_acc:.4f} | Prec={macro_p:.4f} | Rec={macro_r:.4f} | "
              f"F1={macro_f1:.4f} | AUC={auc_macro:.4f} | mAP={mAP:.4f}\n")

        del model
        tf.keras.backend.clear_session()

    # ── Optionally keep top-N by F1 ──────────────────────────────────────────
    if args.top_n and args.top_n < len(model_results):
        sorted_res = sorted(model_results, key=lambda r: r["f1"], reverse=True)
        top_labels = {r["label"] for r in sorted_res[:args.top_n]}
        model_results_plot = [r for r in model_results if r["label"] in top_labels]
        cm_list_plot = [(c, l) for c, l in cm_list if l in top_labels]
        history_paths_plot = [(p, l) for p, l in history_paths if l in top_labels]
        print(f"[INFO] Plotting top-{args.top_n} models by F1-Score")
    else:
        model_results_plot = model_results
        cm_list_plot       = cm_list
        history_paths_plot = history_paths

    # ── Comparison Plots ─────────────────────────────────────────────────────
    print("\n[Generating comparison plots...]")

    # 7. Grouped Bar Comparison
    bar_path = os.path.join(args.output_dir, "comparison_bar.png")
    plot_model_comparison_bar(
        model_results_plot, bar_path,
        title="Model Comparison — Accuracy / Precision / Recall / F1-Score / AUC (Unseen Test Set)"
    )
    print(f"  Saved: {bar_path}")

    # 8. Multi Learning Curves
    if history_paths_plot:
        lc_path = os.path.join(args.output_dir, "multi_learning_curves.png")
        plot_multi_learning_curves(
            history_paths_plot, lc_path,
            title="Validation Learning Curves — All Models"
        )
        print(f"  Saved: {lc_path}")
    else:
        print("  [SKIP] No history CSV files found — skipping Learning Curve overlay.")

    # 9. Multi Confusion Matrices
    cm_grid_path = os.path.join(args.output_dir, "multi_confusion_matrices.png")
    plot_multi_confusion_matrices(
        cm_list_plot, CLASS_NAMES, cm_grid_path,
        normalized=True,
        title="Normalized Confusion Matrices — All Models (Recall %)"
    )
    print(f"  Saved: {cm_grid_path}")

    # 10. Radar Chart
    radar_path = os.path.join(args.output_dir, "radar_comparison.png")
    plot_radar_comparison(
        model_results_plot, radar_path,
        title="Radar Chart — Multi-Model Performance Comparison"
    )
    print(f"  Saved: {radar_path}")

    # 11. Model Size & Latency
    sl_path = os.path.join(args.output_dir, "model_size_latency.png")
    plot_model_size_latency(
        model_results_plot, sl_path,
        title="Model Efficiency — Parameters, File Size & Inference Latency"
    )
    print(f"  Saved: {sl_path}")

    # 12. Summary Table Heatmap
    tbl_path = os.path.join(args.output_dir, "summary_table.png")
    plot_summary_table(
        model_results_plot, tbl_path,
        title="Performance Summary Table — All Models (Unseen Test Set)"
    )
    print(f"  Saved: {tbl_path}")

    # ── Text Summary Report ───────────────────────────────────────────────────
    report_path = os.path.join(args.output_dir, "comparison_report.txt")
    sorted_by_f1 = sorted(model_results, key=lambda r: r["f1"], reverse=True)

    lines = []
    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("\n" + "=" * 90)
    log("  MULTI-MODEL COMPARISON REPORT — Durian Leaf Classification (Unseen Test Set)")
    log("  อ้างอิง: JIKO Unkhair (CNN Durian Leaf) & JUTIF Unsoed (VGG-16 Transfer Learning)")
    log("=" * 90)

    # Header
    col_w = [26, 9, 9, 9, 10, 9, 8, 12, 12, 9]
    headers = ["Model", "Accuracy", "Precision", "Recall", "F1-Score",
               "AUC(Macro)", "mAP", "Params(M)", "Size(MB)", "Lat.(ms)"]
    log("  " + "  ".join(f"{h:<{w}}" for h, w in zip(headers, col_w)))
    log("  " + "  ".join("-" * w for w in col_w))

    for r in sorted_by_f1:
        row = [
            r["label"],
            f"{r['accuracy']:.4f}",
            f"{r['precision']:.4f}",
            f"{r['recall']:.4f}",
            f"{r['f1']:.4f}",
            f"{r['auc_macro']:.4f}",
            f"{r['mAP']:.4f}",
            f"{r['n_params']/1e6:.2f}M",
            f"{r['model_size_mb']:.1f}",
            f"{r['latency_ms']:.2f}",
        ]
        log("  " + "  ".join(f"{v:<{w}}" for v, w in zip(row, col_w)))

    log("\n  [Best by F1-Score] : " + sorted_by_f1[0]["label"] +
        f" (F1 = {sorted_by_f1[0]['f1']:.4f})")
    log("  [Best by Accuracy] : " +
        max(model_results, key=lambda r: r["accuracy"])["label"] +
        f" (Acc = {max(r['accuracy'] for r in model_results):.4f})")
    log("  [Best by AUC]      : " +
        max(model_results, key=lambda r: r["auc_macro"])["label"] +
        f" (AUC = {max(r['auc_macro'] for r in model_results):.4f})")
    log("  [Lowest Latency]   : " +
        min(model_results, key=lambda r: r["latency_ms"])["label"] +
        f" (Lat = {min(r['latency_ms'] for r in model_results):.2f} ms)")
    log("=" * 90)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  Text report : {report_path}")
    print(f"\n[Done] All comparison artifacts saved in '{args.output_dir}'")


if __name__ == "__main__":
    main()
