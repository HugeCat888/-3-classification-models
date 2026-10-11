"""
evaluation/plots.py
---------------------
ฟังก์ชันวาดกราฟและวินิจฉัยประสิทธิภาพโมเดล Classification:

  ─── Single-Model Plots ────────────────────────────────────────────────────
  1.  diagnose_and_plot_learning_curve  – Learning Curve (Loss & Acc) + Diagnosis Box
  2.  plot_roc_curves                  – ROC-AUC ราย Class + Micro/Macro Average
  3.  plot_precision_recall_curves     – PR Curves & AP/mAP
  4.  plot_per_class_metrics           – Grouped Bar: Precision / Recall / F1 ราย Class
  5.  plot_confusion_matrices          – Dual Confusion Matrix (Counts + Normalized %)
  6.  plot_generalization_comparison   – Train vs Val vs Unseen Test

  ─── Multi-Model Comparison Plots ──────────────────────────────────────────
  7.  plot_model_comparison_bar        – แท่งเปรียบเทียบ Acc/Precision/Recall/F1/AUC ทุกโมเดล
  8.  plot_multi_learning_curves       – Learning Curve ซ้อนกันหลายโมเดล
  9.  plot_multi_confusion_matrices    – Confusion Matrix ทุกโมเดลในหน้าเดียว
  10. plot_radar_comparison            – Radar Chart เปรียบเทียบหลายโมเดล
  11. plot_model_size_latency          – Bar chart: Model Size & Latency
  12. plot_summary_table               – Heatmap Summary Table

อ้างอิง (References):
  • CNN Classification of Durian Leaf (JIKO Unkhair)
    https://ejournal.unkhair.ac.id/index.php/jiko/article/view/8576/5217
  • VGG-16 Transfer Learning for Durian Variety Classification (JUTIF Unsoed)
    https://jutif.if.unsoed.ac.id/index.php/jurnal/article/view/5506/1253
"""

import os
from typing import Optional, Tuple, Dict, Any, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

# ---------------------------------------------------------------------------
# Design system
# ---------------------------------------------------------------------------
PALETTE = [
    "#2563EB",  # Royal Blue
    "#10B981",  # Emerald Green
    "#F59E0B",  # Amber
    "#EF4444",  # Crimson Red
    "#8B5CF6",  # Purple
    "#06B6D4",  # Cyan
    "#EC4899",  # Pink
]

FONT_TITLE  = dict(fontsize=14, fontweight="bold", color="#0F172A")
FONT_AXIS   = dict(fontsize=11, color="#334155")
FONT_LEGEND = dict(fontsize=10, framealpha=0.92)
GRID_STYLE  = dict(linestyle="--", alpha=0.45, color="#CBD5E1")
BG_FIG = "#F8FAFC"
BG_AX  = "#FFFFFF"


# ===========================================================================
#  HELPERS
# ===========================================================================

def _finalize(fig, path: str, dpi: int = 160):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _annotate_bars(ax, rects, fmt="{:.3f}", offset=3, fontsize=8.5):
    for rect in rects:
        h = rect.get_height()
        ax.annotate(
            fmt.format(h),
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, offset), textcoords="offset points",
            ha="center", va="bottom",
            fontsize=fontsize, fontweight="bold", color="#1E293B",
        )


# ===========================================================================
#  1) LEARNING CURVE + DIAGNOSIS
# ===========================================================================

def clean_history_df(history_csv_path: str) -> Tuple[pd.DataFrame, Optional[int]]:
    """อ่าน CSVLogger output, รองรับ Phase1/Phase2 restart"""
    df = pd.read_csv(history_csv_path)
    if "epoch" not in df.columns:
        df["epoch_step"] = range(1, len(df) + 1)
        return df, None

    epochs = df["epoch"].tolist()
    zero_indices = [i for i, ep in enumerate(epochs) if ep == 0]

    if len(zero_indices) >= 2:
        p1_start = zero_indices[-2]
        p2_start = zero_indices[-1]
        run_df = df.iloc[p1_start:].copy().reset_index(drop=True)
        split_idx = p2_start - p1_start
    elif len(zero_indices) == 1:
        run_df = df.iloc[zero_indices[0]:].copy().reset_index(drop=True)
        split_idx = None
    else:
        run_df = df.copy().reset_index(drop=True)
        split_idx = None

    run_df["epoch_step"] = range(1, len(run_df) + 1)
    return run_df, split_idx


def diagnose_learning_history(
    df: pd.DataFrame,
    split_step: Optional[int] = None,
    test_loss: Optional[float] = None,
    test_acc: Optional[float] = None,
) -> Dict[str, Any]:
    """วินิจฉัย Overfitting / Underfitting / Good Fit จาก training history"""
    val_loss_col   = "val_loss"     if "val_loss"     in df.columns else None
    val_acc_col    = "val_accuracy" if "val_accuracy" in df.columns else None
    train_loss_col = "loss"         if "loss"         in df.columns else None
    train_acc_col  = "accuracy"     if "accuracy"     in df.columns else None

    if not val_loss_col:
        return {"fit_status": "Unknown", "message": "No validation loss in history"}

    best_idx        = int(df[val_loss_col].idxmin())
    best_row        = df.loc[best_idx]
    best_epoch_step = int(best_row["epoch_step"])
    total_epochs    = len(df)

    best_val_loss      = float(best_row[val_loss_col])
    best_val_acc       = float(best_row[val_acc_col])    if val_acc_col    else None
    train_loss_at_best = float(best_row[train_loss_col]) if train_loss_col else None
    train_acc_at_best  = float(best_row[train_acc_col])  if train_acc_col  else None

    final_row        = df.iloc[-1]
    final_val_loss   = float(final_row[val_loss_col])
    final_val_acc    = float(final_row[val_acc_col])    if val_acc_col    else None
    final_train_loss = float(final_row[train_loss_col]) if train_loss_col else None
    final_train_acc  = float(final_row[train_acc_col])  if train_acc_col  else None

    loss_gap_at_best = (best_val_loss - train_loss_at_best) if train_loss_at_best is not None else 0.0
    acc_gap_at_best  = (train_acc_at_best - best_val_acc)   if (train_acc_at_best and best_val_acc) else 0.0

    if best_val_acc is not None and (best_val_acc < 0.70 or (train_acc_at_best and train_acc_at_best < 0.70)):
        fit_status      = "Underfitting (High Bias)"
        fit_explanation = "โมเดลยังเรียนรู้รูปแบบภาพใบได้ไม่เพียงพอ ควรเพิ่มความจุโมเดลหรือจำนวน epoch"
    elif acc_gap_at_best > 0.10 or (final_val_loss - best_val_loss > 0.15):
        fit_status      = "High Overfitting (High Variance)"
        fit_explanation = (
            f"Train Acc ทิ้งห่าง Val Acc ({acc_gap_at_best:.1%}) "
            f"และ Val Loss ดีดตัวสูงขึ้นหลัง Epoch {best_epoch_step}"
        )
    elif acc_gap_at_best > 0.05:
        fit_status      = "Moderate Overfitting"
        fit_explanation = f"Gap เล็กน้อยระหว่าง Train และ Val ({acc_gap_at_best:.1%}) แต่ยังเสถียร"
    else:
        fit_status      = "Good Fit (Low Bias, Low Variance)"
        fit_explanation = f"Train และ Validation ลู่เข้าสอดคล้องกัน Gap ต่ำ ({acc_gap_at_best:.1%})"

    unseen_status = unseen_explanation = unseen_acc_gap = unseen_loss_gap = None
    if test_acc is not None and best_val_acc is not None:
        unseen_acc_gap  = best_val_acc - test_acc
        unseen_loss_gap = (test_loss - best_val_loss) if test_loss is not None else None
        if unseen_acc_gap <= 0.02:
            unseen_status = "Robust Unseen Generalization"
            unseen_explanation = (
                f"Unseen Test ({test_acc:.1%}) ใกล้เคียง/สูงกว่า Validation ({best_val_acc:.1%}) "
                "— generalize ได้ยอดเยี่ยม"
            )
        elif unseen_acc_gap <= 0.06:
            unseen_status = "Acceptable Unseen Generalization"
            unseen_explanation = (
                f"Unseen Test ({test_acc:.1%}) ลดเล็กน้อยจาก Validation ({best_val_acc:.1%}) "
                f"(Δ = {unseen_acc_gap:.1%}) — ปกติสำหรับชุดข้อมูลใหม่"
            )
        else:
            unseen_status = "Significant Unseen Drop"
            unseen_explanation = (
                f"Unseen Test ({test_acc:.1%}) ดรอปลงจาก Validation ({best_val_acc:.1%}) "
                f"(Δ = {unseen_acc_gap:.1%}) — ควรตรวจเช็ก distribution shift"
            )

    return {
        "total_epochs":        total_epochs,
        "best_epoch_step":     best_epoch_step,
        "split_step":          split_step,
        "best_val_loss":       best_val_loss,
        "best_val_acc":        best_val_acc,
        "train_loss_at_best":  train_loss_at_best,
        "train_acc_at_best":   train_acc_at_best,
        "final_train_loss":    final_train_loss,
        "final_val_loss":      final_val_loss,
        "final_train_acc":     final_train_acc,
        "final_val_acc":       final_val_acc,
        "loss_gap_at_best":    loss_gap_at_best,
        "acc_gap_at_best":     acc_gap_at_best,
        "fit_status":          fit_status,
        "fit_explanation":     fit_explanation,
        "test_loss":           test_loss,
        "test_acc":            test_acc,
        "unseen_acc_gap":      unseen_acc_gap,
        "unseen_loss_gap":     unseen_loss_gap,
        "unseen_status":       unseen_status,
        "unseen_explanation":  unseen_explanation,
    }


def diagnose_and_plot_learning_curve(
    history_csv_path: str,
    output_path: str,
    test_loss: Optional[float] = None,
    test_acc: Optional[float] = None,
    model_label: str = "",
) -> Dict[str, Any]:
    """
    Learning Curve (Loss & Accuracy) พร้อม Diagnosis Box
    - Train vs Validation per epoch
    - Phase 1 / Phase 2 boundary line
    - Best Epoch star marker
    - Unseen Test dashed benchmark line
    อ้างอิง: ทั้งสองงานวิจัยใช้ Training/Validation Curve เป็นหลักฐาน Learning
    """
    df, split_step = clean_history_df(history_csv_path)
    diag = diagnose_learning_history(df, split_step=split_step, test_loss=test_loss, test_acc=test_acc)
    x_steps = df["epoch_step"].values
    prefix = f"{model_label} — " if model_label else ""

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), facecolor=BG_FIG)

    # ── Loss ──────────────────────────────────────────────────────────────
    ax_loss = axes[0]
    ax_loss.set_facecolor(BG_AX)
    if "loss" in df.columns:
        ax_loss.plot(x_steps, df["loss"],     label="Train Loss",      color="#2563EB", lw=2.2)
    if "val_loss" in df.columns:
        ax_loss.plot(x_steps, df["val_loss"], label="Validation Loss",  color="#EA580C", lw=2.2)
    if "loss" in df.columns and "val_loss" in df.columns:
        ax_loss.fill_between(x_steps, df["loss"], df["val_loss"],
                             color="#EA580C", alpha=0.08, label="Generalization Gap")

    best_step = diag["best_epoch_step"]
    ax_loss.scatter([best_step], [diag["best_val_loss"]], color="#DC2626", s=120, zorder=6, marker="*",
                    label=f"Best Epoch {best_step} (Val Loss={diag['best_val_loss']:.4f})")
    if test_loss is not None:
        ax_loss.axhline(test_loss, color="#059669", ls="--", lw=1.8,
                        label=f"Unseen Test Loss: {test_loss:.4f}")
    if split_step and split_step < len(x_steps):
        ax_loss.axvline(split_step, color="#64748B", ls=":", lw=1.5, alpha=0.8)
        ylims = ax_loss.get_ylim()
        ax_loss.text(split_step + 0.3, ylims[1] * 0.92,
                     "Fine-tuning Phase", color="#475569", fontsize=9, style="italic")

    ax_loss.set_title(f"{prefix}Learning Curve: Loss", **FONT_TITLE, pad=12)
    ax_loss.set_xlabel("Epoch (Continuous Step)", **FONT_AXIS)
    ax_loss.set_ylabel("Categorical Cross-Entropy Loss", **FONT_AXIS)
    ax_loss.grid(True, **GRID_STYLE)
    ax_loss.legend(loc="upper right", **FONT_LEGEND)

    # ── Accuracy ──────────────────────────────────────────────────────────
    ax_acc = axes[1]
    ax_acc.set_facecolor(BG_AX)
    if "accuracy" in df.columns:
        ax_acc.plot(x_steps, df["accuracy"],     label="Train Accuracy",      color="#2563EB", lw=2.2)
    if "val_accuracy" in df.columns:
        ax_acc.plot(x_steps, df["val_accuracy"], label="Validation Accuracy",  color="#EA580C", lw=2.2)
    if "accuracy" in df.columns and "val_accuracy" in df.columns:
        ax_acc.fill_between(x_steps, df["accuracy"], df["val_accuracy"],
                            color="#2563EB", alpha=0.08, label="Generalization Gap")

    if diag["best_val_acc"] is not None:
        ax_acc.scatter([best_step], [diag["best_val_acc"]], color="#DC2626", s=120, zorder=6, marker="*",
                       label=f"Best Epoch {best_step} (Val Acc={diag['best_val_acc']:.2%})")
    if test_acc is not None:
        ax_acc.axhline(test_acc, color="#059669", ls="--", lw=1.8,
                       label=f"Unseen Test Acc: {test_acc:.2%}")
    if split_step and split_step < len(x_steps):
        ax_acc.axvline(split_step, color="#64748B", ls=":", lw=1.5, alpha=0.8)

    ax_acc.set_title(f"{prefix}Learning Curve: Accuracy", **FONT_TITLE, pad=12)
    ax_acc.set_xlabel("Epoch (Continuous Step)", **FONT_AXIS)
    ax_acc.set_ylabel("Accuracy", **FONT_AXIS)
    ax_acc.set_ylim(-0.02, 1.05)
    ax_acc.grid(True, **GRID_STYLE)
    ax_acc.legend(loc="lower right", **FONT_LEGEND)

    acc_gap = diag.get("acc_gap_at_best") or 0.0
    diag_line = (
        f"Diagnosis: {diag['fit_status']}  |  "
        f"Best Epoch: {best_step}/{diag['total_epochs']}  |  "
        f"Train\u2013Val Acc Gap: {acc_gap:.1%}"
    )
    if diag.get("unseen_status"):
        diag_line += f"  |  Unseen: {diag['unseen_status']}"
    fig.suptitle(diag_line, fontsize=11, fontweight="bold", color="#1E293B", y=1.01)

    _finalize(fig, output_path)
    return diag


# ===========================================================================
#  2) ROC CURVES
# ===========================================================================

def plot_roc_curves(
    y_true_onehot: np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str],
    output_path: str,
    model_label: str = "",
) -> Dict[str, float]:
    """ROC Curves + AUC (One-vs-Rest per class + Micro/Macro)"""
    n_classes = len(class_names)
    fpr: Dict[Any, np.ndarray] = {}
    tpr: Dict[Any, np.ndarray] = {}
    roc_auc: Dict[Any, float]  = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_probs[:, i])
        roc_auc[i] = float(auc(fpr[i], tpr[i]))

    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_onehot.ravel(), y_probs.ravel())
    roc_auc["micro"] = float(auc(fpr["micro"], tpr["micro"]))

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = float(auc(fpr["macro"], tpr["macro"]))

    fig, ax = plt.subplots(figsize=(8, 7), facecolor=BG_FIG)
    ax.set_facecolor(BG_AX)

    ax.plot(fpr["micro"], tpr["micro"],
            label=f"Micro-average (AUC = {roc_auc['micro']:.4f})",
            color="#0F172A", ls=":", lw=2.5)
    ax.plot(fpr["macro"], tpr["macro"],
            label=f"Macro-average (AUC = {roc_auc['macro']:.4f})",
            color="#475569", ls="-.", lw=2.2)
    for i, cls in enumerate(class_names):
        ax.plot(fpr[i], tpr[i],
                label=f"{cls} (AUC = {roc_auc[i]:.4f})",
                color=PALETTE[i % len(PALETTE)], lw=2.0)
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.5, label="Random Baseline (AUC = 0.5)")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    prefix = f"{model_label} — " if model_label else ""
    ax.set_title(f"{prefix}Multi-Class ROC Curves (Unseen Test Set)", **FONT_TITLE, pad=12)
    ax.set_xlabel("False Positive Rate (FPR)", **FONT_AXIS)
    ax.set_ylabel("True Positive Rate (Recall)", **FONT_AXIS)
    ax.grid(True, **GRID_STYLE)
    ax.legend(loc="lower right", **FONT_LEGEND)

    _finalize(fig, output_path)

    results = {class_names[i]: roc_auc[i] for i in range(n_classes)}
    results["micro_avg"] = roc_auc["micro"]
    results["macro_avg"] = roc_auc["macro"]
    return results


# ===========================================================================
#  3) PRECISION-RECALL CURVES
# ===========================================================================

def plot_precision_recall_curves(
    y_true_onehot: np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str],
    output_path: str,
    model_label: str = "",
) -> Dict[str, float]:
    """PR Curves + AP/mAP บน Unseen Test Set"""
    n_classes = len(class_names)
    precision: Dict[Any, np.ndarray] = {}
    recall:    Dict[Any, np.ndarray] = {}
    ap_scores: Dict[Any, float]      = {}

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_onehot[:, i], y_probs[:, i])
        ap_scores[i] = float(average_precision_score(y_true_onehot[:, i], y_probs[:, i]))

    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_true_onehot.ravel(), y_probs.ravel()
    )
    ap_scores["micro"] = float(average_precision_score(y_true_onehot, y_probs, average="micro"))
    mAP = float(np.mean([ap_scores[i] for i in range(n_classes)]))

    fig, ax = plt.subplots(figsize=(8, 7), facecolor=BG_FIG)
    ax.set_facecolor(BG_AX)

    ax.plot(recall["micro"], precision["micro"],
            label=f"Micro-average (AP = {ap_scores['micro']:.4f})",
            color="#0F172A", ls=":", lw=2.5)
    for i, cls in enumerate(class_names):
        ax.plot(recall[i], precision[i],
                label=f"{cls} (AP = {ap_scores[i]:.4f})",
                color=PALETTE[i % len(PALETTE)], lw=2.0)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    prefix = f"{model_label} — " if model_label else ""
    ax.set_title(f"{prefix}Precision-Recall Curves  (mAP = {mAP:.4f})", **FONT_TITLE, pad=12)
    ax.set_xlabel("Recall", **FONT_AXIS)
    ax.set_ylabel("Precision", **FONT_AXIS)
    ax.grid(True, **GRID_STYLE)
    ax.legend(loc="lower left", **FONT_LEGEND)

    _finalize(fig, output_path)

    results = {class_names[i]: ap_scores[i] for i in range(n_classes)}
    results["micro_ap"] = ap_scores["micro"]
    results["mAP"]      = mAP
    return results


# ===========================================================================
#  4) PER-CLASS METRICS BAR CHART
# ===========================================================================

def plot_per_class_metrics(
    report_dict: Dict[str, Any],
    class_names: List[str],
    output_path: str,
    model_label: str = "",
):
    """
    Grouped Bar: Precision / Recall / F1 ราย Class + Macro Avg
    อ้างอิง: งานวิจัยทั้งสองรายงาน per-class metrics เป็น Table หลัก
    """
    cls_list = list(class_names) + ["macro avg"]
    p_vals, r_vals, f1_vals = [], [], []
    for c in cls_list:
        stats = report_dict.get(c, {})
        p_vals.append(stats.get("precision", 0.0))
        r_vals.append(stats.get("recall",    0.0))
        f1_vals.append(stats.get("f1-score", 0.0))

    x = np.arange(len(cls_list))
    w = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(cls_list) * 1.8), 5.8), facecolor=BG_FIG)
    ax.set_facecolor(BG_AX)

    r1 = ax.bar(x - w, p_vals,  w, label="Precision", color="#2563EB", alpha=0.9)
    r2 = ax.bar(x,      r_vals,  w, label="Recall",    color="#10B981", alpha=0.9)
    r3 = ax.bar(x + w,  f1_vals, w, label="F1-Score",  color="#8B5CF6", alpha=0.9)

    for rects in [r1, r2, r3]:
        _annotate_bars(ax, rects, fmt="{:.3f}", fontsize=8)

    ax.set_ylabel("Score", **FONT_AXIS)
    prefix = f"{model_label} — " if model_label else ""
    ax.set_title(f"{prefix}Per-Class Performance (Unseen Test Set)", **FONT_TITLE, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(cls_list, fontsize=10, color="#1E293B", rotation=20, ha="right")
    ax.set_ylim(0, 1.18)
    ax.grid(axis="y", **GRID_STYLE)
    ax.legend(loc="lower right", **FONT_LEGEND)

    _finalize(fig, output_path)


# ===========================================================================
#  5) CONFUSION MATRICES (DUAL)
# ===========================================================================

def plot_confusion_matrices(
    cm: np.ndarray,
    class_names: List[str],
    output_path: str,
    model_label: str = "",
):
    """
    Dual Confusion Matrix:
    - ซ้าย: Raw Counts
    - ขวา: Normalized by True Class (Recall %)
    อ้างอิง: ทั้งสองงานวิจัยแสดง Confusion Matrix เพื่อระบุคลาสที่สับสน
    """
    cm_norm = cm.astype("float") / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1e-9)
    prefix  = f"{model_label} — " if model_label else ""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), facecolor=BG_FIG)

    panels = [
        (axes[0], cm,      "Blues", f"{prefix}Confusion Matrix (Counts)",
         lambda v: f"{int(v)}", cm.max() / 2),
        (axes[1], cm_norm, "Blues", f"{prefix}Confusion Matrix (Normalized Recall %)",
         lambda v: f"{v:.1%}", 0.5),
    ]
    for ax, data, cmap, title, fmt_fn, threshold in panels:
        ax.set_facecolor(BG_AX)
        vmax = 1.0 if data is cm_norm else data.max()
        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=9.5)
        ax.set_yticklabels(class_names, fontsize=9.5)
        ax.set_xlabel("Predicted Label", **FONT_AXIS)
        ax.set_ylabel("True Label", **FONT_AXIS)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10, color="#0F172A")
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                ax.text(j, i, fmt_fn(v), ha="center", va="center",
                        color="white" if v > threshold else "#0F172A",
                        fontweight="bold", fontsize=9.5)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _finalize(fig, output_path)


def plot_confusion_matrix_counts(
    cm: np.ndarray,
    class_names: List[str],
    output_path: str,
    model_label: str = "",
):
    """
    Confusion Matrix แบบ single panel (Counts เท่านั้น)
    ใช้เมื่อไม่ต้องการ Normalized version
    อ้างอิง: ทั้งสองงานวิจัยแสดง Confusion Matrix เพื่อระบุคลาสที่สับสน
    """
    prefix = f"{model_label} — " if model_label else ""

    fig, ax = plt.subplots(figsize=(7, 6), facecolor=BG_FIG)
    ax.set_facecolor(BG_AX)
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    ax.set_xlabel("Predicted Label", **FONT_AXIS)
    ax.set_ylabel("True Label", **FONT_AXIS)
    ax.set_title(f"{prefix}Confusion Matrix (Counts)", fontsize=13,
                 fontweight="bold", pad=12, color="#0F172A")

    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            ax.text(j, i, f"{int(v)}", ha="center", va="center",
                    color="white" if v > threshold else "#0F172A",
                    fontweight="bold", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _finalize(fig, output_path)


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], output_path: str):
    """Backward-compatibility alias"""
    plot_confusion_matrix_counts(cm, class_names, output_path)


# ===========================================================================
#  6) GENERALIZATION COMPARISON (Train vs Val vs Unseen Test)
# ===========================================================================

def plot_generalization_comparison(
    train_metrics: Dict[str, float],
    val_metrics:   Dict[str, float],
    test_metrics:  Dict[str, float],
    output_path:   str,
    model_label:   str = "",
):
    """
    เปรียบเทียบ Train (Best Epoch) vs Validation (Best) vs Unseen Test
    รายงาน: Accuracy, Precision, Recall, F1-Score, Loss
    """
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "Loss"]
    keys          = ["accuracy", "precision", "recall", "f1", "loss"]

    def _vals(m):
        return [m.get(k, 0.0) for k in keys]

    train_vals = _vals(train_metrics)
    val_vals   = _vals(val_metrics)
    test_vals  = _vals(test_metrics)

    x = np.arange(len(metric_labels))
    w = 0.24

    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor=BG_FIG)
    ax.set_facecolor(BG_AX)

    r1 = ax.bar(x - w, train_vals, w, label="Train (Best Epoch)",      color="#3B82F6", alpha=0.9)
    r2 = ax.bar(x,      val_vals,   w, label="Validation (Best Epoch)", color="#F59E0B", alpha=0.9)
    r3 = ax.bar(x + w,  test_vals,  w, label="Unseen Test Set",         color="#10B981", alpha=0.9)

    for rg in [r1, r2, r3]:
        _annotate_bars(ax, rg, fmt="{:.3f}", fontsize=8.5)

    ax.set_ylabel("Value", **FONT_AXIS)
    prefix = f"{model_label} — " if model_label else ""
    ax.set_title(f"{prefix}Generalization: Train vs Validation vs Unseen Test", **FONT_TITLE, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10.5, color="#1E293B")
    ymax = max(max(train_vals), max(val_vals), max(test_vals))
    ax.set_ylim(0, ymax * 1.28)
    ax.grid(axis="y", **GRID_STYLE)
    ax.legend(loc="upper right", **FONT_LEGEND)

    _finalize(fig, output_path)


# ===========================================================================
#  7) MULTI-MODEL COMPARISON BAR CHART
# ===========================================================================

def plot_model_comparison_bar(
    model_results: List[Dict[str, Any]],
    output_path: str,
    title: str = "Model Comparison — Unseen Test Set",
):
    """
    Grouped Bar: Accuracy / Precision / Recall / F1-Score / AUC(Macro) ทุกโมเดล

    model_results: list of dict, แต่ละ dict ต้องมี:
        "label", "accuracy", "precision", "recall", "f1", "auc_macro"
    อ้างอิง: ทั้งสองงานวิจัยเปรียบเทียบโมเดลผ่าน metric table
    """
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC (Macro)"]
    keys    = ["accuracy", "precision", "recall", "f1", "auc_macro"]

    n_models  = len(model_results)
    n_metrics = len(metrics)
    x         = np.arange(n_metrics)
    total_w   = 0.80
    bar_w     = total_w / n_models
    offsets   = np.linspace(-(total_w - bar_w) / 2, (total_w - bar_w) / 2, n_models)

    fig, ax = plt.subplots(figsize=(13, 6), facecolor=BG_FIG)
    ax.set_facecolor(BG_AX)

    for idx, res in enumerate(model_results):
        color = res.get("color", PALETTE[idx % len(PALETTE)])
        vals  = [res.get(k, 0.0) for k in keys]
        bars  = ax.bar(x + offsets[idx], vals, bar_w,
                       label=res["label"], color=color, alpha=0.88,
                       edgecolor="white", linewidth=0.6)
        _annotate_bars(ax, bars, fmt="{:.3f}", fontsize=7.5)

    ax.set_ylabel("Score", **FONT_AXIS)
    ax.set_title(title, **FONT_TITLE, pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11, color="#1E293B")
    ax.set_ylim(0, 1.22)
    ax.grid(axis="y", **GRID_STYLE)
    ax.legend(loc="lower right", **FONT_LEGEND, ncol=min(n_models, 3))

    _finalize(fig, output_path)


# ===========================================================================
#  8) MULTI-MODEL LEARNING CURVES (OVERLAY)
# ===========================================================================

def plot_multi_learning_curves(
    history_paths: List[Tuple[str, str]],
    output_path: str,
    title: str = "Learning Curves — All Models",
):
    """
    Validation Loss + Accuracy overlay ของหลายโมเดล

    history_paths: list of (csv_path, label)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG_FIG)

    for ax_idx, (ax, y_col, ylabel, title_suffix, loc) in enumerate([
        (axes[0], "val_loss",     "Validation Loss",     "Loss",     "upper right"),
        (axes[1], "val_accuracy", "Validation Accuracy", "Accuracy", "lower right"),
    ]):
        ax.set_facecolor(BG_AX)
        ax.set_xlabel("Epoch (Continuous Step)", **FONT_AXIS)
        ax.set_ylabel(ylabel, **FONT_AXIS)
        ax.set_title(f"{title} — {title_suffix}", **FONT_TITLE, pad=12)
        ax.grid(True, **GRID_STYLE)

        for i, (csv_path, label) in enumerate(history_paths):
            if not os.path.exists(csv_path):
                continue
            df, _ = clean_history_df(csv_path)
            if y_col not in df.columns:
                continue
            color = PALETTE[i % len(PALETTE)]
            ax.plot(df["epoch_step"], df[y_col], label=label, color=color, lw=2.0)

            best_idx = int(df[y_col].idxmin()) if y_col == "val_loss" else int(df[y_col].idxmax())
            best_val = float(df[y_col].iloc[best_idx])
            best_ep  = int(df["epoch_step"].iloc[best_idx])
            ax.scatter([best_ep], [best_val], color=color, s=90, zorder=6, marker="*")

        if ax_idx == 1:
            ax.set_ylim(-0.02, 1.05)
        ax.legend(loc=loc, **FONT_LEGEND)

    _finalize(fig, output_path)


# ===========================================================================
#  9) MULTI-MODEL CONFUSION MATRICES (GRID)
# ===========================================================================

def plot_multi_confusion_matrices(
    cm_list: List[Tuple[np.ndarray, str]],
    class_names: List[str],
    output_path: str,
    normalized: bool = True,
    title: str = "Confusion Matrices — All Models",
):
    """
    Confusion Matrix ของทุกโมเดลในหน้าเดียว (grid)

    cm_list: list of (confusion_matrix_array, model_label)
    """
    n    = len(cm_list)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 5.5 * rows),
                             facecolor=BG_FIG, squeeze=False)

    for idx, (cm, label) in enumerate(cm_list):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        ax.set_facecolor(BG_AX)
        data = cm.astype("float") / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1e-9) if normalized else cm
        vmax = 1.0 if normalized else data.max()
        im = ax.imshow(data, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=8.5)
        ax.set_yticklabels(class_names, fontsize=8.5)
        ax.set_xlabel("Predicted", fontsize=9.5, color="#334155")
        ax.set_ylabel("True",      fontsize=9.5, color="#334155")
        ax.set_title(label, fontsize=12, fontweight="bold", pad=8, color="#0F172A")
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                ax.text(j, i, f"{v:.1%}" if normalized else f"{int(v)}",
                        ha="center", va="center",
                        color="white" if v > 0.5 else "#0F172A",
                        fontweight="bold", fontsize=8.5)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold", color="#0F172A", y=1.01)
    _finalize(fig, output_path)


# ===========================================================================
#  10) RADAR CHART COMPARISON
# ===========================================================================

def plot_radar_comparison(
    model_results: List[Dict[str, Any]],
    output_path: str,
    title: str = "Radar Chart — Multi-Model Comparison",
):
    """
    Spider/Radar Chart: Accuracy, Precision, Recall, F1, AUC(Macro)
    ช่วยให้เห็น trade-off ระหว่างโมเดลได้ชัดเจน
    """
    categories = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC\n(Macro)"]
    keys       = ["accuracy", "precision", "recall", "f1", "auc_macro"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), facecolor=BG_FIG)
    ax.set_facecolor(BG_AX)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, color="#0F172A")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="#64748B")
    ax.grid(color="#CBD5E1", linestyle="--", alpha=0.6)

    for idx, res in enumerate(model_results):
        color = res.get("color", PALETTE[idx % len(PALETTE)])
        vals  = [res.get(k, 0.0) for k in keys]
        vals += vals[:1]
        ax.plot(angles, vals, color=color, lw=2.0, label=res["label"])
        ax.fill(angles, vals, color=color, alpha=0.12)

    ax.set_title(title, fontsize=13, fontweight="bold", color="#0F172A", pad=22)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), **FONT_LEGEND)

    _finalize(fig, output_path)


# ===========================================================================
#  11) MODEL SIZE & LATENCY BAR CHART
# ===========================================================================

def plot_model_size_latency(
    model_results: List[Dict[str, Any]],
    output_path: str,
    title: str = "Model Efficiency — Parameters, Size & Latency",
):
    """
    Parameters (M) / Model Size (MB) / Avg Latency (ms/image) สำหรับทุกโมเดล
    อ้างอิง: งานวิจัยทั้งสองรายงาน model complexity เป็นส่วนหนึ่งของผลการทดลอง
    """
    labels   = [r["label"]           for r in model_results]
    params_m = [r.get("n_params", 0) / 1e6        for r in model_results]
    sizes_mb = [r.get("model_size_mb", 0.0)        for r in model_results]
    latencies= [r.get("latency_ms", 0.0)           for r in model_results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), facecolor=BG_FIG)

    for ax, vals, ylabel, bar_title, color in [
        (axes[0], params_m,  "Millions",   "Trainable Parameters (M)", "#2563EB"),
        (axes[1], sizes_mb,  "MB",         "Model File Size (MB)",     "#10B981"),
        (axes[2], latencies, "ms / image", "Avg. Inference Latency",   "#F59E0B"),
    ]:
        ax.set_facecolor(BG_AX)
        x = np.arange(len(labels))
        bars = ax.bar(x, vals, color=color, alpha=0.88, edgecolor="white")
        _annotate_bars(ax, bars, fmt="{:.2f}", fontsize=9)
        ax.set_ylabel(ylabel, **FONT_AXIS)
        ax.set_title(bar_title, fontsize=12, fontweight="bold", pad=10, color="#0F172A")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, color="#1E293B", rotation=20, ha="right")
        ax.grid(axis="y", **GRID_STYLE)
        ymax = max(vals) if max(vals) > 0 else 1
        ax.set_ylim(0, ymax * 1.30)

    fig.suptitle(title, fontsize=14, fontweight="bold", color="#0F172A")
    _finalize(fig, output_path)


# ===========================================================================
#  12) SUMMARY TABLE HEATMAP
# ===========================================================================

def plot_summary_table(
    model_results: List[Dict[str, Any]],
    output_path: str,
    title: str = "Performance Summary Table — All Models",
):
    """
    Heatmap-table สรุปผลลัพธ์ทุกโมเดล: Accuracy / Precision / Recall / F1 / AUC / mAP
    อ้างอิง: ทั้งสองงานวิจัยนำเสนอผลในรูปแบบตารางเปรียบเทียบ
    """
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC\n(Macro)", "mAP"]
    keys    = ["accuracy", "precision", "recall", "f1", "auc_macro", "mAP"]
    labels  = [r["label"] for r in model_results]
    data    = np.array([[r.get(k, 0.0) for k in keys] for r in model_results])

    fig, ax = plt.subplots(
        figsize=(len(metrics) * 1.8, len(labels) * 0.9 + 1.5),
        facecolor=BG_FIG,
    )
    ax.set_facecolor(BG_AX)
    im = ax.imshow(data, cmap="YlGn", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(metrics)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(metrics, fontsize=11, fontweight="bold", color="#0F172A")
    ax.set_yticklabels(labels,  fontsize=10, color="#1E293B")
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    for i in range(len(labels)):
        for j in range(len(metrics)):
            v = data[i, j]
            ax.text(j, i, f"{v:.4f}", ha="center", va="center",
                    fontsize=9.5, fontweight="bold",
                    color="white" if v > 0.82 else "#0F172A")

    ax.set_title(title, fontsize=13, fontweight="bold", color="#0F172A", pad=14)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    _finalize(fig, output_path)


# ===========================================================================
#  BACKWARD-COMPAT ALIASES
# ===========================================================================

def plot_training_curves(history_csv_path: str, output_path: str):
    diagnose_and_plot_learning_curve(history_csv_path, output_path)
