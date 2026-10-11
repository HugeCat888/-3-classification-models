# Durian Leaf Classification

โปรเจ็คนี้ใช้ **Transfer Learning** จำแนกพันธุ์ทุเรียนจากภาพถ่ายใบ  
รองรับ 3 สถาปัตยกรรม: `EfficientNetB0` · `MobileNetV3 Large` · `ResNet50`

| พันธุ์ | ชื่อ Class |
|--------|-----------|
| หมอนทอง | `Monthong` |
| ชะนี | `Chanee` |
| ก้านยาว | `Kanyao` |
| หนามดำ | `BlackThorn` |
| มูซานคิง | `MusangKing` |

---

## 📁 โครงสร้างโปรเจ็ค

```
durian_leaf_project/
├── data/
│   └── classification/
│       ├── train/          # ชุดฝึกสอน
│       ├── valid/          # ชุด validation (รองรับชื่อ valid/ หรือ validation/)
│       └── test/           # ชุดทดสอบ (Unseen Test Set)
├── models/                 # สถาปัตยกรรมโมเดล
│   ├── __init__.py
│   ├── base.py
│   ├── efficientnet_b0.py
│   ├── mobilenetv3_large.py
│   └── resnet50.py
├── evaluation/
│   ├── evaluate_classifier.py
│   ├── plots.py
│   └── reports/            # ผลลัพธ์ (.png, .txt) บันทึกที่นี่
├── weights/                # checkpoint (.keras) และ training history (.csv)
├── app.py                  # Streamlit Web App
├── train_classifier.py     # สคริปต์เทรน
├── augmentation.py         # Data Augmentation (Noise + Blur + Geometric)
├── datasets.py             # tf.data pipeline + Oversampling
├── preprocessing.py        # Resize + Padding + Model-specific preprocess
├── config.py               # ค่ากลางทั้งโปรเจ็ค
└── requirements.txt
```

---

## 🚀 เริ่มต้นใช้งาน

### ขั้นตอนที่ 1 — ติดตั้ง Dependencies

```bash
# สร้าง virtual environment
python -m venv venv

# เปิดใช้งาน (Windows)
venv\Scripts\activate

# ติดตั้ง packages
pip install -r requirements.txt
```

---

### ขั้นตอนที่ 2 — เตรียม Dataset

จัดโครงสร้างโฟลเดอร์ `data/classification/` ดังนี้:

```
data/classification/
├── train/
│   ├── Monthong/        ← ภาพหมอนทอง
│   ├── Chanee/          ← ภาพชะนี
│   ├── Kanyao/          ← ภาพก้านยาว
│   ├── BlackThorn/      ← ภาพหนามดำ
│   └── MusangKing/      ← ภาพมูซานคิง
├── validation/
│   └── <โครงสร้างเดียวกัน>
└── test/
    └── <โครงสร้างเดียวกัน>
```

> **หมายเหตุ:** วางภาพโดยตรงในโฟลเดอร์ตาม class — ไม่ต้องทำ crop ใดๆ

---

### ขั้นตอนที่ 3 — เทรนโมเดล

#### คำสั่งพื้นฐาน

```bash
python train_classifier.py --model <MODEL_NAME> --augmentation <LEVEL>
```

**`--model`** เลือกสถาปัตยกรรม:

| ค่า | โมเดล |
|-----|-------|
| `efficientnet_b0` | EfficientNet-B0 |
| `mobilenetv3_large` | MobileNet V3 Large |
| `resnet50` | ResNet-50 |

**`--augmentation`** ระดับ Data Augmentation:

| ค่า | รายละเอียด |
|-----|-----------|
| `none` | ไม่มี augmentation (Baseline) |
| `v1` | Flip · Rotation ±3% · Zoom 5% · Translation 5% · **Gaussian Noise** (σ=6) · **Gaussian Blur** (p=30%) |
| `v2` | Flip · Rotation ±6% · Zoom 10% · Translation 8% · Brightness 10% · Contrast 10% · **Gaussian Noise** (σ=12) · **Gaussian Blur** (p=50%) |

> **Gaussian Noise** — สุ่มบวก noise เข้าทุก pixel จำลอง sensor noise และ JPEG artifact  
> **Gaussian Blur** — ใช้ depthwise convolution เพื่อ blur ภาพแบบ stochastic ช่วยให้โมเดลโฟกัสที่ feature ของใบ (สี เส้นใบ เนื้อผิว) แทน background texture

#### ตัวอย่างการเทรน

```bash
# EfficientNetB0 (Baseline)
python train_classifier.py --model efficientnet_b0 --augmentation none

# ResNet50 + Augmentation ระดับกลาง (พร้อม Noise/Blur)
python train_classifier.py --model resnet50 --augmentation v2

# MobileNetV3 พร้อมปรับ epochs
python train_classifier.py --model mobilenetv3_large --augmentation v1 \
    --epochs_head 20 --epochs_finetune 20

# เปิด Oversampling อัตโนมัติ (auto-balance ตาม class ที่มากสุด)
python train_classifier.py --model resnet50 --augmentation v2 --oversample

# ระบุจำนวนภาพต่อ class หลัง oversample เอง
python train_classifier.py --model efficientnet_b0 --augmentation v1 \
    --oversample --oversample_target 300
```

#### Arguments เพิ่มเติม

| Argument | Default | คำอธิบาย |
|----------|---------|----------|
| `--epochs_head` | `15` | จำนวน epoch ของ Phase 1 (Frozen base) |
| `--epochs_finetune` | `15` | จำนวน epoch ของ Phase 2 (Fine-tuning) |
| `--lr_head` | `1e-3` | Learning rate Phase 1 |
| `--lr_finetune` | `1e-5` | Learning rate Phase 2 |
| `--batch_size` | `16` | Batch size |
| `--dropout` | `0.3` | Dropout rate ของ Classification head |
| `--patience` | `8` | Early stopping patience |
| `--unfreeze_layers` | `30` | จำนวน layer สุดท้ายที่ unfreeze ใน Phase 2 |
| `--oversample` | `False` | เปิด Oversampling บน train set (flag) |
| `--oversample_target` | `None` | จำนวนภาพต่อ class หลัง oversample (None = auto ตาม class ที่มากสุด) |

#### ผลลัพธ์ที่ได้

ไฟล์จะบันทึกในโฟลเดอร์ `weights/`:

```
weights/
├── resnet50_v2_best.keras     ← Best checkpoint (ตาม val_loss)
└── resnet50_v2_history.csv    ← Training log (ใช้สำหรับ plot learning curve)
```

---

### ขั้นตอนที่ 4 — ประเมินผล

```bash
python -m evaluation.evaluate_classifier \
    --model <MODEL_NAME> \
    --checkpoint weights/<checkpoint>.keras
```

#### ตัวอย่าง

```bash
# ประเมิน ResNet50 + Augmentation v2
python -m evaluation.evaluate_classifier \
    --model resnet50 \
    --checkpoint weights/resnet50_v2_best.keras

# ระบุ test set และ output folder เอง
python -m evaluation.evaluate_classifier \
    --model efficientnet_b0 \
    --checkpoint weights/efficientnet_b0_v1_best.keras \
    --test_dir data/classification/test \
    --output_dir evaluation/reports

# ข้ามการสร้างกราฟ (เฉพาะตัวเลข metrics)
python -m evaluation.evaluate_classifier \
    --model mobilenetv3_large \
    --checkpoint weights/mobilenetv3_large_none_best.keras \
    --no_plots
```

#### ผลลัพธ์ที่บันทึกใน `evaluation/reports/`

| ไฟล์ | คำอธิบาย |
|------|----------|
| `*_report.txt` | รายงานสรุป metrics, latency, diagnosis ครบถ้วน |
| `*_learning_curve.png` | Learning curve พร้อม diagnosis overfitting/underfitting + Unseen benchmark |
| `*_confusion_matrix.png` | Dual confusion matrix (Raw Counts + Normalized Recall %) |
| `*_roc_curve.png` | ROC Curve ราย class + Micro/Macro average AUC |
| `*_pr_curve.png` | Precision-Recall Curve + Mean Average Precision (mAP) |
| `*_per_class_metrics.png` | Bar chart เปรียบเทียบ Precision / Recall / F1 ราย class |
| `*_generalization_comparison.png` | Train vs Validation vs Unseen Test comparison |

---

### ขั้นตอนที่ 5 — รัน Web App

```bash
streamlit run app.py
```

เปิดเบราว์เซอร์แล้วไปที่ `http://localhost:8501`

**วิธีใช้งาน Web App:**
1. เลือก **Classifier weights** (ไฟล์ `.keras` ใน `weights/`)
2. เลือก **สถาปัตยกรรม** ให้ตรงกับ checkpoint ที่เลือก
3. ปรับจำนวน **Top-K** ผลลัพธ์ที่ต้องการแสดง
4. กด **Browse files** เพื่ออัปโหลดภาพใบทุเรียน (JPG/PNG)
5. ดูผลการจำแนกพันธุ์พร้อม Confidence score

---

## ⚙️ ค่า Default (config.py)

```python
CLASSIFIER_DEFAULTS = dict(
    batch_size=16,
    epochs_head=15,
    epochs_finetune=15,
    lr_head=1e-3,
    lr_finetune=1e-5,
    early_stopping_patience=8,
    unfreeze_last_n_layers=30,
)
```
