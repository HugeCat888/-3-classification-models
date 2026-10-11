"""
app.py
-------
Streamlit application สำหรับ Classification ใบทุเรียน

User Flow:
    Upload JPG/PNG -> Auto-detect architecture from filename
    -> Preprocess -> Classification Model -> Display Top-K Results

รันด้วย:
    streamlit run app.py
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

import augmentation
from config import CLASS_NAMES, CLASS_NAMES_TH, WEIGHTS_DIR
from preprocessing import load_and_preprocess_image

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Durian Leaf Classifier",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Hero header */
    .hero-header {
        background: linear-gradient(135deg, #064e3b 0%, #065f46 40%, #047857 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 8px 32px rgba(6,78,59,0.3);
    }
    .hero-header h1 { font-size: 2rem; font-weight: 700; margin: 0; }
    .hero-header p  { font-size: 0.95rem; opacity: 0.8; margin: 0.4rem 0 0; }

    /* Result card - top prediction */
    .result-card {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border: 2px solid #6ee7b7;
        border-radius: 14px;
        padding: 1.4rem 1.8rem;
        margin-bottom: 1rem;
    }
    .result-card .variety-name {
        font-size: 1.6rem;
        font-weight: 700;
        color: #065f46;
    }
    .result-card .variety-en {
        font-size: 1rem;
        color: #059669;
        margin-top: 0.1rem;
    }

    /* Confidence bar container */
    .conf-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 0.5rem 0;
    }
    .conf-label { font-size: 0.85rem; color: #374151; min-width: 170px; }
    .conf-bar-bg {
        flex: 1;
        height: 10px;
        background: #e5e7eb;
        border-radius: 99px;
        overflow: hidden;
    }
    .conf-bar-fill {
        height: 100%;
        border-radius: 99px;
        transition: width 0.4s ease;
    }
    .conf-pct { font-size: 0.85rem; font-weight: 600; color: #1f2937; min-width: 46px; text-align: right; }

    /* Sidebar styling */
    .sidebar-section {
        background: #f0fdf4;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.8rem;
        border-left: 3px solid #10b981;
    }
    .sidebar-section h4 { margin: 0 0 0.3rem; font-size: 0.82rem; color: #065f46; text-transform: uppercase; letter-spacing: 0.05em; }

    /* Uploader zone */
    [data-testid="stFileUploader"] {
        border: 2px dashed #6ee7b7;
        border-radius: 12px;
        background: #f0fdf4;
        padding: 0.5rem;
    }

    /* Hide default streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
ARCH_KEYWORDS = {
    "efficientnet_b0":   "EfficientNet-B0",
    "mobilenetv3_large": "MobileNet V3 Large",
    "resnet50":          "ResNet-50",
}

CONFIDENCE_COLORS = ["#10b981", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6"]


def detect_arch_from_filename(filename: str) -> str | None:
    """ตรวจ architecture จากชื่อไฟล์อัตโนมัติ"""
    fname = filename.lower()
    for key in ARCH_KEYWORDS:
        if key in fname:
            return key
    return None


@st.cache_resource
def load_classifier(checkpoint_path: str):
    return tf.keras.models.load_model(checkpoint_path)


def confidence_bar_html(label: str, label_en: str, conf: float, color: str, rank: int) -> str:
    pct = conf * 100
    medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][min(rank, 4)]
    return f"""
    <div class="conf-row">
        <span class="conf-label">{medal} {label} <span style="color:#9ca3af;font-size:0.78rem">({label_en})</span></span>
        <div class="conf-bar-bg">
            <div class="conf-bar-fill" style="width:{pct:.1f}%;background:{color};"></div>
        </div>
        <span class="conf-pct">{pct:.1f}%</span>
    </div>
    """


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Hero Header
    st.markdown("""
    <div class="hero-header">
        <h1>🍃 Durian Leaf Classifier</h1>
        <p>Transfer Learning · EfficientNetB0 / MobileNetV3 / ResNet50 · 5 พันธุ์ทุเรียน</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ การตั้งค่า")

        # รายชื่อ checkpoint
        weight_files = sorted([
            f for f in os.listdir(WEIGHTS_DIR) if f.endswith(".keras")
        ]) if os.path.isdir(WEIGHTS_DIR) else []

        if not weight_files:
            st.error("❌ ไม่พบ checkpoint ใน `weights/`\nกรุณารัน `train_classifier.py` ก่อน")
            st.stop()

        st.markdown('<div class="sidebar-section"><h4>📦 Model</h4></div>', unsafe_allow_html=True)

        clf_weight_name = st.selectbox(
            "Checkpoint (.keras)",
            weight_files,
            help="เลือก checkpoint ที่ต้องการใช้",
        )

        # Auto-detect architecture จากชื่อไฟล์
        auto_arch = detect_arch_from_filename(clf_weight_name)
        arch_options = list(ARCH_KEYWORDS.keys())
        default_idx = arch_options.index(auto_arch) if auto_arch else 0

        model_arch = st.selectbox(
            "Architecture",
            arch_options,
            index=default_idx,
            format_func=lambda k: ARCH_KEYWORDS[k],
            help="ตรวจจับจากชื่อไฟล์อัตโนมัติ — เปลี่ยนได้ถ้าชื่อไฟล์ไม่ตรง",
        )

        if auto_arch and auto_arch == model_arch:
            st.success(f"✅ ตรวจพบ: **{ARCH_KEYWORDS[model_arch]}**")
        elif auto_arch and auto_arch != model_arch:
            st.warning("⚠️ Architecture ที่เลือกไม่ตรงกับชื่อไฟล์")

        st.divider()

        st.markdown('<div class="sidebar-section"><h4>🎯 ผลลัพธ์</h4></div>', unsafe_allow_html=True)
        top_k = st.slider("แสดง Top-K Predictions", min_value=1, max_value=5, value=3)
        show_all_probs = st.checkbox("แสดงความน่าจะเป็นทุก class", value=False)

        st.divider()

        # ข้อมูลโมเดล
        ckpt_path = os.path.join(WEIGHTS_DIR, clf_weight_name)
        size_mb = os.path.getsize(ckpt_path) / (1024 * 1024) if os.path.exists(ckpt_path) else 0
        st.caption(f"📁 ขนาดไฟล์: **{size_mb:.1f} MB**")

    # ── โหลดโมเดล ────────────────────────────────────────────────────────────
    classifier = load_classifier(ckpt_path)

    # ── Main layout ───────────────────────────────────────────────────────────
    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("#### 📤 อัปโหลดภาพใบทุเรียน")
        uploaded_file = st.file_uploader(
            "รองรับ JPG, JPEG, PNG",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        if uploaded_file is None:
            st.markdown("""
            <div style="text-align:center;padding:2.5rem 1rem;color:#6b7280;">
                <div style="font-size:3rem;">🍃</div>
                <div style="font-size:0.95rem;margin-top:0.5rem;">ลากและวางภาพ หรือกดเพื่ออัปโหลด</div>
                <div style="font-size:0.8rem;margin-top:0.3rem;color:#9ca3af;">JPG / PNG</div>
            </div>
            """, unsafe_allow_html=True)
            with col_result:
                st.markdown("#### 📊 ผลการจำแนกพันธุ์")
                st.info("รอภาพ...")
            return

        # เปิดภาพ
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception:
            st.error("❌ ไม่สามารถเปิดไฟล์ได้ กรุณาอัปโหลดภาพ JPG/PNG ที่ถูกต้อง")
            return

        st.image(image, caption=f"📷 {uploaded_file.name}", use_container_width=True)

        # ข้อมูลภาพ
        w, h = image.size
        st.caption(f"ขนาดภาพ: {w} × {h} px")

    # ── Inference ─────────────────────────────────────────────────────────────
    with col_result:
        st.markdown("#### 📊 ผลการจำแนกพันธุ์")

        with st.spinner("กำลังวิเคราะห์..."):
            import time
            t0 = time.time()
            input_tensor = load_and_preprocess_image(image, model_arch)
            preds = classifier.predict(input_tensor, verbose=0)[0]
            latency_ms = (time.time() - t0) * 1000

        sorted_indices = np.argsort(preds)[::-1]
        best_idx = int(sorted_indices[0])
        best_name = CLASS_NAMES[best_idx]
        best_name_th = CLASS_NAMES_TH[best_name]
        best_conf = float(preds[best_idx])

        # ── Top prediction card ──
        st.markdown(f"""
        <div class="result-card">
            <div class="variety-name">{best_name_th}</div>
            <div class="variety-en">{best_name}</div>
            <div style="margin-top:0.8rem;font-size:1.1rem;font-weight:600;color:#065f46;">
                Confidence: {best_conf:.1%}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Top-K breakdown ──
        display_count = len(CLASS_NAMES) if show_all_probs else top_k
        st.markdown(f"**Top-{display_count} Predictions**")

        bars_html = ""
        for rank, idx in enumerate(sorted_indices[:display_count]):
            cn = CLASS_NAMES[int(idx)]
            cn_th = CLASS_NAMES_TH[cn]
            conf = float(preds[int(idx)])
            color = CONFIDENCE_COLORS[rank % len(CONFIDENCE_COLORS)]
            bars_html += confidence_bar_html(cn_th, cn, conf, color, rank)

        st.markdown(bars_html, unsafe_allow_html=True)

        st.divider()

        # ── Metadata ──
        meta_col1, meta_col2 = st.columns(2)
        with meta_col1:
            st.metric("⚡ Latency", f"{latency_ms:.0f} ms")
        with meta_col2:
            st.metric("🧠 Model", ARCH_KEYWORDS[model_arch].split()[0])

        # ── Low confidence warning ──
        if best_conf < 0.5:
            st.warning(
                f"⚠️ Confidence ต่ำ ({best_conf:.1%}) — ลองใช้ภาพที่ชัดเจนกว่านี้ "
                "หรือถ่ายใบจากมุมที่เห็นลักษณะของใบชัดเจน"
            )
        elif best_conf < 0.75:
            st.info(f"ℹ️ ผลลัพธ์อยู่ในเกณฑ์ปานกลาง ({best_conf:.1%})")


if __name__ == "__main__":
    main()
