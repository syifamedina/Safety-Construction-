import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import tempfile 
import os

# Page Config
st.set_page_config(
    page_title="Construction Safety Detection",
    page_icon="👷‍♂️",
    layout="wide"
)

# Load Model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "models", "best.pt")
    return YOLO(model_path)

model = load_model()

# Warna per Class
CLASS_COLORS = {
    "helmet"     : (0, 200, 0),   # Hijau
    "no-helmet"  : (0, 0, 220),   # Merah
    "vest"       : (0, 165, 255), # Oranye
    "no-vest"    : (0, 0, 180),   # Merah Gelap
    "person"     : (255, 200, 0), # Biru Muda
}

# Fungsi Deteksi
def detect(image: Image.Image, conf_threshold: float):
    img_array = np.array(image)
    img_bgr   = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    results   = model(img_bgr, conf=conf_threshold)[0]

    counts    = {"person": 0, "helmet": 0, "no-helmet": 0, "vest": 0, "no-vest": 0}

    for box in results.boxes:
        cls_id         = int(box.cls[0])
        cls_name       = model.names[cls_id]
        conf           = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        color = CLASS_COLORS.get(cls_name, (200, 200, 200))
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} {conf:.0%}"
        cv2.putText(img_bgr, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        
        if cls_name in counts:
            counts[cls_name] += 1

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb, counts
    
# Fungsi Analisis APD
def apd_analysis(counts: dict):
    total_person = counts["person"]
    no_helmet    = counts["no-helmet"]
    no_vest      = counts["no-vest"]
    not_safe     = max(no_helmet, no_vest)    # Estimasi pekerja tidak aman

    return {
        "total_person" : total_person,
        "helmet"       : counts["helmet"],
        "no_helmet"    : no_helmet,
        "vest"         : counts["vest"],
        "no_vest"      : no_vest,
        "not_safe"     : not_safe,
        "safe"         : total_person - not_safe if total_person > 0 else 0,
    }

# UI 
st.title("👷‍♂️ Construction Safety Detection")
st.markdown("Deteksi Kelengkapan APD Pekerja Konstruksi secara otomatis menggunakan YOLOv8.")
st.divider()

col_upload, col_setting = st.columns ([3, 1])

with col_setting:
    st.subheader("⚙️ Settings")
    conf = st.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.05)

with col_upload: 
    st.subheader("📤Upload Gambar")
    uploaded = st.file_uploader("Pilih gambar (JPG / PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    col_ori, col_hasil = st.columns(2)
 
    with col_ori:
        st.markdown("**Gambar Original**")
        st.image(image, use_container_width=True)
 
    with st.spinner("🔍 Mendeteksi objek..."):
        result_img, counts = detect(image, conf)
        analysis = apd_analysis(counts)
 
    with col_hasil:
        st.markdown("**Hasil Deteksi**")
        st.image(result_img, use_container_width=True)
 
    st.divider()
    st.subheader("📊 Hasil Analisis APD")
 
    # Metric cards
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("👷 Total Pekerja",  analysis["total_person"])
    m2.metric("✅ Pakai Helm",     analysis["helmet"])
    m3.metric("❌ Tanpa Helm",     analysis["no_helmet"])
    m4.metric("✅ Pakai Rompi",    analysis["vest"])
    m5.metric("❌ Tanpa Rompi",    analysis["no_vest"])
 
    st.divider()
 
    # Summary
    total  = analysis["total_person"]
    unsafe = analysis["not_safe"]
    safe   = analysis["safe"]
 
    if total == 0:
        st.info("ℹ️ Tidak ada pekerja terdeteksi dalam gambar.")
    else:
        pct_safe = (safe / total * 100) if total > 0 else 0
 
        if unsafe == 0:
            st.success(f"✅ Semua **{total} pekerja** terdeteksi lengkap APD-nya!")
        else:
            st.error(f"⚠️ **{unsafe} dari {total} pekerja** tidak lengkap APD-nya!")
 
        st.progress(int(pct_safe), text=f"Tingkat Kepatuhan APD: {pct_safe:.0f}%")
 
        # Tabel detail
        st.markdown("#### 📋 Detail Deteksi")
        detail = {
            "Class"         : ["Person", "Helmet ✅", "No-Helmet ❌", "Vest ✅", "No-Vest ❌"],
            "Jumlah Deteksi": [counts["person"], counts["helmet"],
                               counts["no-helmet"], counts["vest"], counts["no-vest"]],
        }
        st.table(detail)
 
else:
    st.info("👆 Silakan upload gambar pekerja konstruksi untuk memulai deteksi.")