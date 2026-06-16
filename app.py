import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import gdown

# ─── Config ───────────────────────────────────────────────────────────────────
CLASS_NAMES = ["Jahe", "Kencur", "Kunyit", "Lengkuas", "Temulawak"]
IMG_SIZE = (224, 224)

# Ganti dengan File ID Google Drive milik kamu setelah upload model
# Cara: upload best_weight_jamu.weights.h5 ke Google Drive → Share → Anyone with link → salin ID dari URL
GDRIVE_FILE_ID = "YOUR_GDRIVE_FILE_ID_HERE"
WEIGHTS_PATH = "model_tersimpan/best_weight_jamu.weights.h5"

st.set_page_config(
    page_title="Klasifikasi Rimpang Jamu Madura",
    page_icon="🌿",
    layout="centered"
)

# ─── Style ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@400;500&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #1a1a18;
    color: #e8e0d0;
}
h1, h2, h3 {
    font-family: 'Playfair Display', serif;
    color: #d4a843;
}
p, div, label, span {
    font-family: 'Inter', sans-serif;
}
[data-testid="stFileUploader"] {
    background: #2a2a24;
    border: 2px dashed #d4a843;
    border-radius: 12px;
    padding: 1rem;
}
.result-box {
    background: #2a2a24;
    border-left: 4px solid #d4a843;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin-top: 1rem;
}
.confidence-bar-bg {
    background: #3a3a30;
    border-radius: 99px;
    height: 10px;
    margin-top: 4px;
    margin-bottom: 12px;
}
.confidence-bar {
    height: 10px;
    border-radius: 99px;
    background: linear-gradient(90deg, #d4a843, #a87c1e);
}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─── Build model architecture (HARUS sama persis seperti saat training) ───────
@st.cache_resource
def load_model():
    # Download weights dari Google Drive kalau belum ada
    if not os.path.exists(WEIGHTS_PATH):
        os.makedirs("model_tersimpan", exist_ok=True)
        if GDRIVE_FILE_ID == "YOUR_GDRIVE_FILE_ID_HERE":
            st.error("⚠️ Harap ganti GDRIVE_FILE_ID di app.py dengan File ID model kamu.")
            st.stop()
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, WEIGHTS_PATH, quiet=False)

    # Bangun ulang arsitektur (harus identik dengan notebook)
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.3),
        tf.keras.layers.RandomZoom(0.15)
    ])

    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(224, 224, 3), include_top=False, weights=None
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    model.load_weights(WEIGHTS_PATH)
    return model

# ─── Prediction helper ────────────────────────────────────────────────────────
def predict(img: Image.Image, model) -> tuple[str, float, np.ndarray]:
    img_resized = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img_resized, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)
    probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]) * 100, probs

# ─── UI ───────────────────────────────────────────────────────────────────────
st.markdown("## 🌿 Klasifikasi Rimpang Jamu Madura")
st.markdown(
    "Upload foto **irisan rimpang** dan model akan mengidentifikasi jenisnya: "
    "Jahe, Kencur, Kunyit, Lengkuas, atau Temulawak."
)

uploaded = st.file_uploader(
    "Pilih gambar rimpang (JPG / PNG / WEBP)",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded:
    img = Image.open(uploaded)

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.image(img, caption="Gambar yang diupload", use_container_width=True)

    with col2:
        with st.spinner("Menganalisis..."):
            try:
                model = load_model()
                label, conf, probs = predict(img, model)
            except Exception as e:
                st.error(f"Error saat prediksi: {e}")
                st.stop()

        conf_color = "#4caf7d" if conf >= 80 else "#e09a3a" if conf >= 50 else "#e05a4a"
        st.markdown(f"""
        <div class="result-box">
            <div style="font-size:13px;text-transform:uppercase;letter-spacing:2px;color:#888;margin-bottom:4px">Prediksi</div>
            <div style="font-size:2rem;font-weight:700;color:{conf_color}">{label}</div>
            <div style="font-size:13px;color:#aaa;margin-top:4px">Keyakinan: <b style="color:{conf_color}">{conf:.1f}%</b></div>
            <div class="confidence-bar-bg">
                <div class="confidence-bar" style="width:{conf:.0f}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Probabilitas per kelas**")
        for i, cls in enumerate(CLASS_NAMES):
            bar_pct = float(probs[i]) * 100
            bar_color = "#d4a843" if cls == label else "#3a3a30"
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
                <span style="width:80px;font-size:13px;color:#ccc">{cls}</span>
                <div style="flex:1;background:#2a2a24;border-radius:99px;height:8px">
                    <div style="width:{bar_pct:.1f}%;background:{bar_color};height:8px;border-radius:99px"></div>
                </div>
                <span style="width:40px;text-align:right;font-size:12px;color:#aaa">{bar_pct:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

st.divider()
st.markdown(
    "<div style='text-align:center;font-size:12px;color:#555'>"
    "Transfer Learning · EfficientNetB0 · CRISP-DM · Jamu Madura"
    "</div>",
    unsafe_allow_html=True
)
