import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Konfigurasi Halaman
st.set_page_config(page_title="Klasifikasi Rimpang Jamu", page_icon="🌿")

st.title("🌿 Sistem Klasifikasi Rimpang Jamu Madura")
st.write("Unggah foto irisan rimpang untuk diprediksi oleh model AI.")

# Daftar kelas (sesuaikan urutannya dengan output modelmu)
class_names = ['Jahe', 'Kencur', 'Kunyit', 'Lengkuas', 'Temulawak']

# Fungsi untuk memuat model (di-cache agar tidak loading terus-menerus)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model_jamu.h5")

model = load_model()

# Widget Upload Gambar
uploaded_file = st.file_uploader("Pilih file gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar yang diunggah
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah', use_column_width=True)
    
    st.write("⏳ Memproses prediksi...")
    
    # Preprocessing gambar agar sesuai dengan input model (224x224)
    img = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prediksi
    predictions = model.predict(img_array)
    idx_pred = np.argmax(predictions[0])
    confidence = predictions[0][idx_pred] * 100
    
    # Tampilkan Hasil
    st.success(f"**Hasil Prediksi: {class_names[idx_pred]}**")
    st.info(f"Tingkat Kepercayaan (Confidence): {confidence:.2f}%")
