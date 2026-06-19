import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Konfigurasi Halaman
st.set_page_config(page_title="Klasifikasi Rimpang Jamu", page_icon="🌿")

# Judul Aplikasi
st.title("🌿 Klasifikasi Jenis Rimpang Jamu Madura")
st.write("Aplikasi ini menggunakan Deep Learning (EfficientNetB0) untuk mengklasifikasikan irisan rimpang ke dalam 5 kategori: **Jahe, Kencur, Kunyit, Lengkuas, dan Temulawak**.")

# Fungsi untuk memuat model (di-cache agar lebih cepat & hemat memori)
# Ubah nama fungsinya sedikit (misal ditambah angka 2) agar cache lama dibuang
@st.cache_resource
def load_model_2():
    model = tf.keras.models.load_model('model_jamu.keras')
    return model

# Load model dengan nama fungsi yang baru
try:
    model = load_model_2()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")

# Daftar Kelas sesuai dengan urutan di dataset Anda
class_names = ['Jahe', 'Kencur', 'Kunyit', 'Lengkuas', 'Temulawak']

# Upload Gambar
uploaded_file = st.file_uploader("Unggah gambar irisan rimpang (JPG/PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan Gambar
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah', use_container_width=True)
    
    st.write("Memproses dan memprediksi...")
    
    # Preprocessing Gambar
    # Ubah ukuran ke 224x224 sesuai input EfficientNetB0
    img = image.resize((224, 224))
    img_array = np.array(img)
    
    # Pastikan gambar memiliki 3 channel (RGB)
    if len(img_array.shape) == 2: # Jika grayscale
        img_array = np.stack((img_array,)*3, axis=-1)
    elif img_array.shape[-1] == 4: # Jika ada channel alpha (RGBA)
        img_array = img_array[..., :3]
        
    # Tambah dimensi batch menjadi (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0) 
    
    # Prediksi
    predictions = model.predict(img_array)
    idx_pred = np.argmax(predictions[0])
    predicted_class = class_names[idx_pred]
    confidence = predictions[0][idx_pred] * 100
    
    # Tampilkan Hasil
    st.success(f"**Prediksi: {predicted_class}**")
    st.info(f"Tingkat Kepercayaan (Confidence): {confidence:.2f}%")
