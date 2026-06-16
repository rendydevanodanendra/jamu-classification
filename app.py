import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Klasifikasi Rimpang", page_icon="🌿")
st.title("🌿 Klasifikasi Rimpang Jamu")

class_names = ['Jahe', 'Kencur', 'Kunyit', 'Lengkuas', 'Temulawak']

# Membangun ulang arsitektur persis seperti di code lu, lalu load bobotnya
@st.cache_resource
def load_model_and_weights():
    # Augmentasi
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.3),
        tf.keras.layers.RandomZoom(0.15)
    ])
    
    # Base model
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(224, 224, 3), 
        include_top=False, 
        weights=None # None karena kita akan timpa dengan weights lu
    )
    
    # Custom head
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(5, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    # Load file weights lu (pastikan nama file h5-nya sesuai yang ada di GitHub)
    model.load_weights("model_jamu.h5") 
    
    return model

model = load_model_and_weights()

uploaded_file = st.file_uploader("Upload gambar rimpang...", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah')
    
    img = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    idx_pred = np.argmax(predictions[0])
    
    st.success(f"**Prediksi: {class_names[idx_pred]}**")
    st.info(f"Confidence: {predictions[0][idx_pred] * 100:.2f}%")
