import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image

# Load the model
MODEL_PATH = '/Users/luthfirakan/Downloads/leaf_disease_classification_model.h5'
model = load_model(MODEL_PATH)

# Define the class labels based on your training data
class_labels = ["Healthy (Sehat)", "Powdery (Sakit)", "Rust (Sakit)"]

# Function to preprocess the image
def preprocess_image(image, target_size=(225, 225)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image.astype('float32') / 255.0
    return image

# Function to make prediction
def model_predict(img, model):
    img = preprocess_image(img)
    preds = model.predict(img)
    return preds

# Set page title and background color
st.set_page_config(page_title='EMILIA', page_icon=':leaves:', layout='wide', initial_sidebar_state='expanded')

# Add custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: #f9f9f9;
        background-image: url('https://example.com/background-pattern.jpg');
        background-repeat: repeat;
    }
    .st-bb {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    }
    .st-dd {
        padding: 10px;
        border-radius: 10px;
    }
    .title {
        text-align: center;
        color: #008080;
        font-size: 36px;
        padding: 20px;
    }
    .subtitle {
        text-align: center;
        color: #696969;
        font-size: 18px;
    }
    .footer {
        text-align: center;
        color: #808080;
        font-size: 12px;
        padding: 20px;
    }
    .description {
        text-align: center;
        color: #333333;
        font-size: 16px;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add title and description
st.markdown("<h1 class='title'>(EMILIA) Deteksi Penyakit Pada Tanaman Apel Berdasarkan Citra Tekstur Daun</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Unggah gambar untuk mengetahui kondisi kesehatan daun!</p>", unsafe_allow_html=True)

# Add description
description_text = """
Halo! Ini adalah EMILIA. Ini merupakan aplikasi berbasis web yang bisa digunakan untuk mendeteksi penyakit pada tanaman apel. Program ini bekerja dengan mendeteksi tekstur daun lalu mengklasifikasikannya menjadi : Healthy (Sehat), Powdery (Sakit), dan Rust (Sakit). Teman-teman bisa mengetahui kondisi apa yang terjadi pada tanaman apel hanya dengan mengunggah gambarnya saja. Yuk, unggah gambarnya!
"""
st.markdown(f"<p class='description'>{description_text}</p>", unsafe_allow_html=True)

# Add file uploader
uploaded_file = st.file_uploader("Pilih Gambar Daun...", type=["jpg", "jpeg", "png"])

# If image is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Display uploaded image
    st.subheader('Gambar daun yang telah diunggah')
    st.image(image, caption='Gambar terunggah.', use_column_width=True)

    # Classify image
    st.write("")
    st.write("Mendeteksi...")
    preds = model_predict(image, model)
    result = np.argmax(preds, axis=1)
    st.write(f"Hasil Prediksi: {class_labels[result[0]]}")

# Add footer
st.markdown("<p class='footer'>Made with ❤️ by Luthfi Rakan Nabila</p>", unsafe_allow_html=True)
