import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model dan scaler
model_folder = 'models/'
with open(f'{model_folder}best_gb_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open(f'{model_folder}scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Judul aplikasi
st.title("Prediksi Harga Rumah Jakarta Selatan")

# Input fitur
st.header("Masukkan Fitur Rumah")
lt = st.number_input("Luas Tanah (m²)", min_value=10, value=100, step=1, help="Luas tanah minimal 10 m²")
lb = st.number_input("Luas Bangunan (m²)", min_value=10, value=100, step=1, help="Luas bangunan minimal 10 m²")
jkt = st.number_input("Jumlah Kamar Tidur", min_value=1, value=2, step=1, help="Jumlah kamar tidur minimal 1")
jkm = st.number_input("Jumlah Kamar Mandi", min_value=1, value=1, step=1, help="Jumlah kamar mandi minimal 1")
grs = st.selectbox("Garasi", options=[0, 1], format_func=lambda x: "Ada" if x == 1 else "Tidak Ada", help="Garasi: 1 untuk ada, 0 untuk tidak ada")

# Prediksi harga rumah
if st.button("Prediksi Harga"):
    # Data input pengguna
    input_data = pd.DataFrame([[lt, lb, jkt, jkm, grs]], columns=['LT', 'LB', 'JKT', 'JKM', 'GRS'])

    # Transformasi data
    input_scaled = scaler.transform(input_data)

    # Prediksi
    predicted_log = model.predict(input_scaled)
    predicted_price = np.expm1(predicted_log)  # Konversi dari log

    # Tampilkan hasil prediksi
    st.success(f"Prediksi Harga Rumah: Rp {predicted_price[0]:,.2f}")
