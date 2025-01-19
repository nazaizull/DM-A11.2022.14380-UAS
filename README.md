**DOKUMENTASI PROJECT DM**

### 1. Judul / Topik Project dan Identitas Lengkap


**Topik**: Penerapan Machine Learning dengan model Linear Regression, Random Forest, dan Gradient Boosting untuk Prediksi Harga Rumah di Jakarta Selatan
**Nama**: Naza Izul Haki
**NIM**: A11.2022.14380
**Mata Kuliah**: Data Mining  
**Tugas Akhir Semester (UAS)**  

---

### 2. Ringkasan dan Permasalahan Project + Tujuan yang Akan Dicapai + Model / Alur Penyelesaian

#### Ringkasan
Proyek ini bertujuan untuk membangun model prediksi harga rumah menggunakan beberapa model algoritma seperti Linear Regression, Random Forest, dan Gradient Boosting Regressor. Dataset yang digunakan mencakup fitur-fitur seperti luas tanah, luas bangunan, jumlah kamar tidur, jumlah kamar mandi, dan keberadaan garasi. Prediksi harga rumah di daerah Jakarta Selatan yang akurat akan membantu pembeli dan penjual dalam menentukan nilai wajar properti.

#### Permasalahan Project
Menentukan harga rumah yang akurat berdasarkan fitur-fitur properti merupakan tantangan yang kompleks. Variasi luas tanah, bangunan, dan fasilitas rumah dapat memengaruhi harga dengan signifikan. Tantangan utama adalah menciptakan model yang dapat menangkap hubungan kompleks ini untuk memberikan estimasi harga yang tepat.

#### Tujuan yang Akan Dicapai
1. Membangun model prediksi harga rumah berbasis beberapa algoritma.
2. Mengoptimalkan parameter model untuk mendapatkan performa terbaik.
3. Mengevaluasi performa model menggunakan metrik R² dan Mean Squared Error (MSE).
4. Memberikan wawasan tentang fitur yang paling berpengaruh terhadap harga rumah.
5. Mengembangkan model prediksi harga rumah yang akurat berdasarkan fitur-fitur properti yang tersedia. Model ini diharapkan dapat membantu calon pembeli, penjual, atau investor dalam menentukan nilai rumah dengan lebih akurat.

#### Alur / Model Penyelesaian

1. **Pengumpulan Data**: Dataset harga rumah dikumpulkan dari https://www.kaggle.com/datasets/wisnuanggara/daftar-harga-rumah
2. **Eksplorasi Data (EDA)**: Analisis data dilakukan untuk memahami distribusi fitur dan hubungan antar fitur.
3. **Preprocessing**: Melakukan normalisasi data dan menangani nilai kategorikal.
4. **Modeling**: Membangun model menggunakan Linear Regression, Random Forest, dan Gradient Boosting.
5. **Evaluasi**: Mengevaluasi model dengan data uji.
6. **Kesimpulan**: Menyimpulkan hasil dan implikasi model.

Untuk bagan alur ada di file alurpenyelesaian.png
Mulai -> Load Data -> EDA dan Preprocessing Data -> Modelling -> Evaluasi Model -> Kesimpulan -> Selesai

---

### 3. Penjelasan Dataset, EDA, dan Proses Features Dataset

#### Dataset
Dataset harga rumah jaksel terdiri dari 7 kolom dengan jumlah data yaitu 1003 data. Kolom tersebut terdiri dari:
- `LT`: Luas Tanah (m²)
- `LB`: Luas Bangunan (m²)
- `JKT`: Jumlah Kamar Tidur
- `JKM`: Jumlah Kamar Mandi
- `GRS`: Keberadaan Garasi (1: Ada, 0: Tidak Ada)
- `HARGA`: Harga Rumah (Target)
- `KOTA`: Lokasi rumah, dalam hal ini seluruhnya berada di wilayah Jakarta Selatan (disingkat "JAKSEL").

#### Eksplorasi Data dan Analisis (EDA)
EDA dilakukan untuk memahami distribusi data, mengidentifikasi outlier, dan melihat hubungan antar fitur. Heatmap korelasi digunakan untuk memahami hubungan antara fitur dengan target `HARGA`.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = 'HARGA RUMAH JAKSEL.xlsx'
df = pd.read_excel(file_path, header=1, usecols=['HARGA', 'LT', 'LB', 'JKT', 'JKM', 'GRS'])

# Info dataset
df.info()

# Statistik deskriptif
df.describe()

# Visualisasi distribusi harga rumah
plt.figure(figsize=(10, 6))
sns.histplot(df['HARGA'], kde=True, bins=30)
plt.title("Distribusi Harga Rumah")
plt.show()

# Korelasi antar fitur
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Korelasi Antar Fitur")
plt.show()
```

#### Proses Features Dataset
- Mengubah fitur kategorikal `GRS` menjadi numerik.
- Melakukan normalisasi data menggunakan `StandardScaler` untuk memastikan skala yang seragam.

```python
from sklearn.preprocessing import StandardScaler

# Encoding fitur GRS
df['GRS'] = df['GRS'].map({'ADA': 1, 'TIDAK ADA': 0})

# Definisi fitur dan target
X = df[['LT', 'LB', 'JKT', 'JKM', 'GRS']]
y = df['HARGA']

# Normalisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

### 4. Proses Learning / Modeling

Tiga model digunakan untuk membandingkan performa: Linear Regression, Random Forest, dan Gradient Boosting. Model terbaik ditentukan berdasarkan evaluasi metrik R² dan MSE.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Split data menjadi train dan test
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(x_train, y_train)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)

# Gradient Boosting
gb_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
gb_model.fit(x_train, y_train)
```

---

### 5. Performa Model

Evaluasi dilakukan untuk membandingkan performa model. Metrik yang digunakan adalah R² dan Mean Squared Error (MSE).

```python
# Evaluasi Linear Regression
y_pred_lr = lr_model.predict(x_test)
r2_lr = r2_score(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)

# Evaluasi Random Forest
y_pred_rf = rf_model.predict(x_test)
r2_rf = r2_score(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)

# Evaluasi Gradient Boosting
y_pred_gb = gb_model.predict(x_test)
r2_gb = r2_score(y_test, y_pred_gb)
mse_gb = mean_squared_error(y_test, y_pred_gb)

# Tampilkan hasil
print(f"Linear Regression - R²: {r2_lr:.4f}, MSE: {mse_lr:.2f}")
print(f"Random Forest - R²: {r2_rf:.4f}, MSE: {mse_rf:.2f}")
print(f"Gradient Boosting - R²: {r2_gb:.4f}, MSE: {mse_gb:.2f}")
```

---

### 6. Diskusi Hasil dan Kesimpulan

#### Diskusi Hasil
- Model Gradient Boosting memberikan performa terbaik dengan R² tertinggi dan MSE terendah dibandingkan dengan Linear Regression dan Random Forest.
- Fitur `LB` (Luas Bangunan) dan `LT` (Luas Tanah) memiliki pengaruh terbesar terhadap harga rumah berdasarkan analisis Feature Importance.

```python
# Feature Importance untuk Gradient Boosting
features = ['LT', 'LB', 'JKT', 'JKM', 'GRS']
importance = gb_model.feature_importances_

plt.figure(figsize=(10, 6))
plt.barh(features, importance, color='skyblue')
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()
```

#### Kesimpulan
1. Gradient Boosting adalah model terbaik untuk prediksi harga rumah dengan performa terbaik dibandingkan Linear Regression dan Random Forest.
2. Model ini dapat digunakan untuk membantu pembeli atau penjual rumah menentukan nilai properti secara akurat.
3. Penambahan fitur geografis atau ekonomis mungkin dapat meningkatkan performa model di masa depan.
