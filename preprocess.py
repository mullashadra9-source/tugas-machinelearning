import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Membaca dataset (file CSV harus berada satu folder dengan program)
df = pd.read_csv("AirQualityUCI.csv", sep=";", decimal=",")
df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
print("=== STRUKTUR DATA AWAL ===")
print("Ukuran data:", df.shape)
print("\nTipe data setiap kolom:\n", df.dtypes)

# 2. Mengubah sentinel -200 menjadi missing value
numeric_cols = df.select_dtypes(include="number").columns
df[numeric_cols] = df[numeric_cols].replace(-200, np.nan)

# Hapus baris yang seluruh nilainya kosong (misalnya baris pemisah di akhir CSV)
df = df.dropna(axis=0, how="all").copy()
print("\n=== MISSING VALUE SETELAH -200 DIUBAH MENJADI NaN ===")
print(df.isna().sum())

# 3. Duplikasi: laporkan dahulu, tidak dihapus otomatis tanpa pemeriksaan
duplicate_count = int(df.duplicated().sum())
print("\nJumlah baris duplikat:", duplicate_count)

# 4. Tampilkan 5 baris teratas dan terbawah setelah pembersihan baris kosong
print("\n=== 5 BARIS TERATAS SETELAH PEMBERSIHAN AWAL ===")
print(df.head(5).to_string(index=False))
print("\n=== 5 BARIS TERBAWAH SETELAH PEMBERSIHAN AWAL ===")
print(df.tail(5).to_string(index=False))

# 5. Hapus kolom yang >80% missing, kecuali CO karena diperlukan untuk target
missing_ratio = df.isna().mean()
drop_cols = [c for c in df.columns if missing_ratio[c] > 0.80 and c != "CO(GT)"]
print("\nKolom yang dihapus karena missing >80%:", drop_cols)
df = df.drop(columns=drop_cols)

# Baris tanpa CO tidak dapat diberi label target yang valid
df = df.dropna(subset=["CO(GT)"]).copy()

def kategori_co(nilai):
    if nilai <= 2:
        return "Rendah"
    elif nilai <= 4:
        return "Sedang"
    return "Tinggi"

df["Kualitas_Udara"] = df["CO(GT)"].apply(kategori_co)
print("\n=== DISTRIBUSI TARGET ===")
print(df["Kualitas_Udara"].value_counts())
print("\nPersentase target (%):")
print((df["Kualitas_Udara"].value_counts(normalize=True) * 100).round(2))

# 6. Hanya fitur numerik; CO dikeluarkan untuk mencegah target leakage
feature_cols = [c for c in df.select_dtypes(include="number").columns if c != "CO(GT)"]
X, y = df[feature_cols], df["Kualitas_Udara"]

# 7. Split sebelum imputasi dan standardisasi
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", GaussianNB())
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n=== HASIL KLASIFIKASI GAUSSIAN NAIVE BAYES ===")
print("Jumlah data valid:", len(df))
print("Jumlah fitur:", len(feature_cols))
print("Fitur:", feature_cols)
print("Training:", len(X_train), "| Testing:", len(X_test))
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification report:\n", classification_report(y_test, y_pred, zero_division=0))
print("Confusion matrix (urutan Rendah, Sedang, Tinggi):")
print(confusion_matrix(y_test, y_pred, labels=["Rendah", "Sedang", "Tinggi"]))

# Simpan dataset berlabel untuk dokumentasi; standardisasi diterapkan saat pipeline fit/predict
df.to_csv("AirQuality_Preprocessed.csv", index=False)
print("\nDataset berlabel disimpan sebagai AirQuality_Preprocessed.csv")