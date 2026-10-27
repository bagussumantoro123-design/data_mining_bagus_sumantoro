import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import pairwise_distances

st.set_page_config(
    page_title="SPK Prediksi Risiko Diabetes",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("🩺 SPK Prediksi Risiko Diabetes")
st.caption("Sistem Pendukung Keputusan menggunakan metode K-Nearest Neighbors (KNN)")
st.info(
    "Aplikasi ini merupakan contoh pembelajaran berdasarkan dataset kecil pada soal. "
    "Hasilnya bukan diagnosis medis dan tidak boleh digunakan untuk menentukan kondisi kesehatan sebenarnya."
)

# Dataset dari gambar soal
data = {
    "Pasien": ["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09", "P10"],
    "Usia": [25, 30, 35, 40, 45, 50, 55, 28, 38, 48],
    "BMI": [21.5, 23.1, 25.0, 27.5, 30.2, 31.5, 33.0, 22.5, 26.8, 29.5],
    "Glukosa": [90, 95, 105, 120, 140, 155, 165, 92, 115, 145],
    "Tekanan Darah": [110, 115, 120, 125, 135, 140, 145, 112, 122, 130],
    "Riwayat Keluarga": [0, 0, 0, 1, 1, 1, 1, 0, 0, 1],
    "Kelas": ["Rendah", "Rendah", "Rendah", "Tinggi", "Tinggi", "Tinggi", "Tinggi", "Rendah", "Rendah", "Tinggi"]
}
df = pd.DataFrame(data)

fitur = ["Usia", "BMI", "Glukosa", "Tekanan Darah", "Riwayat Keluarga"]
X = df[fitur]
y = df["Kelas"]

st.subheader("1. Dataset Pasien")
with st.expander("Lihat 10 data pasien yang digunakan"):
    tampilan = df.copy()
    tampilan["Riwayat Keluarga"] = tampilan["Riwayat Keluarga"].map({0: "Tidak", 1: "Ada"})
    st.dataframe(tampilan, use_container_width=True, hide_index=True)
    st.caption("Riwayat keluarga: 0 = Tidak ada, 1 = Ada.")

st.subheader("2. Masukkan Data Pasien Baru")
with st.form("form_pasien"):
    col1, col2 = st.columns(2)
    with col1:
        usia = st.number_input("Usia (tahun)", min_value=1, max_value=120, value=42, step=1)
        bmi = st.number_input("BMI", min_value=10.0, max_value=80.0, value=28.5, step=0.1, format="%.1f")
        glukosa = st.number_input("Glukosa", min_value=20, max_value=600, value=130, step=1)
    with col2:
        tekanan = st.number_input("Tekanan darah", min_value=40, max_value=300, value=128, step=1)
        riwayat_label = st.selectbox("Riwayat diabetes dalam keluarga", ["Ada", "Tidak ada"], index=0)
        k = st.selectbox("Jumlah tetangga (K)", [1, 3, 5, 7, 9], index=1)
    submitted = st.form_submit_button("🔎 Prediksi Risiko", use_container_width=True)

if submitted:
    pasien_baru = pd.DataFrame([{
        "Usia": usia,
        "BMI": bmi,
        "Glukosa": glukosa,
        "Tekanan Darah": tekanan,
        "Riwayat Keluarga": 1 if riwayat_label == "Ada" else 0
    }], columns=fitur)

    # Standardisasi diperlukan karena skala tiap atribut berbeda.
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=min(k, len(df))))
    ])
    model.fit(X, y)
    prediksi = model.predict(pasien_baru)[0]
    probabilitas = model.predict_proba(pasien_baru)[0]
    kelas_model = list(model.named_steps["knn"].classes_)
    confidence = float(probabilitas[kelas_model.index(prediksi)]) * 100

    st.subheader("3. Hasil Prediksi")
    if prediksi == "Tinggi":
        st.error("Hasil klasifikasi dataset: RISIKO TINGGI")
    else:
        st.success("Hasil klasifikasi dataset: RISIKO RENDAH")

    c1, c2 = st.columns(2)
    c1.metric("K yang digunakan", min(k, len(df)))
    c2.metric("Proporsi suara kelas terpilih", f"{confidence:.1f}%")
    st.caption(
        "Persentase di atas adalah proporsi probabilitas dari model KNN pada dataset contoh, "
        "bukan peluang medis seseorang mengalami diabetes."
    )

    # Hitung jarak setelah fitur distandardisasi agar tetangga terdekat bisa ditampilkan.
    scaler = model.named_steps["scaler"]
    X_scaled = scaler.transform(X)
    pasien_scaled = scaler.transform(pasien_baru)
    distances = pairwise_distances(pasien_scaled, X_scaled, metric="euclidean")[0]
    tetangga = df.copy()
    tetangga["Jarak Euclidean (setelah standardisasi)"] = distances
    tetangga = tetangga.sort_values("Jarak Euclidean (setelah standardisasi)").head(min(k, len(df)))
    tetangga["Riwayat Keluarga"] = tetangga["Riwayat Keluarga"].map({0: "Tidak", 1: "Ada"})

    st.subheader("4. Tetangga Terdekat")
    st.dataframe(
        tetangga[["Pasien", "Usia", "BMI", "Glukosa", "Tekanan Darah",
                  "Riwayat Keluarga", "Kelas", "Jarak Euclidean (setelah standardisasi)"]]
        .round({"Jarak Euclidean (setelah standardisasi)": 3}),
        use_container_width=True,
        hide_index=True
    )

    hitung = tetangga["Kelas"].value_counts()
    rendah = int(hitung.get("Rendah", 0))
    tinggi = int(hitung.get("Tinggi", 0))
    st.write(f"Suara tetangga terdekat: **Rendah = {rendah}** dan **Tinggi = {tinggi}**.")

st.divider()
st.subheader("Cara membaca data")
st.markdown(
    """
    - **BMI**: indeks massa tubuh yang dimasukkan sebagai angka.
    - **Glukosa** dan **tekanan darah**: masukkan nilai sesuai data yang tersedia.
    - **Riwayat keluarga**: pilih *Ada* jika terdapat riwayat diabetes dalam keluarga.
    - **KNN**: membandingkan pasien baru dengan data pasien yang sudah memiliki kelas.
    - Fitur distandardisasi sebelum menghitung jarak agar perbedaan skala angka tidak mendominasi perhitungan.
    """
)
st.warning(
    "Dataset hanya berisi 10 baris contoh, sehingga hasil dapat berubah ketika dataset atau nilai K berubah. "
    "Untuk penilaian kesehatan, konsultasikan hasil pemeriksaan kepada tenaga kesehatan."
)
st.caption("Proyek pembelajaran Sistem Pendukung Keputusan • Python + Streamlit")
