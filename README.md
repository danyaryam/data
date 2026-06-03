# PsyScreening Dashboard — EDA & Insights

Proyek analisis data  dan visualisasi untuk data skrining psikologis menggunakan Python, menampilkan analisis notebook Jupyter dan dashboard Streamlit interaktif.


## 🔧 Instalasi

### Prasyarat
- Python 3.8 atau lebih tinggi
- Manajer paket pip

### Langkah-langkah Setup

1. **Clone atau navigasi ke direktori proyek**
   ```bash
   cd /path/to/project
   ```

2. **Buat virtual environment (direkomendasikan)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Di Windows: venv\Scripts\activate
   ```

3. **Pasang dependensi**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Menjalankan Proyek

### Jalankan Dashboard Streamlit
```bash
streamlit run app.py
```
Dashboard akan terbuka di browser default Anda di `http://localhost:8501`

### Jalankan Jupyter Notebook
```bash
jupyter notebook PsyScreening.ipynb
```

## 📊 Overview Data

### File
- **survey.csv**: Data survei mentah yang berisi respons skrining psikologis
- **data_dictionary.csv**: Dokumentasi lengkap variabel, tipe, dan deskripsi
- **processed_data.csv**: Dataset yang telah dibersihkan dan siap untuk analisis dan pemodelan

## 🐳 Deployment Docker

Bangun dan jalankan menggunakan Docker:
```bash
docker build -t psy-screening .
docker run -p 8501:8501 psy-screening
```


## 👨‍💻 Contoh Penggunaan

### Akses melalui Dashboard
1. Jalankan `streamlit run app.py`
2. Gunakan sidebar interaktif untuk memfilter data
3. Jelajahi visualisasi dan metrik
4. Ekspor insight sesuai kebutuhan

### Analisis melalui Notebook
1. Jalankan `jupyter notebook PsyScreening.ipynb`
2. Jalankan sel secara berurutan untuk analisis langkah demi langkah
3. Ubah kode untuk analisis khusus
4. Tambahkan visualisasi baru sesuai kebutuhan
