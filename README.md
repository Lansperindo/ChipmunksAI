# Breast Tumor Diagnosis Classification Using Gaussian Naive Bayes Based on Morphological Features

> Proyek akhir Kecerdasan Buatan, DTETI Universitas Gadjah Mada (2026)
> **Kelompok:** _[nomor]_
> **Anggota:** Gadiza Aliefya Yunanda (23/514971/TK/56562), Anju Agrita Sidabutar (23/515046/TK/56574), Inge Arista Robiyanto (23/515142/TK/56600)
> **Demo langsung:** _[tautan Streamlit Community Cloud]_

Sistem klasifikasi tumor payudara (**jinak / Benign** vs **ganas / Malignant**) berbasis
**Gaussian Naive Bayes** (Lecture 5, Uncertainty & Bayesian Reasoning), dilatih pada dataset
Wisconsin Diagnostic Breast Cancer. Aplikasi web menampilkan probabilitas posterior dan
**breakdown kontribusi tiap fitur**, sehingga keputusan model dapat ditelusuri oleh
pengguna non-programmer.

![Beranda aplikasi](docs/screenshots/app_beranda.png)

### Fitur aplikasi

| Tab | Isi |
|---|---|
| **Mulai di sini** | Tujuan aplikasi, cara memakai, daftar menu, ringkasan model, dan batasan |
| **Periksa sampel** | Pilih sampel data uji / isi ciri sendiri / unggah CSV banyak sampel (dengan confusion matrix, ROC, dan metrik untuk file itu bila berlabel). Menampilkan peluang, keputusan, *timbangan bukti* (waterfall), kurva Gaussian per ciri, kasus serupa, panel **"Bagaimana model menghitung angka ini?"** berisi 4 langkah Bayes, unduh laporan hasil (HTML), dan riwayat pemeriksaan |
| **Simulasi** | Ubah nilai ciri dan lihat peluang berubah; kurva peluang saat satu ciri digeser |
| **Jelajahi ciri** | Histogram data latih vs kurva Gaussian model per ciri, daya pisah (AUC) tiap ciri, ciri yang berkorelasi |
| **Akurasi** | Metrik, confusion matrix, ROC, 5-fold CV, diagram keandalan peluang, sampel yang salah ditebak |
| **Data & istilah** | Sumber dan grafik dataset aktif, kamus ciri, glosarium |
| **Tentang** | Tim, konfigurasi model, model card, verifikasi terhadap scikit-learn |

### Sumber data aplikasi

Secara bawaan aplikasi memakai dataset proyek ini (**WDBC**) sehingga demo langsung siap dipakai.
Lewat panel samping (**Sumber data**) pengguna juga dapat mengunggah CSV penyakit lain: pilih kolom
label, nilai yang dianggap kasus positif, dan ciri numerik yang dipakai. Model Gaussian Naive Bayes
tim dilatih ulang pada dataset itu dan seluruh tab menyesuaikan nama kelasnya. Syarat berkas:
minimal dua kolom angka sebagai ciri dan satu kolom label berisi tepat dua nilai.

Batas keputusan (threshold) dapat diubah dari panel samping; semua tab ikut diperbarui, dan
panel samping menampilkan langsung efeknya (sensitivitas, spesifisitas, jumlah kasus terlewat).

### Aksesibilitas dan tampilan ponsel

- Semua pasangan warna teks/latar diuji terhadap WCAG 2.1 AA (rasio kontras minimum 4,5:1).
  Rasio terendah yang dipakai adalah 5,0:1; teks utama 16,9:1.
- Informasi tidak pernah disampaikan lewat warna saja: setiap hasil disertai ikon dan teks.
- Kotak hasil memakai `role="status"` dan `aria-live`, bar dan ikon dekoratif diberi
  `aria-label` atau `aria-hidden`, dan fokus keyboard diberi garis tebal yang terlihat.
- Tata letak diuji pada viewport 390 px: panel samping otomatis tertutup, tab dapat digeser,
  kolom menumpuk, dan grafik tidak memaksa halaman melebar.

![Tampilan di ponsel](docs/screenshots/app_mobile.png)

![Hasil prediksi](docs/screenshots/app_prediksi.png)
![Timbangan bukti](docs/screenshots/app_breakdown.png)
![Simulasi](docs/screenshots/app_simulasi.png)

## Dataset

| | |
|---|---|
| Sumber | Wolberg, Street & Mangasarian (1995), *Breast Cancer Wisconsin (Diagnostic)*, UCI ML Repository, [doi:10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B) |
| Sampel | 569 citra FNA massa payudara (tanpa kolom ID; urutan baris diacak) |
| Label | `Malignant` (212), `Benign` (357) |
| Fitur | 10 ciri inti sel × {mean, se, worst} = 30 fitur kontinu |
| File | `data/wdbc.csv` (pemisah kolom `;`) |

## Struktur repositori

```
├── app.py                  # Aplikasi web Streamlit
├── src/
│   ├── gnb_manual.py       # ★ Gaussian Naive Bayes dari nol (kode tim, NumPy murni)
│   ├── naive_bayes.py      # Adapter antarmuka (nama method seragam untuk app & test)
│   ├── project_info.py     # Judul, tim, link GitHub/demo (dipakai aplikasi)
│   ├── config.py           # ★ Keputusan desain: fitur, prior, threshold
│   ├── data_utils.py       # Loading data, label mapping, split terstratifikasi
│   └── metrics.py          # Akurasi, sensitivitas, spesifisitas, ROC/AUC
├── scripts/
│   ├── eda.py              # Grafik & ringkasan eksplorasi data
│   ├── evaluate.py         # Evaluasi hold-out + k-fold CV → results/metrics.json
│   └── gnb_analysis.py     # Analisis lengkap tim: parameter, CV, worked example, cek asumsi Gaussian
├── tests/test_naive_bayes.py
├── data/wdbc.csv
├── figures/                # Output grafik
├── results/                # Output metrik (JSON)
├── docs/model_card.md      # Penjelasan model (tampil di tab "Tentang")
└── docs/Lampiran_Teknis_GaussianNB.docx  # Lampiran laporan: parameter, metrik, kamus fitur
```

## Cara menjalankan

```bash
git clone https://github.com/<user>/<repo>.git
cd <repo>
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/eda.py              # grafik eksplorasi data
python -m pytest tests/ -v         # cek implementasi Naive Bayes
python scripts/evaluate.py         # metrik hold-out & cross-validation
python scripts/gnb_analysis.py     # analisis lengkap + grafik untuk laporan
streamlit run app.py               # buka http://localhost:8501
```

## Deploy (Streamlit Community Cloud, gratis)

1. Push repositori ini ke GitHub (public).
2. Buka <https://share.streamlit.io>, login dengan GitHub, pilih **New app**.
3. Pilih repositori, branch `main`, main file `app.py`, lalu **Deploy**.
4. Salin URL aplikasi ke bagian atas README ini dan ke laporan.

## Hasil

Split hold-out terstratifikasi 80/20 (seed 42, 113 sampel uji) dan 5-fold stratified CV,
seluruh 30 fitur, prior dari data latih, threshold 0,5 (lihat `src/config.py`).

| Metrik | Hold-out | 5-fold CV (mean ± std) |
|---|---|---|
| Akurasi | 0.956 | 0.938 ± 0.018 |
| Sensitivitas (Malignant) | 0.905 | 0.891 ± 0.044 |
| Spesifisitas (Benign) | 0.986 | 0.966 ± 0.021 |
| Presisi (PPV) | 0.974 | 0.942 ± 0.033 |
| F1 | 0.938 | 0.915 ± 0.026 |
| ROC AUC | 0.991 | 0.987 ± 0.007 |

Confusion matrix hold-out: TP = 38, FN = 4, FP = 1, TN = 70.
Prediksi model tim 100% identik dengan `sklearn.naive_bayes.GaussianNB` (selisih probabilitas maks. ~1e-16).

## Penggunaan AI tools

_[DIISI TIM dengan jujur.]_ Catatan: web app (`app.py`), `data_utils.py`, `metrics.py`,
`eda.py`, `evaluate.py`, unit test, adapter `naive_bayes.py`, README, template laporan/slide,
serta perbaikan bug pada `gnb_analysis.py` dibuat/dibantu Claude (Anthropic). Tuliskan juga
dengan jujur asal-usul `gnb_manual.py` dan bagian lain yang kalian kerjakan.

## Disclaimer

Prototipe akademik. **Bukan** alat diagnosis medis.
