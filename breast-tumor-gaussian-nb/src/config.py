"""
config.py — SEMUA KEPUTUSAN DESAIN TIM ADA DI SINI.

Nilai di bawah sama dengan yang dipakai script analisis awal tim
(30 fitur, prior dari data latih, keputusan argmax = threshold 0,5). Setiap nilai yang kalian pakai di versi final harus
kalian putuskan sendiri dan kalian justifikasi di Bab "Design Justification"
pada laporan (dan harus bisa kalian jelaskan saat oral defense).
"""

# Fitur yang dipakai model.
#   None -> seluruh 30 fitur di dataset.
#   Atau list nama kolom, mis. ["radius_mean", "texture_mean", ...]
# Pertimbangkan: korelasi antar-fitur vs asumsi conditional independence.
FEATURES: list[str] | None = None

# Prior kelas [P(Benign), P(Malignant)].
#   None -> diserahkan ke implementasi model (mis. diestimasi dari data latih).
#   Atau tuple dua angka yang berjumlah 1.
# Pertimbangkan: proporsi dataset vs prevalensi di populasi yang dituju.
PRIORS: tuple[float, float] | None = None

# Threshold keputusan: prediksi Malignant jika P(Malignant | x) >= THRESHOLD.
# Pertimbangkan: biaya false negative vs false positive dalam konteks medis.
THRESHOLD: float = 0.5

# Protokol evaluasi.
TEST_SIZE: float = 0.2
RANDOM_SEED: int = 42
CV_FOLDS: int = 5
