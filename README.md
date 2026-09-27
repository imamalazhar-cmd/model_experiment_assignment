# AI Model Experiment & Evaluation: Sentiment Analysis Ulasan Pelanggan

Eksperimen yang membandingkan dua pendekatan AI untuk klasifikasi sentimen ulasan pelanggan e-commerce (positif/negatif):

1. **Model klasik Machine Learning**: TF-IDF + Logistic Regression (Scikit-learn)
2. **Large Language Model via API**: Gemini (zero-shot prompting)

Program: AI Engineering Bootcamp · Tools: Python, Pandas, Scikit-learn, Gemini AI API, VS Code, GitHub

---

## 1. Problem Statement

| Komponen | Keterangan |
|---|---|
| **Objective** | Mengklasifikasikan sentimen ulasan pelanggan secara otomatis pada halaman produk e-commerce, sebagai dasar memilih pendekatan yang akan dikembangkan ke produksi |
| **Target / label** | `sentiment` → `positif` atau `negatif` (binary classification) |
| **Input** | `review_text` (teks ulasan berbahasa Indonesia) |
| **Batasan & asumsi** | Hanya 2 kelas (tanpa netral); dataset asli tidak dimodifikasi; kedua pendekatan diuji pada test set yang sama; Gemini dijalankan dengan kuota free tier; tidak ada training Deep Learning dari nol |

## 2. Dataset

File: `data/customer_reviews_sentiment.csv`

| Atribut | Nilai |
|---|---|
| Jumlah baris | 200 |
| Kolom | `review_id`, `product_name`, `review_text`, `sentiment` |
| Missing value | 0 |
| Distribusi label | positif 110 (55%) · negatif 90 (45%) |
| Kalimat unik | **40** (20 positif, 20 negatif), diulang pada produk berbeda |

> **Temuan penting:** dataset hanya berisi 40 kalimat template yang diulang. Dengan `train_test_split` acak (random_state=42, stratify), **97,5% kalimat di test set juga muncul di train set** (data leakage). Hal ini memengaruhi interpretasi hasil model klasik (lihat bagian 5).

## 3. Struktur Project

```
model-experiment-assignment/
├── data/
│   └── customer_reviews_sentiment.csv
├── notebook/
│   └── experiment_notebook.ipynb
├── documentation/
│   └── model_comparison_summary.png   (opsional)
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

## 4. Ringkasan Eksperimen

**Split data:** `train_test_split(test_size=0.2, random_state=42, stratify=y)` menghasilkan **160 train / 40 test** (18 negatif, 22 positif). Test set yang sama dipakai untuk kedua pendekatan.

### Pendekatan 1: Model Klasik (Scikit-learn)
| Tahap | Detail |
|---|---|
| Preprocessing | `TfidfVectorizer()` (default: lowercase, unigram) |
| Model | `LogisticRegression()` |
| Training | fit pada 160 data latih |
| Waktu train + predict | **±0,12 detik** |

### Pendekatan 2: LLM API (Gemini)
| Parameter | Nilai | Alasan |
|---|---|---|
| Model | `gemini-3.5-flash-lite` | varian flash-lite untuk teks: cepat dan hemat kuota |
| Teknik prompt | zero-shot | task sederhana (2 label eksplisit), tanpa data latih |
| `temperature` | 0.2 | output rendah variasi agar konsisten untuk klasifikasi (disarankan 0 untuk hasil deterministik penuh) |
| Normalisasi output | `strip()` + `lower()` | menyeragamkan jawaban dengan label dataset |

Prompt yang digunakan:
```
Klasifikasikan sentimen ulasan berikut sebagai "positif" atau "negatif".
Jawab hanya dengan satu kata: positif atau negatif.

Ulasan: "{review_text}"
Sentimen:
```

## 5. Hasil Evaluasi & Perbandingan

| Pendekatan | Accuracy | Precision (macro) | Recall (macro) | F1-Score (macro) | Waktu | Perlu Training | Biaya |
|---|---|---|---|---|---|---|---|
| Model Klasik (TF-IDF + LogReg) | 1.00 | 1.00 | 1.00 | 1.00 | ±0,12 detik | Ya | ~0 (lokal) |
| LLM API (Gemini) | 1.00 | 1.00 | 1.00 | 1.00 | puluhan detik–menit* | Tidak | Per token / kuota API |

\*Dibatasi free tier 5 request/menit, sehingga butuh jeda antar request.

**Confusion matrix (kedua pendekatan identik):**

| | Prediksi negatif | Prediksi positif |
|---|---|---|
| **Aktual negatif** | 18 | 0 |
| **Aktual positif** | 0 | 22 |

**Interpretasi metrik:**
- **Accuracy 1.00**: seluruh 40 ulasan uji diklasifikasikan benar.
- **Precision 1.00**: tidak ada false positive; setiap label yang diprediksi selalu benar.
- **Recall 1.00**: tidak ada ulasan yang terlewat; semua ulasan negatif dan positif berhasil dikenali.
- **F1-Score 1.00**: keseimbangan precision dan recall sempurna.
- Metrik memakai **macro average** agar kelas positif dan negatif berbobot sama.

**Catatan kritis:** skor sempurna model klasik **tidak mencerminkan kemampuan generalisasi**. Karena 97,5% kalimat test sudah ada di train, model cenderung *menghafal*. Pada uji tambahan dengan split per kalimat unik (`StratifiedGroupKFold`, tanpa overlap), accuracy model klasik turun ke **~0,63** karena gagal memahami **negasi**:
- "Produknya rapi dan bersih, **tidak ada cacat** sama sekali." (positif) → diprediksi negatif
- "Bahan murahan, **tidak sesuai** dengan harga yang dibayarkan." (negatif) → diprediksi positif

## 6. Analisis Trade-off

| Aspek | Model Klasik | LLM API (Gemini) |
|---|---|---|
| Performa (dataset ini) | 1.00 (terbantu leakage) | 1.00 (tanpa training) |
| Effort implementasi | Perlu data berlabel, vectorizer, training | Cukup prompt + API key |
| Kecepatan | Sangat cepat (<1 detik untuk 40 data) | Lambat: latency jaringan + rate limit |
| Biaya | ~0 setelah training | Gratis terbatas; berbayar per token saat volume besar |
| Stabilitas operasional | Offline, dikendalikan penuh | Bergantung server, kuota, dan versi model |
| Pemahaman konteks/negasi | Lemah | Kuat |
| Privasi data | Data tetap internal | Data dikirim ke pihak ketiga |

**Kendala nyata selama eksperimen Gemini:**
1. `404 NOT_FOUND`: model `gemini-2.0-flash` sudah dipensiunkan.
2. `503 UNAVAILABLE`: server sibuk (*high demand*).
3. `429 RESOURCE_EXHAUSTED`: batas free tier 5 request/menit terlampaui.
4. Output `None`: salah memilih varian model audio (`-tts`).

Semua kendala ini **tidak terjadi pada model klasik** dan menunjukkan adanya *operational overhead* pada LLM API.

## 7. Limitation

**Model klasik:** tidak memahami negasi dan konteks; terbatas pada vocabulary data latih (slang, typo, emoji tidak dikenali); rawan menghafal pada dataset kecil dan berulang; perlu retraining jika pola bahasa berubah.

**LLM API:** lambat dan dibatasi kuota; biaya naik seiring volume; format output bisa tidak konsisten sehingga perlu normalisasi; model dapat dipensiunkan sewaktu-waktu; bersifat black box; ada isu privasi data pelanggan.

**Dataset:** hanya 40 kalimat unik yang bersifat template; tidak ada kelas netral atau ulasan campuran; test set kecil sehingga metrik sensitif terhadap 1–2 kesalahan.

## 8. Rekomendasi Technical Approach

**Rekomendasi: model klasik sebagai sistem utama produksi, dengan Gemini sebagai pendukung (hybrid).**

Alasan:
1. **Performa setara** pada dataset ini (F1 1.00 vs 1.00), sehingga keputusan ditentukan oleh faktor operasional.
2. **Kecepatan dan biaya**: model klasik memproses 40 data dalam ±0,12 detik dengan biaya ~0. Gemini dibatasi 5 request/menit dan berbayar per token. Untuk ribuan ulasan per hari di halaman produk, model klasik jauh lebih layak.
3. **Stabilitas**: model klasik bebas dari risiko model dipensiunkan, server sibuk, dan kuota habis, yang semuanya dialami langsung selama eksperimen.

Namun, karena model klasik terbukti lemah pada negasi saat diuji tanpa leakage (~0,63), langkah pengembangan berikutnya adalah:
- **Perbanyak dan perkaya data latih** dengan ulasan nyata yang beragam, lalu evaluasi ulang dengan split per kalimat unik.
- **Gunakan Gemini sebagai asisten labeling** untuk memberi label awal pada ulasan baru, divalidasi manusia, lalu dipakai untuk retraining model klasik.
- **Gunakan Gemini sebagai fallback** untuk prediksi dengan confidence rendah dari model klasik (misal `predict_proba` < 0,6), agar kasus ambigu atau bernegasi tetap tertangani tanpa membebani biaya API untuk seluruh ulasan.

## 9. Cara Menjalankan

Prasyarat: Python 3.10+ dan API key Gemini dari [Google AI Studio](https://aistudio.google.com).

```bash
# 1. Clone repository
git clone https://github.com/<username>/assignment-model-experiment-<nama-peserta>.git
cd assignment-model-experiment-<nama-peserta>

# 2. Buat & aktifkan virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependency
pip install -r requirements.txt

# 4. Siapkan API key
cp .env.example .env               # lalu isi: GEMINI_API_KEY=your_api_key_here

# 5. Jalankan notebook
jupyter notebook notebook/experiment_notebook.ipynb
```

Di notebook, pilih kernel `.venv`, lalu jalankan **Run All**. Pemanggilan Gemini memakan waktu beberapa menit karena batas free tier 5 request/menit.

> File `.env` tidak ikut di-commit (terdaftar di `.gitignore`). Jangan menaruh API key langsung di notebook.
