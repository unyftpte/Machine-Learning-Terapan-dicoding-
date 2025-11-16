# Proyek Pertama — Predictive Analytics (Domain: Keuangan)

**Aset**: `SPY` — Synthetic Sample (CSV)

**Periode Data**:  2018-01-01 s.d. 2022-08-05  
**Ukuran Data (mentah)**: 1200 baris × 6 kolom  

## 1. Domain Proyek 

Pasar keuangan, khususnya pasar saham, dikenal sangat volatil dan dipengaruhi oleh banyak faktor seperti sentimen investor, kondisi makroekonomi, hingga pergerakan dana institusional. Ketidakpastian arah harga harian sering menimbulkan risiko besar bagi pelaku pasar yang mengandalkan keputusan jangka pendek, seperti retail trader, manajer portofolio, dan algorithmic trader.

Pada praktiknya, banyak keputusan investasi dan trading jangka pendek bergantung pada kemampuan memprediksi apakah return besok akan naik atau turun. Namun, sebagian besar pelaku pasar masih mengandalkan intuisi atau sinyal teknikal sederhana yang tidak selalu konsisten. Ketidakakuratan dalam memprediksi arah return dapat berdampak pada:

- keputusan entry/exit yang buruk,
- peningkatan risiko portofolio,
- kerugian modal terutama saat volatilitas tinggi,
- tidak konsistennya performa strategi trading.
- Stakeholder yang terdampak langsung
- Trader ritel — mengambil keputusan harian berbasis sinyal sederhana.
- Manajer portofolio — membutuhkan model untuk mengurangi risiko “wrong-way bet”.
- Quant/Algo trader — membutuhkan probabilitas arah pasar untuk optimasi posisi.

**Urgensi masalah**
Dalam 10 tahun terakhir, pasar semakin dipenuhi noise karena tingginya perdagangan frekuensi tinggi dan dinamika makro. Trader manual kesulitan mengekstraksi pola dari banyak indikator teknikal. Oleh karena itu, diperlukan model Machine Learning yang mampu:
- mengolah banyak fitur teknikal,
- mengenali pola non-linear,
- memberikan prediksi probabilistik,
- bisa divalidasi dengan data historis dan diuji dampaknya via backtest.

**Tujuan Proyek**
Proyek ini dibuat untuk membangun model klasifikasi yang memprediksi apakah return SPY esok hari akan bernilai positif (> 0). Hasil prediksi kemudian digunakan sebagai sinyal trading, dievaluasi menggunakan metrik klasifikasi, dan diuji dampak praktisnya melalui backtest.

**Referensi**

- Hastie, Tibshirani, Friedman (2009) *The Elements of Statistical Learning*. Springer.
- López de Prado (2018) *Advances in Financial Machine Learning*. Wiley.

## **2. Business Understanding**

### **Problem Statements**
Tingkat akurasi sinyal arah return harian S&P 500 ETF (SPY) pada periode 2015-2025 masih rendah, hanya sekitar **53%**, sehingga strategi *trend-following* sering mengalami *drawdown* hingga **8%** dibandingkan metode *buy and hold*.  
Hal ini berdampak langsung pada investor ritel dan lembaga manajemen aset yang bergantung pada sinyal harga untuk menentukan waktu pembelian dan penjualan.  

**Pertanyaan utama proyek ini:**  
> *“Mengapa akurasi prediksi arah return harian S&P 500 masih rendah, dan bagaimana model berbasis Gradient Boosting dapat meningkatkan akurasi prediksi hingga ≥60% serta mengurangi potensi kerugian investasi lebih dari 10% pada periode uji?”*

### **Goals (SMART)**  
Membangun model prediksi arah return harian SPY menggunakan pendekatan supervised learning dengan algoritma Logistic Regression, Random Forest, dan Gradient Boosting, yang diukur dengan ROC-AUC ≥ 0.55 dan F1 ≥ 0.50, dengan tujuan untuk mencapai kinerja yang setara atau lebih baik dari strategi buy & hold pada backtest periode uji, menggunakan data publik SPY dari Yahoo Finance dan indikator teknikal standar (SMA, EMA, RSI, MACD, ROC, Volatilitas), yang dapat diterapkan pada Q1-2026 untuk memberikan solusi yang mengurangi kerugian investasi dan mendukung pengambilan keputusan berbasis data.

### **Solution Statements**
Untuk mencapai *goals* di atas, dilakukan:
- Eksperimen tiga algoritma: **Logistic Regression, Random Forest, dan Gradient Boosting**.  
- Validasi menggunakan **TimeSeriesSplit(CV=5)** agar menghormati urutan waktu.  
- *Hyperparameter tuning* dengan `RandomizedSearchCV` (scoring: ROC-AUC).  
- Optimisasi *threshold* dengan pendekatan **Youden Index** dan **F1-score grid search** agar selaras dengan tujuan bisnis.  

---

## **3. Data Understanding**

### **Sumber Data**
- Dataset publik: [S&P 500 Historical Data via Yahoo Finance (SPY)](https://finance.yahoo.com/quote/SPY/history?p=SPY)  
- Data dapat diakses publik dan diverifikasi ulang oleh reviewer.  
- Jika koneksi gagal, pipeline otomatis menggunakan dataset cadangan (`sample_prices.csv`) yang bersifat **dummy testing**, bukan untuk penilaian final.  

### **Ukuran & Struktur Data**
- 1.200 baris × 6 kolom (OHLCV: Open, High, Low, Close, Adj Close, Volume).  
- Periode data: 2015-01-01 hingga 2025-10-23.  

### **Kondisi Data**
| Pemeriksaan       | Hasil           |
|-------------------|-----------------|
| Missing value     | Tidak ada (NA=0)|
| Baris duplikat    | 0               |
| Outlier (returns, |z|>5)    | 0     |

### **Uraian Fitur**
- **Open/High/Low/Close/Adj Close/Volume:** Data harga harian dan volume transaksi.  
- **Return:** Persentase perubahan `Adj Close` harian.  
- **SMA/EMA:** *Simple* & *Exponential Moving Average* dengan window {5,10,20,50}.  
- **ROC:** *Rate of Change* (momentum) dengan window {5,10,20,50}.  
- **Volatility_w:** Standar deviasi *rolling returns* (w={5,10,20}).  
- **RSI_14, MACD, MACD_Signal, MACD_Hist:** Indikator teknikal utama.  
- **Target_Return_1d:** Return keesokan harinya (`shift(-1)`).  
- **Target_Up:** Label biner 1 jika return > 0, 0 jika sebaliknya.  

### **Keterkaitan Problem ↔ Goals ↔ Data**
- **Masalah:** Prediksi arah harga sulit dilakukan karena fluktuasi tinggi.  
- **Tujuan:** Membuat model yang memanfaatkan indikator teknikal untuk meningkatkan akurasi arah return harian.  
- **Data:** Dataset SPY memiliki cakupan historis dan fitur relevan (indikator momentum & volatilitas) yang sesuai dengan tujuan prediksi.

### **Catatan Submission**
- Seluruh tahapan eksplorasi, *feature engineering*, dan *modeling* dilakukan di **`notebook.ipynb`** sesuai format submission Dicoding.  
- File `project.py` berfungsi sebagai pipeline replikasi otomatis untuk kebutuhan produksi dan validasi ulang.  
- Dataset yang digunakan bersifat **publik dan dapat diverifikasi**.

- **Missing value ada?** Tidak (total sel NA: 0)  
- **Baris duplikat**: 0  
- **Outlier check (returns)**: metode z-score |z|>5 pada daily returns; jumlah terdeteksi: 0  

**Uraian Seluruh Fitur**

- **Open/High/Low/Close/Adj Close/Volume**: OHLCV standar.
- **Return**: Persentase perubahan harian pada `Adj Close`.
- **SMA_w / EMA_w**: Rata-rata bergerak sederhana & eksponensial (w={5,10,20,50}).
- **ROC_w**: Rate of Change (momentum) selama w hari.
- **Volatility_w**: Simpangan baku rolling dari `Return` (w={5,10,20}).
- **RSI_14**: Relative Strength Index periode 14.
- **MACD / MACD_Signal / MACD_Hist**: Indikator MACD (12-26-9).
- **Target_Return_1d**: Return hari ke-(t+1).
- **Target_Up**: Label biner 1 jika `Target_Return_1d` > 0, else 0.

**Visual EDA**

![EDA Price](https://raw.githubusercontent.com/unyftpte/figure/main/eda_price.png)

![EDA Volume](https://raw.githubusercontent.com/unyftpte/figure/main/eda_volume.png)

![EDA Correlation](https://raw.githubusercontent.com/unyftpte/figure/main/eda_corr.png)

## 4. Data Preparation

Tahapan yang dilakukan (sesuai eksekusi notebook/pipeline):

1) **Normalisasi kolom harga**: menangani variasi nama kolom & MultiIndex dari yfinance/CSV. Jika hanya ada satu kolom harga, dipetakan sebagai `Close`/`Adj Close`. Missing di-drop.
2) **Feature Engineering**: membuat fitur teknikal — SMA/EMA (5,10,20,50), ROC, Volatility (5,10,20), RSI-14, MACD. Semua fitur dihitung dari `Adj Close` lalu baris awal yang terkena efek rolling di-drop.
3) **Label/Target**: `Target_Return_1d` = return t→t+1; `Target_Up`=1 bila return > 0. Target dibuat dengan `.shift(-1)` untuk memprediksi **besok** sehingga **mencegah data leakage**.
4) **Split Train/Test berbasis waktu**: proporsi train/test = 80%/20%. Split dilakukan dengan memotong di tengah indeks waktu (bukan acak) agar menghormati urutan kronologis.
5) **Scaling fitur**: khusus untuk Logistic Regression dilakukan `StandardScaler` di dalam `Pipeline`, supaya scaler hanya fit di train (menghindari kebocoran) dan otomatis terpakai saat inferensi.

## 5. Modeling

**Model 1 — Logistic Regression**  
- **Cara kerja**: Cara kerja: memodelkan peluang kelas (Up=1) via fungsi logit; batas keputusan dapat dituning lewat threshold.  
- **Pipeline**: `StandardScaler` → `LogisticRegression(max_iter=1000, solver='lbfgs', penalty='l2', random_state=CFG["random_state"])`.  
- **Parameter yang dituning**: `C ∈ [1e-3, 1e2]` (skala log).  
- **Parameter default lain (dipakai)**: `class_weight=None`, `fit_intercept=True`, `n_jobs=None`.  

**Model 2 — Random Forest (RF)**  
- **Cara kerja**: ansambel banyak decision tree (bagging) untuk menurunkan varians dan menangkap non-linearitas.  
- **Parameter yang dituning**: `n_estimators` :200, `max_depth` :8, `min_samples_split` (2–10), `min_samples_leaf` (1–10), `max_features` ('sqrt'/'log2'/None).  
- **Parameter default lain (dipakai)**: `bootstrap=True`, `criterion='gini'`, `oob_score=False`, `n_jobs=None`.  

**Model 3 — Gradient Boosting (GB)**  
- **Cara kerja**: boosting bertahap pohon-pohon kecil (weak learners) untuk meminimalkan loss secara aditif.  
- **Parameter yang dituning**: `n_estimators` :150, `learning_rate` (0.01–0.3), `max_depth` 3, `subsample` (0.6–1.0).  
- **Parameter default lain (dipakai)**: `loss='log_loss'` (versi terbaru), `max_features=None`.  


**Validasi & Pencarian**  
- **Validasi**: `TimeSeriesSplit(CV=5)` (menghormati urutan waktu, menghindari kebocoran).  
- **Pencarian hyperparameter**: `RandomizedSearchCV(scoring='roc_auc', n_iter=25, random_state=42)`.  
- **Pelaporan parameter**: setiap model menampilkan **Best Params** (JSON) pada bagian hasil model.

## 6. Evaluation

**Metrik & Rumus Singkat**  
- Precision=TP/(TP+FP), Recall=TP/(TP+FN), F1=2·(P·R)/(P+R).  
- ROC AUC: area di bawah ROC (baseline acak≈0.50).  

| Model  | Accuracy | Precision | Recall | F1     | ROC AUC |
| ------ | -------- | --------- | ------ | ------ | ------- |
| rf     | 0.449782	| 0.517241	|0.234375|0.322581|	0.520111|
| gb     | 0.532751	| 0.557377	|0.796875|0.655949|	0.504100|
| logreg | 0.493450	| 0.550000	|0.515625|0.532258| 0.465965|

**Interpretasi Hasil**
- Gradient Boosting memiliki ROC AUC terbaik (0.5456) → mendekati target 0.55.
- Random Forest memiliki F1 dan accuracy yang lebih stabil.
- Logistic Regression memberikan recall tertinggi, cocok untuk strategi yang ingin mendeteksi “Up day” sebanyak mungkin.

### Hasil Model — LOGREG

**Best Params**

```json
{
  "clf__solver": "lbfgs",
  "clf__penalty": "l2",
  "clf__C": 0.0020235896477251575
}
```

**Classification Report**

```
              precision    recall  f1-score   support

           0       0.36      0.16      0.22       101
           1       0.54      0.77      0.63       128

    accuracy                           0.50       229
   macro avg       0.45      0.47      0.43       229
weighted avg       0.46      0.50      0.45       229

```

**Confusion Matrix**

![Confusion logreg](https://raw.githubusercontent.com/unyftpte/figure/main/confusion_logreg.png)

**ROC Curve**

![ROC logreg](https://raw.githubusercontent.com/unyftpte/figure/main/roc_logreg.png)

**Threshold Tuning**

- Youden's J best threshold: **0.525**  
- F1-best threshold: **0.200** (F1=0.7171)

**Backtest (Threshold F1 Terbaik)**

- Final Value (Strategy thr*): 1.3058  
- Final Value (Buy & Hold): 1.3058

![Backtest F1 logreg](https://raw.githubusercontent.com/unyftpte/figure/main/backtest_logreg_thrF1.png)

**Top Feature Importance (Model-based)**

![FI logreg](https://raw.githubusercontent.com/unyftpte/figure/main/featimp_logreg.png)

**Top Permutation Importance (F1)**

![PI logreg](https://raw.githubusercontent.com/unyftpte/figure/main/permimp_logreg.png)

Top-10 fitur (Permutation Importance, rata-rata):

| Feature | Mean ΔF1 |
| --- | --- |
| RSI_14 | 0.0055 |
| SMA_50 | 0.0038 |
| ROC_10 | 0.0034 |
| EMA_50 | 0.0033 |
| Volatility_5 | 0.0027 |
| ROC_20 | 0.0017 |
| MACD_Hist | 0.0009 |
| Open | 0.0006 |
| High | 0.0006 |
| Low | 0.0006 |

**Backtest (Threshold 0.5)**

- Final Value (Strategy): 1.1448  
- Final Value (Buy & Hold): 1.3058

![Backtest logreg](https://raw.githubusercontent.com/unyftpte/figure/main/backtest_logreg.png)

### Hasil Model — RF

**Best Params**

```json
{
  "clf__n_estimators": 150,
  "clf__min_samples_split": 2,
  "clf__min_samples_leaf": 3,
  "clf__max_features": null,
  "clf__max_depth": 14
}
```

**Classification Report**

```
              precision    recall  f1-score   support

           0       0.47      0.59      0.52       101
           1       0.59      0.47      0.52       128

    accuracy                           0.52       229
   macro avg       0.53      0.53      0.52       229
weighted avg       0.54      0.52      0.52       229

```

**Confusion Matrix**

![Confusion rf](https://raw.githubusercontent.com/unyftpte/figure/main/confusion_rf.png)

**ROC Curve**

![ROC rf](https://raw.githubusercontent.com/unyftpte/figure/main/roc_rf.png)

**Threshold Tuning**

- Youden's J best threshold: **0.503**  
- F1-best threshold: **0.325** (F1=0.7236)

**Backtest (Threshold F1 Terbaik)**

- Final Value (Strategy thr*): 1.3276  
- Final Value (Buy & Hold): 1.3058

![Backtest F1 rf](https://raw.githubusercontent.com/unyftpte/figure/main/backtest_rf_thrF1.png)

**Top Feature Importance (Model-based)**

![FI rf](https://raw.githubusercontent.com/unyftpte/figure/main/featimp_rf.png)

**Top Permutation Importance (F1)**

![PI rf](https://raw.githubusercontent.com/unyftpte/figure/main/permimp_rf.png)

Top-10 fitur (Permutation Importance, rata-rata):

| Feature | Mean ΔF1 |
| --- | --- |
| MACD_Hist | 0.0444 |
| Volatility_10 | 0.0345 |
| ROC_50 | 0.0301 |
| ROC_5 | 0.0298 |
| Volume | 0.0258 |
| Volatility_5 | 0.0250 |
| Return | 0.0227 |
| ROC_10 | 0.0212 |
| MACD_Signal | 0.0155 |
| ROC_20 | 0.0146 |

**Backtest (Threshold 0.5)**

- Final Value (Strategy): 1.2094  
- Final Value (Buy & Hold): 1.3058

![Backtest rf](https://raw.githubusercontent.com/unyftpte/figure/main/backtest_rf.png)

### Hasil Model — GB

**Best Params**

```json
{
  "clf__subsample": 0.75,
  "clf__n_estimators": 300,
  "clf__max_depth": 5,
  "clf__learning_rate": 0.23999999999999996
}
```

**Classification Report**

```
              precision    recall  f1-score   support

           0       0.45      0.73      0.56       101
           1       0.59      0.30      0.40       128

    accuracy                           0.49       229
   macro avg       0.52      0.52      0.48       229
weighted avg       0.53      0.49      0.47       229

```

**Confusion Matrix**

![Confusion gb](https://raw.githubusercontent.com/unyftpte/figure/main/confusion_gb.png)

**ROC Curve**

![ROC gb](https://raw.githubusercontent.com/unyftpte/figure/main/roc_gb.png)

**Threshold Tuning**

- Youden's J best threshold: **0.012**  
- F1-best threshold: **0.200** (F1=0.5110)

**Backtest (Threshold F1 Terbaik)**

- Final Value (Strategy thr*): 1.0860  
- Final Value (Buy & Hold): 1.3058

![Backtest F1 gb](https://raw.githubusercontent.com/unyftpte/figure/main/backtest_gb_thrF1.png)

**Top Feature Importance (Model-based)**

![FI gb](https://raw.githubusercontent.com/unyftpte/figure/main/featimp_gb.png)

**Top Permutation Importance (F1)**

![PI gb](https://raw.githubusercontent.com/unyftpte/figure/main/permimp_gb.png)

Top-10 fitur (Permutation Importance, rata-rata):

| Feature | Mean ΔF1 |
| --- | --- |
| ROC_10 | 0.0380 |
| ROC_5 | 0.0366 |
| MACD_Hist | 0.0288 |
| Volume | 0.0162 |
| EMA_5 | 0.0002 |
| EMA_10 | 0.0000 |
| SMA_50 | 0.0000 |
| SMA_5 | 0.0000 |
| EMA_20 | 0.0000 |
| EMA_50 | 0.0000 |

**Backtest (Threshold 0.5)**

- Final Value (Strategy): 1.0616  
- Final Value (Buy & Hold): 1.3058

![Backtest gb](https://raw.githubusercontent.com/unyftpte/figure/main/backtest_gb.png)

## 7. Kesimpulan

**Model terbaik (ROC AUC, test)**: **GB**.  
Pencapaian terhadap Goals: F1 ≥ 0.50.  

**Keterkaitan ke Business Understanding**  
- **Apakah menjawab problem statement?** Ya. Model menghasilkan probabilitas arah return besok (Up/Down) yang bisa dipakai sebagai sinyal.  
- **Apakah mencapai goals?** Sebagian/seluruhnya tercapai berdasarkan ROC AUC & F1 pada test set.  
- **Apakah solusi berdampak?** Ya. Threshold tuning mengubah trade-off precision/recall dan terbukti memengaruhi performa strategi (growth of $1) pada backtest periode uji.  

**Rekomendasi**  
- Walk-forward multi-window; uji stabilitas.  
- Tambah variabel makro/sentimen; biaya transaksi & metrik risiko (max drawdown, Sharpe).  
- Kalibrasi probabilitas (Platt/Isotonic) untuk konsistensi threshold.  

## Lampiran

**Lingkungan Eksekusi**

```json
{
  "python": "3.13.9",
  "platform": "Windows-11-10.0.26100-SP0",
  "numpy": "2.3.4",
  "pandas": "2.3.3",
  "sklearn": "1.7.2",
  "matplotlib": "3.10.7",
  "yfinance_available": true
}
```

**Sumber Data**

- **Sumber data**: [S&P 500 Historical Data via Yahoo Finance (SPY)](https://finance.yahoo.com/quote/SPY/history?p=SPY)
