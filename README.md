# EnerjiSon — Elektrik Üretim, Tüketim ve Fiyat Tahmini

Türkiye elektrik piyasasına ait üretim, tüketim ve piyasa takas fiyatı (AOF) verilerini 2021-2025 yılları arasında birleştirip; klasik zaman serisi, makine öğrenmesi, derin öğrenme ve hibrit (stacking) yaklaşımlarıyla saatlik/günlük tahmin modelleri kuran kapsamlı bir enerji analitiği projesi.

## Proje Hakkında

Proje, EPİAŞ şeffaflık platformu formatındaki ham CSV verilerini (üretim, tüketim, fiyat) hava durumu verileriyle birleştirerek; **tüketim miktarı** ve **piyasa fiyatı** için gelecek tahminleri üreten modeller geliştirmeyi amaçlıyor. SARIMAX'tan Temporal Fusion Transformer'a (TFT) kadar geniş bir model yelpazesi denenmiştir.

## Proje Yapısı

```
EnerjiSon/
├── Üretim/                    # 2021-2025 saatlik gerçek zamanlı üretim verileri (3'er aylık CSV dosyaları)
├── Tüketim/                   # 2021-2025 saatlik gerçek zamanlı tüketim verileri
├── Fiyat/                     # 2021-2025 saatlik ağırlıklı ortalama fiyat (AOF) verileri
├── Hava/
│   └── hava.ipynb             # Hava durumu verisinin işlenmesi
├── claude/
│   ├── veri_birlestirme.ipynb # Üretim+tüketim+fiyat+hava verisinin nihai birleştirilmesi
│   └── elektrik_hava_son.csv  # Birleştirilmiş nihai veri seti
├── Modeller/
│   ├── ml.ipynb               # Klasik makine öğrenmesi modelleri
│   ├── dl.ipynb                # Derin öğrenme modelleri (LSTM/CNN-LSTM vb.)
│   ├── time.ipynb             # SARIMAX ve VARMAX zaman serisi modelleri
│   ├── hybrid.ipynb           # Hibrit/stacking modeller ve TFT
│   ├── tft_ckpts/             # Temporal Fusion Transformer checkpoint dosyaları (final)
│   └── *.png                  # Tahmin, önem (importance) ve tanı grafikleri
├── verionisleme.ipynb          # Genel veri ön işleme
├── tumveri.ipynb                # Ham verilerin okunup birleştirildiği ana notebook
├── tüm_üretim.csv / tüm_tüketim.csv / tüm_fiyat.csv  # Üretim, tüketim ve fiyat verilerinin ayrı ayrı birleştirilmiş halleri
├── birlesik_veri.csv            # Üretim+tüketim+fiyat birleştirilmiş ara veri seti
└── *.pdf                        # Bitirme çalışması raporu
```

## Yöntem

### 1. Veri Toplama ve Birleştirme (`tumveri.ipynb`, `veri_birlestirme.ipynb`)
- Üretim, tüketim ve fiyat verilerinin (`;` ayraçlı, Türkçe ondalık formatlı CSV'ler) tek tek okunup birleştirilmesi
- Türkçe sayı formatlarının (`29.489,46` → `29489.46`) düzeltilmesi
- Tarih-saat alanlarının birleştirilip `Datetime` sütununa dönüştürülmesi
- Üretim, tüketim, fiyat ve hava durumu verilerinin `Datetime` üzerinden `merge` edilmesi
- Bütünlük doğrulama (beklenen satır sayısı kontrolü)

### 2. Özellik Mühendisliği
- Döngüsel zaman kodlaması (saat/gün/ay için sin-cos dönüşümü)
- Hedef değişken için lag özellikleri (24s, 48s, 168s)
- Hafta sonu / tatil gibi ikili (binary) özellikler
- Fourier bazlı takvim özellikleri (haftalık ve yıllık mevsimsellik)

### 3. Modelleme (`Modeller/` klasörü)
- **Klasik ML (`ml.ipynb`):** Ağaç tabanlı ve doğrusal modeller, `TimeSeriesSplit` ile çapraz doğrulama
- **Derin Öğrenme (`dl.ipynb`):** PyTorch tabanlı LSTM/CNN-LSTM modelleri, `RobustScaler` ile normalizasyon
- **Klasik Zaman Serisi (`time.ipynb`):** SARIMAX ve VARMAX, günlük resample edilmiş veri üzerinde
- **Hibrit/Gelişmiş (`hybrid.ipynb`):** Optuna ile hiperparametre optimizasyonu, Temporal Fusion Transformer (TFT), zaman bazlı fold'larla model birleştirme (stacking), meta-model (`meta_model.joblib`)

### 4. Değerlendirme
- Fiyat tahmini için özelleştirilmiş metrikler (düşük fiyat saatlerinin MAPE/SMAPE hesaplamasından hariç tutulması)
- RMSE, MAE, MAPE, SMAPE, NRMSE metrikleri
- Tahmin, önem skoru (feature importance) ve kalıntı (residual) diyagnostik grafikleri (`Modeller/*.png`)

## Kullanılan Teknolojiler

- Python 3, pandas, numpy
- scikit-learn (klasik ML, ölçekleme, çapraz doğrulama)
- PyTorch & PyTorch Lightning (derin öğrenme, TFT)
- statsmodels (SARIMAX, VARMAX)
- Optuna (hiperparametre optimizasyonu)
- matplotlib (görselleştirme)

## Kurulum ve Çalıştırma

Bu projeyi çalıştırmak için yukarıdaki kütüphanelerin (pandas, numpy, scikit-learn, torch, pytorch-lightning, statsmodels, optuna, matplotlib) kendi Python ortamınıza kurulu olması gerekir.

Notebook'ları sırasıyla çalıştırın:

```bash
jupyter notebook tumveri.ipynb          # 1. Ham verileri birleştir
jupyter notebook claude/veri_birlestirme.ipynb   # 2. Hava verisiyle birleştir
jupyter notebook Modeller/ml.ipynb      # 3a. Klasik ML modelleri
jupyter notebook Modeller/dl.ipynb      # 3b. Derin öğrenme modelleri
jupyter notebook Modeller/time.ipynb    # 3c. Zaman serisi modelleri
jupyter notebook Modeller/hybrid.ipynb  # 3d. Hibrit/TFT modelleri
```

## Çıktılar

- Tüketim ve fiyat için karşılaştırmalı model performans tabloları
- TFT dikkat (attention) ve önem (importance) grafikleri
- SARIMAX ve CNN-LSTM tahmin grafikleri
- Eğitilmiş model checkpoint'leri (`tft_ckpts/`)

## Not

- Kod içerisindeki dosya yolları (`C:/Users/kerem/Desktop/...`) yerel makineye özeldir; projeyi çalıştırmadan önce kendi veri yolunuzla güncellemeniz gerekir.
- `lightning_logs/` klasörü deneme eğitim loglarından oluşur ve tekrar üretilebilir; **repoya yüklemeden önce bu klasörü silmeniz veya `.gitignore` ile hariç tutmanız gerekir** (şu an proje klasöründe mevcut).
- `tft_ckpts/` altında final modellerin yanı sıra ara sürüm (`-v1`, `-v2` gibi) checkpoint dosyaları da bulunuyor; repoya yüklemeden önce yalnızca final checkpoint'leri (`tft_Fiyat_best.ckpt`, `tft_Tuketim_best.ckpt`) bırakıp diğerlerini silmeniz önerilir.
- Proje klasöründe `birleşik_veri.csv` adında iki farklı kopya (biri Türkçe karakterli, biri karaktersiz dosya adıyla) bulunuyor; bunlardan yalnızca birini repoya eklemeniz, diğerini silmeniz önerilir.
- Büyük checkpoint (`.ckpt`, `.pth`) dosyalarını repoya yüklemeden önce `.gitignore` ile hariç tutmanız; gerekirse Git LFS veya harici depolama kullanmanız önerilir.
