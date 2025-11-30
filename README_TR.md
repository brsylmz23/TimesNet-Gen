# 🌊 TimesNet-Gen: Sismik Dalga Formu Üretimi

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Point-cloud latent space mixing kullanarak gerçekçi sismik zaman serileri üreten derin öğrenme framework'ü**

---

## 🚀 Hızlı Başlangıç

### Kurulum
```bash
git clone https://github.com/YOUR_USERNAME/TimesNet-Gen.git
cd TimesNet-Gen
pip install -r requirements.txt
```

### Model İndir
📥 Pre-trained modeli indir: **[Buraya link ekle]**

Şuraya koy: `checkpoints/timesnet_pointcloud_phase1_final.pth`

### Sample Üret
```bash
python generate_samples.py
```

**Bu kadar!** ✅ 1-2 dakikada 250 sentetik sismik sinyal üretildi!

---

## 📊 Çıktılar

```
generated_samples/
├── generated_timeseries_npz/
│   ├── station_0205_generated_timeseries.npz  (50 sample)
│   ├── station_1716_generated_timeseries.npz  (50 sample)
│   ├── station_2020_generated_timeseries.npz  (50 sample)
│   ├── station_3130_generated_timeseries.npz  (50 sample)
│   └── station_4628_generated_timeseries.npz  (50 sample)
└── preview_plots/
    └── [10 karşılaştırma grafiği]
```

---

## 🎯 Kullanım Örnekleri

### Daha Fazla Sample Üret
```bash
python generate_samples.py --num_samples 100  # 500 sample
python generate_samples.py --num_samples 200  # 1000 sample
```

### Belirli İstasyonlar İçin
```bash
python generate_samples.py --stations 0205 1716 --num_samples 50
```

### Demo Çalıştır
```bash
cd examples
python demo_quick_start.py
```

---

## 📖 Dokümantasyon

- 🚀 **[QUICK_UPLOAD.md](QUICK_UPLOAD.md)** - GitHub'a hızlı yükleme (5 dk)
- 📚 **[GITHUB_UPLOAD_GUIDE.md](GITHUB_UPLOAD_GUIDE.md)** - Detaylı yükleme rehberi
- 📖 **[GETTING_STARTED.md](GETTING_STARTED.md)** - Kullanım kılavuzu
- 📝 **[docs/GENERATION_README.md](docs/GENERATION_README.md)** - Tüm detaylar

---

## 🏗️ Mimari

TimesNet-Gen, yenilikçi **point-cloud generation** yaklaşımı kullanır:

1. **Encoder**: Gerçek sismik sinyallerden latent özellikler çıkarır
2. **Point-Cloud Mixing**: Aynı istasyondan K adet latent noktayı ortalar
3. **Decoder**: 3-kanallı zaman serisi yeniden oluşturur (E-W, N-S, U-D)

### Yenilik: Latent Space Sürekliliği
- **Phase 0**: Encoder/decoder'ı reconstruction loss ile eğit
- **Phase 1**: Eğitim sırasında latent özelliklere Gaussian noise ekle
- **Sonuç**: Düzgün, sürekli latent space → gerçekçi interpolasyonlar

---

## 📁 Proje Yapısı

```
TimesNet-Gen/
├── generate_samples.py         # 🚀 Ana inference scripti
├── untitled1_gen.py            # 🏋️  Eğitim scripti
├── models/                     # 🧠 Model tanımları
├── docs/                       # 📖 Dokümantasyon
├── examples/                   # 📚 Demo scriptleri
└── checkpoints/                # 💾 Model checkpoint'leri
```

---

## 📧 İletişim

- **GitHub Issues**: [Issue aç](https://github.com/YOUR_USERNAME/TimesNet-Gen/issues)
- **Email**: your.email@example.com

---

## 📝 Lisans

MIT License - Detaylar için [LICENSE](LICENSE) dosyasına bakın.

---

**Sismoloji topluluğu için ❤️ ile yapıldı**
