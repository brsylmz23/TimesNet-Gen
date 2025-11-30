# ⚡ Hızlı Yükleme Rehberi (5 Dakika)

## 🎯 Sadece Bu Komutları Çalıştır!

### 1️⃣ GitHub'da Repo Oluştur
1. https://github.com/new adresine git
2. **Repository name:** `TimesNet-Gen`
3. **Public** seç
4. **❌ Hiçbir şeyi initialize etme**
5. **Create repository**

### 2️⃣ Terminal'de Çalıştır

```bash
# Proje klasörüne git
cd "/Applications/Projects/DeepEQ/Detection of P and S Waves in Strong Motion Earthquake Data/TimesNet-Gen"

# Git başlat
git init
git add .
git commit -m "Initial commit: TimesNet-Gen generative seismic model"
git branch -M main

# GitHub'a bağlan (URL'i kendi URL'inle değiştir!)
git remote add origin https://github.com/KULLANICI_ADIN/TimesNet-Gen.git

# Yükle!
git push -u origin main
```

### 3️⃣ Kimlik Doğrulama
- **Username:** GitHub kullanıcı adın
- **Password:** Personal Access Token (PAT)
  - PAT oluştur: https://github.com/settings/tokens
  - "Generate new token (classic)" → `repo` yetkisini seç
  - Token'ı kopyala ve şifre yerine yapıştır

### 4️⃣ Kontrol Et
https://github.com/KULLANICI_ADIN/TimesNet-Gen

---

## 📝 Sonra Yapılacaklar

### Placeholder'ları Değiştir
```bash
# brsylmz23'i değiştir
cd TimesNet-Gen
find . -name "*.md" -type f -exec sed -i '' 's/brsylmz23/GERÇEK_KULLANICI_ADIN/g' {} +

# Commit ve push
git add .
git commit -m "Update repository links"
git push
```

### Model Linki Ekle
1. Model dosyasını Google Drive'a yükle
2. Link'i `checkpoints/README.md`'ye ekle
3. Commit ve push

---

## 🎉 Bitti!

Repo'n hazır: `https://github.com/KULLANICI_ADIN/TimesNet-Gen`

**Detaylı rehber için:** `GITHUB_UPLOAD_GUIDE.md`

