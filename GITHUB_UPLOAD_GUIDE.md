# 📤 GitHub'a Yükleme Rehberi

Bu rehber, **TimesNet-Gen** projesini kendi GitHub hesabınıza nasıl yükleyeceğinizi adım adım gösterir.

---

## 🎯 Ön Hazırlık

### 1. GitHub Hesabı Kontrolü
- GitHub hesabınız var mı? → https://github.com
- Yoksa, ücretsiz hesap oluşturun

### 2. Git Kurulumu Kontrolü
Terminal'de şunu çalıştırın:
```bash
git --version
```

Eğer kurulu değilse:
- **macOS:** `brew install git` veya Xcode Command Line Tools
- **Windows:** https://git-scm.com/download/win
- **Linux:** `sudo apt install git` veya `sudo yum install git`

---

## 📋 Adım Adım Yükleme

### ADIM 1: GitHub'da Yeni Repo Oluştur

1. **GitHub'a giriş yap:** https://github.com
2. **Sağ üstteki "+" butonuna tıkla** → "New repository"
3. **Repository bilgilerini gir:**
   - **Repository name:** `TimesNet-Gen`
   - **Description:** `Generative seismic waveform synthesis using TimesNet architecture`
   - **Public** veya **Private** seç (önerim: Public)
   - **❌ Initialize this repository with:** Hiçbir şeyi seçme (README, .gitignore, license)
   - **Create repository** butonuna tıkla

4. **Repo URL'ini kopyala:**
   - Sayfada göreceksin: `https://github.com/YOUR_USERNAME/TimesNet-Gen.git`
   - Bu URL'i not al!

---

### ADIM 2: Projeyi Git ile Hazırla

Terminal'i aç ve şu komutları sırayla çalıştır:

```bash
# 1. Proje klasörüne git
cd "/Applications/Projects/DeepEQ/Detection of P and S Waves in Strong Motion Earthquake Data/TimesNet-Gen"

# 2. Git deposu başlat
git init

# 3. Git kullanıcı bilgilerini ayarla (ilk kez kullanıyorsan)
git config user.name "Adın Soyadın"
git config user.email "email@example.com"

# 4. Tüm dosyaları staging area'ya ekle
git add .

# 5. İlk commit'i oluştur
git commit -m "Initial commit: TimesNet-Gen generative seismic model"

# 6. Ana branch'i 'main' olarak ayarla
git branch -M main

# 7. GitHub repo'nu remote olarak ekle (URL'i kendi URL'inle değiştir!)
git remote add origin https://github.com/YOUR_USERNAME/TimesNet-Gen.git

# 8. GitHub'a yükle!
git push -u origin main
```

---

### ADIM 3: GitHub Kimlik Doğrulama

`git push` komutunu çalıştırdığında, GitHub kimlik doğrulama isteyecek:

#### Seçenek A: Personal Access Token (Önerilen)
1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. "Generate new token" → "Generate new token (classic)"
3. **Note:** `TimesNet-Gen Upload`
4. **Expiration:** 90 days (veya istediğin süre)
5. **Scopes:** ✅ `repo` (tüm repo yetkilerini seç)
6. "Generate token" → **Token'ı kopyala** (bir daha göremezsin!)
7. Terminal'de şifre sorduğunda, bu token'ı yapıştır

#### Seçenek B: SSH Key
```bash
# SSH key oluştur
ssh-keygen -t ed25519 -C "email@example.com"

# Public key'i kopyala
cat ~/.ssh/id_ed25519.pub

# GitHub → Settings → SSH and GPG keys → New SSH key
# Kopyaladığın key'i yapıştır

# Remote URL'i SSH'a çevir
git remote set-url origin git@github.com:YOUR_USERNAME/TimesNet-Gen.git
git push -u origin main
```

---

### ADIM 4: Yüklemeyi Doğrula

1. **GitHub repo sayfana git:** `https://github.com/YOUR_USERNAME/TimesNet-Gen`
2. **Dosyaların yüklendiğini kontrol et:**
   - ✅ README.md görünüyor mu?
   - ✅ Klasörler (models/, docs/, examples/) var mı?
   - ✅ Python dosyaları (.py) görünüyor mu?

3. **README'nin düzgün görüntülendiğini kontrol et:**
   - Badges göründü mü?
   - Görseller yüklendi mi?
   - Markdown formatı doğru mu?

---

## 🎨 Repo'yu Güzelleştir

### 1. Topics/Tags Ekle
Repo sayfanda → "About" bölümünün yanındaki ⚙️ (Settings) → Topics:
- `deep-learning`
- `seismology`
- `pytorch`
- `generative-model`
- `time-series`
- `earthquake`
- `waveform-synthesis`

### 2. Description Ekle
"About" → Description:
```
Generative seismic waveform synthesis using TimesNet architecture with point-cloud latent space mixing
```

### 3. Website Ekle (Varsa)
"About" → Website: Paper link, demo site, vb.

---

## 📥 Model ve Data Linklerini Ekle

### Model Checkpoint'i Yükle

**Seçenek 1: Google Drive**
1. Model dosyasını (`timesnet_pointcloud_phase1_final.pth`) Google Drive'a yükle
2. Dosyaya sağ tıkla → "Get link" → "Anyone with the link"
3. Link'i kopyala
4. `checkpoints/README.md` dosyasını güncelle:
   ```markdown
   📥 **Download Link:** [Google Drive](https://drive.google.com/file/d/YOUR_FILE_ID/view)
   ```

**Seçenek 2: Hugging Face**
1. https://huggingface.co hesabı oluştur
2. "New Model" → Model adı: `timesnet-gen`
3. Model dosyasını yükle
4. Link'i README'ye ekle:
   ```markdown
   📥 **Download Link:** [Hugging Face](https://huggingface.co/YOUR_USERNAME/timesnet-gen)
   ```

**Seçenek 3: Zenodo**
1. https://zenodo.org hesabı oluştur
2. "New upload" → Model dosyasını yükle
3. DOI al ve README'ye ekle

### Placeholder'ları Güncelle

Şu dosyalarda `YOUR_USERNAME` ve link placeholder'larını değiştir:
- `README.md`
- `GETTING_STARTED.md`
- `checkpoints/README.md`
- `data/README.md`

```bash
# Otomatik değiştirmek için (macOS/Linux):
cd "/Applications/Projects/DeepEQ/Detection of P and S Waves in Strong Motion Earthquake Data/TimesNet-Gen"

# YOUR_USERNAME'i değiştir
find . -name "*.md" -type f -exec sed -i '' 's/YOUR_USERNAME/GERÇEK_KULLANICI_ADIN/g' {} +

# Değişiklikleri commit et ve push et
git add .
git commit -m "Update repository links and usernames"
git push
```

---

## 🔄 Güncellemeler İçin

Proje üzerinde değişiklik yaptıktan sonra:

```bash
# 1. Değişiklikleri staging area'ya ekle
git add .

# 2. Commit oluştur (açıklayıcı mesaj yaz)
git commit -m "Update: açıklama buraya"

# 3. GitHub'a yükle
git push
```

### Örnek Commit Mesajları:
```bash
git commit -m "Add: New visualization feature"
git commit -m "Fix: Bug in data loader"
git commit -m "Update: Documentation improvements"
git commit -m "Refactor: Code cleanup"
```

---

## 🚨 Sorun Giderme

### "Permission denied (publickey)"
→ SSH key'ini doğru ekledin mi? Seçenek B'yi tekrar kontrol et.

### "fatal: remote origin already exists"
```bash
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/TimesNet-Gen.git
```

### "Updates were rejected"
```bash
git pull origin main --allow-unrelated-histories
git push -u origin main
```

### Büyük dosya hatası (>100 MB)
```bash
# .gitignore'a ekle
echo "*.pth" >> .gitignore
git rm --cached checkpoints/*.pth
git commit -m "Remove large checkpoint files"
git push
```

### Yanlış dosya yükledim
```bash
# Dosyayı git'ten kaldır (diskten silmez)
git rm --cached dosya_adi
git commit -m "Remove unwanted file"
git push
```

---

## ✅ Son Kontrol Listesi

Yüklemeden önce kontrol et:

- [ ] `.gitignore` dosyası var ve doğru yapılandırılmış
- [ ] Büyük dosyalar (*.pth, *.mat) .gitignore'da
- [ ] README.md düzgün görünüyor
- [ ] Tüm placeholder'lar (YOUR_USERNAME) değiştirilmiş
- [ ] Model download linki eklendi
- [ ] Email adresi güncellendi
- [ ] LICENSE dosyası var
- [ ] requirements.txt güncel

---

## 🎉 Tebrikler!

Repo'n artık yayında! 🚀

**Repo URL'in:** `https://github.com/YOUR_USERNAME/TimesNet-Gen`

### Sonraki Adımlar:
1. ⭐ Kendi repo'na star ver (istatistikler için)
2. 📢 README'yi paylaş (Twitter, LinkedIn, vb.)
3. 📝 Paper'da repo linkini belirt
4. 🔔 "Watch" butonuna tıkla (issue bildirimlerini al)
5. 📊 GitHub Actions ekle (CI/CD için)

---

**İyi şanslar! 🌟**

