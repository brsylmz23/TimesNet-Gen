# Pre-trained Model Checkpoints

## Required Files

Place the following files in this directory to run `generate_samples.py`:

### 1. Phase 1 Model (Required)
- **File:** `timesnet_pointcloud_phase1_final.pth`
- **Size:** ~50 MB
- **Description:** Fine-tuned model on 5 seismic stations
- **Download:** [Link will be provided]

### 2. Latent Bank (Required)
- **File:** `latent_bank_phase1.npz`
- **Size:** ~100 MB
- **Description:** Pre-computed latent vectors for all 5 stations (364 samples total)
- **Contains:**
  - Latent vectors: (N_samples, 6000, 128) per station
  - Means: (N_samples, 6000, 128) per station
  - Stdevs: (N_samples, 6000, 128) per station

### 3. Encoder Statistics (Required)
- **File:** `../pcgen_stats/encoder_feature_std.npy`
- **Size:** ~1 KB
- **Description:** Encoder feature standard deviations for noise injection
- **Note:** Place this file in `./pcgen_stats/` directory (one level up)

---

## Download Instructions

**Option 1: Direct Download**
```bash
# Download from [Google Drive / Zenodo / Hugging Face]
# Place files in this directory
```

**Option 2: Command Line**
```bash
cd checkpoints/

# Download Phase 1 model
wget [URL]/timesnet_pointcloud_phase1_final.pth

# Download latent bank
wget [URL]/latent_bank_phase1.npz

# Download encoder stats
cd ../pcgen_stats/
wget [URL]/encoder_feature_std.npy
```

---

## Verification

After downloading, verify the files:

```bash
ls -lh checkpoints/
# Should show:
# timesnet_pointcloud_phase1_final.pth (~50 MB)
# latent_bank_phase1.npz (~100 MB)

ls -lh pcgen_stats/
# Should show:
# encoder_feature_std.npy (~1 KB)
```

---

## File Structure

```
TimesNet-Gen/
├── checkpoints/
│   ├── timesnet_pointcloud_phase1_final.pth  ← Phase 1 model
│   └── latent_bank_phase1.npz                ← Latent bank
└── pcgen_stats/
    └── encoder_feature_std.npy               ← Encoder stats
```

---

## Notes

- **No training required:** These files are sufficient to run `generate_samples.py`
- **Phase 0 model not needed:** Only Phase 1 (fine-tuned) model is used for generation
- **No real data required:** Latent bank contains all necessary information for generation

---

## Quick Test

After downloading, test the setup:

```bash
python generate_samples.py --num_samples 5
```

If successful, you should see:
```
[INFO] Loaded encoder_std from ./pcgen_stats/encoder_feature_std.npy
[INFO] Using latent bank: ./checkpoints/latent_bank_phase1.npz
[INFO] Loaded 90 latent vectors for station 0205
...
```
