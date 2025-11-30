# TimesNet-PointCloud: Quick Start Guide

## 🚀 Two-Step Workflow

### Step 1: Train the Model (One-Time Setup)

First, train the model and save it:

```bash
python untitled1_gen.py
```

This will:
- ✅ Train Phase 0 (base autoencoder)
- ✅ Train Phase 1 (noise injection fine-tuning)
- ✅ Save trained model to `./checkpoints/timesnet_pointcloud_phase1_final.pth`
- ✅ Generate 50 samples per station as evaluation
- ✅ Create diagnostic plots and statistics

**Time**: ~30-60 minutes on GPU

**You only need to do this once!** The saved model can be used for unlimited generation later.

---

### Step 2: Generate Samples (Anytime, Fast!)

After training, generate as many samples as you want:

```bash
# Generate 50 samples per station (default)
python generate_samples.py

# Generate 100 samples per station
python generate_samples.py --num_samples 100

# Generate 200 samples per station
python generate_samples.py --num_samples 200
```

This will:
- ✅ Load pre-trained model from `./checkpoints/timesnet_pointcloud_phase1_final.pth`
- ✅ Generate N samples per station (5 stations × N samples)
- ✅ Save NPZ files to `./generated_samples/generated_timeseries_npz/`
- ✅ Create preview plots in `./generated_samples/preview_plots/`

**Time**: ~1-2 minutes on GPU, ~5-10 minutes on CPU

**No retraining needed!** Run this as many times as you want with different `--num_samples`.

---

## 📦 What You Get

After running `python generate_samples.py --num_samples 50`, you'll have:

### Generated Samples (NPZ files)
```
./generated_samples/
├── generated_timeseries_npz/
│   ├── station_0205_generated_timeseries.npz  (50 samples)
│   ├── station_1716_generated_timeseries.npz  (50 samples)
│   ├── station_2020_generated_timeseries.npz  (50 samples)
│   ├── station_3130_generated_timeseries.npz  (50 samples)
│   └── station_4628_generated_timeseries.npz  (50 samples)
└── preview_plots/
    ├── station_0205_generated_sample_1.png
    ├── station_0205_generated_sample_2.png
    ├── station_1716_generated_sample_1.png
    └── ... (2 plots per station)
```

**Total**: 5 stations × 50 samples = **250 generated seismic signals**

Each NPZ file contains:
- `generated_signals`: (50, 6000, 3) - 50 samples, 6000 time steps, 3 channels (E-W, N-S, U-D)
- `real_names`: List of source filenames used for mixing
- `station_id`: Station identifier (e.g., '0205')

---

## 🔄 Complete Workflow Example

```bash
# 1. Train once (first time only)
python untitled1_gen.py
# ⏱️  Takes ~30-60 minutes
# 💾 Saves model to ./checkpoints/timesnet_pointcloud_phase1_final.pth

# 2. Generate samples (as many times as you want!)
python generate_samples.py
# ⏱️  Takes ~1-2 minutes
# 📦 Creates 5 × 50 = 250 samples (default)

python generate_samples.py --num_samples 100
# ⏱️  Takes ~2-3 minutes
# 📦 Creates 5 × 100 = 500 samples

python generate_samples.py --num_samples 200
# ⏱️  Takes ~3-5 minutes
# 📦 Creates 5 × 200 = 1000 samples
```

**That's it!** No need for shell scripts or complex commands. Just run `python generate_samples.py` and you're done! 🎉

---

## 📊 Load and Use Generated Samples

```python
import numpy as np
import matplotlib.pyplot as plt

# Load generated samples for station 0205
data = np.load('./generated_samples/generated_timeseries_npz/station_0205_generated_timeseries.npz')

generated_signals = data['generated_signals']  # Shape: (50, 6000, 3)
station_id = str(data['station_id'])           # '0205'

print(f"Station: {station_id}")
print(f"Number of samples: {generated_signals.shape[0]}")  # 50
print(f"Time steps: {generated_signals.shape[1]}")         # 6000
print(f"Channels: {generated_signals.shape[2]}")           # 3 (E-W, N-S, U-D)

# Access first sample, E-W channel
ew_signal = generated_signals[0, :, 0]  # Shape: (6000,)

# Plot it
fs = 100  # Sampling frequency (Hz)
time = np.arange(len(ew_signal)) / fs
plt.figure(figsize=(12, 4))
plt.plot(time, ew_signal)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (cm/s²)')
plt.title(f'Generated E-W Signal - Station {station_id}')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 🎯 Key Features

- **Fast Inference**: Generate hundreds of samples in minutes (no training required)
- **K-Sample Mixing**: Generates new samples by mixing encoder features from k=5 real samples
- **Station-Specific**: Each generated sample is conditioned on a specific station
- **Realistic**: Preserves statistical properties and frequency characteristics of real data
- **Flexible**: Generate any number of samples per station with a single parameter

---

## 📖 Full Documentation

For detailed documentation, see:
- **[GENERATION_README.md](./GENERATION_README.md)** - Complete guide with all options
- **[generate_samples.py](./generate_samples.py)** - Standalone inference script
- **[untitled1_gen.py](./untitled1_gen.py)** - Training script (run once to train the model)

---

## ⚙️ Requirements

- Python 3.7+
- PyTorch 1.10+
- NumPy
- Matplotlib
- Pre-trained model checkpoint (download or train using `untitled1_gen.py`)
- CUDA (optional, but recommended for faster generation)

---

## 🐛 Troubleshooting

### "CUDA out of memory"
Reduce batch size in `GenArgs` class (line 30 in `untitled1_gen.py`):
```python
self.batch_size = 16  # Default is 32
```

### "No module named 'TimesNet_PointCloud'"
Make sure `TimesNet_PointCloud.py` is in the same directory.

### "Checkpoint not found"
First run `python untitled1_gen.py` to train the model.

---

## 📧 Support

For issues or questions, please open an issue on GitHub.

---

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@article{your_paper,
  title={Your Paper Title},
  author={Your Name},
  journal={Your Journal},
  year={2025}
}
```

