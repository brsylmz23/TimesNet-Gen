# 🚀 Getting Started with TimesNet-Gen

Welcome! This guide will help you get up and running in **5 minutes**.

---

## ⚡ Quick Setup (3 Steps)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Download Pre-trained Model
📥 Download the checkpoint from: **[Add your link here]**

Place it here:
```
checkpoints/timesnet_pointcloud_phase1_final.pth
```

### Step 3: Generate Samples
```bash
python generate_samples.py
```

**That's it!** ✅

---

## 📦 What You Get

After running `generate_samples.py`, you'll have:

```
generated_samples/
├── generated_timeseries_npz/
│   ├── station_0205_generated_timeseries.npz  (50 samples × 3 channels × 6000 timesteps)
│   ├── station_1716_generated_timeseries.npz
│   ├── station_2020_generated_timeseries.npz
│   ├── station_3130_generated_timeseries.npz
│   └── station_4628_generated_timeseries.npz
└── preview_plots/
    ├── station_0205_preview_R0_G0.png
    ├── station_0205_preview_R0_G1.png
    └── ... (10 plots total)
```

**Total:** 250 synthetic seismic waveforms (5 stations × 50 samples)

---

## 🎯 Common Use Cases

### 1. Generate More Samples
```bash
# 100 samples per station (500 total)
python generate_samples.py --num_samples 100

# 200 samples per station (1000 total)
python generate_samples.py --num_samples 200
```

### 2. Generate for Specific Stations
```bash
# Only stations 0205 and 1716
python generate_samples.py --stations 0205 1716 --num_samples 50
```

### 3. Run Demo Script
```bash
cd examples
python demo_quick_start.py
```

### 4. Visualize Results
```bash
python plot_combined_hvsr_all_sources.py
```

---

## 📊 Understanding the Output

### NPZ Files
Each `.npz` file contains:
- `generated_signals`: NumPy array of shape `(N, 3, 6000)`
  - N = number of samples
  - 3 = channels (E-W, N-S, U-D)
  - 6000 = timesteps (30 seconds @ 200 Hz)

**Load in Python:**
```python
import numpy as np

data = np.load('generated_samples/generated_timeseries_npz/station_2020_generated_timeseries.npz')
signals = data['generated_signals']  # Shape: (50, 3, 6000)

# Get first sample, E-W channel
ew_signal = signals[0, 0, :]  # Shape: (6000,)
```

### Preview Plots
- Compare real vs generated waveforms
- 2 generated samples per station
- Visual quality check

---

## 🔧 Configuration

### Change Model Path
Edit `generate_samples.py`:
```python
parser.add_argument('--checkpoint', type=str, 
                    default='./checkpoints/timesnet_pointcloud_phase1_final.pth')
```

### Change Data Path
```python
parser.add_argument('--data_root', type=str, 
                    default=r"D:\Baris\new_Ps_Vs30/")
```

### Change Output Directory
```bash
python generate_samples.py --output_dir ./my_outputs
```

---

## 🎓 Learn More

### Documentation
- 📖 **[README.md](README.md)** - Project overview
- 📚 **[docs/QUICKSTART.md](docs/QUICKSTART.md)** - Quick reference
- 📝 **[docs/GENERATION_README.md](docs/GENERATION_README.md)** - Detailed guide

### Examples
- 🐍 **[examples/demo_quick_start.py](examples/demo_quick_start.py)** - Python demo
- 📓 **[examples/demo_notebook.ipynb](examples/demo_notebook.ipynb)** - Jupyter notebook

### Training
- 🏋️ **[untitled1_gen.py](untitled1_gen.py)** - Train your own model

---

## ❓ Troubleshooting

### "Checkpoint not found"
**Solution:** Download the pre-trained model from the link above and place it in `checkpoints/`

### "Module not found"
**Solution:** Install dependencies:
```bash
pip install -r requirements.txt
```

### "CUDA out of memory"
**Solution:** The model will automatically use CPU if CUDA is unavailable. For faster generation, use a GPU with at least 4GB VRAM.

### "No samples generated"
**Solution:** Check that your data path is correct:
```bash
python generate_samples.py --data_root /path/to/your/data
```

---

## 🆘 Need Help?

- 📧 **Email:** your.email@example.com
- 🐛 **Issues:** [GitHub Issues](https://github.com/brsylmz23/TimesNet-Gen/issues)
- 💬 **Discussions:** [GitHub Discussions](https://github.com/brsylmz23/TimesNet-Gen/discussions)

---

## 🎉 Success!

If you see this output, you're all set:

```
================================================================================
TimesNet-PointCloud Sample Generation
================================================================================
Checkpoint: ./checkpoints/timesnet_pointcloud_phase1_final.pth
Target stations: ['0205', '1716', '2020', '3130', '4628']
Samples per station: 50
...
================================================================================
Generation Complete!
================================================================================
Generated samples saved to: ./generated_samples/generated_timeseries_npz
Preview plots saved to: ./generated_samples/preview_plots
================================================================================
```

**Next steps:**
1. Explore the generated NPZ files
2. Run the demo script: `python examples/demo_quick_start.py`
3. Visualize results: `python plot_combined_hvsr_all_sources.py`
4. Read the full documentation

---

**Happy generating! 🌊⚡**

