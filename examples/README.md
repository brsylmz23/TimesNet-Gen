# Examples

This directory contains example scripts and notebooks to help you get started with TimesNet-PointCloud.

## 📂 Contents

### 1. `demo_quick_start.py`
**Quick demonstration script** - Generates and visualizes synthetic seismic waveforms.

```bash
# First, generate some samples
cd ..
python generate_samples.py --num_samples 10

# Then run the demo
cd examples
python demo_quick_start.py
```

**What it does:**
- Loads generated samples from a specific station
- Plots time series (E-W, N-S, U-D channels)
- Computes and plots Fourier Amplitude Spectra
- Displays statistical summary
- Saves output figure

**Output:** `demo_output_station_XXXX.png`

---

### 2. `demo_notebook.ipynb` (Coming Soon)
**Interactive Jupyter notebook** - Step-by-step tutorial with visualizations.

```bash
# Install Jupyter if needed
pip install jupyter

# Launch notebook
jupyter notebook demo_notebook.ipynb
```

**What it covers:**
- Loading pre-trained models
- Generating samples
- Time series visualization
- Frequency analysis (FAS, HVSR)
- Statistical analysis
- Multi-station comparison

---

### 3. `sample_outputs/`
Example outputs from the generation process.

---

## 🚀 Quick Start

### Option 1: Python Script (Fastest)
```bash
cd examples
python demo_quick_start.py
```

### Option 2: Jupyter Notebook (Interactive)
```bash
cd examples
jupyter notebook demo_notebook.ipynb
```

### Option 3: Command Line (Direct)
```bash
# Generate 50 samples per station
python generate_samples.py --num_samples 50

# Check outputs
ls generated_samples/generated_timeseries_npz/
ls generated_samples/preview_plots/
```

---

## 📊 Expected Outputs

After running the demo, you should see:

1. **Time Series Plots**: 3-channel waveforms (E-W, N-S, U-D)
2. **Frequency Spectra**: FAS for each channel
3. **Statistics**: Mean/std of dominant frequency and amplitude
4. **Saved Figure**: `demo_output_station_XXXX.png`

---

## 🔧 Customization

### Change Station
Edit `demo_quick_start.py`:
```python
station_id = '2020'  # Change to: 0205, 1716, 2020, 3130, 4628
```

### Generate More Samples
```bash
python ../generate_samples.py --num_samples 100
```

### Analyze Different Channels
Edit the channel index in the script:
```python
channel_idx = 0  # 0: E-W, 1: N-S, 2: U-D
```

---

## 📚 Additional Resources

- **Full Documentation**: `../docs/GENERATION_README.md`
- **Quick Start Guide**: `../docs/QUICKSTART.md`
- **Main README**: `../README.md`

---

## ❓ Troubleshooting

### "Generated samples not found"
Run the generation script first:
```bash
cd ..
python generate_samples.py --num_samples 10
```

### "Checkpoint not found"
Download the pre-trained model:
- See `../checkpoints/README.md` for download link
- Place the `.pth` file in `../checkpoints/`

### "Module not found"
Install dependencies:
```bash
cd ..
pip install -r requirements.txt
```

---

**Happy generating! 🎉**

