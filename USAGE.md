# Usage Guide

## Quick Start (3 Steps)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Pre-trained Models

Place these files in `./checkpoints/`:
- `timesnet_pointcloud_phase1_final.pth`
- `latent_bank_phase1.npz`

Place this file in `./pcgen_stats/`:
- `encoder_feature_std.npy`

### 3. Generate Samples

```bash
python generate_samples.py --num_samples 50
```

---

## Command-Line Options

```bash
python generate_samples.py \
    --checkpoint ./checkpoints/timesnet_pointcloud_phase1_final.pth \
    --latent_bank ./checkpoints/latent_bank_phase1.npz \
    --num_samples 50 \
    --stations 0205 1716 2020 3130 4628 \
    --output_dir ./generated_samples
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--checkpoint` | `timesnet_pointcloud_phase1_final.pth` | Model checkpoint path |
| `--latent_bank` | `latent_bank_phase1.npz` | Latent bank path |
| `--num_samples` | `50` | Number of samples per station |
| `--stations` | `0205 1716 2020 3130 4628` | Station IDs to generate |
| `--output_dir` | `./generated_samples` | Output directory |

---

## Output Structure

```
./generated_samples/
├── generated_timeseries_npz/
│   ├── station_0205_generated_timeseries.npz
│   │   ├── generated_signals: (50, 3, 6000)  # 3-channel waveforms
│   │   ├── f0_timesnet: (50,)                # Dominant frequencies
│   │   ├── pdf_timesnet: (20,)               # f₀ histogram
│   │   ├── hvsr_freq_timesnet: (400,)        # HVSR frequency grid
│   │   └── hvsr_median_timesnet: (400,)      # Median HVSR curve
│   └── ...
│
├── preview_plots/
│   └── station_XXXX_preview_N.png            # Quick visualization
│
└── hvsr_analysis/
    ├── combined_hvsr_f0_station_XXXX.png     # HVSR + f₀ comparison
    ├── extended_timesnet_15x15.png           # Similarity matrix
    └── real_vs_generated_timeseries_pairs/   # Detailed comparisons
```

---

## Loading Generated Data

```python
import numpy as np

# Load generated samples
data = np.load('./generated_samples/generated_timeseries_npz/station_0205_generated_timeseries.npz')

# Access waveforms (50 samples, 3 channels, 6000 timesteps)
waveforms = data['generated_signals']  # Shape: (50, 3, 6000)

# Access f₀ values
f0_values = data['f0_timesnet']  # Shape: (50,)

# Access HVSR curve
hvsr_freq = data['hvsr_freq_timesnet']    # Frequency grid
hvsr_curve = data['hvsr_median_timesnet']  # Median HVSR
```

---

## Examples

### Generate 100 samples for all stations
```bash
python generate_samples.py --num_samples 100
```

### Generate for specific stations only
```bash
python generate_samples.py --stations 0205 1716 --num_samples 50
```

### Use custom checkpoint
```bash
python generate_samples.py \
    --checkpoint ./my_models/custom_model.pth \
    --latent_bank ./my_models/custom_latent_bank.npz
```

---

## Troubleshooting

### Error: "Latent bank not found"
- Ensure `latent_bank_phase1.npz` is in `./checkpoints/`
- Check the path with `--latent_bank` argument

### Error: "Checkpoint not found"
- Ensure `timesnet_pointcloud_phase1_final.pth` is in `./checkpoints/`
- Check the path with `--checkpoint` argument

### Error: "encoder_feature_std.npy not found"
- Ensure `encoder_feature_std.npy` is in `./pcgen_stats/`
- This file is required for noise injection

---

## Performance

- **Generation Speed**: ~10-20 seconds for 250 samples (50 per station × 5 stations)
- **Memory Usage**: ~2 GB GPU / 4 GB RAM
- **Disk Space**: ~100 MB for latent bank, ~50 MB per 50 samples output

---

## Next Steps

- Analyze generated waveforms with your own tools
- Compare HVSR characteristics with real data
- Use generated samples for data augmentation
- Fine-tune on your own seismic stations

