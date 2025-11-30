# TimesNet-PointCloud: Sample Generation

This guide explains how to train and generate seismic time series samples using TimesNet-PointCloud model.

## Quick Start

### Step 1: Train the Model (One-Time)

Train the model and save it for future use:

```bash
python untitled1_gen.py
```

This will:
1. **Phase 0**: Train base autoencoder model
2. **Phase 1**: Fine-tune with noise injection
3. **Evaluate**: Generate 50 samples per station for evaluation
4. **Save**: Model checkpoint saved to `./checkpoints/timesnet_pointcloud_phase1_final.pth`
5. **Output**: Evaluation results saved to `./gen_eval_results_base/phase1/`

**Training time**: ~30-60 minutes on GPU

**Important**: You only need to run this once! The saved model can be reused for unlimited generation.

---

### Step 2: Generate Samples (Fast Inference)

After training, generate as many samples as you need:

```bash
# Generate 50 samples per station (default)
python generate_samples.py

# Generate 100 samples per station
python generate_samples.py --num_samples 100

# Custom checkpoint and output directory
python generate_samples.py --checkpoint ./checkpoints/timesnet_pointcloud_phase1_final.pth --num_samples 200 --output_dir ./my_samples
```

**Generation time**: ~1-2 minutes on GPU, ~5-10 minutes on CPU

**No retraining needed!** Run this script as many times as you want.

### Full Options

```bash
# All options with defaults
python generate_samples.py \
    --checkpoint ./checkpoints/timesnet_pointcloud_phase1_final.pth \
    --num_samples 50 \
    --output_dir ./generated_samples \
    --num_preview 2 \
    --stations 0205 1716 2020 3130 4628 \
    --data_root "D:\Baris\new_Ps_Vs30/" \
    --seq_len 6000 \
    --pcgen_k 5

# But usually you only need:
python generate_samples.py --num_samples 100
```

## Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--checkpoint` | str | `./checkpoints/timesnet_pointcloud_phase1_final.pth` | Path to pre-trained model checkpoint (.pth file) |
| `--num_samples` | int | 50 | Number of samples to generate per station |
| `--output_dir` | str | `./generated_samples` | Output directory for all results |
| `--num_preview` | int | 2 | Number of preview plots to create per station |
| `--stations` | list | `0205 1716 2020 3130 4628` | Target station IDs |
| `--data_root` | str | `D:\Baris\new_Ps_Vs30/` | Root path to seismic data |
| `--seq_len` | int | 6000 | Sequence length (time steps) |
| `--pcgen_k` | int | 5 | Number of real samples to mix for each generated sample |

## Output Structure

### After Training (`untitled1_gen.py`)

```
./
├── checkpoints/
│   └── timesnet_pointcloud_phase1_final.pth  # Trained model (use for inference)
├── gen_eval_results_base/
│   ├── phase0/  # Phase 0 results
│   │   ├── training_loss.png
│   │   └── ...
│   └── phase1/  # Phase 1 results + generated samples
│       ├── generated_timeseries_npz/
│       │   ├── station_0205_generated_timeseries.npz  (50 samples)
│       │   ├── station_1716_generated_timeseries.npz  (50 samples)
│       │   ├── station_2020_generated_timeseries.npz  (50 samples)
│       │   ├── station_3130_generated_timeseries.npz  (50 samples)
│       │   └── station_4628_generated_timeseries.npz  (50 samples)
│       ├── real_generated_timeseries_pairs/
│       │   └── ... (comparison plots)
│       ├── training_loss.png
│       └── ...
└── pcgen_stats/
    └── encoder_feature_std.npy  # Encoder statistics
```

### After Additional Generation (`generate_samples.py`)

```
generated_samples/
├── generated_timeseries_npz/
│   ├── station_0205_generated_timeseries.npz
│   ├── station_1716_generated_timeseries.npz
│   ├── station_2020_generated_timeseries.npz
│   ├── station_3130_generated_timeseries.npz
│   └── station_4628_generated_timeseries.npz
└── preview_plots/
    ├── station_0205_generated_sample_1.png
    ├── station_0205_generated_sample_2.png
    └── ...
```

### NPZ File Contents

Each NPZ file contains:
- `generated_signals`: numpy array of shape `(num_samples, 6000, 3)` - Generated time series
  - Dimension 0: Sample index (0 to num_samples-1)
  - Dimension 1: Time steps (0 to 5999)
  - Dimension 2: Channels (0=E-W, 1=N-S, 2=U-D)
- `real_names`: List of real signal filenames used for mixing
- `station_id`: Station identifier (e.g., '0205')

### Loading Generated Samples

```python
import numpy as np

# Load generated samples
data = np.load('generated_samples/generated_timeseries_npz/station_0205_generated_timeseries.npz')
generated_signals = data['generated_signals']  # shape: (50, 6000, 3)
real_names = data['real_names']
station_id = str(data['station_id'])

print(f"Station: {station_id}")
print(f"Number of samples: {generated_signals.shape[0]}")
print(f"Sequence length: {generated_signals.shape[1]}")
print(f"Channels: {generated_signals.shape[2]} (E-W, N-S, U-D)")

# Access first sample, E-W channel
ew_channel = generated_signals[0, :, 0]
```

## Generation Method

The script uses **k-sample encoder feature mixing**:

1. For each generated sample, randomly select `k` real samples from the target station
2. Encode all `k` samples using the trained encoder
3. Average the encoder features to create a "mixed" latent representation
4. Decode the mixed features to generate a new time series

This approach ensures that generated samples:
- Are station-specific (only mix samples from the same station)
- Preserve realistic characteristics from real data
- Create novel variations through feature mixing

## Requirements

- PyTorch
- NumPy
- Matplotlib
- TimesNet_PointCloud model definition
- data_loader.py and data_loader_gen.py modules

## Workflow Summary

### Recommended Workflow

```bash
# Step 1: Train once (one-time setup, ~30-60 min)
python untitled1_gen.py
# Output: ./checkpoints/timesnet_pointcloud_phase1_final.pth

# Step 2: Generate samples anytime (fast, ~1-2 min)
python generate_samples.py --num_samples 50
python generate_samples.py --num_samples 100
python generate_samples.py --num_samples 200 --output_dir ./batch2
```

**Key Point**: Training is done once, then you can generate unlimited samples quickly!

## Example: Generate 100 Samples

```bash
python generate_samples.py \
    --checkpoint ./checkpoints/timesnet_pointcloud_phase1_final.pth \
    --num_samples 100 \
    --output_dir ./my_generated_data \
    --num_preview 5
```

This will:
- Generate 100 samples per station (5 stations × 100 = 500 total samples)
- Save NPZ files to `./my_generated_data/generated_timeseries_npz/`
- Create 5 preview plots per station in `./my_generated_data/preview_plots/`

## Notes

- **GPU**: The script automatically uses GPU if available
- **Speed**: Generating 50 samples per station takes ~1-2 minutes on GPU
- **Memory**: Requires ~2GB GPU memory for default settings
- **Reproducibility**: Set random seed in the script for reproducible results

## Troubleshooting

### "Checkpoint not found"
Make sure the path to your checkpoint file is correct. Use absolute paths if needed.

### "No samples found for station"
Check that your `--data_root` points to the correct directory containing MAT files.

### CUDA out of memory
Reduce `--num_samples` or process stations one at a time.

## Citation

If you use this code, please cite:
```
[Your paper citation here]
```

