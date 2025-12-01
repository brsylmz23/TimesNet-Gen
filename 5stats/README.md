# 5stats - Demo Data

This directory contains seismic data for the 5 fine-tuned stations used in TimesNet-Gen.

## Stations

- **0205**: Station 0205 samples
- **1716**: Station 1716 samples
- **2020**: Station 2020 samples
- **3130**: Station 3130 samples
- **4628**: Station 4628 samples

## Data Format

Each `.mat` file contains 3-channel seismic time series:
- **Channel 1**: East-West (E-W)
- **Channel 2**: North-South (N-S)
- **Channel 3**: Up-Down (U-D)

**Sampling rate**: 200 Hz  
**Format**: MATLAB `.mat` files

## Usage

The `generate_samples.py` script automatically loads data from this directory:

```python
python generate_samples.py --num_samples 50
```

The script will:
1. Search for `*0205*.mat`, `*1716*.mat`, `*2020*.mat`, `*3130*.mat`, `*4628*.mat`
2. Load valid files (skips corrupted/incompatible files)
3. Generate new samples using point-cloud mixing

## File Naming Convention

Files are named with the format:
```
YYYYMMDDHHMMSS_SSSS.mat
```

Where:
- `YYYYMMDDHHMMSS`: Timestamp
- `SSSS`: Station ID (e.g., 0205, 1716, 2020, 3130, 4628)

## Adding Your Own Data

To use your own seismic data:

1. Convert to `.mat` format with 3 channels (E, N, U)
2. Ensure station ID is in the filename
3. Place files in this directory
4. Run `generate_samples.py`

## Data Statistics

| Station | Files | Dominant Frequency (f₀) |
|---------|-------|------------------------|
| 0205    | ~23   | 2.6 Hz                 |
| 1716    | ~24   | 6.4 Hz                 |
| 2020    | ~37   | 5.1 Hz                 |
| 3130    | ~22   | 12.8 Hz                |
| 4628    | ~10   | 1.8 Hz                 |

**Total**: ~115 files

## Notes

- Some files may be skipped if they have incompatible formats
- Minimum 5-10 valid files per station recommended for generation
- Files are automatically resampled to 6000 time steps (30 seconds @ 200 Hz)

