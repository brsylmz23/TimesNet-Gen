# Data Directory

Place your seismic data files here.

## Expected Data Format

The model expects seismic time series data in MATLAB `.mat` format with the following structure:
- **3 channels**: East-West (E), North-South (N), Up-Down (U)
- **Sampling rate**: 200 Hz
- **Duration**: Variable (will be resampled to 6000 time steps = 30 seconds)

## Data Organization

```
data/
├── station_0205/
│   ├── sample_001.mat
│   ├── sample_002.mat
│   └── ...
├── station_1716/
├── station_2020/
├── station_3130/
└── station_4628/
```

## Download Dataset

Due to the large size of seismic datasets, the data is hosted separately:

📥 **Download Link:** [Add your data repository link here]

## Alternative: Use Your Own Data

You can use your own seismic data by:
1. Converting it to `.mat` format
2. Ensuring 3-channel structure (E, N, U)
3. Updating the `data_root` path in `generate_samples.py`

For data preprocessing scripts, see the `examples/` directory.

