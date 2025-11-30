# Demo Data Requirements

To run `generate_samples.py`, you need seismic data in MATLAB `.mat` format.

## Minimum Data for Demo

For a quick demo, you need **at least 5-10 samples per station**:

```
data/
├── AFAD.0205.02_XXXXXX.mat  (5-10 files for station 0205)
├── AFAD.1716.17_XXXXXX.mat  (5-10 files for station 1716)
├── AFAD.2020.20_XXXXXX.mat  (5-10 files for station 2020)
├── AFAD.3130.31_XXXXXX.mat  (5-10 files for station 3130)
└── AFAD.4628.46_XXXXXX.mat  (5-10 files for station 4628)
```

**Total:** ~25-50 `.mat` files (minimum for demo)

## Data Format

Each `.mat` file should contain:
- **3 channels**: East-West (E), North-South (N), Up-Down (U)
- **Sampling rate**: 200 Hz
- **Format**: NumPy array or MATLAB matrix
- **Shape**: (3, N) where N is number of time steps

## Example Data Structure

```matlab
% In MATLAB
data = struct();
data.E = rand(1, 6000);  % East-West channel
data.N = rand(1, 6000);  % North-South channel
data.U = rand(1, 6000);  % Up-Down channel
save('AFAD.2020.20_example.mat', 'data');
```

## Where to Get Data

1. **Use your own seismic data**: Convert to `.mat` format
2. **Download from repository**: [Add link to data repository]
3. **Contact for sample data**: [Add contact email]

## Quick Test Without Real Data

If you don't have data yet, the script will show an error but you can still:
1. Check if the model loads correctly
2. Verify the code structure
3. See what outputs would be generated

## Data Organization

The script expects data in this structure:
```
data_root/
├── AFAD.0205.02_180316225621S3014_EV_0007.mat
├── AFAD.0205.02_170112020218P1512_EV_0012.mat
├── AFAD.1716.17_XXXXXX.mat
└── ...
```

Station ID is extracted from the filename (e.g., `0205` from `AFAD.0205.02_...`)

## Updating Data Path

Edit `generate_samples.py` line 255:
```python
parser.add_argument('--data_root', type=str, 
                    default=r"./data/",  # Change this to your data path
                    help='Root path to seismic data')
```

Or run with custom path:
```bash
python generate_samples.py --data_root /path/to/your/data
```

