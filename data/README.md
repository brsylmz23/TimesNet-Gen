# Data Directory

This project **does not require any real data** to run the public demo in `generate_samples.py`.
All generation is done from the pre-computed latent bank:
- `checkpoints/timesnet_pointcloud_phase1_final.pth`
- `checkpoints/latent_bank_phase1.npz`

This `data/` folder is only needed if you:
- want to experiment with your **own seismic records**, or
- plan to extend the code for **fine-tuning** on new stations.

## Using Your Own Data (Optional)

If you add data here, a good convention is:
- MATLAB `.mat` files
- 3 channels: East–West (E), North–South (N), Up–Down (U)
- Variable length (you can resample to 6000 samples if you follow the original pipeline)

Example layout (purely illustrative):

```
data/
├── station_XXXX/
│   ├── event_001.mat
│   ├── event_002.mat
│   └── ...
└── station_YYYY/
```

Currently, the public demo script **does not read from `data/`** by default.
You can safely leave this folder empty when using the GitHub demo.


