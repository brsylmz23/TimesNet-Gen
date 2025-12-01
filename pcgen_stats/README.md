# Encoder Statistics

## Required File

Place the following file in this directory:

### encoder_feature_std.npy

- **Size:** ~1 KB
- **Description:** Per-dimension standard deviations of encoder features
- **Shape:** (128,) - one std value per latent dimension
- **Usage:** Used for Gaussian noise injection during sample generation
- **Download:** [Link will be provided]

---

## Purpose

This file contains the standard deviations computed from the encoder outputs during Phase 0 training. It is used to add realistic noise to the latent space during generation, ensuring diverse and realistic samples.

---

## Verification

After downloading:

```bash
python -c "import numpy as np; data = np.load('pcgen_stats/encoder_feature_std.npy'); print(f'Shape: {data.shape}, Min: {data.min():.4f}, Max: {data.max():.4f}')"
```

Expected output:
```
Shape: (128,), Min: 0.XXXX, Max: 0.XXXX
```

