# TimesNet-Gen: Seismic Waveform Generation via Point-Cloud Aggregation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📄 Paper

**[Paper Title]**  
Authors: [Your Names]  
arXiv: [Link will be added upon publication]  
Conference/Journal: [Venue]

> **Abstract:** [Brief 2-3 sentence summary of your work]

---

## 🎯 Key Features

- **Point-Cloud Generation**: Bootstrap aggregation of latent vectors for diverse sample generation
- **Station-Specific**: Fine-tuned on 5 seismic stations with distinct site characteristics
- **Fast Inference**: Generate 50 samples in ~10 seconds (no real data required)
- **Latent Bank**: Pre-computed latent representations for instant generation
- **HVSR Preservation**: Maintains site-specific frequency characteristics (f₀)

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/yourusername/TimesNet-Gen.git
cd TimesNet-Gen
pip install -r requirements.txt
```

### 2. Download Pre-trained Model

Download the pre-trained checkpoint and latent bank:

```bash
# Download from [Google Drive / Zenodo / etc.]
# Place files in ./checkpoints/
```

Required files:
- `timesnet_pointcloud_phase1_final.pth` (Model checkpoint)
- `latent_bank_phase1.npz` (Pre-computed latent vectors)
- `encoder_feature_std.npy` (For noise injection)

### 3. Generate Samples

```bash
# Generate 50 samples per station (5 stations)
python generate_samples.py --num_samples 50

# Generate for specific stations
python generate_samples.py --stations 0205 1716 --num_samples 100
```

**Output:**
- Generated waveforms (NPZ format)
- HVSR curves and f₀ distributions
- Comparison plots (Real vs Generated)

---

## 📊 Model Architecture

<p align="center">
  <img src="figures/TimesNetgenModelErdem.png" alt="TimesNet-Gen Architecture" width="800"/>
</p>

**Key Components:**
- **Encoder**: TimesBlock-based feature extraction
- **Latent Space**: Point-cloud aggregation with bootstrap sampling
- **Decoder**: Reconstruction with temporal dynamics preservation
- **Noise Injection**: Gaussian noise scaled by encoder statistics

---


## 📈 Results

### Station-Specific f₀ Values

| Station | Target f₀ (Hz) |
|---------|----------------|
| 2020    | 5.1            |
| 4628    | 1.8            |
| 0205    | 2.6            |
| 1716    | 6.4            |
| 3130    | 12.8           |

### Sample Outputs

<p align="center">
  <img src="examples/sample_outputs/example_comparison.png" alt="Real vs Generated" width="600"/>
</p>

*Real (blue) vs Generated (orange) waveforms with Fourier Amplitude Spectra*

---

## 📁 Project Structure

```
TimesNet-Gen/
├── generate_samples.py                  # Main inference script
├── TimesNet_PointCloud.py               # Point-cloud generation model
├── plot_combined_hvsr_all_sources.py    # HVSR analysis and plotting
├── checkpoints/                         # Pre-trained models
│   ├── timesnet_pointcloud_phase1_final.pth
│   └── latent_bank_phase1.npz
├── pcgen_stats/                         # Encoder statistics
│   └── encoder_feature_std.npy
├── figures/                             # Architecture diagrams
├── requirements.txt
└── README.md
```

---

## 🔧 Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Matplotlib
- SciPy

See `requirements.txt` for complete list.

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{yourname2025timesnetgen,
  title={TimesNet-Gen: Seismic Waveform Generation via Point-Cloud Aggregation},
  author={Your Name and Co-authors},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

---

## 📧 Contact

For questions or issues, please:
- Open an issue on GitHub
- Email: brsylmz23@hotmail.com

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with ❤️ for the seismology community
</p>
