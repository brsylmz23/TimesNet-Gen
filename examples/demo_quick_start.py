"""
TimesNet-PointCloud Quick Start Demo
=====================================

This script demonstrates how to generate synthetic seismic waveforms
using the pre-trained TimesNet-PointCloud model.

Usage:
    python demo_quick_start.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

print("="*80)
print("TimesNet-PointCloud Demo")
print("="*80)

# Configuration
station_id = '2020'  # Change to: 0205, 1716, 2020, 3130, 4628
npz_path = f'../generated_samples/generated_timeseries_npz/station_{station_id}_generated_timeseries.npz'

# Check if samples exist
if not os.path.exists(npz_path):
    print("\n❌ Generated samples not found!")
    print("Please run the generation script first:")
    print("    cd ..")
    print("    python generate_samples.py --num_samples 10")
    sys.exit(1)

# Load generated samples
print(f"\n📂 Loading samples from: {npz_path}")
data = np.load(npz_path)
generated_signals = data['generated_signals']  # Shape: (N, 3, 6000)

print(f"✅ Loaded {len(generated_signals)} samples")
print(f"   Shape: {generated_signals.shape}")
print(f"   Channels: E-W, N-S, U-D")
print(f"   Duration: 30 seconds @ 200 Hz")

# Select a random sample
sample_idx = np.random.randint(0, len(generated_signals))
sample = generated_signals[sample_idx]  # Shape: (3, 6000)

print(f"\n🎲 Randomly selected sample {sample_idx + 1}/{len(generated_signals)}")

# Time axis
time = np.arange(6000) / 200.0

# Create figure with time series and FAS
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

channel_names = ['E-W', 'N-S', 'U-D']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Plot time series (left column)
for i, (name, color) in enumerate(zip(channel_names, colors)):
    ax = fig.add_subplot(gs[i, 0])
    ax.plot(time, sample[i], color=color, linewidth=0.8)
    ax.set_ylabel(f'{name}\nAmplitude (m/s²)', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 30)
    if i == 0:
        ax.set_title('Time Series', fontsize=12, fontweight='bold')
    if i < 2:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel('Time (s)', fontsize=10, fontweight='bold')

# Plot FAS (right column)
for i, (name, color) in enumerate(zip(channel_names, colors)):
    ax = fig.add_subplot(gs[i, 1])
    
    # Compute FAS
    signal = sample[i]
    yf = fft(signal)
    xf = fftfreq(len(signal), 1/200)[:len(signal)//2]
    fas = 2.0/len(signal) * np.abs(yf[:len(signal)//2])
    
    ax.plot(xf, fas, color=color, linewidth=1.2)
    ax.set_ylabel(f'{name}\nAmplitude (m/s²)', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 25)
    if i == 0:
        ax.set_title('Fourier Amplitude Spectrum', fontsize=12, fontweight='bold')
    if i < 2:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel('Frequency (Hz)', fontsize=10, fontweight='bold')
    
    # Mark dominant frequency
    mask = (xf > 0.5) & (xf < 25)
    dominant_freq = xf[mask][np.argmax(fas[mask])]
    ax.axvline(dominant_freq, color='red', linestyle='--', alpha=0.5, linewidth=1)

# Statistics subplot
ax_stats = fig.add_subplot(gs[3, :])
ax_stats.axis('off')

# Compute statistics
dominant_freqs = []
peak_amps = []
for sig in generated_signals:
    # E-W channel
    signal = sig[0]
    peak_amps.append(np.max(np.abs(signal)))
    
    # Dominant frequency
    yf = fft(signal)
    xf = fftfreq(len(signal), 1/200)[:len(signal)//2]
    fas = 2.0/len(signal) * np.abs(yf[:len(signal)//2])
    mask = (xf > 0.5) & (xf < 25)
    dominant_freqs.append(xf[mask][np.argmax(fas[mask])])

stats_text = f"""
📊 Statistics for Station {station_id} ({len(generated_signals)} samples):

   • Dominant Frequency (f₀): {np.mean(dominant_freqs):.2f} ± {np.std(dominant_freqs):.2f} Hz
   • Peak Amplitude: {np.mean(peak_amps):.4f} ± {np.std(peak_amps):.4f} m/s²
   • Min/Max f₀: {np.min(dominant_freqs):.2f} / {np.max(dominant_freqs):.2f} Hz
   • Min/Max Amplitude: {np.min(peak_amps):.4f} / {np.max(peak_amps):.4f} m/s²
"""

ax_stats.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
              verticalalignment='center', bbox=dict(boxstyle='round', 
              facecolor='wheat', alpha=0.3))

fig.suptitle(f'Generated Seismic Waveform - Station {station_id}', 
             fontsize=14, fontweight='bold')

plt.savefig(f'demo_output_station_{station_id}.png', dpi=150, bbox_inches='tight')
print(f"\n💾 Saved figure: demo_output_station_{station_id}.png")

plt.show()

print("\n" + "="*80)
print("✅ Demo complete!")
print("="*80)
print("\n📚 Next steps:")
print("   1. Try different stations: Edit 'station_id' variable")
print("   2. Generate more samples: python ../generate_samples.py --num_samples 100")
print("   3. Explore the Jupyter notebook: demo_notebook.ipynb")
print("   4. Read documentation: ../docs/GENERATION_README.md")
print("\n" + "="*80)
