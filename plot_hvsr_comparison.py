"""
Simple HVSR comparison plotting for generated samples.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def compute_hvsr(signal_3ch, fs=200.0, smooth_window=21):
    """
    Compute HVSR from 3-channel signal (3, N).
    
    Args:
        signal_3ch: (3, N) array [E-W, N-S, U-D]
        fs: sampling frequency
        smooth_window: smoothing window size
    
    Returns:
        freqs, hvsr_curve
    """
    # Compute FFT for each channel
    n = signal_3ch.shape[1]
    freqs = np.fft.rfftfreq(n, 1/fs)
    
    # Fourier amplitude spectra
    fas_ew = np.abs(np.fft.rfft(signal_3ch[0]))
    fas_ns = np.abs(np.fft.rfft(signal_3ch[1]))
    fas_ud = np.abs(np.fft.rfft(signal_3ch[2]))
    
    # Horizontal component (geometric mean of E-W and N-S)
    horizontal = np.sqrt(fas_ew * fas_ns)
    
    # Vertical component
    vertical = fas_ud
    
    # HVSR
    hvsr = horizontal / (vertical + 1e-10)
    
    # Smooth with moving average
    if smooth_window > 1:
        hvsr = np.convolve(hvsr, np.ones(smooth_window)/smooth_window, mode='same')
    
    return freqs, hvsr


def find_dominant_frequency(freqs, hvsr, freq_range=(0.5, 25)):
    """Find dominant frequency (f0) in HVSR curve."""
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    if not np.any(mask):
        return 0.0
    idx = np.argmax(hvsr[mask])
    return freqs[mask][idx]


def plot_hvsr_comparison(generated_npz_dir, output_dir, station_f0_dict=None):
    """
    Plot HVSR comparison between real and generated samples.
    
    Args:
        generated_npz_dir: Directory with station_XXXX_generated_timeseries.npz files
        output_dir: Output directory for plots
        station_f0_dict: Dict of {station_id: f0_hz} for reference
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Default f0 values
    if station_f0_dict is None:
        station_f0_dict = {
            '0205': 2.6,
            '1716': 6.4,
            '2020': 5.1,
            '3130': 12.8,
            '4628': 1.8
        }
    
    # Find all NPZ files
    npz_files = {}
    for station_id in station_f0_dict.keys():
        npz_path = os.path.join(generated_npz_dir, f'station_{station_id}_generated_timeseries.npz')
        if os.path.exists(npz_path):
            npz_files[station_id] = npz_path
    
    if len(npz_files) == 0:
        print("[WARN] No NPZ files found for HVSR plotting")
        return
    
    print(f"\n[INFO] Creating HVSR comparison plots for {len(npz_files)} stations...")
    
    # Create comparison figure
    n_stations = len(npz_files)
    fig, axes = plt.subplots(1, n_stations, figsize=(5*n_stations, 4))
    if n_stations == 1:
        axes = [axes]
    
    for ax, (station_id, npz_path) in zip(axes, npz_files.items()):
        # Load generated samples
        data = np.load(npz_path)
        generated_signals = data['generated_signals']  # (N, 3, 6000)
        
        print(f"[INFO] Station {station_id}: Computing HVSR for {len(generated_signals)} samples...")
        
        # Compute HVSR for all generated samples
        hvsr_curves = []
        f0_values = []
        
        for i in range(len(generated_signals)):
            signal_3ch = generated_signals[i]  # (3, 6000)
            freqs, hvsr = compute_hvsr(signal_3ch)
            hvsr_curves.append(hvsr)
            f0 = find_dominant_frequency(freqs, hvsr)
            f0_values.append(f0)
        
        # Compute mean and std
        hvsr_curves = np.array(hvsr_curves)
        hvsr_mean = np.mean(hvsr_curves, axis=0)
        hvsr_std = np.std(hvsr_curves, axis=0)
        
        # Plot
        ax.plot(freqs, hvsr_mean, 'b-', linewidth=2, label='Generated (mean)')
        ax.fill_between(freqs, hvsr_mean - hvsr_std, hvsr_mean + hvsr_std, 
                        alpha=0.3, color='blue', label='Generated (±1σ)')
        
        # Mark dominant frequency
        mean_f0 = np.mean(f0_values)
        ax.axvline(mean_f0, color='blue', linestyle='--', linewidth=1.5, 
                  label=f'Generated f₀={mean_f0:.1f} Hz')
        
        # Mark reference f0
        if station_id in station_f0_dict:
            ref_f0 = station_f0_dict[station_id]
            ax.axvline(ref_f0, color='red', linestyle='--', linewidth=1.5,
                      label=f'Reference f₀={ref_f0:.1f} Hz')
        
        ax.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
        ax.set_ylabel('HVSR', fontsize=12, fontweight='bold')
        ax.set_title(f'Station {station_id}', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 25)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'hvsr_comparison_all_stations.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Saved HVSR comparison plot: {output_path}")
    
    # Create f0 comparison bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    stations = []
    generated_f0s = []
    reference_f0s = []
    
    for station_id, npz_path in npz_files.items():
        data = np.load(npz_path)
        generated_signals = data['generated_signals']
        
        # Compute f0 for all samples
        f0_values = []
        for i in range(len(generated_signals)):
            signal_3ch = generated_signals[i]
            freqs, hvsr = compute_hvsr(signal_3ch)
            f0 = find_dominant_frequency(freqs, hvsr)
            f0_values.append(f0)
        
        stations.append(station_id)
        generated_f0s.append(np.mean(f0_values))
        reference_f0s.append(station_f0_dict.get(station_id, 0))
    
    x = np.arange(len(stations))
    width = 0.35
    
    ax.bar(x - width/2, reference_f0s, width, label='Reference', color='red', alpha=0.7)
    ax.bar(x + width/2, generated_f0s, width, label='Generated', color='blue', alpha=0.7)
    
    ax.set_xlabel('Station', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dominant Frequency f₀ (Hz)', fontsize=12, fontweight='bold')
    ax.set_title('Dominant Frequency Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(stations)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'f0_comparison_bar.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Saved f0 comparison plot: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Plot HVSR comparison')
    parser.add_argument('--npz_dir', type=str, default='./generated_samples/generated_timeseries_npz',
                       help='Directory with NPZ files')
    parser.add_argument('--output_dir', type=str, default='./generated_samples/hvsr_plots',
                       help='Output directory')
    args = parser.parse_args()
    
    plot_hvsr_comparison(args.npz_dir, args.output_dir)

