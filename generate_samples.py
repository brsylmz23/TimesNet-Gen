#!/usr/bin/env python3
"""
Simplified inference script for TimesNet-PointCloud generative model.
Only loads data for the 5 fine-tuned stations.

Usage:
    python generate_samples_simple.py --num_samples 50
"""
import os
import argparse
import torch
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import glob
import scipy.io as sio


class SimpleArgs:
    """Configuration for generation."""
    def __init__(self):
        # Model architecture
        self.seq_len = 6000
        self.d_model = 128
        self.d_ff = 256
        self.e_layers = 2
        self.d_layers = 2
        self.num_kernels = 6
        self.top_k = 2
        self.dropout = 0.1
        self.latent_dim = 256
        
        # System
        self.use_gpu = torch.cuda.is_available()
        self.seed = 0
        
        # Point-cloud generation
        self.pcgen_k = 5
        self.pcgen_jitter_std = 0.0


def load_mat_file(filepath, seq_len=6000):
    """Load and preprocess a .mat file."""
    try:
        mat_data = sio.loadmat(filepath)
        
        # Try different possible field names
        for key in ['data', 'signal', 'waveform', 'acc']:
            if key in mat_data:
                data = mat_data[key]
                break
        else:
            # Use first non-metadata key
            data = [v for k, v in mat_data.items() if not k.startswith('__')][0]
        
        # Ensure shape is (3, N)
        if data.shape[0] != 3:
            data = data.T
        
        # Resample to seq_len
        if data.shape[1] != seq_len:
            from scipy import signal as sp_signal
            data_resampled = np.zeros((3, seq_len))
            for i in range(3):
                data_resampled[i] = sp_signal.resample(data[i], seq_len)
            data = data_resampled
        
        return torch.FloatTensor(data)
    
    except Exception as e:
        print(f"[ERROR] Failed to load {filepath}: {e}")
        return None


def load_model(checkpoint_path, args):
    """Load pre-trained TimesNet-PointCloud model."""
    from models.TimesNet_PointCloud import TimesNetPointCloud
    
    # Create model config
    class ModelConfig:
        def __init__(self, args):
            self.seq_len = args.seq_len
            self.pred_len = 0
            self.enc_in = 3
            self.c_out = 3
            self.d_model = args.d_model
            self.d_ff = args.d_ff
            self.num_kernels = args.num_kernels
            self.top_k = args.top_k
            self.e_layers = args.e_layers
            self.d_layers = args.d_layers
            self.dropout = args.dropout
            self.embed = 'timeF'
            self.freq = 'h'
            self.latent_dim = args.latent_dim
    
    config = ModelConfig(args)
    model = TimesNetPointCloud(config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    if args.use_gpu:
        model = model.cuda()
    
    print(f"[INFO] Model loaded successfully from {checkpoint_path}")
    return model


def generate_samples_for_station(model, station_files, num_samples, args):
    """Generate samples using point-cloud mixing."""
    if len(station_files) == 0:
        return None, None
    
    generated_signals = []
    real_names_used = []
    
    # Load all station files
    station_data = []
    for fpath in station_files:
        data = load_mat_file(fpath, args.seq_len)
        if data is not None:
            station_data.append((data, os.path.basename(fpath)))
    
    if len(station_data) == 0:
        print(f"[WARN] Could not load any files")
        return None, None
    
    print(f"[INFO] Loaded {len(station_data)} real samples")
    
    with torch.no_grad():
        for _ in range(num_samples):
            # Randomly select k samples
            k = min(args.pcgen_k, len(station_data))
            selected_indices = np.random.choice(len(station_data), size=k, replace=False)
            
            # Encode all k samples
            encoder_features = []
            names = []
            for idx in selected_indices:
                x, fname = station_data[idx]
                x = x.unsqueeze(0)  # Add batch dimension
                if args.use_gpu:
                    x = x.cuda()
                
                # Encode
                enc_out, means_b, stdev_b = model.encode_features_for_reconstruction(x)
                encoder_features.append(enc_out)
                names.append(fname)
            
            # Mix encoder features (average)
            mixed_features = torch.mean(torch.cat(encoder_features, dim=0), dim=0, keepdim=True)
            
            # Optional jitter
            if args.pcgen_jitter_std > 0:
                noise = torch.randn_like(mixed_features) * args.pcgen_jitter_std
                mixed_features = mixed_features + noise
            
            # Decode to generate signal
            x_first, _ = station_data[selected_indices[0]]
            x_first = x_first.unsqueeze(0)
            if args.use_gpu:
                x_first = x_first.cuda()
            _, means_b, stdev_b = model.encode_features_for_reconstruction(x_first)
            
            generated = model.project_features_for_reconstruction(mixed_features, means_b, stdev_b)
            
            # Store
            generated_signals.append(generated.squeeze(0).cpu().numpy())
            real_names_used.append(names)
    
    return np.array(generated_signals), real_names_used


def save_generated_samples(generated_signals, real_names, station_id, output_dir):
    """Save generated samples to NPZ file."""
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f'station_{station_id}_generated_timeseries.npz')
    np.savez_compressed(
        output_path,
        generated_signals=generated_signals,
        real_names=real_names,
        station_id=station_id
    )
    print(f"[INFO] Saved {len(generated_signals)} generated samples to {output_path}")


def plot_sample_preview(generated_signals, station_id, output_dir, num_preview=2):
    """Create preview plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(min(num_preview, len(generated_signals))):
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        signal = generated_signals[i]
        
        channel_names = ['E-W', 'N-S', 'U-D']
        for ch, (ax, name) in enumerate(zip(axes, channel_names)):
            ax.plot(signal[ch], linewidth=0.8)
            ax.set_ylabel(f'{name}\nAmplitude', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time Steps', fontsize=10, fontweight='bold')
        fig.suptitle(f'Generated Sample - Station {station_id}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f'station_{station_id}_preview_{i}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"[INFO] Saved {min(num_preview, len(generated_signals))} preview plots to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate seismic samples (simplified version)')
    parser.add_argument('--checkpoint', type=str, 
                        default=r'D:\Baris\codes\Time-Series-Library-main\checkpoints\timesnet_pointcloud_phase1_final.pth',
                        help='Path to pre-trained model checkpoint')
    parser.add_argument('--num_samples', type=int, default=50, 
                        help='Number of samples to generate per station')
    parser.add_argument('--output_dir', type=str, default='./generated_samples', 
                        help='Output directory')
    parser.add_argument('--num_preview', type=int, default=2, 
                        help='Number of preview plots per station')
    parser.add_argument('--stations', type=str, nargs='+', default=['0205', '1716', '2020', '3130', '4628'],
                        help='Target station IDs')
    parser.add_argument('--data_root', type=str, default=r"D:\Baris\new_Ps_Vs30/", 
                        help='Root path to seismic data')
    
    args_cli = parser.parse_args()
    
    # Check checkpoint
    if not os.path.exists(args_cli.checkpoint):
        print(f"\n{'='*80}")
        print(f"❌ ERROR: Checkpoint not found!")
        print(f"{'='*80}")
        print(f"\nLooking for: {args_cli.checkpoint}")
        return
    
    # Create configuration
    args = SimpleArgs()
    
    print("="*80)
    print("TimesNet-Gen Sample Generation (Simplified)")
    print("="*80)
    print(f"Checkpoint: {args_cli.checkpoint}")
    print(f"Target stations: {args_cli.stations}")
    print(f"Samples per station: {args_cli.num_samples}")
    print(f"Output directory: {args_cli.output_dir}")
    print("="*80)
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load model
    model = load_model(args_cli.checkpoint, args)
    
    # Load data files for selected stations only
    print("\n[INFO] Loading data for selected stations...")
    
    all_station_files = {}
    for station_id in args_cli.stations:
        # Find all .mat files for this station
        pattern = os.path.join(args_cli.data_root, f"*{station_id}*.mat")
        station_files = glob.glob(pattern)
        
        if len(station_files) == 0:
            print(f"[WARN] No files found for station {station_id}")
        else:
            print(f"[INFO] Found {len(station_files)} files for station {station_id}")
            all_station_files[station_id] = station_files
    
    if len(all_station_files) == 0:
        print(f"\n❌ ERROR: No data files found in {args_cli.data_root}")
        return
    
    # Create output directories
    npz_output_dir = os.path.join(args_cli.output_dir, 'generated_timeseries_npz')
    plot_output_dir = os.path.join(args_cli.output_dir, 'preview_plots')
    
    # Generate samples for each station
    print("\n[INFO] Generating samples...")
    for station_id in args_cli.stations:
        if station_id not in all_station_files:
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing Station: {station_id}")
        print(f"{'='*60}")
        
        generated_signals, real_names = generate_samples_for_station(
            model, all_station_files[station_id], args_cli.num_samples, args
        )
        
        if generated_signals is not None:
            # Save to NPZ
            save_generated_samples(generated_signals, real_names, station_id, npz_output_dir)
            
            # Create preview plots
            plot_sample_preview(generated_signals, station_id, plot_output_dir, args_cli.num_preview)
    
    print("\n" + "="*80)
    print("Generation Complete!")
    print("="*80)
    print(f"Generated samples saved to: {npz_output_dir}")
    print(f"Preview plots saved to: {plot_output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()

