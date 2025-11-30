#!/usr/bin/env python3
"""
Quick inference script for TimesNet-PointCloud generative model.
Loads a pre-trained checkpoint and generates seismic time series samples.

Usage:
    # Generate 50 samples per station (default)
    python generate_samples.py
    
    # Generate 100 samples per station
    python generate_samples.py --num_samples 100
    
    # Generate 200 samples per station
    python generate_samples.py --num_samples 200
    
    # Use custom checkpoint
    python generate_samples.py --checkpoint ./my_model.pth --num_samples 50
"""
import os
import argparse
import torch
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

try:
    from data_provider.data_loader import generate_and_save_station_splits_mat
except Exception:
    from data_loader import generate_and_save_station_splits_mat
try:
    from data_provider.data_loader_gen import GenMatDataset
except Exception:
    from data_loader_gen import GenMatDataset


class GenArgs:
    """Configuration for generation."""
    def __init__(self):
        # Data and model architecture
        self.root_path = r"D:\Baris\new_Ps_Vs30/"
        self.seq_len = 6000
        self.batch_size = 32
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

        # Point-cloud generation options
        self.pcgen_k = 5  # number of latent points to mix
        self.pcgen_per_station = True  # restrict mixes within same station pool
        self.pcgen_jitter_std = 0.0  # optional Gaussian jitter on mixed z


def load_model(checkpoint_path, args):
    """Load pre-trained TimesNet-PointCloud model from checkpoint."""
    from TimesNet_PointCloud import TimesNetPointCloud
    
    # Create model config
    class ModelConfig:
        def __init__(self, args):
            self.seq_len = args.seq_len
            self.pred_len = 0
            self.enc_in = 3  # E, N, U channels
            self.c_out = 3
            self.d_model = args.d_model
            self.d_ff = args.d_ff
            self.num_kernels = args.num_kernels
            self.top_k = args.top_k
            self.e_layers = args.e_layers
            self.dropout = args.dropout
            self.embed = 'timeF'
            self.freq = 'h'
    
    config = ModelConfig(args)
    model = TimesNetPointCloud(config)
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[INFO] Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
        print(f"[INFO] Loaded model state dict")
    
    if args.use_gpu:
        model = model.cuda()
    
    model.eval()
    print(f"[INFO] Model loaded successfully from {checkpoint_path}")
    return model


def generate_samples_for_station(model, dataset, station_id, num_samples, args):
    """
    Generate samples for a specific station using k-sample encoder feature mixing.
    
    Args:
        model: TimesNetPointCloud model
        dataset: GenMatDataset
        station_id: Target station ID (e.g., '0205')
        num_samples: Number of samples to generate
        args: Generation arguments
    
    Returns:
        generated_signals: numpy array of shape (num_samples, seq_len, 3)
        real_names: list of real signal filenames used for mixing
    """
    # Get all samples for this station
    station_indices = [i for i, (_, sid, _) in enumerate(dataset) if dataset.station_id_map[int(sid)] == station_id]
    
    if len(station_indices) == 0:
        print(f"[WARN] No samples found for station {station_id}")
        return None, None
    
    print(f"[INFO] Station {station_id}: Found {len(station_indices)} real samples, generating {num_samples} new samples")
    
    generated_signals = []
    real_names_used = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            # Randomly select k samples from this station
            k = min(args.pcgen_k, len(station_indices))
            selected_indices = np.random.choice(station_indices, size=k, replace=False)
            
            # Encode all k samples
            encoder_features = []
            names = []
            for idx in selected_indices:
                x, sid, fname = dataset[idx]
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
            # Use means and stdev from the first sample
            x_first, _, _ = dataset[selected_indices[0]]
            x_first = x_first.unsqueeze(0)
            if args.use_gpu:
                x_first = x_first.cuda()
            _, means_b, stdev_b = model.encode_features_for_reconstruction(x_first)
            
            generated = model.project_features_for_reconstruction(mixed_features, means_b, stdev_b)
            
            # Convert to numpy
            generated_np = generated.squeeze(0).cpu().numpy()  # (seq_len, 3)
            generated_signals.append(generated_np)
            real_names_used.append(names)
    
    generated_signals = np.stack(generated_signals, axis=0)  # (num_samples, seq_len, 3)
    return generated_signals, real_names_used


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
    """Create preview plots of generated samples."""
    os.makedirs(output_dir, exist_ok=True)
    
    num_preview = min(num_preview, len(generated_signals))
    fs = 100.0  # Sampling frequency in Hz
    
    for i in range(num_preview):
        sig = generated_signals[i]  # (seq_len, 3)
        T = sig.shape[0]
        t = np.arange(T) / fs
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        labels = ['E-W', 'N-S', 'U-D']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for ch in range(3):
            axes[ch].plot(t, sig[:, ch], color=colors[ch], linewidth=1.0, alpha=0.9)
            axes[ch].set_ylabel(labels[ch], fontsize=12, fontweight='bold')
            axes[ch].grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        fig.suptitle(f'Generated Sample - Station {station_id} (#{i+1})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f'station_{station_id}_generated_sample_{i+1}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print(f"[INFO] Saved {num_preview} preview plots to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate seismic time series samples from pre-trained TimesNet-PointCloud model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 50 samples (default)
  python generate_samples.py
  
  # Generate 100 samples
  python generate_samples.py --num_samples 100
  
  # Generate with custom checkpoint and output
  python generate_samples.py --checkpoint ./my_model.pth --num_samples 200 --output_dir ./results
  
  # Generate for specific stations only
  python generate_samples.py --stations 0205 1716 --num_samples 50
        """
    )
    parser.add_argument('--checkpoint', type=str, 
                        default=r'D:\Baris\codes\Time-Series-Library-main\checkpoints\timesnet_pointcloud_phase1_final.pth',
                        help='Path to pre-trained model checkpoint')
    parser.add_argument('--num_samples', type=int, default=50, 
                        help='Number of samples to generate per station (default: 50)')
    parser.add_argument('--output_dir', type=str, default='./generated_samples', 
                        help='Output directory for generated samples')
    parser.add_argument('--num_preview', type=int, default=2, 
                        help='Number of preview plots per station (default: 2)')
    parser.add_argument('--stations', type=str, nargs='+', default=['0205', '1716', '2020', '3130', '4628'],
                        help='Target station IDs (default: all 5 stations)')
    parser.add_argument('--data_root', type=str, default=r"./data/", 
                        help='Root path to seismic data (see data/DEMO_DATA_INFO.md for format)')
    parser.add_argument('--seq_len', type=int, default=6000, 
                        help='Sequence length in time steps (default: 6000)')
    parser.add_argument('--pcgen_k', type=int, default=5, 
                        help='Number of real samples to mix for each generated sample (default: 5)')
    
    args_cli = parser.parse_args()
    
    # Check if checkpoint exists
    if not os.path.exists(args_cli.checkpoint):
        print(f"\n{'='*80}")
        print(f"❌ ERROR: Checkpoint not found!")
        print(f"{'='*80}")
        print(f"\nLooking for: {args_cli.checkpoint}")
        print("\n📁 Available checkpoints:")
        checkpoint_dir = './checkpoints'
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
            if checkpoints:
                for ckpt in sorted(checkpoints):
                    ckpt_path = os.path.join(checkpoint_dir, ckpt)
                    size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
                    print(f"  ✓ {ckpt_path} ({size_mb:.1f} MB)")
            else:
                print("  ⚠ No checkpoints found in ./checkpoints/")
        else:
            print("  ⚠ ./checkpoints/ directory does not exist")
        
        # Check benchmarks folder too
        benchmark_dir = './benchmarks'
        if os.path.exists(benchmark_dir):
            benchmarks = [f for f in os.listdir(benchmark_dir) if f.endswith('.pth')]
            if benchmarks:
                print("\n📁 Checkpoints in ./benchmarks/:")
                for ckpt in sorted(benchmarks):
                    ckpt_path = os.path.join(benchmark_dir, ckpt)
                    size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
                    print(f"  ✓ {ckpt_path} ({size_mb:.1f} MB)")
        
        print("\n💡 To train a model first, run:")
        print("   python untitled1_gen.py")
        print(f"\n{'='*80}\n")
        return
    
    # Create configuration
    args = GenArgs()
    args.root_path = args_cli.data_root
    args.seq_len = args_cli.seq_len
    args.pcgen_k = args_cli.pcgen_k
    
    print("="*80)
    print("TimesNet-PointCloud Sample Generation")
    print("="*80)
    print(f"Checkpoint: {args_cli.checkpoint}")
    print(f"Target stations: {args_cli.stations}")
    print(f"Samples per station: {args_cli.num_samples}")
    print(f"Output directory: {args_cli.output_dir}")
    print(f"K-sample mixing: {args.pcgen_k}")
    print("="*80)
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load model
    model = load_model(args_cli.checkpoint, args)
    
    # Create dataset splits
    print("\n[INFO] Preparing dataset...")
    split_out_dir = f"./station_splits_gen_inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(split_out_dir, exist_ok=True)
    
    try:
        generate_and_save_station_splits_mat(
            root_dir=args.root_path,
            selected_stations=args_cli.stations,
            output_dir=split_out_dir,
            seq_len=args.seq_len,
            copy_files=False,
        )
    except Exception as e:
        print(f"[WARN] Dataset split creation: {e}")
    
    # Load test dataset
    test_csv = os.path.join(split_out_dir, 'test_list.csv')
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")
    
    test_ds = GenMatDataset(
        csv_path=test_csv,
        root_dir=args.root_path,
        seq_len=args.seq_len,
        return_filename=True
    )
    print(f"[INFO] Loaded test dataset: {len(test_ds)} samples")
    
    # Create output directories
    npz_output_dir = os.path.join(args_cli.output_dir, 'generated_timeseries_npz')
    plot_output_dir = os.path.join(args_cli.output_dir, 'preview_plots')
    
    # Generate samples for each station
    print("\n[INFO] Generating samples...")
    for station_id in args_cli.stations:
        print(f"\n{'='*60}")
        print(f"Processing Station: {station_id}")
        print(f"{'='*60}")
        
        generated_signals, real_names = generate_samples_for_station(
            model, test_ds, station_id, args_cli.num_samples, args
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


if __name__ == '__main__':
    main()

