#!/usr/bin/env python3
"""
Simplified inference script for TimesNet-Gen.
Only loads data for the 5 fine-tuned stations.

Usage:
    python generate_samples.py --num_samples 50
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


def _iter_np_arrays(obj):
    """Recursively iterate through numpy arrays in nested structures."""
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            for item in obj.flat:
                yield from _iter_np_arrays(item)
        else:
            yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_np_arrays(v)
    elif isinstance(obj, np.void):
        if obj.dtype.names:
            for name in obj.dtype.names:
                yield from _iter_np_arrays(obj[name])


def _find_3ch_from_arrays(arrays):
    """Find 3-channel array from list of arrays."""
    # Prefer arrays that are 2D with a 3-channel dimension
    for arr in arrays:
        if isinstance(arr, np.ndarray) and arr.ndim == 2 and (arr.shape[0] == 3 or arr.shape[1] == 3):
            return arr
    # Otherwise, try to find three 1D arrays of same length
    one_d = [a for a in arrays if isinstance(a, np.ndarray) and a.ndim == 1]
    for i in range(len(one_d)):
        for j in range(i + 1, len(one_d)):
            for k in range(j + 1, len(one_d)):
                if one_d[i].shape[0] == one_d[j].shape[0] == one_d[k].shape[0]:
                    return np.stack([one_d[i], one_d[j], one_d[k]], axis=0)
    return None


def load_mat_file(filepath, seq_len=6000, debug=False):
    """Load and preprocess a .mat file (using data_loader_gen.py logic)."""
    try:
        if debug:
            print(f"\n[DEBUG] Loading: {os.path.basename(filepath)}")
        
        # Load with squeeze_me and struct_as_record like data_loader_gen.py
        mat = sio.loadmat(filepath, squeeze_me=True, struct_as_record=False)
        
        if debug:
            print(f"[DEBUG] Keys in mat file: {[k for k in mat.keys() if not k.startswith('__')]}")
        
        # Check if 'EQ' is a struct with nested 'anEQ' structure (like in data_loader_gen.py)
        if 'EQ' in mat:
            try:
                eq_obj = mat['EQ']
                
                if debug:
                    print(f"[DEBUG] EQ type: {type(eq_obj)}")
                    print(f"[DEBUG] EQ shape: {eq_obj.shape if hasattr(eq_obj, 'shape') else 'N/A'}")
                
                # Since struct_as_record=False, EQ is a mat_struct object
                # Access with attributes, not subscripts
                if hasattr(eq_obj, 'anEQ'):
                    dataset = eq_obj.anEQ
                    if debug:
                        print(f"[DEBUG] Found anEQ, type: {type(dataset)}")
                    
                    if hasattr(dataset, 'Accel'):
                        accel = dataset.Accel
                        
                        if debug:
                            print(f"[DEBUG] Found Accel: type={type(accel)}, shape={accel.shape if hasattr(accel, 'shape') else 'N/A'}")
                        
                        if isinstance(accel, np.ndarray):
                            # Transpose to (3, N) if needed
                            if accel.ndim == 2:
                                if accel.shape[1] == 3:
                                    accel = accel.T
                                
                                if accel.shape[0] == 3:
                                    data = accel
                                    if debug:
                                        print(f"[DEBUG] ✅ Successfully extracted 3-channel data! Shape: {data.shape}")
                                    
                                    # Resample if needed
                                    if data.shape[1] != seq_len:
                                        from scipy import signal as sp_signal
                                        data_resampled = np.zeros((3, seq_len), dtype=np.float32)
                                        for i in range(3):
                                            data_resampled[i] = sp_signal.resample(data[i], seq_len)
                                        data = data_resampled
                                        if debug:
                                            print(f"[DEBUG] Resampled to {seq_len} samples")
                                    
                                    return torch.FloatTensor(data)
                                else:
                                    if debug:
                                        print(f"[DEBUG] Unexpected Accel shape[0]: {accel.shape[0]} (expected 3)")
                            else:
                                if debug:
                                    print(f"[DEBUG] Accel is not 2D: ndim={accel.ndim}")
                    else:
                        if debug:
                            print(f"[DEBUG] anEQ has no 'Accel' attribute")
                            if hasattr(dataset, '__dict__'):
                                print(f"[DEBUG] anEQ attributes: {list(vars(dataset).keys())}")
                else:
                    if debug:
                        print(f"[DEBUG] EQ has no 'anEQ' attribute")
                        if hasattr(eq_obj, '__dict__'):
                            print(f"[DEBUG] EQ attributes: {list(vars(eq_obj).keys())}")
                        
            except Exception as e:
                if debug:
                    import traceback
                    print(f"[DEBUG] Could not parse EQ structure: {e}")
                    print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        
        arrays = list(_iter_np_arrays(mat))
        
        if debug:
            print(f"[DEBUG] Found {len(arrays)} arrays")
            for i, arr in enumerate(arrays[:5]):  # Show first 5
                if isinstance(arr, np.ndarray):
                    print(f"[DEBUG]   Array {i}: shape={arr.shape}, dtype={arr.dtype}")
        
        # Common direct keys first
        for key in ['signal', 'data', 'sig', 'x', 'X', 'signal3c', 'acc', 'NS', 'EW', 'UD']:
            if key in mat and isinstance(mat[key], np.ndarray):
                arrays.insert(0, mat[key])
                if debug:
                    print(f"[DEBUG] Found key '{key}': shape={mat[key].shape}")
        
        # Find 3-channel array
        data = _find_3ch_from_arrays(arrays)
        
        if data is None:
            if debug:
                print(f"[DEBUG] Could not find 3-channel array!")
            return None
        
        if debug:
            print(f"[DEBUG] Found 3-channel data: shape={data.shape}")
        
        # Ensure shape is (3, N)
        if data.shape[0] != 3 and data.shape[1] == 3:
            data = data.T
            if debug:
                print(f"[DEBUG] Transposed to: shape={data.shape}")
        
        if data.shape[0] != 3:
            if debug:
                print(f"[DEBUG] Wrong number of channels: {data.shape[0]}")
            return None
        
        # Resample to seq_len
        if data.shape[1] != seq_len:
            from scipy import signal as sp_signal
            data_resampled = np.zeros((3, seq_len), dtype=np.float32)
            for i in range(3):
                data_resampled[i] = sp_signal.resample(data[i], seq_len)
            data = data_resampled
            if debug:
                print(f"[DEBUG] Resampled to: shape={data.shape}")
        
        if debug:
            print(f"[DEBUG] ✅ Successfully loaded!")
        
        return torch.FloatTensor(data)
    
    except Exception as e:
        if debug:
            print(f"[DEBUG] ❌ Exception: {e}")
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


def generate_samples_from_latent_bank(model, latent_bank_path, station_id, num_samples, args, encoder_std=None):
    """
    Generate samples directly from pre-computed latent bank.
    NO REAL DATA NEEDED!
    
    Args:
        model: TimesNet model
        latent_bank_path: Path to latent_bank_phase1.npz
        station_id: Station ID (e.g., '0205')
        num_samples: Number of samples to generate
        args: Model arguments
        encoder_std: Encoder std vector for noise injection
    
    Returns:
        generated_signals: (num_samples, 3, seq_len) array
        real_names_used: List of lists indicating which latent vectors were used
    """
    print(f"[INFO] Loading latent bank from {latent_bank_path}...")
    
    try:
        latent_data = np.load(latent_bank_path)
    except Exception as e:
        print(f"[ERROR] Could not load latent bank: {e}")
        return None, None
    
    # Load latent vectors for this station
    latents_key = f'latents_{station_id}'
    means_key = f'means_{station_id}'
    stdev_key = f'stdev_{station_id}'
    
    if latents_key not in latent_data:
        print(f"[ERROR] Station {station_id} not found in latent bank!")
        print(f"Available stations: {[k.replace('latents_', '') for k in latent_data.keys() if k.startswith('latents_')]}")
        return None, None
    
    latents = latent_data[latents_key]  # (N_samples, seq_len, d_model)
    means = latent_data[means_key]      # (N_samples, seq_len, d_model)
    stdevs = latent_data[stdev_key]     # (N_samples, seq_len, d_model)
    
    print(f"[INFO] Loaded {len(latents)} latent vectors for station {station_id}")
    print(f"[INFO] Generating {num_samples} samples via bootstrap aggregation...")
    
    generated_signals = []
    real_names_used = []
    
    model.eval()
    with torch.no_grad():
        for i in range(num_samples):
            # Bootstrap: randomly select k latent vectors with replacement
            k = min(args.pcgen_k, len(latents))
            selected_indices = np.random.choice(len(latents), size=k, replace=True)
            
            # Mix latent features (average)
            selected_latents = latents[selected_indices]  # (k, seq_len, d_model)
            selected_means = means[selected_indices]      # (k, seq_len, d_model)
            selected_stdevs = stdevs[selected_indices]    # (k, seq_len, d_model)

            mixed_features = np.mean(selected_latents, axis=0)  # (seq_len, d_model)
            mixed_means = np.mean(selected_means, axis=0)       # (seq_len, d_model)
            mixed_stdevs = np.mean(selected_stdevs, axis=0)     # (seq_len, d_model)

            # NOTE: Do NOT add noise during generation (matching untitled1_gen.py)
            # untitled1_gen.py only uses noise during TRAINING (Phase 1), not during generation
            # if encoder_std is not None:
            #     noise = np.random.randn(*mixed_features.shape) * encoder_std
            #     mixed_features = mixed_features + noise

            # Convert to torch tensors
            mixed_features_torch = torch.from_numpy(mixed_features).float().unsqueeze(0)  # (1, seq_len, d_model)
            means_b = torch.from_numpy(mixed_means).float().unsqueeze(0)   # (1, seq_len, d_model)
            stdev_b = torch.from_numpy(mixed_stdevs).float().unsqueeze(0)  # (1, seq_len, d_model)

            if args.use_gpu:
                mixed_features_torch = mixed_features_torch.cuda()
                means_b = means_b.cuda()
                stdev_b = stdev_b.cuda()

            # Decode
            xg = model.project_features_for_reconstruction(mixed_features_torch, means_b, stdev_b)
            
            # Store - transpose to (3, 6000)
            generated_np = xg.squeeze(0).cpu().numpy().T  # (6000, 3) → (3, 6000)
            generated_signals.append(generated_np)
            
            # Track which latent indices were used
            real_names_used.append([f"latent_{idx}" for idx in selected_indices])
            
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{num_samples} samples...")
    
    return np.array(generated_signals), real_names_used


def _preprocess_component_boore(data: np.ndarray, fs: float, corner_freq: float, filter_order: int = 2) -> np.ndarray:
    """Boore (2005) style preprocessing: detrend (linear), zero-padding, high-pass Butterworth (zero-phase)."""
    from scipy.signal import butter, filtfilt
    x = np.asarray(data, dtype=np.float64)
    n = x.shape[0]
    # Linear detrend
    t = np.arange(n, dtype=np.float64)
    t_mean = t.mean()
    x_mean = x.mean()
    denom = np.sum((t - t_mean) ** 2)
    slope = 0.0 if denom == 0 else float(np.sum((t - t_mean) * (x - x_mean)) / denom)
    intercept = float(x_mean - slope * t_mean)
    x_detr = x - (slope * t + intercept)
    # Zero-padding
    Tzpad = (1.5 * filter_order) / max(corner_freq, 1e-6)
    pad_samples = int(round(Tzpad * fs))
    x_pad = np.concatenate([np.zeros(pad_samples, dtype=np.float64), x_detr, np.zeros(pad_samples, dtype=np.float64)])
    # High-pass filter (zero-phase)
    normalized = corner_freq / (fs / 2.0)
    normalized = min(max(normalized, 1e-6), 0.999999)
    b, a = butter(filter_order, normalized, btype='high')
    y = filtfilt(b, a, x_pad)
    return y


def _konno_ohmachi_smoothing(spectrum: np.ndarray, freq: np.ndarray, b: float = 40.0) -> np.ndarray:
    """Konno-Ohmachi smoothing as in MATLAB reference (O(n^2))."""
    f = np.asarray(freq, dtype=np.float64).reshape(-1)
    s = np.asarray(spectrum, dtype=np.float64).reshape(-1)
    f = np.where(f == 0.0, 1e-12, f)
    n = f.shape[0]
    out = np.zeros_like(s)
    for i in range(n):
        w = np.exp(-b * (np.log(f / f[i])) ** 2)
        w[~np.isfinite(w)] = 0.0
        denom = np.sum(w)
        out[i] = 0.0 if denom == 0 else float(np.sum(w * s) / denom)
    return out


def _compute_hvsr_simple(signal: np.ndarray, fs: float = 100.0):
    """Compute HVSR curve using MATLAB-style pipeline (Boore HP filter + FAS + Konno-Ohmachi)."""
    try:
        if signal.ndim != 2 or signal.shape[1] != 3:
            return None, None
        if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
            return None, None

        # Preprocess components (Boore 2005): detrend + zero-padding + high-pass (0.05 Hz)
        ew = _preprocess_component_boore(signal[:, 0], fs, 0.05, 2)
        ns = _preprocess_component_boore(signal[:, 1], fs, 0.05, 2)
        ud = _preprocess_component_boore(signal[:, 2], fs, 0.05, 2)

        n = int(min(len(ew), len(ns), len(ud)))
        if n < 16:
            return None, None
        ew = ew[:n]; ns = ns[:n]; ud = ud[:n]

        # FFT amplitudes and linear frequency grid
        half = n // 2
        if half <= 1:
            return None, None
        freq = (np.arange(0, half, dtype=np.float64)) * (fs / n)
        amp_ew = np.abs(np.fft.fft(ew))[:half]
        amp_ns = np.abs(np.fft.fft(ns))[:half]
        amp_ud = np.abs(np.fft.fft(ud))[:half]

        # Horizontal combination via geometric mean, then Konno-Ohmachi smoothing
        combined_h = np.sqrt(np.maximum(amp_ew, 0.0) * np.maximum(amp_ns, 0.0))
        sm_h = _konno_ohmachi_smoothing(combined_h, freq, 40.0)
        sm_v = _konno_ohmachi_smoothing(amp_ud, freq, 40.0)

        sm_v_safe = np.where(sm_v <= 0.0, 1e-12, sm_v)
        sm_hvsr = sm_h / sm_v_safe

        # Limit to 1-20 Hz band
        mask = (freq >= 1.0) & (freq <= 20.0)
        if not np.any(mask):
            return None, None
        return freq[mask], sm_hvsr[mask]
    except Exception:
        return None, None


def save_generated_samples(generated_signals, real_names, station_id, output_dir):
    """Save generated samples to NPZ file with HVSR and f0 data."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute HVSR and f0 for all generated signals
    f0_list = []
    hvsr_curves = []
    fs = 100.0
    
    print(f"[INFO] Computing HVSR and f0 for {len(generated_signals)} generated samples...")
    for idx, sig in enumerate(generated_signals):
        # sig is (3, T), need to transpose to (T, 3)
        sig_t = sig.T  # (T, 3)
        freq, hvsr = _compute_hvsr_simple(sig_t, fs)
        if freq is not None and hvsr is not None:
            hvsr_curves.append((freq, hvsr))
            # f0 = frequency at max HVSR
            max_idx = np.argmax(hvsr)
            f0 = float(freq[max_idx])
            f0_list.append(f0)
    
    # Build median HVSR curve on a fixed frequency grid (1-20 Hz, 400 points for consistency)
    hvsr_freq = None
    hvsr_median = None
    if hvsr_curves:
        # Use a fixed frequency grid for consistency with other plots
        hvsr_freq = np.linspace(1.0, 20.0, 400)
        # Interpolate all curves to common grid
        hvsr_matrix = []
        for freq, hvsr in hvsr_curves:
            hvsr_interp = np.interp(hvsr_freq, freq, hvsr, left=hvsr[0], right=hvsr[-1])
            hvsr_matrix.append(hvsr_interp)
        hvsr_median = np.median(np.vstack(hvsr_matrix), axis=0)
    
    # Build f0 histogram (PDF)
    f0_bins = np.linspace(1.0, 20.0, 21)
    f0_array = np.array(f0_list)
    f0_hist, _ = np.histogram(f0_array, bins=f0_bins)
    f0_pdf = f0_hist.astype(float)
    f0_sum = f0_pdf.sum()
    if f0_sum > 0:
        f0_pdf = f0_pdf / f0_sum
    
    # Save timeseries NPZ with HVSR data
    output_path = os.path.join(output_dir, f'station_{station_id}_generated_timeseries.npz')
    np.savez_compressed(
        output_path,
        generated_signals=generated_signals,
        signals_generated=generated_signals,  # Alias for compatibility
        real_names=real_names,
        station_id=station_id,
        station=station_id,  # Alias for compatibility
        f0_timesnet=f0_array,
        f0_bins=f0_bins,
        pdf_timesnet=f0_pdf,
        hvsr_freq_timesnet=hvsr_freq if hvsr_freq is not None else np.array([]),
        hvsr_median_timesnet=hvsr_median if hvsr_median is not None else np.array([]),
    )
    print(f"[INFO] Saved {len(generated_signals)} generated samples to {output_path}")
    if len(f0_list) > 0:
        print(f"[INFO]   - f0 samples: {len(f0_list)}, median f0: {np.median(f0_array):.2f} Hz")
    else:
        print(f"[INFO]   - No valid HVSR computed")


def fine_tune_model(model, all_station_files, args, encoder_std, epochs=10, lr=1e-4):
    """
    Fine-tune the model on 5 stations with noise injection.
    Matches Phase 1 training in untitled1_gen.py exactly.
    """
    print("\n" + "="*80)
    print("Phase 1: Fine-Tuning with Noise Injection")
    print("="*80)
    
    # Prepare data loader
    all_data = []
    for station_id, files in all_station_files.items():
        for fpath in files:
            data = load_mat_file(fpath, args.seq_len, debug=False)
            if data is not None:
                all_data.append(data)
    
    if len(all_data) == 0:
        print("[WARN] No data loaded for fine-tuning!")
        return model
    
    print(f"[INFO] Loaded {len(all_data)} samples for fine-tuning")
    
    # Create optimizer (matching untitled1_gen.py Phase 1)
    batch_size = 32
    weight_decay = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # AMP scaler (matching untitled1_gen.py)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.use_gpu))
    
    # Gradient clipping (matching untitled1_gen.py)
    grad_clip = 1.0
    
    train_losses_p1 = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_rec = 0.0
        num_batches = 0
        
        # Shuffle data
        np.random.shuffle(all_data)
        
        for i in range(0, len(all_data), batch_size):
            batch = all_data[i:i+batch_size]
            if len(batch) == 0:
                continue
            
            # Stack batch
            x_list = []
            for sig in batch:
                # sig is (3, 6000), transpose to (6000, 3)
                x_list.append(sig.transpose(0, 1))
            
            x = torch.stack(x_list, dim=0)  # (batch, 6000, 3)
            if args.use_gpu:
                x = x.cuda()
            
            # Zero gradients (matching untitled1_gen.py)
            optimizer.zero_grad(set_to_none=True)
            
            # Forward with AMP and noise injection (matching untitled1_gen.py Phase 1)
            with torch.cuda.amp.autocast(enabled=(args.use_gpu)):
                enc_out, means_b, stdev_b = model.encode_features_for_reconstruction(x)
                
                # Add noise if encoder_std available (matching untitled1_gen.py line 945-948)
                if encoder_std is not None:
                    std_vec = torch.from_numpy(encoder_std).to(enc_out.device).float()
                    noise = torch.randn_like(enc_out) * std_vec.view(1, 1, -1) * 1.0  # noise_std_scale=1.0
                    enc_out = enc_out + noise
                
                # Decode
                x_hat = model.project_features_for_reconstruction(enc_out, means_b, stdev_b)
                
                # Reconstruction loss (MSE, matching untitled1_gen.py)
                loss_rec = torch.nn.functional.mse_loss(x_hat, x)
                loss = loss_rec
            
            # Backward with gradient scaling (matching untitled1_gen.py)
            scaler.scale(loss).backward()
            
            # Gradient clipping (matching untitled1_gen.py)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            
            # Optimizer step with scaler (matching untitled1_gen.py)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += float(loss.detach().cpu())
            total_rec += float(loss_rec.detach().cpu())
            num_batches += 1
        
        # Scheduler step (matching untitled1_gen.py)
        scheduler.step()
        
        avg_loss = total_loss / max(1, num_batches)
        avg_rec = total_rec / max(1, num_batches)
        train_losses_p1.append(avg_loss)
        print(f"[P1] epoch {epoch+1}/{epochs} loss={avg_loss:.4f} (rec={avg_rec:.4f})")
    
    print("[INFO] Phase 1 fine-tuning complete!")
    
    # Save fine-tuned model (matching untitled1_gen.py Phase 1 checkpoint)
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    fine_tuned_path = os.path.join(checkpoint_dir, 'timesnet_pointcloud_phase1_finetuned.pth')
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses_phase1': train_losses_p1,
        'phase': 'phase1'
    }, fine_tuned_path)
    print(f"[INFO] ✓ Fine-tuned model saved to: {fine_tuned_path}")
    
    return model


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
    parser.add_argument('--latent_bank', type=str, 
                        default=r'D:\Baris\codes\Time-Series-Library-main\checkpoints\latent_bank_phase1.npz',
                        help='Path to latent bank NPZ file')
    parser.add_argument('--num_samples', type=int, default=50, 
                        help='Number of samples to generate per station')
    parser.add_argument('--output_dir', type=str, default='./generated_samples', 
                        help='Output directory')
    parser.add_argument('--num_preview', type=int, default=2, 
                        help='Number of preview plots per station')
    parser.add_argument('--stations', type=str, nargs='+', default=['0205', '1716', '2020', '3130', '4628'],
                        help='Target station IDs')
    parser.add_argument('--data_root', type=str, default=r"D:\Baris\5stats/", 
                        help='Root path to seismic data (only needed if --fine_tune is used)')
    parser.add_argument('--fine_tune', action='store_true',
                        help='Fine-tune the model before generation (use with Phase 0 checkpoint)')
    parser.add_argument('--fine_tune_epochs', type=int, default=10,
                        help='Number of fine-tuning epochs')
    parser.add_argument('--fine_tune_lr', type=float, default=1e-4,
                        help='Learning rate for fine-tuning')
    
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
    
    # Try to load encoder_std from Phase 0 (only needed if fine-tuning)
    encoder_std_path = './pcgen_stats/encoder_feature_std.npy'
    encoder_std = None
    if os.path.exists(encoder_std_path):
        encoder_std = np.load(encoder_std_path)
        print(f"[INFO] Loaded encoder_std from {encoder_std_path} (shape: {encoder_std.shape})")
        print(f"[INFO] encoder_std loaded (used only for fine-tuning, NOT for generation)")
    else:
        print(f"[INFO] No encoder_std found (not needed for generation, only for fine-tuning)")
    
    # Check if latent bank exists
    if not os.path.exists(args_cli.latent_bank):
        print(f"\n❌ ERROR: Latent bank not found!")
        print(f"Looking for: {args_cli.latent_bank}")
        print(f"\nPlease run untitled1_gen.py first to generate the latent bank.")
        return
    
    print(f"[INFO] Using latent bank: {args_cli.latent_bank}")
    
    # Fine-tune if requested (requires real data)
    if args_cli.fine_tune:
        print("\n[INFO] Fine-tuning enabled! Loading real data...")
        
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
        
        model = fine_tune_model(model, all_station_files, args, encoder_std, 
                               epochs=args_cli.fine_tune_epochs, 
                               lr=args_cli.fine_tune_lr)
    
    # Create output directories
    npz_output_dir = os.path.join(args_cli.output_dir, 'generated_timeseries_npz')
    plot_output_dir = os.path.join(args_cli.output_dir, 'preview_plots')
    
    # Generate samples for each station (from latent bank)
    print("\n[INFO] Generating samples from latent bank...")
    for station_id in args_cli.stations:
        print(f"\n{'='*60}")
        print(f"Processing Station: {station_id}")
        print(f"{'='*60}")
        
        generated_signals, real_names = generate_samples_from_latent_bank(
            model, args_cli.latent_bank, station_id, args_cli.num_samples, args, encoder_std
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
    
    # Debug: Show how many samples were generated per station
    print("\n[DEBUG] Generated samples per station:")
    for station_id in args_cli.stations:
        npz_path = os.path.join(npz_output_dir, f'station_{station_id}_generated_timeseries.npz')
        if os.path.exists(npz_path):
            try:
                data = np.load(npz_path, allow_pickle=True)
                if 'signals_generated' in data:
                    n_samples = data['signals_generated'].shape[0]
                    print(f"  Station {station_id}: {n_samples} samples")
            except Exception as e:
                print(f"  Station {station_id}: Error loading NPZ - {e}")
    print("="*80)
    
    # Create HVSR comparison plots (import plot_combined_hvsr_all_sources and call main)
    print("\n[INFO] Creating HVSR comparison plots (matrices, HVSR curves, f0 distributions)...")
    print("[INFO] Only plotting TimesNet-Gen vs Real (no Recon/VAE)")
    try:
        import sys
        # Import the plotting module
        import plot_combined_hvsr_all_sources as hvsr_plotter
        
        # Override sys.argv to pass arguments to the plotter
        # Only provide gen_dir and gen_ts_dir, explicitly disable others with empty strings
        original_argv = sys.argv
        sys.argv = [
            'plot_combined_hvsr_all_sources.py',
            '--gen_dir', npz_output_dir,  # Use our generated NPZs as gen_dir (they now have HVSR/f0 data)
            '--gen_ts_dir', npz_output_dir,  # Also use for timeseries plots
            '--out', os.path.join(args_cli.output_dir, 'hvsr_analysis'),
            '--recon_dir', '',  # Explicitly empty to disable auto-default
            '--vae_dir', '',    # Explicitly empty to disable auto-default
            '--vae_gen_dir', '',  # Explicitly empty to disable auto-default
        ]
        
        # Call the main plotting function
        hvsr_plotter.main()
        
        # Restore original argv
        sys.argv = original_argv
        
        print(f"[INFO] ✅ HVSR analysis complete! Plots saved to: {os.path.join(args_cli.output_dir, 'hvsr_analysis')}")
    except Exception as e:
        import traceback
        print(f"[WARN] Could not create HVSR plots: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()

