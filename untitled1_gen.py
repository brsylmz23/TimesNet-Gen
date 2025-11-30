#!/usr/bin/env python3
import os
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

# Helper: reconstruction loss function
import torch.nn.functional as F
def reconstruction_loss(x_hat: torch.Tensor, x: torch.Tensor, loss_type: str = 'l1') -> torch.Tensor:
    if loss_type == 'l1':
        return F.l1_loss(x_hat, x)
    return F.mse_loss(x_hat, x)


class GenArgs:
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
        
        # Training configuration
        self.train_epochs = 50
        self.learning_rate = 3e-4
        self.weight_decay = 1e-4
        self.grad_clip = 1.0
        
        # VAE loss configuration (β-ELBO)
        self.kl_beta_max = 0.05
        self.kl_warmup_epochs = 30
        self.recon_loss_type = 'mse'  # 'l1' or 'mse' (original TimesNet uses MSE)
        
        # System
        self.use_gpu = torch.cuda.is_available()
        self.use_amp = False  # Disable mixed precision (cuFFT constraint)
        self.seed = 0

        # AE / point-cloud generation options
        self.ae_mode = True  # if True: train as AE (no KL), use mu deterministically
        self.pcgen_enable = True  # enable point-cloud generation path in eval
        self.pcgen_k = 5  # number of latent points to mix
        self.pcgen_per_station = True  # restrict mixes within same station pool
        self.pcgen_jitter_std = 0.0  # optional Gaussian jitter on mixed z

        # Phase-0/Phase-1 latent noise injection (encoder feature space)
        self.latent_stats_path = './pcgen_stats/encoder_feature_std.npy'  # per-dim std (d_model,)
        self.noise_injection_enable = False  # enable in Phase 1 to add noise using Phase 0 std
        self.noise_std_scale = 1.0  # multiplier over stored std
        
        # Number of samples to generate per station after training
        self.num_generated_samples = 50  # Generate 50 samples per station by default


def main():
    args = GenArgs()
    split_out_dir = f"./station_splits_gen/afad_{args.seq_len}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # Force 5 stations to TEST (evaluation set); others to TRAIN
    test_stations = ['0205', '1716', '2020', '3130', '4628']
    os.makedirs(split_out_dir, exist_ok=True)
    try:
        print(f"[gen][splits] Creating splits to {split_out_dir}")
        print(f"[gen][splits] Forcing {test_stations} to TEST; others to TRAIN")
        generate_and_save_station_splits_mat(
            root_dir=args.root_path,
            selected_stations=test_stations,
            output_dir=split_out_dir,
            seq_len=args.seq_len,
            copy_files=False,
        )
    except Exception as e:
        print(f"[gen][splits][warn] {e}")

    # Split train into train/val (80/20)
    train_csv_orig = os.path.join(split_out_dir, 'train_list.csv')
    train_csv_split = os.path.join(split_out_dir, 'train_split.csv')
    val_csv_split = os.path.join(split_out_dir, 'val_split.csv')
    
    # Read all train items and shuffle
    import csv
    train_items = []
    with open(train_csv_orig, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        train_items = [row for row in reader if len(row) >= 2]
    
    np.random.seed(args.seed)
    np.random.shuffle(train_items)
    split_idx = int(len(train_items) * 0.8)
    train_split = train_items[:split_idx]
    val_split = train_items[split_idx:]
    
    with open(train_csv_split, 'w') as f:
        f.write('name,station\n')
        for row in train_split:
            f.write(','.join(row) + '\n')
    with open(val_csv_split, 'w') as f:
        f.write('name,station\n')
        for row in val_split:
            f.write(','.join(row) + '\n')
    
    print(f"[gen] Train: {len(train_split)}, Val: {len(val_split)}")
    
    train_ds = GenMatDataset(args.root_path, train_csv_split, args.seq_len)
    val_ds = GenMatDataset(args.root_path, val_csv_split, args.seq_len)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Test dataset for evaluation
    test_csv = os.path.join(split_out_dir, 'test_list.csv')
    test_ds = GenMatDataset(args.root_path, test_csv, args.seq_len)

    # Reconstruction-only TimesNet point-cloud variant (keeps original TimesNet intact)
    from TimesNet_PointCloud import TimesNetPointCloud as TimesNetModel
    class _Cfg:
        def __init__(self):
            self.task_name = 'anomaly_detection'
            self.seq_len = args.seq_len
            self.pred_len = 0
            self.top_k = args.top_k
            self.d_model = args.d_model
            self.d_ff = args.d_ff
            self.num_kernels = args.num_kernels
            self.e_layers = args.e_layers
            self.dropout = args.dropout
            self.enc_in = 3
            self.c_out = 3
            self.embed = 'fixed'
            self.freq = 'h'
            self.label_len = 0
            self.num_class = 0
            self.use_ps_heads = False

    base_cfg = _Cfg()
    model = TimesNetModel(base_cfg)
    if args.use_gpu:
        model = model.cuda()
        print(f"[GPU] Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("[CPU] CUDA not available, using CPU")

    # Phase 0: compute encoder feature std across training set
    @torch.no_grad()
    def compute_encoder_feature_std(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader,
                                    save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.eval()
        n = 0
        # Streaming mean/var
        mean_acc = torch.zeros(args.d_model, device=('cuda' if args.use_gpu else 'cpu'))
        m2_acc = torch.zeros(args.d_model, device=('cuda' if args.use_gpu else 'cpu'))
        for x, sid, _ in data_loader:
            if args.use_gpu:
                x = x.cuda()
            enc_out, _, _ = model.encode_features_for_reconstruction(x)
            f = enc_out.mean(dim=1)  # (B, d_model)
            b = f.shape[0]
            # Update per-dim Welford
            n_new = n + b
            delta = f.mean(dim=0) - mean_acc
            mean_acc = mean_acc + delta * (b / max(1, n_new))
            # For M2, compute sum of squared differences
            diff = f - mean_acc.unsqueeze(0)
            m2_acc = m2_acc + (diff.pow(2).sum(dim=0))
            n = n_new
        var = m2_acc / max(1, (n - 1))
        std = torch.sqrt(torch.clamp(var, min=1e-8))
        np.save(save_path, std.detach().cpu().numpy())
        print(f"[phase0] Saved encoder feature std to {save_path}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # Cosine decay scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.train_epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.use_gpu and args.use_amp))

    print("[TimesNet-Gen] start training...")
    train_losses = []

    for epoch in range(args.train_epochs):
        model.train()
        total = 0.0
        total_rec = 0.0
        total_kl = 0.0
        for x, sid, _ in train_loader:
            if args.use_gpu:
                x = x.cuda(); sid = sid.cuda()
            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(args.use_gpu and args.use_amp)):
                # Encode → optional noise injection using Phase-0 std → project
                enc_out, means_b, stdev_b = model.encode_features_for_reconstruction(x)
                if args.noise_injection_enable and os.path.exists(args.latent_stats_path):
                    std_vec = torch.from_numpy(np.load(args.latent_stats_path)).to(enc_out.device).float()
                    noise = torch.randn_like(enc_out) * std_vec.view(1, 1, -1) * float(args.noise_std_scale)
                    enc_out = enc_out + noise
                x_hat = model.project_features_for_reconstruction(enc_out, means_b, stdev_b)
                loss_rec = reconstruction_loss(x_hat, x, loss_type=args.recon_loss_type)
                loss_kl = torch.zeros((), device=x_hat.device)
                beta = 0.0
                loss = loss_rec
            scaler.scale(loss).backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optim)
            scaler.update()
            total += float(loss.detach().cpu())
            total_rec += float(loss_rec.detach().cpu())
            total_kl += float(loss_kl.detach().cpu())
        scheduler.step()
        avg_loss = total / len(train_loader)
        avg_rec = total_rec / len(train_loader)
        avg_kl = total_kl / len(train_loader)
        train_losses.append(avg_loss)
        print(f"[TimesNet-Gen] epoch {epoch+1}/{args.train_epochs} loss={avg_loss:.4f} (rec={avg_rec:.4f}, kl={avg_kl:.4f}, beta={beta:.3f})")

    print("[TimesNet-Gen] training done.")
    # After training: Phase-0 std computation with trained weights
    try:
        print(f"[phase0] Computing encoder feature std across training set (post-training)...")
        tmp_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        compute_encoder_feature_std(model, tmp_loader, args.latent_stats_path)
    except Exception as e:
        print(f"[phase0][warn] std computation failed: {e}")
    
    # Plot training loss (save under base_dir/phase0)
    out_dir_base = './gen_eval_results_base'
    out_dir = os.path.join(out_dir_base, 'phase0')
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', linewidth=2, markersize=6, color='steelblue')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.title('TimesNet-Gen Base Training Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'training_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[TimesNet-Gen] training loss plot saved to {out_dir}/training_loss.png")

    # -------- Posterior Gaussianity Diagnostics --------
    @torch.no_grad()
    def analyze_posterior_gaussianity(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader,
                                      out_dir: str, max_samples: int = 2000):
        import math
        import numpy as np
        try:
            from scipy import stats as scipy_stats
        except Exception:
            scipy_stats = None
        os.makedirs(out_dir, exist_ok=True)
        model.eval()
        feats = []
        num_collected = 0
        for x, sid, _ in data_loader:
            if args.use_gpu:
                x = x.cuda(); sid = sid.cuda()
            enc_out, _, _ = model.encode_features_for_reconstruction(x)
            feats.append(enc_out.mean(dim=1).detach().cpu().numpy())
            num_collected += x.shape[0]
            if num_collected >= max_samples:
                break
        if not feats:
            return
        Z = np.concatenate(feats, axis=0)  # (N, d_model)
        MU = Z
        LV = np.zeros_like(Z)
        N, D = Z.shape
        # Save summary stats for mu/logvar/sigma
        mu_mean = MU.mean(axis=0)
        mu_std  = MU.std(axis=0)
        lv_mean = LV.mean(axis=0)
        sigma_mean = np.exp(0.5 * lv_mean)
        import csv
        with open(os.path.join(out_dir, 'latent_mu_logvar_summary.csv'), 'w', newline='') as fcsv:
            w = csv.writer(fcsv)
            w.writerow(['dim', 'mu_mean', 'mu_std', 'logvar_mean', 'sigma_mean'])
            for d in range(D):
                w.writerow([d, float(mu_mean[d]), float(mu_std[d]), float(lv_mean[d]), float(sigma_mean[d])])
        # Histograms for first few dims with prior overlay
        dims_to_plot = min(6, D)
        x_plot = np.linspace(-4, 4, 401)
        prior_pdf = (1.0/np.sqrt(2*np.pi)) * np.exp(-0.5 * x_plot**2)
        for d in range(dims_to_plot):
            plt.figure(figsize=(6, 4))
            plt.hist(Z[:, d], bins=40, density=True, alpha=0.6, color='gray', label=f'z[{d}]')
            plt.plot(x_plot, prior_pdf, 'r--', linewidth=2, label='N(0,1)')
            plt.title(f'Latent dim {d} histogram vs N(0,1) (N={N})')
            plt.grid(True, alpha=0.3); plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'latent_hist_dim{d}.png'), dpi=200, bbox_inches='tight')
            plt.close()
        # Correlation heatmap
        Z_std = (Z - Z.mean(axis=0, keepdims=True)) / (Z.std(axis=0, keepdims=True) + 1e-8)
        C = np.corrcoef(Z_std, rowvar=False)
        plt.figure(figsize=(6, 5))
        im = plt.imshow(C, vmin=-1, vmax=1, cmap='coolwarm')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title('Latent correlation matrix (should ~ I)')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'latent_corr_heatmap.png'), dpi=220, bbox_inches='tight')
        plt.close()
        # KS test vs N(0,1) and per-dim KL
        per_dim_kl = 0.5 * (np.exp(LV) + MU**2 - 1.0 - LV)  # (N,D)
        with open(os.path.join(out_dir, 'latent_per_dim_kl.csv'), 'w', newline='') as fcsv:
            w = csv.writer(fcsv)
            w.writerow(['dim', 'kl_mean'])
            for d in range(D):
                w.writerow([d, float(per_dim_kl[:, d].mean())])
        if scipy_stats is not None:
            with open(os.path.join(out_dir, 'latent_ks_test.csv'), 'w', newline='') as fcsv:
                w = csv.writer(fcsv)
                w.writerow(['dim', 'ks_stat', 'p_value'])
                for d in range(D):
                    ks_stat, p_val = scipy_stats.kstest(Z[:, d], 'norm')
                    w.writerow([d, float(ks_stat), float(p_val)])
        # Overlapping histograms: 6 dimensions in different colors on same plot
        dims_to_plot_combined = min(6, D)
        colors = ['steelblue', 'darkorange', 'green', 'red', 'purple', 'brown']
        
        plt.figure(figsize=(10, 6))
        x_range = np.linspace(-4, 4, 401)
        prior_pdf = (1.0/np.sqrt(2*np.pi)) * np.exp(-0.5 * x_range**2)
        plt.plot(x_range, prior_pdf, 'k--', linewidth=2, label='N(0,1) prior', alpha=0.7)
        
        for d in range(dims_to_plot_combined):
            plt.hist(Z[:, d], bins=50, density=True, alpha=0.4, color=colors[d], 
                    label=f'z[{d}]', edgecolor='none')
        
        plt.xlabel('Latent Value', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title(f'Latent Space: First 6 Dimensions Distribution (N={N})', fontsize=13)
        plt.legend(loc='upper right', fontsize=10, ncol=2)
        plt.grid(True, alpha=0.3)
        plt.xlim(-4, 4)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'latent_distributions_first6_combined.png'), dpi=220, bbox_inches='tight')
        plt.close()
        
        # 2D scatter plots: visualize latent space clustering (pairwise)
        # Plot first 3 pairs individually
        for pair_idx in range(3):
            d1 = pair_idx * 2
            d2 = pair_idx * 2 + 1
            if d2 < dims_to_plot_combined:
                plt.figure(figsize=(7, 6))
                plt.scatter(Z[:, d1], Z[:, d2], alpha=0.3, s=10, c=colors[pair_idx], edgecolors='none')
                plt.xlabel(f'z[{d1}]', fontsize=11)
                plt.ylabel(f'z[{d2}]', fontsize=11)
                plt.title(f'Latent Space: z[{d1}] vs z[{d2}] (N={N})', fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.axhline(0, color='gray', linewidth=0.8, linestyle='--')
                plt.axvline(0, color='gray', linewidth=0.8, linestyle='--')
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f'latent_scatter_z{d1}_z{d2}.png'), dpi=200, bbox_inches='tight')
                plt.close()
        
        # t-SNE 2D projection (colored by station if available)
        try:
            from sklearn.manifold import TSNE
            print(f"[diag] Computing t-SNE projection...")
            Z_tsne = TSNE(n_components=2, random_state=args.seed, perplexity=min(30, N//4)).fit_transform(Z[:min(1000, N)])
            plt.figure(figsize=(8, 7))
            plt.scatter(Z_tsne[:, 0], Z_tsne[:, 1], alpha=0.5, s=15, c='steelblue', edgecolors='none')
            plt.xlabel('t-SNE Dimension 1', fontsize=11)
            plt.ylabel('t-SNE Dimension 2', fontsize=11)
            plt.title(f'Latent Space (t-SNE 2D) - N={min(1000, N)} samples', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'latent_tsne_2d.png'), dpi=220, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"[diag] t-SNE failed: {e}")
        
        print(f"[diag] Posterior Gaussianity plots saved to {out_dir}")

    analyze_posterior_gaussianity(model, val_loader, os.path.join(out_dir, 'posterior_diagnostics'), max_samples=2000)

    # -------- Helper functions for HVSR/f0 (MATLAB-like pipeline from transfer_learning_station_cond.py) --------
    @torch.no_grad()
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

    @torch.no_grad()
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

    @torch.no_grad()
    def _compute_hvsr(signal: np.ndarray, fs: float = 100.0):
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

    @torch.no_grad()
    def _estimate_f0(freq: np.ndarray, hvsr: np.ndarray):
        if freq is None or hvsr is None or len(freq) == 0:
            return None
        return float(freq[int(np.argmax(hvsr))])

    # -------- Reconstruction sanity check on 5 test stations --------
    @torch.no_grad()
    def plot_reconstruction_samples(model: torch.nn.Module, dataset: GenMatDataset, target_stations: list,
                                    out_dir: str, max_per_station: int = 10):
        os.makedirs(out_dir, exist_ok=True)
        model.eval()
        # Map station index to code
        id_to_station = dataset.id_to_station
        collected = {st: 0 for st in target_stations}
        total = len(dataset)
        for idx in range(total):
            x, sid, name = dataset[idx]
            st = id_to_station[int(sid.item())]
            if st not in target_stations:
                continue
            if collected[st] >= max_per_station:
                continue
            x_batch = x.unsqueeze(0)
            sid_batch = sid.unsqueeze(0)
            if args.use_gpu:
                x_batch = x_batch.cuda(); sid_batch = sid_batch.cuda()
            # Use deterministic path for plotting to reduce variance
            x_hat = model(x_batch, sid_batch)
            if isinstance(x_hat, tuple):
                x_hat = x_hat[0]
            xr = x_batch.squeeze(0).detach().cpu().numpy()
            xh = x_hat.squeeze(0).detach().cpu().numpy()
            # Signals and HVSR
            freq_r, hvsr_r = _compute_hvsr(xr)
            freq_h, hvsr_h = _compute_hvsr(xh)
            # Plot 2x2
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            t = np.arange(xr.shape[0]) / 100.0
            # Plot raw signals (no additional centering; normalization was applied inside the model only)
            for ch, label in enumerate(['E', 'N', 'U']):
                axes[0, 0].plot(t, xr[:, ch], label=f'{label} (Real)', alpha=0.7)
            axes[0, 0].set_title(f'Real Signal - Station {st}')
            axes[0, 0].set_xlabel('Time (s)'); axes[0, 0].set_ylabel('Amplitude')
            axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)
            for ch, label in enumerate(['E', 'N', 'U']):
                axes[0, 1].plot(t, xh[:, ch], label=f'{label} (Recon)', alpha=0.7)
            axes[0, 1].set_title('Reconstruction (TimesNet-Gen)')
            axes[0, 1].set_xlabel('Time (s)'); axes[0, 1].set_ylabel('Amplitude')
            axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)
            if freq_r is not None and hvsr_r is not None:
                axes[1, 0].plot(freq_r, hvsr_r, color='steelblue', linewidth=2, label='Real HVSR')
                f0r = _estimate_f0(freq_r, hvsr_r)
                if f0r is not None:
                    axes[1, 0].axvline(f0r, color='red', linestyle='--', linewidth=1.5, label=f'f0={f0r:.2f} Hz')
                axes[1, 0].set_xlim(1, 20)
                axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)
            if freq_h is not None and hvsr_h is not None:
                axes[1, 1].plot(freq_h, hvsr_h, color='seagreen', linewidth=2, label='Recon HVSR')
                f0h = _estimate_f0(freq_h, hvsr_h)
                if f0h is not None:
                    axes[1, 1].axvline(f0h, color='red', linestyle='--', linewidth=1.5, label=f'f0={f0h:.2f} Hz')
                axes[1, 1].set_xlim(1, 20)
                axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)
            plt.tight_layout()
            fname = f'recon_check_{st}_{collected[st]+1:02d}.png'
            plt.savefig(os.path.join(out_dir, fname), dpi=160, bbox_inches='tight')
            plt.close()
            collected[st] += 1
            # Early exit if all collected
            if all(collected[s] >= max_per_station for s in target_stations):
                break
        print(f"[diag] Reconstruction samples saved to {out_dir}")

    plot_reconstruction_samples(model, test_ds, test_stations, os.path.join(out_dir, 'recon_sanity_check'), max_per_station=10)

    # -------- Base generative extended-matrix evaluation (no conditioning) --------
    @torch.no_grad()
    def evaluate_extended_base(model: torch.nn.Module, dataset: GenMatDataset, out_dir: str,
                               samples_per_station: int = 100, target_stations: list = None):
        os.makedirs(out_dir, exist_ok=True)
        model.eval()
        id_to_station = dataset.id_to_station
        # Filter to target stations if provided
        if target_stations:
            id_to_station = [st for st in id_to_station if st in target_stations]
        S = len(id_to_station)
        print(f"[eval] Evaluating {S} stations with {samples_per_station} samples each...")

        # Real f0 per station (subset)
        real_f0 = {st: [] for st in id_to_station}
        real_signals = {st: [] for st in id_to_station}
        real_hvsr_curves = {st: [] for st in id_to_station}
        real_names = {st: [] for st in id_to_station}
        idxs = np.random.choice(len(dataset), size=min(500, len(dataset)), replace=False)
        print(f"[eval] Computing real f0 from {len(idxs)} samples...")
        for i in idxs:
            x, sid, name = dataset[i]
            st = id_to_station[int(sid.item())]
            freq, hvsr = _compute_hvsr(x.numpy())
            if freq is None:
                continue
            f0 = _estimate_f0(freq, hvsr)
            if f0 is not None:
                real_f0[st].append(f0)
                real_signals[st].append(x.numpy())
                real_hvsr_curves[st].append((freq, hvsr))
                try:
                    real_names[st].append(str(name))
                except Exception:
                    real_names[st].append(f"real_{st}_{len(real_f0[st]):03d}")

        # Generated f0 per station
        # If AE/point-cloud mode is enabled, draw z from latent point-cloud (mu bank)
        # Otherwise sample z ~ N(0, I)
        gen_f0 = {st: [] for st in id_to_station}
        gen_signals = {st: [] for st in id_to_station}
        gen_hvsr_curves = {st: [] for st in id_to_station}
        gen_names = {st: [] for st in id_to_station}
        print(f"[eval] Generating {samples_per_station} samples per station...")

        # Precompute station index lists for on-the-fly encoder feature mixing
        station_to_indices = {}
        for idx_all in range(len(dataset)):
            _, sid_all, _ = dataset[idx_all]
            st_code = dataset.id_to_station[int(sid_all.item())]
            station_to_indices.setdefault(st_code, []).append(idx_all)
        for st_idx, st in enumerate(id_to_station):
            if (st_idx + 1) % 10 == 0 or st_idx == 0:
                print(f"  Processing station {st_idx+1}/{S}: {st}")
            for sample_idx in range(samples_per_station):
                # Choose z
                if args.pcgen_enable:
                    # On-the-fly k-sample feature mixing (encoder feature level)
                    if args.pcgen_per_station:
                        cand_indices = station_to_indices.get(st, list(range(len(dataset))))
                    else:
                        cand_indices = list(range(len(dataset)))
                    if len(cand_indices) == 0:
                        cand_indices = list(range(len(dataset)))
                    k = min(args.pcgen_k, len(cand_indices))
                    pick = np.random.choice(cand_indices, size=k, replace=False)
                    enc_list = []
                    means_list = []
                    stdev_list = []
                    for pi in pick:
                        x_i, _, _ = dataset[pi]
                        x_b = x_i.unsqueeze(0)
                        if args.use_gpu:
                            x_b = x_b.cuda()
                        with torch.no_grad():
                            enc_i, means_i, stdev_i = model.encode_features_for_reconstruction(x_b)
                        enc_list.append(enc_i)
                        means_list.append(means_i)
                        stdev_list.append(stdev_i)
                    # Equal-weight average of encoder sequences and stats
                    enc_out = torch.stack(enc_list, dim=0).mean(dim=0)
                    means_mix = torch.stack(means_list, dim=0).mean(dim=0)
                    stdev_mix = torch.stack(stdev_list, dim=0).mean(dim=0)
                    if args.pcgen_jitter_std > 0:
                        enc_out = enc_out + torch.randn_like(enc_out) * args.pcgen_jitter_std
                    xg = model.project_features_for_reconstruction(enc_out, means=means_mix, stdev=stdev_mix)
                else:
                    # Fallback: single random encoder feature from station
                    cand_indices = station_to_indices.get(st, list(range(len(dataset))))
                    if len(cand_indices) == 0:
                        cand_indices = list(range(len(dataset)))
                    pi = int(np.random.choice(cand_indices, size=1)[0])
                    x_i, _, _ = dataset[pi]
                    x_b = x_i.unsqueeze(0)
                    if args.use_gpu:
                        x_b = x_b.cuda()
                    with torch.no_grad():
                        enc_out, means_i, stdev_i = model.encode_features_for_reconstruction(x_b)
                    xg = model.project_features_for_reconstruction(enc_out, means=means_i, stdev=stdev_i)
                xg = xg.squeeze(0).detach().cpu().numpy()
                freq, hvsr = _compute_hvsr(xg)
                if freq is None:
                    continue
                f0 = _estimate_f0(freq, hvsr)
                if f0 is not None:
                    gen_f0[st].append(f0)
                    gen_signals[st].append(xg)
                    gen_hvsr_curves[st].append((freq, hvsr))
                    gen_names[st].append(f"gen_{st}_{len(gen_f0[st]):03d}")

        # Debug: print f0 counts per station
        print(f"[eval] f0 counts per station:")
        for st in id_to_station:
            print(f"  {st}: Real={len(real_f0[st])}, Gen={len(gen_f0[st])}")
            if len(real_f0[st]) > 0:
                print(f"    Real f0 range: [{np.min(real_f0[st]):.2f}, {np.max(real_f0[st]):.2f}]")
            if len(gen_f0[st]) > 0:
                print(f"    Gen f0 range: [{np.min(gen_f0[st]):.2f}, {np.max(gen_f0[st]):.2f}]")

        # Station-wise HVSR/f0 distributions: save plots and NPZ bundles (Phase 1)
        print(f"[eval] Generating station-wise HVSR distribution plots and NPZ bundles...")
        hvsr_dir = os.path.join(out_dir, 'station_hvsr_distributions')
        os.makedirs(hvsr_dir, exist_ok=True)

        def _safe_hist(vals, bins):
            arr = np.asarray(vals, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return np.zeros(len(bins) - 1, dtype=float), bins
            hist, edges = np.histogram(arr, bins=bins)
            pdf = hist.astype(float)
            s = float(pdf.sum())
            if s > 0:
                pdf = pdf / s
            return pdf, edges

        def _median_curve(curves):
            if not curves:
                return None, None
            target_f = np.linspace(1.0, 20.0, 400)
            mats = []
            for f, h in curves:
                if f is None or h is None:
                    continue
                try:
                    h_i = np.interp(target_f, f, h, left=np.nan, right=np.nan)
                    mats.append(h_i)
                except Exception:
                    continue
            if not mats:
                return None, None
            M = np.vstack(mats)
            M = M[np.all(np.isfinite(M), axis=1)]
            if M.size == 0:
                return None, None
            return target_f, np.median(M, axis=0)

        f_bins = np.linspace(1.0, 20.0, 40)
        for st in id_to_station:
            vals_r = real_f0.get(st, [])
            vals_g = gen_f0.get(st, [])
            names_r = real_names.get(st, [])
            names_g = gen_names.get(st, [])

            pdf_r, edges = _safe_hist(vals_r, f_bins)
            pdf_g, _ = _safe_hist(vals_g, f_bins)
            centers = 0.5 * (edges[:-1] + edges[1:])

            fr, hr = _median_curve(real_hvsr_curves.get(st, []))
            fg, hg = _median_curve(gen_hvsr_curves.get(st, []))

            # Plot
            plt.figure(figsize=(12, 8))
            ax1 = plt.gca()
            w = (edges[1] - edges[0])
            if np.any(pdf_r):
                ax1.bar(centers - 0.22 * w, pdf_r, width=0.22 * w, alpha=0.6, color='steelblue', label='Real f0 PDF')
            if np.any(pdf_g):
                ax1.bar(centers, pdf_g, width=0.22 * w, alpha=0.6, color='seagreen', label='TimesNet-Gen f0 PDF')
            ax1.set_xlabel('f0 (Hz)'); ax1.set_ylabel('Probability'); ax1.set_xlim(1, 20)
            ax1.grid(True, alpha=0.3)

            if fr is not None and hr is not None:
                ax2 = ax1.twinx()
                ax2.plot(fr, hr, color='navy', linewidth=2, label='Real HVSR (median)')
                if fg is not None and hg is not None:
                    ax2.plot(fg, hg, color='darkgreen', linewidth=2, linestyle='--', label='TimesNet-Gen HVSR (median)')
                ax2.set_ylabel('HVSR'); ax2.set_xlim(1, 20)
                h1, l1 = ax1.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                ax2.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=10)

            plt.title(f'Station {st}: f0 PDFs and HVSR curves (Real vs TimesNet-Gen)')
            plt.tight_layout()
            plt.savefig(os.path.join(hvsr_dir, f'hvsr_f0_station_{st}.png'), dpi=220, bbox_inches='tight')
            plt.close()

            # NPZ bundle
            out_npz = os.path.join(hvsr_dir, f'hvsr_f0_station_{st}.npz')
            np.savez_compressed(
                out_npz,
                station=st,
                f0_real=np.asarray(vals_r, dtype=np.float32),
                f0_timesnet=np.asarray(vals_g, dtype=np.float32),
                f0_bins=edges.astype(np.float32),
                pdf_real=pdf_r.astype(np.float32),
                pdf_timesnet=pdf_g.astype(np.float32),
                hvsr_freq_real=(fr.astype(np.float32) if fr is not None else np.array([], dtype=np.float32)),
                hvsr_median_real=(hr.astype(np.float32) if hr is not None else np.array([], dtype=np.float32)),
                hvsr_freq_timesnet=(fg.astype(np.float32) if fg is not None else np.array([], dtype=np.float32)),
                hvsr_median_timesnet=(hg.astype(np.float32) if hg is not None else np.array([], dtype=np.float32)),
                real_names=np.array(names_r, dtype=object),
                gen_names=np.array(names_g, dtype=object),
            )
        print(f"[eval] Saved per-station plots and NPZ bundles to {hvsr_dir}")

        # === NEW: per-station NPZ with generated time-series (20 samples) ===
        try:
            gen_ts_dir = os.path.join(out_dir, 'generated_timeseries_npz')
            os.makedirs(gen_ts_dir, exist_ok=True)
            max_gen_per_station_npz = 50
            for st in id_to_station:
                gen_list = gen_signals.get(st, [])
                if not gen_list:
                    continue
                n_keep = min(max_gen_per_station_npz, len(gen_list))
                sigs_gen = np.asarray(gen_list[:n_keep], dtype=np.float32)
                np.savez_compressed(
                    os.path.join(gen_ts_dir, f'station_{st}_generated_timeseries.npz'),
                    station=st,
                    signals_generated=sigs_gen,
                )
            print(f"[eval] Saved generated time-series NPZ bundles to {gen_ts_dir}")
        except Exception as e:
            print(f"[eval][warn] saving generated timeseries NPZ bundles failed: {e}")

        # === NEW: per-station time-series figures (Real×Gen combinations) ===
        # For each station: use first two real and first two generated samples (if available)
        # and plot combinations: (Real1,Gen1), (Real1,Gen2), (Real2,Gen1).
        try:
            comb_ts_dir = os.path.join(out_dir, 'generated_timeseries_pairs')
            os.makedirs(comb_ts_dir, exist_ok=True)
            fs_plot = 100.0
            for st in id_to_station:
                r_list = real_signals.get(st, [])
                g_list = gen_signals.get(st, [])
                if len(r_list) < 2 or len(g_list) < 2:
                    continue
                # indices: real0, real1, gen0, gen1
                combos = [(0, 0), (0, 1), (1, 0)]
                for r_idx, g_idx in combos:
                    try:
                        sig_real = np.asarray(r_list[r_idx], dtype=np.float32)
                        sig_gen = np.asarray(g_list[g_idx], dtype=np.float32)
                        if sig_real.ndim != 2 or sig_real.shape[1] != 3:
                            continue
                        T = sig_real.shape[0]
                        t = np.arange(T, dtype=np.float32) / fs_plot
                        fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
                        labels = ['E', 'N', 'U']
                        for ch in range(3):
                            axes[ch].plot(t, sig_real[:, ch], label='Real', color='blue', linewidth=0.8, alpha=0.9)
                            axes[ch].plot(t, sig_gen[:, ch], label='Generated', color='red', linewidth=0.8, alpha=0.9)
                            axes[ch].set_ylabel(labels[ch])
                            axes[ch].grid(True, alpha=0.3)
                            if ch == 0:
                                axes[ch].legend(loc='upper right', fontsize=8)
                        axes[-1].set_xlabel('Time (s)')
                        fig.suptitle(
                            f'Station {st} – Real{r_idx+1} vs Gen{g_idx+1}',
                            fontsize=12
                        )
                        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                        out_png = os.path.join(
                            comb_ts_dir,
                            f'timeseries_station_{st}_R{r_idx+1}_G{g_idx+1}.png'
                        )
                        plt.savefig(out_png, dpi=200, bbox_inches='tight')
                        plt.close(fig)
                    except Exception as e:
                        print(f"[eval][warn] plotting Real{r_idx+1}×Gen{g_idx+1} failed for station {st}: {e}")
            print(f"[eval] Saved Real×Generated time-series pair plots to {comb_ts_dir}")
        except Exception as e:
            print(f"[eval][warn] could not create Real×Gen time-series plots: {e}")

        # Individual sample plots: Real vs Generated (signal + HVSR)
        print(f"[eval] Generating individual sample plots...")
        sample_dir = os.path.join(out_dir, 'individual_samples')
        os.makedirs(sample_dir, exist_ok=True)
        for st in id_to_station:
            n_real = min(10, len(real_signals[st]))
            n_gen = min(2, len(gen_signals[st]))
            for i in range(n_real):
                sig_real = real_signals[st][i]
                freq_real, hvsr_real = real_hvsr_curves[st][i]
                f0_real = _estimate_f0(freq_real, hvsr_real)
                
                # Match with a generated sample (same index if available)
                if i < n_gen:
                    sig_gen = gen_signals[st][i]
                    freq_gen, hvsr_gen = gen_hvsr_curves[st][i]
                    f0_gen = _estimate_f0(freq_gen, hvsr_gen)
                else:
                    sig_gen = None

                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                # Top: Signals
                t = np.arange(sig_real.shape[0]) / 100.0
                for ch, label in enumerate(['E', 'N', 'U']):
                    axes[0, 0].plot(t, sig_real[:, ch], label=f'{label} (Real)', alpha=0.7)
                axes[0, 0].set_title(f'Real Signal - Station {st} (Sample {i+1})')
                axes[0, 0].set_xlabel('Time (s)'); axes[0, 0].set_ylabel('Amplitude')
                axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

                if sig_gen is not None:
                    t_gen = np.arange(sig_gen.shape[0]) / 100.0
                    for ch, label in enumerate(['E', 'N', 'U']):
                        axes[0, 1].plot(t_gen, sig_gen[:, ch], label=f'{label} (Gen)', alpha=0.7)
                    axes[0, 1].set_title(f'Generated Signal - Station {st} (Sample {i+1})')
                    axes[0, 1].set_xlabel('Time (s)'); axes[0, 1].set_ylabel('Amplitude')
                    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)
                else:
                    axes[0, 1].text(0.5, 0.5, 'No Generated Match', ha='center', va='center', transform=axes[0, 1].transAxes)
                    axes[0, 1].set_title('Generated Signal')

                # Bottom: HVSR
                axes[1, 0].plot(freq_real, hvsr_real, color='steelblue', linewidth=2, label='Real HVSR')
                if f0_real is not None:
                    axes[1, 0].axvline(f0_real, color='red', linestyle='--', linewidth=1.5, label=f'f0={f0_real:.2f} Hz')
                axes[1, 0].set_xlim(1, 20); axes[1, 0].set_xlabel('Frequency (Hz)')
                axes[1, 0].set_ylabel('HVSR'); axes[1, 0].set_title('Real HVSR')
                axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

                if sig_gen is not None:
                    axes[1, 1].plot(freq_gen, hvsr_gen, color='seagreen', linewidth=2, label='Generated HVSR')
                    if f0_gen is not None:
                        axes[1, 1].axvline(f0_gen, color='red', linestyle='--', linewidth=1.5, label=f'f0={f0_gen:.2f} Hz')
                    axes[1, 1].set_xlim(1, 20); axes[1, 1].set_xlabel('Frequency (Hz)')
                    axes[1, 1].set_ylabel('HVSR'); axes[1, 1].set_title('Generated HVSR')
                    axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)
                else:
                    axes[1, 1].text(0.5, 0.5, 'No Generated Match', ha='center', va='center', transform=axes[1, 1].transAxes)
                    axes[1, 1].set_title('Generated HVSR')

                plt.tight_layout()
                plt.savefig(os.path.join(sample_dir, f'sample_{st}_{i+1:02d}_real_vs_gen.png'), dpi=150, bbox_inches='tight')
                plt.close()

    # Phase 0 results under base_dir/phase0; Phase 1 under base_dir/phase1
    out_dir_base = './gen_eval_results_base'
    os.makedirs(out_dir_base, exist_ok=True)

    # Phase 0: only reconstruction metrics and visuals; skip generation
    out_dir_p0 = os.path.join(out_dir_base, 'phase0')
    os.makedirs(out_dir_p0, exist_ok=True)
    evaluate_extended_base(model, test_ds, out_dir_p0, samples_per_station=0, target_stations=test_stations)
    print(f"[P0] evaluation saved under {out_dir_p0}")

    # Phase 1: fine-tune with noise injection and then evaluate (recon + generation)
    out_dir_p1 = os.path.join(out_dir_base, 'phase1')
    os.makedirs(out_dir_p1, exist_ok=True)
    print("[TimesNet-PointCloud] Phase 1: fine-tuning with noise injection...")
    train_losses_p1 = []
    optim_p1 = torch.optim.AdamW(model.parameters(), lr=getattr(args, 'phase1_lr', 1e-4), weight_decay=args.weight_decay)
    scheduler_p1 = torch.optim.lr_scheduler.CosineAnnealingLR(optim_p1, T_max=getattr(args, 'phase1_epochs', 10))
    args.noise_injection_enable = True
    for epoch in range(getattr(args, 'phase1_epochs', 10)):
        model.train()
        total = 0.0
        total_rec = 0.0
        for x, sid, _ in train_loader:
            if args.use_gpu:
                x = x.cuda(); sid = sid.cuda()
            optim_p1.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(args.use_gpu and args.use_amp)):
                enc_out, means_b, stdev_b = model.encode_features_for_reconstruction(x)
                if os.path.exists(args.latent_stats_path):
                    std_vec = torch.from_numpy(np.load(args.latent_stats_path)).to(enc_out.device).float()
                    noise = torch.randn_like(enc_out) * std_vec.view(1, 1, -1) * float(getattr(args, 'noise_std_scale', 1.0))
                    enc_out = enc_out + noise
                x_hat = model.project_features_for_reconstruction(enc_out, means_b, stdev_b)
                loss_rec = reconstruction_loss(x_hat, x, loss_type=args.recon_loss_type)
                loss = loss_rec
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optim_p1)
            scaler.update()
            total += float(loss.detach().cpu())
            total_rec += float(loss_rec.detach().cpu())
        scheduler_p1.step()
        avg_loss = total / len(train_loader)
        avg_rec = total_rec / len(train_loader)
        train_losses_p1.append(avg_loss)
        print(f"[P1] epoch {epoch+1}/{getattr(args, 'phase1_epochs', 10)} loss={avg_loss:.4f} (rec={avg_rec:.4f})")

    # Save Phase 1 loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses_p1) + 1), train_losses_p1, marker='o', linewidth=2, markersize=6, color='seagreen')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.title('Phase 1 Training Loss (noise injection)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir_p1, 'training_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[P1] training loss plot saved to {out_dir_p1}/training_loss.png")

    # Phase 1 diagnostics and evaluation (reconstruction + generation)
    analyze_posterior_gaussianity(model, val_loader, os.path.join(out_dir_p1, 'posterior_diagnostics'), max_samples=2000)
    plot_reconstruction_samples(model, test_ds, test_stations, os.path.join(out_dir_p1, 'recon_sanity_check'), max_per_station=10)
    
    # Get number of samples to generate from args (default 50)
    num_samples_to_generate = getattr(args, 'num_generated_samples', 50)
    evaluate_extended_base(model, test_ds, out_dir_p1, samples_per_station=num_samples_to_generate, target_stations=test_stations)
    print(f"[P1] evaluation saved under {out_dir_p1}")
    
    # Save Phase 1 checkpoint (final model for inference)
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_p1_path = os.path.join(checkpoint_dir, 'timesnet_pointcloud_phase1_final.pth')
    torch.save({
        'epoch': args.train_epochs + getattr(args, 'phase1_epochs', 10),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim_p1.state_dict(),
        'train_losses_phase0': train_losses,
        'train_losses_phase1': train_losses_p1,
        'args': vars(args),
        'phase': 'phase1'
    }, checkpoint_p1_path)
    print(f"\n{'='*80}")
    print(f"Training & Generation Complete!")
    print(f"{'='*80}")
    print(f"✓ Model checkpoint saved: {checkpoint_p1_path}")
    print(f"✓ Generated {num_samples_to_generate} samples per station ({len(test_stations)} stations)")
    print(f"✓ Results saved to: {out_dir_p1}")
    print(f"\nGenerated NPZ files:")
    npz_dir = os.path.join(out_dir_p1, 'generated_timeseries_npz')
    if os.path.exists(npz_dir):
        for st in test_stations:
            npz_file = os.path.join(npz_dir, f'station_{st}_generated_timeseries.npz')
            if os.path.exists(npz_file):
                print(f"  - {npz_file}")
    print(f"\nTo generate more samples later, use:")
    print(f"  python generate_samples.py --checkpoint {checkpoint_p1_path} --num_samples 50")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()


