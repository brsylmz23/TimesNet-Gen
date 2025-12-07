import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import re
import scipy.io as sio
try:
    from scipy.stats import gaussian_kde  # for KDE overlays
except Exception:
    gaussian_kde = None


# Jensen-Shannon similarity helper (module-level)
def _jensen_shannon_similarity(pdf_a, pdf_b):
    """
    Compute JS similarity (1 - JS divergence) between two histogram PDFs.
    Returns similarity in [0, 1] where 1 = identical, 0 = completely different.
    """
    try:
        from scipy.spatial.distance import jensenshannon
        a = np.asarray(pdf_a, dtype=float).ravel()
        b = np.asarray(pdf_b, dtype=float).ravel()
        if a.size != b.size or a.size == 0:
            return 0.0
        # Ensure probability distributions
        a = a / a.sum() if a.sum() > 0 else a
        b = b / b.sum() if b.sum() > 0 else b
        eps = 1e-10
        a = a + eps; b = b + eps
        a = a / a.sum(); b = b / b.sum()
        js_div = jensenshannon(a, b, base=2)
        return float(1.0 - js_div)
    except Exception:
        return 0.0


# Normalized cross-correlation between two matrices (module-level)
def _normalized_corr2(A: np.ndarray, B: np.ndarray) -> float:
    try:
        a = np.asarray(A, dtype=float).ravel()
        b = np.asarray(B, dtype=float).ravel()
        if a.size == 0 or b.size == 0 or a.size != b.size:
            return 0.0
        if np.allclose(np.std(a), 0) or np.allclose(np.std(b), 0):
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])
    except Exception:
        return 0.0


def _safe_get(d: Dict[str, np.ndarray], key: str, default: np.ndarray) -> np.ndarray:
    try:
        return d[key]
    except Exception:
        return default


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    try:
        if a.shape != b.shape:
            n = min(a.size, b.size)
            a = a[:n]
            b = b[:n]
        if np.all(a == a[0]) or np.all(b == b[0]):
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])
    except Exception:
        return 0.0


def _get_first(d: Dict[str, np.ndarray], keys: List[str]) -> Optional[np.ndarray]:
    for k in keys:
        try:
            v = d.get(k)
        except Exception:
            v = None
        if v is not None:
            return v
    return None


def _find_first_key(d: Dict[str, np.ndarray], prefixes: List[str]) -> Optional[str]:
    try:
        keys = list(d.keys())
    except Exception:
        return None
    for p in prefixes:
        for k in keys:
            if k.startswith(p):
                return k
    return None


def _canonicalize_station(st_raw: str) -> str:
    try:
        m = re.findall(r'(\d{4})', str(st_raw))
        return m[-1] if m else str(st_raw)
    except Exception:
        return str(st_raw)


def _load_npz_map(dir_path: str, pattern: str = 'hvsr_f0_station_*.npz') -> Dict[str, Dict[str, np.ndarray]]:
    station_to_data: Dict[str, Dict[str, np.ndarray]] = {}
    if not dir_path or not os.path.isdir(dir_path):
        return station_to_data

    # Search recursively to catch nested outputs
    patterns = [
        pattern,
        'station_*.npz',
        '*.npz',
        os.path.join('**', pattern),
        os.path.join('**', 'station_*.npz'),
        os.path.join('**', '*.npz'),
    ]
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(dir_path, pat), recursive=True))
    # Deduplicate while preserving order
    seen = set()
    files_unique = []
    for f in files:
        if f not in seen:
            files_unique.append(f)
            seen.add(f)

    print(f"[auto] Scanning {dir_path} -> {len(files_unique)} npz files")
    for fp in files_unique:
        try:
            data = np.load(fp, allow_pickle=True)
            basename = os.path.basename(fp)
            st = None
            # Prefer station field inside NPZ if present
            if 'station' in data:
                st_val = data['station']
                try:
                    if isinstance(st_val, (bytes, bytearray)):
                        st = _canonicalize_station(st_val.decode('utf-8', errors='ignore'))
                    elif isinstance(st_val, np.ndarray):
                        # handle array of strings or bytes
                        elem = st_val.item() if st_val.shape == () else st_val[0]
                        if isinstance(elem, (bytes, bytearray)):
                            st = _canonicalize_station(elem.decode('utf-8', errors='ignore'))
                        else:
                            st = _canonicalize_station(str(elem))
                    else:
                        st = _canonicalize_station(str(st_val))
                except Exception:
                    st = None
            if st is None:
                # Fallback: parse from filename
                st_raw = basename.replace('hvsr_f0_station_', '').replace('.npz', '')
                st = _canonicalize_station(st_raw)

            if st:
                station_to_data[st] = {k: data[k] for k in data.files}
        except Exception as e:
            print(f"[debug] Failed to load {fp}: {e}")
            continue

    print(f"[auto] Found stations in {dir_path}: {sorted(list(station_to_data.keys()))}")
    return station_to_data


def _load_generated_timeseries_npz(dir_path: str) -> Dict[str, np.ndarray]:
    """
    Load generated time-series NPZ bundles of the form
    'station_XXXX_generated_timeseries.npz' with keys:
      - station
      - signals_generated (or similar): (N_gen, T, 3)
    Returns: {station_code: signals_generated_array}
    """
    ts_map: Dict[str, np.ndarray] = {}
    if not dir_path or not os.path.isdir(dir_path):
        return ts_map
    pattern = os.path.join(dir_path, 'station_*_generated_timeseries.npz')
    files = sorted(glob.glob(pattern))
    print(f"[gen-ts] Scanning {dir_path} -> {len(files)} generated timeseries NPZ files")
    for fp in files:
        try:
            data = np.load(fp, allow_pickle=True)
            st = None
            if 'station' in data:
                try:
                    st_val = data['station']
                    if isinstance(st_val, (bytes, bytearray)):
                        st = _canonicalize_station(st_val.decode('utf-8', errors='ignore'))
                    elif isinstance(st_val, np.ndarray):
                        elem = st_val.item() if st_val.shape == () else st_val[0]
                        if isinstance(elem, (bytes, bytearray)):
                            st = _canonicalize_station(elem.decode('utf-8', errors='ignore'))
                        else:
                            st = _canonicalize_station(str(elem))
                    else:
                        st = _canonicalize_station(str(st_val))
                except Exception:
                    st = None
            if st is None:
                base = os.path.basename(fp)
                st_raw = base.replace('station_', '').replace('_generated_timeseries.npz', '')
                st = _canonicalize_station(st_raw)
            sig_key = None
            for k in data.files:
                if k.lower().startswith('signals'):
                    sig_key = k
                    break
            if st and sig_key:
                sigs = np.asarray(data[sig_key], dtype=float)
                ts_map[st] = sigs
        except Exception as e:
            print(f"[gen-ts][warn] failed to load {fp}: {e}")
    print(f"[gen-ts] Found stations in {dir_path}: {sorted(list(ts_map.keys()))}")
    return ts_map


def _ensure_bins(pdf: np.ndarray) -> np.ndarray:
    if pdf is None:
        return np.array([0.0, 1.0], dtype=float)
    if pdf.ndim == 0:
        return np.array([0.0, 1.0], dtype=float)
    return pdf


def _kde_from_samples(samples: np.ndarray, bins: np.ndarray) -> np.ndarray:
    # Simple histogram-based density as fallback if KDE not available
    if samples.size == 0:
        return np.zeros(bins.size - 1, dtype=float)
    hist, _ = np.histogram(samples, bins=bins, density=True)
    return hist


def _rebin_to_bins(pdf: np.ndarray, src_bins: Optional[np.ndarray], dst_bins: np.ndarray) -> np.ndarray:
    if pdf is None:
        return None
    # If no src_bins or same length-1 as dst, return as is
    if src_bins is None or pdf.size == dst_bins.size - 1:
        return pdf
    try:
        src_centers = 0.5 * (src_bins[:-1] + src_bins[1:])
        dst_centers = 0.5 * (dst_bins[:-1] + dst_bins[1:])
        rebinned = np.interp(dst_centers, src_centers, pdf, left=0.0, right=0.0)
        widths = np.diff(dst_bins)
        area = float(np.sum(rebinned * widths))
        if area > 0:
            rebinned = rebinned / area
        return rebinned
    except Exception:
        return pdf


def _preprocess_component_boore(data: np.ndarray, fs: float, corner_freq: float, filter_order: int = 2) -> np.ndarray:
    # Linear detrend + zero padding + simple high-pass via FFT masking fallback
    x = np.asarray(data, dtype=np.float64)
    n = x.shape[0]
    t = np.arange(n, dtype=np.float64)
    t_mean = t.mean(); x_mean = x.mean()
    denom = np.sum((t - t_mean) ** 2)
    slope = 0.0 if denom == 0 else float(np.sum((t - t_mean) * (x - x_mean)) / denom)
    intercept = float(x_mean - slope * t_mean)
    x_detr = x - (slope * t + intercept)
    # zero pad
    pad_samples = int(round((1.5 * filter_order) / max(corner_freq, 1e-6) * fs))
    x_pad = np.concatenate([np.zeros(pad_samples, dtype=np.float64), x_detr, np.zeros(pad_samples, dtype=np.float64)])
    return x_pad


def _konno_ohmachi_smoothing(spectrum: np.ndarray, freq: np.ndarray, b: float = 40.0) -> np.ndarray:
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


def _compute_hvsr(signal: np.ndarray, fs: float = 100.0):
    try:
        if signal.ndim != 2 or signal.shape[1] != 3:
            return None, None
        if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
            return None, None
        ew = _preprocess_component_boore(signal[:, 0], fs, 0.05, 2)
        ns = _preprocess_component_boore(signal[:, 1], fs, 0.05, 2)
        ud = _preprocess_component_boore(signal[:, 2], fs, 0.05, 2)
        n = int(min(len(ew), len(ns), len(ud)))
        if n < 16:
            return None, None
        ew = ew[:n]; ns = ns[:n]; ud = ud[:n]
        half = n // 2
        if half <= 1:
            return None, None
        freq = (np.arange(0, half, dtype=np.float64)) * (fs / n)
        amp_ew = np.abs(np.fft.fft(ew))[:half]
        amp_ns = np.abs(np.fft.fft(ns))[:half]
        amp_ud = np.abs(np.fft.fft(ud))[:half]
        combined_h = np.sqrt(np.maximum(amp_ew, 0.0) * np.maximum(amp_ns, 0.0))
        sm_h = _konno_ohmachi_smoothing(combined_h, freq, 40.0)
        sm_v = _konno_ohmachi_smoothing(amp_ud, freq, 40.0)
        sm_v_safe = np.where(sm_v <= 0.0, 1e-12, sm_v)
        sm_hvsr = sm_h / sm_v_safe
        mask = (freq >= 1.0) & (freq <= 20.0)
        if not np.any(mask):
            return None, None
        return freq[mask], sm_hvsr[mask]
    except Exception:
        return None, None

def plot_station(
    station: str,
    gen: Optional[Dict[str, np.ndarray]],
    recon: Optional[Dict[str, np.ndarray]],
    vae: Optional[Dict[str, np.ndarray]],
    out_dir: str,
    vae_base_dir: Optional[str] = None,
    vae_gen_base_dir: Optional[str] = None,
    recon_dir: Optional[str] = None,
):
    os.makedirs(out_dir, exist_ok=True)

    # Resolve f0 bins (prefer gen, then recon, then vae)
    f0_bins = None
    for src in (gen, recon, vae):
        if src is not None and 'f0_bins' in src:
            f0_bins = src['f0_bins']
            break
    if f0_bins is None:
        f0_bins = np.linspace(1.0, 20.0, 40)

    bin_centers = 0.5 * (f0_bins[:-1] + f0_bins[1:])
    width = (f0_bins[1] - f0_bins[0]) * 0.27

    # PDFs (keys exactly per our exporters)
    # - Gen (untitled1_gen.py, phase1): pdf_real, pdf_timesnet, f0_real, f0_timesnet, hvsr_freq_real, hvsr_median_real, hvsr_freq_timesnet, hvsr_median_timesnet
    # - Recon (transfer_learning_station_cond.py): pdf_real, pdf_timesnet, pdf_vae, f0_real, f0_timesnet, f0_vae, hvsr_freq_real, hvsr_median_real, hvsr_freq_timesnet, hvsr_median_timesnet
    pdf_real = None
    pdf_gen_timesnet = None  # TimesNet-Gen (from gen_dir NPZs)

    # Prefer provided PDFs; if missing, derive from samples if available
    recon_bins = None
    gen_bins = None

    if recon is not None:
        # Get real data from recon NPZ
        pdf_real = recon.get('pdf_real')
        if pdf_real is None and 'f0_real' in recon:
            pdf_real = _kde_from_samples(np.asarray(recon['f0_real']).ravel(), f0_bins)
        recon_bins = recon.get('f0_bins') if 'f0_bins' in recon else None

    if gen is not None:
        # Exact keys per untitled1_gen NPZ spec
        if pdf_real is None:
            pdf_real = gen.get('pdf_real')
        pdf_gen_timesnet = gen.get('pdf_timesnet') if pdf_gen_timesnet is None else pdf_gen_timesnet
        if pdf_real is None and 'f0_real' in gen:
            pdf_real = _kde_from_samples(np.asarray(gen['f0_real']).ravel(), f0_bins)
        if pdf_gen_timesnet is None and 'f0_timesnet' in gen:
            vals = np.asarray(gen['f0_timesnet']).ravel()
            print(f"[debug:{station}] gen f0 key='f0_timesnet', count={vals.size}, min={vals.min() if vals.size>0 else 'nan'}, max={vals.max() if vals.size>0 else 'nan'}")
            pdf_gen_timesnet = _kde_from_samples(vals, f0_bins)
        gen_bins = gen.get('f0_bins') if 'f0_bins' in gen else None

    # Do NOT rebin; plot as read. Ensure arrays are at least arrays; if None, use zeros with own bins later.
    pdf_real = np.asarray(pdf_real) if pdf_real is not None else None
    pdf_gen_timesnet = np.asarray(pdf_gen_timesnet) if pdf_gen_timesnet is not None else None

    # Final shape sanity (using each own bins)
    try:
        lr = (pdf_real.size if pdf_real is not None else 0)
        lg = (pdf_gen_timesnet.size if pdf_gen_timesnet is not None else 0)
        print(f"[debug:{station}] lens real={lr}, gen={lg}")
    except Exception:
        pass

    # Correlations for diagnostics
    # Compute correlations only when lengths match
    def _corr_if_match(a, b):
        if a is None or b is None:
            return float('nan')
        if a.size != b.size:
            return float('nan')
        return _safe_corr(a, b)
    corr_gr = _corr_if_match(pdf_gen_timesnet, pdf_real)
    print(f"[debug:{station}] corr Gen-Real={corr_gr:.3f}")

    # Color palette (consistent across top/bottom)
    color_real = 'steelblue'      # Real
    color_gen = 'indianred'       # TimesNet-Gen

    # HVSR median curves
    hfreq = None
    h_med_real = None
    h_med_gen = None

    # Prefer recon for real frequency
    for src in (recon, gen):
        if src is not None and 'hvsr_freq_real' in src:
            hfreq = src['hvsr_freq_real']
            break
    if hfreq is None:
        hfreq = np.linspace(1.0, 20.0, 256)

    if recon is not None:
        if h_med_real is None:
            h_med_real = recon.get('hvsr_median_real', None)
    if gen is not None:
        if h_med_real is None:
            h_med_real = gen.get('hvsr_median_real')
        if h_med_gen is None:
            h_med_gen = gen.get('hvsr_median_timesnet')

    # Align optional curves to real frequency grid if they have their own freq arrays
    def _align_to(freq_src: Optional[np.ndarray], values: Optional[np.ndarray], freq_dst: np.ndarray) -> Optional[np.ndarray]:
        try:
            if values is None:
                return None
            if freq_src is None or values.size == 0:
                return None
            return np.interp(freq_dst, freq_src, values, left=np.nan, right=np.nan)
        except Exception:
            return None

    # Attempt to fetch source freq arrays for each median curve
    f_real_src = None
    f_gen_src = None
    if recon is not None and 'hvsr_freq_real' in recon:
        f_real_src = recon.get('hvsr_freq_real')
    elif gen is not None and 'hvsr_freq_real' in gen:
        f_real_src = gen.get('hvsr_freq_real')
    if gen is not None and 'hvsr_freq_timesnet' in gen:
        f_gen_src = gen.get('hvsr_freq_timesnet')

    # Build unified hfreq if needed
    if hfreq is None:
        if f_real_src is not None:
            hfreq = f_real_src
        else:
            hfreq = np.linspace(1.0, 20.0, 256)

    # Align medians to hfreq
    if h_med_real is not None and f_real_src is not None:
        h_med_real = _align_to(f_real_src, h_med_real, hfreq)
    if h_med_gen is not None and f_gen_src is not None:
        h_med_gen = _align_to(f_gen_src, h_med_gen, hfreq)

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Top: f0 bar + KDE-like density (we use histogram density)
    # Draw bars (rectangles) at each source's bin centers; no rebinning
    drawn_labels = set()
    def _draw_pdf_bars(ax, bins, pdf, color, label):
        try:
            if bins is None or pdf is None:
                return
            bins = np.asarray(bins)
            pdf = np.asarray(pdf)
            if pdf.size != bins.size - 1:
                return
            centers = 0.5 * (bins[:-1] + bins[1:])
            # Use median bin width scaled to avoid over-coverage
            bin_w = float(np.median(np.diff(bins))) if bins.size > 1 else 0.2
            width = bin_w * 0.75
            lbl = label if label not in drawn_labels else None
            ax.bar(centers, pdf, width=width, color=color, edgecolor=color, linewidth=0.8, alpha=0.30, label=lbl, zorder=2)
            if lbl is not None:
                drawn_labels.add(label)
        except Exception:
            return

    # Collect raw f0 samples from NPZs
    samples_real = None
    if recon is not None and 'f0_real' in recon:
        samples_real = recon.get('f0_real')
    elif gen is not None and 'f0_real' in gen:
        samples_real = gen.get('f0_real')
    samples_gen = gen.get('f0_timesnet') if (gen is not None and 'f0_timesnet' in gen) else None

    # Labels for histogram bars
    label_pdf_real = 'Real f0'
    label_pdf_gen = 'TimesNet-Gen f0'

    _draw_pdf_bars(ax1, recon_bins if pdf_real is not None else None, pdf_real, color_real, label_pdf_real)
    _draw_pdf_bars(ax1, gen_bins, pdf_gen_timesnet, color_gen, label_pdf_gen)

    # KDE overlays from raw samples (no rebinning)
    def _plot_kde(ax, samples, color, label):
        try:
            if gaussian_kde is None or samples is None:
                return
            s = np.asarray(samples, dtype=float).ravel()
            s = s[np.isfinite(s)]
            if s.size < 2:
                return
            x_kde = np.linspace(1.0, 20.0, 400)
            kde = gaussian_kde(s)
            y = kde(x_kde)
            lbl = label if label not in drawn_labels else None
            ax.plot(x_kde, y, color=color, linewidth=2.0, alpha=0.9, label=lbl, zorder=3)
            if lbl is not None:
                drawn_labels.add(label)
        except Exception:
            return

    # Density curves (smoothed, without sample counts)
    _plot_kde(ax1, samples_real, color_real, 'Real f0 Density')
    _plot_kde(ax1, samples_gen, color_gen, 'TimesNet-Gen f0 Density')

    ax1.set_xlabel('f0 (Hz)')
    ax1.set_ylabel('Probability Density')
    ax1.set_title(f'f0 Distribution Comparison (corr Gen-Real={corr_gr:.2f}) - Station {station}', fontsize=16, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=13, framealpha=0.95)
    ax1.set_xlabel('f0 (Hz)', fontsize=15, fontweight='bold')
    ax1.set_ylabel('Probability Density', fontsize=15, fontweight='bold')
    ax1.tick_params(labelsize=13)
    # Do not force common x-limits; keep default to avoid hiding differing ranges

    # Bottom: Median HVSR curves - Only Real vs Gen
    # Plot as read: use source freq arrays if available; avoid interpolation
    if h_med_real is not None:
        # Use hfreq if h_med_real was aligned, otherwise use original frequency array
        if recon is not None and 'hvsr_freq_real' in recon:
            freq_real = hfreq if (f_real_src is not None and len(h_med_real) == len(hfreq)) else recon.get('hvsr_freq_real', hfreq)
        elif gen is not None and 'hvsr_freq_real' in gen:
            freq_real = hfreq if (f_real_src is not None and len(h_med_real) == len(hfreq)) else gen.get('hvsr_freq_real', hfreq)
        else:
            freq_real = hfreq  # Fallback to hfreq if no real frequency array found
        if freq_real is not None and len(freq_real) == len(h_med_real):
            ax2.plot(freq_real, h_med_real, color=color_real, linewidth=2, label='Real HVSR')
    if gen is not None and h_med_gen is not None:
        # Use hfreq if h_med_gen was aligned, otherwise use original frequency array
        freq_gen = hfreq if (f_gen_src is not None and len(h_med_gen) == len(hfreq)) else gen.get('hvsr_freq_timesnet', hfreq)
        if freq_gen is not None and len(freq_gen) == len(h_med_gen):
            ax2.plot(freq_gen, h_med_gen, color=color_gen, linestyle='--', linewidth=2, label='TimesNet-Gen HVSR')
    ax2.set_xlabel('Frequency (Hz)', fontsize=15, fontweight='bold')
    ax2.set_ylabel('HVSR', fontsize=15, fontweight='bold')
    ax2.set_title(f'Median HVSR Curves - Station {station}', fontsize=16, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=13, framealpha=0.95)
    ax2.tick_params(labelsize=13)
    ax2.set_xlim(1.0, 20.0)
    ax2.set_yscale('log')

    plt.tight_layout()
    out_path = os.path.join(out_dir, f'combined_hvsr_f0_station_{station}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    try:
        print(f"[plot] {station}: sums real={float(np.sum(pdf_real)):.3f}, gen={float(np.sum(pdf_gen_timesnet)):.3f}")
        print(f"[debug:{station}] keys gen={list(gen.keys()) if gen else None}\n    recon={list(recon.keys()) if recon else None}\n    vae={list(vae.keys()) if vae else None}")
    except Exception as e:
        print(f"[debug:{station}] print failed: {e}")

    # Additional single figure: Average HVSR curves (Real vs TimesNet-Recon vs VAE-Recon)
    try:
        fig3, ax = plt.subplots(1, 1, figsize=(12, 6))
        # Plot median HVSR curves
        if h_med_real is not None and hfreq is not None:
            ax.plot(hfreq, h_med_real, color=color_real, linewidth=2.5, label='Real HVSR', alpha=0.9)
        if h_med_recon is not None:
            if 'hvsr_freq_timesnet' in recon and recon.get('hvsr_freq_timesnet') is not None:
                freq_tmp = recon.get('hvsr_freq_timesnet')
                h_tmp = recon.get('hvsr_median_timesnet')
                if freq_tmp is not None and h_tmp is not None:
                    ax.plot(freq_tmp, h_tmp, color=color_recon, linewidth=2.5, linestyle='--', label='TimesNet-Recon HVSR', alpha=0.9)
        if h_med_vae is not None and hfreq is not None:
            # VAE median may already be aligned to hfreq or may need alignment
            ax.plot(hfreq, h_med_vae, color=color_vae, linewidth=2.5, linestyle='-.', label='VAE-Recon HVSR', alpha=0.9)
        ax.set_xlabel('Frequency (Hz)', fontsize=15, fontweight='bold')
        ax.set_ylabel('HVSR', fontsize=15, fontweight='bold')
        ax.set_title(f'Average HVSR (Real vs TimesNet-Recon vs VAE-Recon) - Station {station}', fontsize=16, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=13, framealpha=0.95)
        ax.tick_params(labelsize=13)
        ax.set_xlim(1.0, 20.0)
        ax.set_yscale('log')
        out_path3 = os.path.join(out_dir, f'average_hvsr_recon_sources_station_{station}.png')
        plt.tight_layout()
        plt.savefig(out_path3, dpi=300, bbox_inches='tight')
        plt.close(fig3)
    except Exception as e:
        print(f"[plot:{station}] average HVSR plot failed: {e}")

def _plot_real_generated_timeseries_pairs(
    gen_ts_map: Dict[str, np.ndarray],
    gen_map: Dict[str, Dict[str, np.ndarray]],
    out_dir: str,
    mat_root: Optional[str] = None,
    max_gen_per_station: int = 2,
) -> None:
    """
    For each station, compare **all available real signals** (loaded from
    original MAT files using `real_names` in gen NPZs) with up to
    `max_gen_per_station` generated signals.

    - Real: loaded via MAT using names in gen_map[st]['real_names'] -> (N_real, T, 3)
    - Gen:  gen_ts_map[st]                                          -> (N_gen,  T, 3)
    """
    if not gen_ts_map:
        print("[gen-ts] No generated timeseries map provided; skipping real-vs-gen plots.")
        return

    # Infer MAT root if not provided
    if not mat_root or not os.path.isdir(mat_root):
        candidates = [
            r"D:\Baris\new_Ps_Vs30",
            os.path.join(os.getcwd(), "new_Ps_Vs30"),
        ]
        for c in candidates:
            if os.path.isdir(c):
                mat_root = c
                break
    if not mat_root or not os.path.isdir(mat_root):
        print("[gen-ts][warn] No valid MAT root directory found; cannot load real signals.")
        return

    out_ts = os.path.join(out_dir, 'real_vs_generated_timeseries_pairs')
    os.makedirs(out_ts, exist_ok=True)
    fs = 100.0  # Hz

    def _compute_fas_single_channel(sig: np.ndarray, fs_local: float, channel: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Fourier amplitude spectrum (RFFT) for a single channel.
        Args:
            sig: (T, 3) signal array where columns are [E, N, U] (East-West, North-South, Up-Down)
            fs_local: sampling frequency
            channel: 0=E (East-West), 1=N (North-South), 2=U (Up-Down/Vertical)
        Returns freqs (F,) and amplitude (F,) for the specified channel."""
        try:
            T = sig.shape[0]
            freqs = np.fft.rfftfreq(T, d=1.0 / fs_local)
            amps = np.abs(np.fft.rfft(sig[:, channel]))  # Single channel FFT
            return freqs, amps
        except Exception:
            return np.array([]), np.array([])

    def _load_real_from_mat(mat_name: str) -> Optional[np.ndarray]:
        try:
            path = os.path.join(mat_root, mat_name)
            if not path.lower().endswith('.mat'):
                path = path + '.mat'
            if not os.path.exists(path):
                for sub in ['train', 'val', 'test']:
                    alt = os.path.join(mat_root, sub, os.path.basename(path))
                    if os.path.exists(alt):
                        path = alt
                        break
            if not os.path.exists(path):
                cand = glob.glob(os.path.join(mat_root, '**', os.path.basename(path)), recursive=True)
                if cand:
                    path = cand[0]
            if not os.path.exists(path):
                print(f"[gen-ts][warn] MAT file not found for {mat_name} under {mat_root}")
                return None
            mat = sio.loadmat(path)
            if 'signal' in mat:
                signal = mat['signal']
                if signal.ndim != 2:
                    return None
                if signal.shape[0] == 3:
                    signal = signal.T
            elif 'EQ' in mat:
                try:
                    dataset = mat['EQ'][0][0]['anEQ']
                    accel = dataset['Accel'][0][0]
                    if accel.ndim != 2 or accel.shape[1] != 3:
                        return None
                    signal = accel
                except Exception:
                    return None
            else:
                return None
            if signal.shape[1] != 3:
                return None
            return np.asarray(signal, dtype=float)
        except Exception as e:
            print(f"[gen-ts][warn] failed to load MAT {mat_name}: {e}")
            return None

    for st, gen_sigs in gen_ts_map.items():
        meta = gen_map.get(st, {})
        real_names = meta.get('real_names', None)
        if real_names is None:
            print(f"[gen-ts][warn] gen NPZ for station {st} has no 'real_names'; skipping.")
            continue

        # Flatten real_names into a Python string list
        try:
            real_name_list = [str(n) for n in np.asarray(real_names).ravel()]
        except Exception:
            real_name_list = []
        if not real_name_list:
            print(f"[gen-ts][warn] empty real_names for station {st}; skipping.")
            continue

        # Load as many real signals as we can from MAT (no hard limit; later compare all vs first 2 gen)
        real_sigs_named: List[Tuple[str, np.ndarray]] = []
        for nm in real_name_list:
            sig = _load_real_from_mat(nm)
            if sig is not None:
                sig_arr = np.asarray(sig, dtype=float)
                real_sigs_named.append((nm, sig_arr))
        if not real_sigs_named:
            print(f"[gen-ts][warn] could not load any real signals for station {st} from MAT; skipping.")
            continue

        # Debug: list lengths and names to see which are not fixed-length
        lengths = [sig.shape[0] for (_, sig) in real_sigs_named]
        unique_lengths = sorted(set(lengths))
        print(f"[gen-ts][debug:{st}] real signal lengths (unique): {unique_lengths}")
        if len(unique_lengths) > 1:
            # Show up to first 10 with their lengths
            print(f"[gen-ts][debug:{st}] sample (name, len) pairs:")
            for nm, sig in real_sigs_named[:10]:
                print(f"    {nm}: T={sig.shape[0]}")

        # Align all real signals to a common target length along time axis so that stacking works
        try:
            # Generated signals are produced with a fixed seq_len (6000) in untitled1_gen.py,
            # so we align real signals to T_target=6000 for fair comparison.
            T_target = 6000
            aligned_real = []
            for nm, sig in real_sigs_named:
                T = sig.shape[0]
                if T > T_target:
                    # center-crop to T_target (match MatSplitDataset behavior conceptually)
                    start = (T - T_target) // 2
                    sig = sig[start:start + T_target, :]
                elif T < T_target:
                    # pad zeros at the end
                    pad = np.zeros((T_target - T, sig.shape[1]), dtype=float)
                    sig = np.concatenate([sig, pad], axis=0)
                aligned_real.append(sig)
            real_sigs = np.stack(aligned_real, axis=0)
        except Exception as e:
            print(f"[gen-ts][warn] aligning real signals failed for station {st}: {e}")
            continue
        gen_sigs = np.asarray(gen_sigs, dtype=float)

        if real_sigs.ndim != 3 or real_sigs.shape[2] != 3:
            print(f"[gen-ts][warn] unexpected real shape for station {st}: {real_sigs.shape}; skipping.")
            continue
        if gen_sigs.ndim != 3 or gen_sigs.shape[2] != 3:
            print(f"[gen-ts][warn] unexpected generated shape for station {st}: {gen_sigs.shape}; skipping.")
            continue

        n_real_total = real_sigs.shape[0]
        n_gen_total = gen_sigs.shape[0]
        if n_real_total == 0 or n_gen_total == 0:
            continue

        # Limit to at most 50 real and 20 generated
        max_real = 50
        max_gen = 20
        n_real = min(max_real, n_real_total)
        n_gen = min(max_gen, n_gen_total)
        
        # Total plots: each real vs each generated -> n_real * n_gen
        total_plots = n_real * n_gen
        print(f"[gen-ts] Station {st}: {n_real} real × {n_gen} gen -> {total_plots} plots")

        for r_idx in range(n_real):
            for g_idx in range(n_gen):
                try:
                    sig_r = np.asarray(real_sigs[r_idx], dtype=float)
                    sig_g = np.asarray(gen_sigs[g_idx], dtype=float)
                    if sig_r.ndim != 2 or sig_r.shape[1] != 3 or sig_g.ndim != 2 or sig_g.shape[1] != 3:
                        continue

                    T = min(sig_r.shape[0], sig_g.shape[0])
                    sig_r = sig_r[:T]
                    sig_g = sig_g[:T]
                    t = np.arange(T, dtype=float) / fs

                    # 2x2 subplot layout:
                    #   Top-left: Real E-W time series
                    #   Top-right: Generated E-W time series
                    #   Bottom-left: Real E-W FAS
                    #   Bottom-right: Generated E-W FAS
                    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                    real_color = 'tab:blue'
                    gen_color = 'tab:orange'
                    ew_channel = 0  # E (East-West) channel - verified from np.stack([e, n, u], axis=1)

                    # Extract E-W channel only (no conversion, just label as cm/s²)
                    sig_r_ew = sig_r[:, ew_channel]
                    sig_g_ew = sig_g[:, ew_channel]

                    # Common y-limit for time series (E-W only)
                    y_max_ts = float(
                        max(
                            np.max(np.abs(sig_r_ew)) if sig_r_ew.size else 0.0,
                            np.max(np.abs(sig_g_ew)) if sig_g_ew.size else 0.0,
                        )
                    )
                    if y_max_ts <= 0:
                        y_max_ts = 1.0

                    # Top-left: Real E-W time series
                    axes[0, 0].plot(t, sig_r_ew, color=real_color, linewidth=1.5, alpha=0.8)
                    axes[0, 0].set_ylabel('Amplitude (cm/s²)', fontsize=18, fontweight='bold')
                    axes[0, 0].grid(True, alpha=0.3)
                    axes[0, 0].set_ylim(-1.05 * y_max_ts, 1.05 * y_max_ts)
                    axes[0, 0].tick_params(labelsize=14, width=1.5, length=6)

                    # Top-right: Generated E-W time series
                    axes[0, 1].plot(t, sig_g_ew, color=gen_color, linewidth=1.5, alpha=0.8)
                    axes[0, 1].set_ylabel('', fontsize=18, fontweight='bold')  # Empty label
                    axes[0, 1].grid(True, alpha=0.3)
                    axes[0, 1].set_ylim(-1.05 * y_max_ts, 1.05 * y_max_ts)
                    axes[0, 1].tick_params(labelsize=14, width=1.5, length=6)

                    # Compute FAS for E-W channel only (no conversion, just label as cm/s²)
                    f_r, fas_r = _compute_fas_single_channel(sig_r, fs, channel=ew_channel)
                    f_g, fas_g = _compute_fas_single_channel(sig_g, fs, channel=ew_channel)

                    # Bottom-left: Real E-W FAS
                    if f_r.size and fas_r.size:
                        axes[1, 0].plot(f_r, fas_r, color=real_color, linewidth=1.5, alpha=0.9)
                    axes[1, 0].set_ylabel('Amplitude (cm/s²)', fontsize=18, fontweight='bold')
                    axes[1, 0].set_xlabel('Frequency (Hz)', fontsize=18, fontweight='bold')
                    axes[1, 0].grid(True, alpha=0.3)
                    axes[1, 0].tick_params(labelsize=14, width=1.5, length=6)

                    # Bottom-right: Generated E-W FAS
                    if f_g.size and fas_g.size:
                        axes[1, 1].plot(f_g, fas_g, color=gen_color, linewidth=1.5, alpha=0.9)
                    axes[1, 1].set_ylabel('', fontsize=18, fontweight='bold')  # Empty label
                    axes[1, 1].set_xlabel('Frequency (Hz)', fontsize=18, fontweight='bold')
                    axes[1, 1].grid(True, alpha=0.3)
                    axes[1, 1].tick_params(labelsize=14, width=1.5, length=6)

                    # Match FAS axes limits if both available
                    if f_r.size and fas_r.size and f_g.size and fas_g.size:
                        f_min = 0.0
                        f_max = float(max(f_r.max(), f_g.max()))
                        a_max = float(max(fas_r.max(), fas_g.max()))
                        if a_max <= 0:
                            a_max = 1.0
                        axes[1, 0].set_xlim(f_min, f_max)
                        axes[1, 0].set_ylim(0.0, 1.05 * a_max)
                        axes[1, 1].set_xlim(f_min, f_max)
                        axes[1, 1].set_ylim(0.0, 1.05 * a_max)

                    # No title, just tight layout
                    plt.tight_layout()

                    out_png = os.path.join(out_ts, f'timeseries_{st}_R{r_idx+1:03d}_G{g_idx+1:02d}.png')
                    plt.savefig(out_png, dpi=220, bbox_inches='tight')
                    plt.close(fig)
                except Exception as e:
                    print(f"[gen-ts][warn] plotting Real{r_idx+1}×Gen{g_idx+1} failed for station {st}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Plot combined HVSR and f0 distributions from gen/recon/VAE NPZs.')
    parser.add_argument('--gen_dir', type=str, default='',
                        help='Directory with generation NPZs (e.g., gen_eval_results_base/phase1/station_hvsr_distributions)')
    parser.add_argument('--recon_dir', type=str, default='',
                        help='Directory with reconstruction NPZs (e.g., eval_results/station_freq_<timestamp>)')
    parser.add_argument('--vae_dir', type=str, default='',
                        help='Directory with VAE NPZs (optional)')
    parser.add_argument('--out', type=str, default='./combined_hvsr_plots_all',
                        help='Output directory for combined plots')
    parser.add_argument('--vae_gen_dir', type=str, default='',
                        help='Directory containing VAE-Gen ASC folders (e.g., D:/Baris/codes/Time-Series-Library-main/baris_gen)')
    parser.add_argument('--gen_ts_dir', type=str, default='',
                        help='Directory with per-station generated timeseries NPZs (station_XXXX_generated_timeseries.npz)')
    args = parser.parse_args()

    # Always show output
    import sys
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    
    try:

        # Auto-defaults so the script runs without CLI args
        if not args.gen_dir:
            gen_candidates = [
                r'D:\Baris\codes\Time-Series-Library-main\gen_eval_results_base\phase1\station_hvsr_distributions',
                os.path.join(os.getcwd(), 'gen_eval_results_base', 'phase1', 'station_hvsr_distributions'),
            ]
            for c in gen_candidates:
                if os.path.isdir(c):
                    args.gen_dir = c
                    break

        if not args.recon_dir:
            recon_candidates = [
                r'D:\Baris\codes\Time-Series-Library-main\eval_results\station_freq_20251018_001223',
                os.path.join(os.getcwd(), 'eval_results', 'station_freq_20251018_001223'),
            ]
            for c in recon_candidates:
                if os.path.isdir(c):
                    args.recon_dir = c
                    break

        if not args.vae_dir:
            vae_candidates = [
                r'D:\Baris\codes\Time-Series-Library-main\baris_second_train',
                os.path.join(os.getcwd(), 'baris_second_train'),
            ]
            for c in vae_candidates:
                if os.path.isdir(c):
                    args.vae_dir = c
                    break

        if not args.vae_gen_dir:
            vae_gen_candidates = [
                r'D:\Baris\codes\Time-Series-Library-main\baris_gen',
                os.path.join(os.getcwd(), 'baris_gen'),
            ]
            for c in vae_gen_candidates:
                if os.path.isdir(c):
                    args.vae_gen_dir = c
                    break

        if not args.gen_ts_dir:
            gen_ts_candidates = [
                r'D:\Baris\codes\Time-Series-Library-main\gen_eval_results_base\phase1\generated_timeseries_npz',
                os.path.join(os.getcwd(), 'gen_eval_results_base', 'phase1', 'generated_timeseries_npz'),
            ]
            for c in gen_ts_candidates:
                if os.path.isdir(c):
                    args.gen_ts_dir = c
                    break

        gen_map = _load_npz_map(args.gen_dir)
        recon_map = _load_npz_map(args.recon_dir)
        vae_map = _load_npz_map(args.vae_dir) if args.vae_dir and os.path.isdir(args.vae_dir) else {}

        # If gen_map is empty (None in logs), still proceed by using recon/vae station list
        stations: List[str] = sorted(set(recon_map.keys()) | set(vae_map.keys()) | set(gen_map.keys()))
        os.makedirs(args.out, exist_ok=True)
        # Provide VAE base dir hint to plot_station for fallback median HVSR
        vae_base_dir = args.vae_dir if args.vae_dir and os.path.isdir(args.vae_dir) else None
        vae_gen_base_dir = args.vae_gen_dir if args.vae_gen_dir and os.path.isdir(args.vae_gen_dir) else None

        for st in stations:
            gen = gen_map.get(st) if gen_map else None
            recon = recon_map.get(st)
            vae = vae_map.get(st)
            if gen is None and gen_map:
                # Try alternative station keys from gen
                try:
                    alt = None
                    for k in gen_map.keys():
                        if k.endswith(st):
                            alt = k
                            break
                    if alt is not None:
                        gen = gen_map.get(alt)
                except Exception:
                    pass
            plot_station(st, gen, recon, vae, args.out, vae_base_dir=vae_base_dir, vae_gen_base_dir=vae_gen_base_dir, recon_dir=args.recon_dir)

        # ================= Real vs Generated time-series pairs =================
        try:
            gen_ts_map = _load_generated_timeseries_npz(args.gen_ts_dir) if args.gen_ts_dir else {}
            if gen_ts_map:
                _plot_real_generated_timeseries_pairs(gen_ts_map, gen_map, args.out, mat_root=None, max_gen_per_station=2)
        except Exception as e:
            print(f"[gen-ts][warn] real-vs-generated timeseries plotting failed: {e}")
        # ================= Generation Extended Matrix (Real vs TimesNet-Gen) =================
        # DISABLED: Matrix plotting removed per user request to avoid MAT file lookup errors
        pass  # Matrix plotting disabled
        
        # ================= 15x15 Extended Matrices: TimesNet trio and VAE trio =================
        # DISABLED: Extended matrix plotting removed per user request
        pass  # Extended matrix plotting disabled
    finally:
        pass  # No cleanup needed


if __name__ == '__main__':
    main()


