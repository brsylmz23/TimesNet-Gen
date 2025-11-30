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
    pdf_recon_timesnet = None  # TimesNet-Recon (from recon_dir NPZs)
    pdf_recon_vae = None  # VAE-Recon (from recon_dir NPZs or vae_dir NPZs)

    # Prefer provided PDFs; if missing, derive from samples if available
    recon_bins = None
    gen_bins = None
    vae_bins = None
    pdf_recon_vae_source = None  # 'recon' or 'vae'

    if recon is not None:
        # Exact keys per station_cond NPZ spec
        pdf_real = recon.get('pdf_real')
        pdf_recon_timesnet = recon.get('pdf_timesnet')
        if pdf_recon_vae is None and 'pdf_vae' in recon:
            pdf_recon_vae = recon.get('pdf_vae')
            pdf_recon_vae_source = 'recon'
        if pdf_real is None and 'f0_real' in recon:
            pdf_real = _kde_from_samples(np.asarray(recon['f0_real']).ravel(), f0_bins)
        if pdf_recon_timesnet is None and 'f0_timesnet' in recon:
            pdf_recon_timesnet = _kde_from_samples(np.asarray(recon['f0_timesnet']).ravel(), f0_bins)
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

    if vae is not None:
        # If separate VAE NPZs are provided, prefer explicit keys
        if pdf_recon_vae is None:
            pdf_recon_vae = vae.get('pdf_vae')
            if pdf_recon_vae is not None:
                pdf_recon_vae_source = 'vae'
        if pdf_recon_vae is None and 'f0_vae' in vae:
            vals = np.asarray(vae['f0_vae']).ravel()
            print(f"[debug:{station}] vae f0 key='f0_vae', count={vals.size}, min={vals.min() if vals.size>0 else 'nan'}, max={vals.max() if vals.size>0 else 'nan'}")
            pdf_recon_vae = _kde_from_samples(vals, f0_bins)
        vae_bins = vae.get('f0_bins') if 'f0_bins' in vae else None

    # Do NOT rebin; plot as read. Ensure arrays are at least arrays; if None, use zeros with own bins later.
    pdf_real = np.asarray(pdf_real) if pdf_real is not None else None
    pdf_gen_timesnet = np.asarray(pdf_gen_timesnet) if pdf_gen_timesnet is not None else None
    pdf_recon_timesnet = np.asarray(pdf_recon_timesnet) if pdf_recon_timesnet is not None else None
    pdf_recon_vae = np.asarray(pdf_recon_vae) if pdf_recon_vae is not None else None

    # Final shape sanity (using each own bins)
    try:
        lr = (pdf_real.size if pdf_real is not None else 0)
        lg = (pdf_gen_timesnet.size if pdf_gen_timesnet is not None else 0)
        lq = (pdf_recon_timesnet.size if pdf_recon_timesnet is not None else 0)
        lv = (pdf_recon_vae.size if pdf_recon_vae is not None else 0)
        print(f"[debug:{station}] lens real={lr}, gen={lg}, recon={lq}, vae={lv}")
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
    corr_rr = _corr_if_match(pdf_recon_timesnet, pdf_real)
    corr_vr = _corr_if_match(pdf_recon_vae, pdf_real)
    print(f"[debug:{station}] corr Gen-Real={corr_gr:.3f}, Recon-Real={corr_rr:.3f}, VAE-Real={corr_vr:.3f}")

    # Color palette (consistent across top/bottom)
    color_real = 'steelblue'      # Real
    color_gen = 'indianred'       # TimesNet-Gen
    color_recon = 'darkorange'    # TimesNet-Recon
    color_vae = 'seagreen'        # VAE-Recon
    color_vae_gen = 'mediumpurple'  # VAE-Gen

    # HVSR median curves
    hfreq = None
    h_med_real = None
    h_med_gen = None
    h_med_recon = None
    h_med_vae = None

    # Prefer recon for real frequency
    for src in (recon, gen, vae):
        if src is not None and 'hvsr_freq_real' in src:
            hfreq = src['hvsr_freq_real']
            break
    if hfreq is None:
        hfreq = np.linspace(1.0, 20.0, 256)

    if recon is not None:
        if h_med_real is None:
            h_med_real = recon.get('hvsr_median_real', None)
        tmp_recon = _get_first(recon, ['hvsr_median_recon', 'hvsr_median_timesnet'])
        if tmp_recon is not None:
            h_med_recon = tmp_recon
        # Try to get VAE median HVSR if present in recon NPZ
        if h_med_vae is None:
            h_med_vae = recon.get('hvsr_median_vae') if 'hvsr_median_vae' in recon else None
    if gen is not None:
        if h_med_real is None:
            h_med_real = gen.get('hvsr_median_real')
        if h_med_gen is None:
            h_med_gen = gen.get('hvsr_median_timesnet')
    if vae is not None and h_med_vae is None:
        # If VAE median exists (optional)
        h_med_vae = vae.get('hvsr_median_vae')

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
    f_recon_src = None
    f_vae_src = None
    if recon is not None and 'hvsr_freq_real' in recon:
        f_real_src = recon.get('hvsr_freq_real')
    elif gen is not None and 'hvsr_freq_real' in gen:
        f_real_src = gen.get('hvsr_freq_real')
    if gen is not None and 'hvsr_freq_timesnet' in gen:
        f_gen_src = gen.get('hvsr_freq_timesnet')
    if recon is not None and 'hvsr_freq_timesnet' in recon:
        f_recon_src = recon.get('hvsr_freq_timesnet')
    if vae is not None and 'hvsr_freq_vae' in vae:
        f_vae_src = vae.get('hvsr_freq_vae')

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
    if h_med_recon is not None and f_recon_src is not None:
        h_med_recon = _align_to(f_recon_src, h_med_recon, hfreq)
    if h_med_vae is not None and f_vae_src is not None:
        h_med_vae = _align_to(f_vae_src, h_med_vae, hfreq)

    # Fallback: compute VAE-Recon median HVSR directly from VAE .asc triplets if NPZ lacks it
    if (h_med_vae is None or not np.any(np.isfinite(h_med_vae))) and vae_base_dir and os.path.isdir(vae_base_dir):
        try:
            # Prefer explicit mapping as used in transfer_learning_station_cond.py
            station_to_vae_dir = {
                '0205': '0205_generated_5_class_trial_reconstru_condfs',
                '1716': '1716_generated_5_class_trial_reconstru_condfs',
                '2020': '2020_generated_5_class_trial_reconstru_condfs',
                '3130': '3130_generated_5_class_trial_reconstru_condfs',
                '4628': '4628_generated_5_class_trial_reconstru_condfs',
            }
            station_dir = None
            mapped = station_to_vae_dir.get(station)
            if mapped:
                cand = os.path.join(vae_base_dir, mapped)
                if os.path.isdir(cand):
                    station_dir = cand
            # If mapping failed, try glob patterns
            if station_dir is None:
                search_patterns = [
                    os.path.join(vae_base_dir, f"{station}_generated*reconstru*"),
                    os.path.join(vae_base_dir, '**', f"{station}_generated*reconstru*"),
                ]
                candidate_dirs = []
                for pat in search_patterns:
                    candidate_dirs.extend(glob.glob(pat, recursive=True))
                for d in candidate_dirs:
                    if os.path.isdir(d):
                        station_dir = d
                        break
            if station_dir:
                e_files = sorted([f for f in os.listdir(station_dir) if f.endswith('_Channel_E.asc')])
                curves = []
                max_samples = 12
                print(f"[debug:{station}] VAE fallback: reading up to {max_samples} triplets from {station_dir} (found {len(e_files)} E files)")
                for idx, ef in enumerate(e_files[:max_samples]):
                    base = ef[:-len('_Channel_E.asc')]
                    n_path = os.path.join(station_dir, base + '_Channel_N.asc')
                    u_path = os.path.join(station_dir, base + '_Channel_U.asc')
                    e_path = os.path.join(station_dir, ef)
                    if not (os.path.exists(e_path) and os.path.exists(n_path) and os.path.exists(u_path)):
                        continue
                    try:
                        e = np.loadtxt(e_path, dtype=np.float64)
                        n = np.loadtxt(n_path, dtype=np.float64)
                        u = np.loadtxt(u_path, dtype=np.float64)
                        T = int(min(len(e), len(n), len(u)))
                        if T < 16:
                            continue
                        sig = np.stack([e[:T], n[:T], u[:T]], axis=1)
                        fva, hva = _compute_hvsr(sig, fs=100.0)
                        if fva is None or hva is None:
                            continue
                        curves.append((fva, hva))
                        if (idx + 1) % 3 == 0:
                            print(f"[debug:{station}]   processed {idx + 1}/{min(max_samples, len(e_files))} triplets")
                    except Exception:
                        continue
                if curves:
                    # Build median on unified grid hfreq
                    if hfreq is None or hfreq.size == 0:
                        # Use first curve's freq
                        hfreq = curves[0][0]
                    mats = []
                    for fva, hva in curves:
                        try:
                            # Interp with edge-value extension to avoid NaNs
                            y = np.interp(hfreq, fva, hva, left=float(hva[0]), right=float(hva[-1]))
                            mats.append(y)
                        except Exception:
                            pass
                    if mats:
                        M = np.vstack(mats)
                        h_med_vae = np.median(M, axis=0)
                        print(f"[debug:{station}] VAE median HVSR computed from {len(mats)} files in {station_dir}")
                    else:
                        print(f"[debug:{station}] VAE fallback: no valid interpolations (curves={len(curves)})")
            else:
                print(f"[debug:{station}] No VAE directory found under {vae_base_dir} for fallback median HVSR")
        except Exception as e:
            print(f"[debug:{station}] VAE fallback median HVSR failed: {e}")

    # NEW: VAE-Gen from ASC directories
    h_med_vae_gen = None
    samples_vae_gen = []
    pdf_vae_gen = None
    if vae_gen_base_dir and os.path.isdir(vae_gen_base_dir):
        try:
            vae_gen_map = {
                '0205': '0205_generated_5_class_trial_condition_append',
                '1716': '1716_generated_5_class_trial_condition_append',
                '2020': '2020_generated_5_class_trial_condition_append',
                '3130': '3130_generated_5_class_trial_condition_append',
                '4628': '4628_generated_5_class_trial_condition_append',
            }
            vae_gen_dir = None
            mapped = vae_gen_map.get(station)
            if mapped:
                cand = os.path.join(vae_gen_base_dir, mapped)
                if os.path.isdir(cand):
                    vae_gen_dir = cand
            if vae_gen_dir is None:
                pats = [
                    os.path.join(vae_gen_base_dir, f"{station}_generated*condition_append*"),
                    os.path.join(vae_gen_base_dir, '**', f"{station}_generated*condition_append*"),
                ]
                cands = []
                for p in pats:
                    cands.extend(glob.glob(p, recursive=True))
                for d in cands:
                    if os.path.isdir(d):
                        vae_gen_dir = d; break
            if vae_gen_dir:
                e_files = sorted([f for f in os.listdir(vae_gen_dir) if f.endswith('_Channel_E.asc')])
                curves = []
                max_samples = 12
                print(f"[debug:{station}] VAE-Gen: reading up to {max_samples} triplets from {vae_gen_dir} (found {len(e_files)} E files)")
                for idx, ef in enumerate(e_files[:max_samples]):
                    base = ef[:-len('_Channel_E.asc')]
                    n_path = os.path.join(vae_gen_dir, base + '_Channel_N.asc')
                    u_path = os.path.join(vae_gen_dir, base + '_Channel_U.asc')
                    e_path = os.path.join(vae_gen_dir, ef)
                    if not (os.path.exists(e_path) and os.path.exists(n_path) and os.path.exists(u_path)):
                        continue
                    try:
                        e = np.loadtxt(e_path, dtype=np.float64)
                        n = np.loadtxt(n_path, dtype=np.float64)
                        u = np.loadtxt(u_path, dtype=np.float64)
                        T = int(min(len(e), len(n), len(u)))
                        if T < 16:
                            continue
                        sig = np.stack([e[:T], n[:T], u[:T]], axis=1)
                        fva, hva = _compute_hvsr(sig, fs=100.0)
                        if fva is None or hva is None:
                            continue
                        curves.append((fva, hva))
                        # f0 from HVSR max
                        try:
                            f0_idx = int(np.argmax(hva))
                            f0_val = float(fva[f0_idx])
                            if np.isfinite(f0_val):
                                samples_vae_gen.append(f0_val)
                        except Exception:
                            pass
                    except Exception:
                        continue
                # median HVSR for VAE-Gen
                if curves:
                    # choose reference grid as first curve
                    fref = curves[0][0]
                    mats = []
                    for fva, hva in curves:
                        try:
                            y = np.interp(fref, fva, hva, left=float(hva[0]), right=float(hva[-1]))
                            mats.append(y)
                        except Exception:
                            pass
                    if mats:
                        h_med_vae_gen = np.median(np.vstack(mats), axis=0)
                        print(f"[debug:{station}] VAE-Gen median HVSR computed from {len(mats)} files")
                    # Build PDF for VAE-Gen using common f0_bins for consistency
                    try:
                        vals = np.asarray(samples_vae_gen, dtype=float)
                        vals = vals[np.isfinite(vals)]
                        if vals.size > 0 and f0_bins is not None:
                            hist, _ = np.histogram(vals, bins=f0_bins)
                            pdf = hist.astype(float)
                            s = pdf.sum()
                            pdf_vae_gen = pdf / s if s > 0 else pdf
                    except Exception:
                        pdf_vae_gen = None
            else:
                print(f"[debug:{station}] No VAE-Gen directory found under {vae_gen_base_dir}")
        except Exception as e:
            print(f"[debug:{station}] VAE-Gen reading failed: {e}")

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
    samples_recon = recon.get('f0_timesnet') if (recon is not None and 'f0_timesnet' in recon) else None
    samples_vae = recon.get('f0_vae') if (recon is not None and 'f0_vae' in recon) else None
    # VAE-Gen samples already collected from ASC
    samples_vae_gen_arr = np.asarray(samples_vae_gen, dtype=float) if samples_vae_gen else None

    # Labels for histogram bars
    label_pdf_real = 'Real f0'
    label_pdf_gen = 'TimesNet-Gen f0'
    label_pdf_recon = 'TimesNet-Recon f0'
    label_pdf_vae = 'VAE-Recon f0'
    label_pdf_vae_gen = 'VAE-Gen f0'

    _draw_pdf_bars(ax1, recon_bins if pdf_real is not None else None, pdf_real, color_real, label_pdf_real)
    _draw_pdf_bars(ax1, gen_bins, pdf_gen_timesnet, color_gen, label_pdf_gen)
    _draw_pdf_bars(ax1, recon_bins, pdf_recon_timesnet, color_recon, label_pdf_recon)
    _draw_pdf_bars(ax1, recon_bins if pdf_recon_vae_source == 'recon' else vae_bins, pdf_recon_vae, color_vae, label_pdf_vae)
    # VAE-Gen PDF bars (uses common f0_bins histogram built above)
    _draw_pdf_bars(ax1, f0_bins, pdf_vae_gen, color_vae_gen, label_pdf_vae_gen)

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
    _plot_kde(ax1, samples_recon, color_recon, 'TimesNet-Recon f0 Density')
    _plot_kde(ax1, samples_vae, color_vae, 'VAE-Recon f0 Density')
    _plot_kde(ax1, samples_vae_gen_arr, color_vae_gen, 'VAE-Gen f0 Density')

    ax1.set_xlabel('f0 (Hz)')
    ax1.set_ylabel('Probability Density')
    ax1.set_title(f'f0 Distribution Comparison (corr Gen-Real={corr_gr:.2f}, Recon-Real={corr_rr:.2f}, VAE-Real={corr_vr:.2f}) - Station {station}', fontsize=16, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=13, framealpha=0.95)
    ax1.set_xlabel('f0 (Hz)', fontsize=15, fontweight='bold')
    ax1.set_ylabel('Probability Density', fontsize=15, fontweight='bold')
    ax1.tick_params(labelsize=13)
    # Do not force common x-limits; keep default to avoid hiding differing ranges

    # Bottom: Median HVSR curves
    # Plot as read: use source freq arrays if available; avoid interpolation
    if recon is not None and 'hvsr_freq_real' in recon and h_med_real is not None:
        ax2.plot(recon.get('hvsr_freq_real'), h_med_real, color=color_real, linewidth=2, label='Real HVSR')
    elif gen is not None and 'hvsr_freq_real' in gen and h_med_real is not None:
        ax2.plot(gen.get('hvsr_freq_real'), h_med_real, color=color_real, linewidth=2, label='Real Median HVSR')
    if gen is not None and 'hvsr_freq_timesnet' in gen and h_med_gen is not None:
        ax2.plot(gen.get('hvsr_freq_timesnet'), h_med_gen, color=color_gen, linestyle='--', linewidth=2, label='TimesNet-Gen HVSR')
    if recon is not None and 'hvsr_freq_timesnet' in recon and h_med_recon is not None:
        ax2.plot(recon.get('hvsr_freq_timesnet'), h_med_recon, color=color_recon, linestyle='-.', linewidth=2, label='TimesNet-Recon HVSR')
    # VAE-Recon: either from NPZ freq or fallback computed curves median (already in h_med_vae with hfreq grid)
    if vae is not None and 'hvsr_freq_vae' in vae and h_med_vae is not None:
        ax2.plot(vae.get('hvsr_freq_vae'), h_med_vae, color=color_vae, linestyle=':', linewidth=2, label='VAE-Recon HVSR')
    elif h_med_vae is not None:
        # Fallback median from ASC (computed above); uses hfreq grid
        ax2.plot(hfreq, h_med_vae, color=color_vae, linestyle=':', linewidth=2, label='VAE-Recon HVSR')
    else:
        if vae is not None:
            print(f"[debug:{station}] No VAE median HVSR found in NPZ; skipping bottom VAE curve.")
    # VAE-Gen HVSR
    if h_med_vae_gen is not None:
        # Use reference frequency grid fref from earlier; reuse hfreq if lengths match, else plot against fref
        try:
            if 'fref' in locals() and len(h_med_vae_gen) == len(fref):
                ax2.plot(fref, h_med_vae_gen, color=color_vae_gen, linestyle=':', linewidth=2, label='VAE-Gen HVSR')
            else:
                # fallback: plot against indices scaled to 1..20 for visibility
                ax2.plot(np.linspace(1.0, 20.0, len(h_med_vae_gen)), h_med_vae_gen, color=color_vae_gen, linestyle=':', linewidth=2, label='VAE-Gen HVSR')
        except Exception:
            pass
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
        print(f"[plot] {station}: sums real={float(np.sum(pdf_real)):.3f}, gen={float(np.sum(pdf_gen_timesnet)):.3f}, recon={float(np.sum(pdf_recon_timesnet)):.3f}, vae={float(np.sum(pdf_recon_vae)):.3f}")
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
        try:
            if gen_map:
                # Common f0 bins for histogram PDFs (1..20 Hz, 20 bins)
                f0_bins = np.linspace(1.0, 20.0, 21)
                stations_sorted = sorted(list(gen_map.keys()))
                S = len(stations_sorted)
                # Build per-station PDFs from raw samples for robustness
                def _hist_pdf(vals, bins):
                    v = np.asarray(vals, dtype=float).ravel()
                    v = v[np.isfinite(v)]
                    if v.size == 0:
                        return np.zeros(bins.size - 1, dtype=float)
                    h, _ = np.histogram(v, bins=bins)
                    h = h.astype(float)
                    s = h.sum()
                    return (h / s) if s > 0 else h
                H_real = {}
                H_gen = {}
                for st in stations_sorted:
                    d = gen_map.get(st, {})
                    vals_r = d.get('f0_real') if 'f0_real' in d else np.array([])
                    vals_g = d.get('f0_timesnet') if 'f0_timesnet' in d else np.array([])
                    H_real[st] = _hist_pdf(vals_r, f0_bins)
                    H_gen[st] = _hist_pdf(vals_g, f0_bins)
                # Safe correlation
                def _safe_corr_pdf(a, b):
                    a = np.asarray(a, dtype=float).ravel()
                    b = np.asarray(b, dtype=float).ravel()
                    if a.size != b.size or a.size == 0:
                        return 0.0
                    if np.allclose(a.std(), 0) or np.allclose(b.std(), 0):
                        return 0.0
                    return float(np.corrcoef(a, b)[0, 1])
                # SxS: Real vs Gen
                M_gen = np.zeros((S, S), dtype=float)
                for i, si in enumerate(stations_sorted):
                    hi = H_real.get(si, np.zeros(len(f0_bins) - 1))
                    for j, sj in enumerate(stations_sorted):
                        hj = H_gen.get(sj, np.zeros(len(f0_bins) - 1))
                        M_gen[i, j] = _jensen_shannon_similarity(hi, hj)
                # 2Sx2S extended with interleaved [R, Gen]
                def ridx(k):
                    return 2 * k
                def gidx(k):
                    return 2 * k + 1
                M_ext_gen = np.zeros((2 * S, 2 * S), dtype=float)
                for i, si in enumerate(stations_sorted):
                    for j, sj in enumerate(stations_sorted):
                        # R-R
                        M_ext_gen[ridx(i), ridx(j)] = _jensen_shannon_similarity(H_real.get(si, np.zeros(len(f0_bins) - 1)), H_real.get(sj, np.zeros(len(f0_bins) - 1)))
                        # R-Gen
                        M_ext_gen[ridx(i), gidx(j)] = _jensen_shannon_similarity(H_real.get(si, np.zeros(len(f0_bins) - 1)), H_gen.get(sj, np.zeros(len(f0_bins) - 1)))
                        # Gen-R
                        M_ext_gen[gidx(i), ridx(j)] = _jensen_shannon_similarity(H_gen.get(si, np.zeros(len(f0_bins) - 1)), H_real.get(sj, np.zeros(len(f0_bins) - 1)))
                        # Gen-Gen
                        M_ext_gen[gidx(i), gidx(j)] = _jensen_shannon_similarity(H_gen.get(si, np.zeros(len(f0_bins) - 1)), H_gen.get(sj, np.zeros(len(f0_bins) - 1)))
                # Save CSVs (plain)
                try:
                    # SxS
                    with open(os.path.join(args.out, 'gen_result_matrix.csv'), 'w') as fcsv:
                        fcsv.write(',' + ','.join(stations_sorted) + '\n')
                        for i, si in enumerate(stations_sorted):
                            row = [f"{M_gen[i, j]:.6f}" for j in range(S)]
                            fcsv.write(si + ',' + ','.join(row) + '\n')
                    # 2Sx2S extended
                    row_labels = []
                    col_labels = []
                    for s in stations_sorted:
                        row_labels.extend([f'R:{s}', f'Gen:{s}'])
                        col_labels.extend([f'R:{s}', f'Gen:{s}'])
                    with open(os.path.join(args.out, 'gen_result_matrix_extended.csv'), 'w') as fcsv:
                        fcsv.write(',' + ','.join(col_labels) + '\n')
                        for i in range(2 * S):
                            row = [f"{M_ext_gen[i, j]:.6f}" for j in range(2 * S)]
                            fcsv.write(row_labels[i] + ',' + ','.join(row) + '\n')
                except Exception:
                    pass
                # Save heatmaps
                try:
                    # SxS heatmap with numeric annotations (like transfer script)
                    # Build ideal SxS identity for Real vs Gen (same-station match)
                    I_gen = np.eye(S, dtype=float)
                    gen_score = _normalized_corr2(M_gen, I_gen)
                    plt.figure(figsize=(max(8, S), max(6, S)))
                    im = plt.imshow(M_gen, cmap='gray_r', aspect='auto', vmin=0.0, vmax=1.0)
                    plt.colorbar(im, fraction=0.046, pad=0.04, label='Jensen-Shannon Similarity')
                    plt.xticks(ticks=np.arange(S), labels=stations_sorted, rotation=45, ha='right', fontsize=12)
                    plt.yticks(ticks=np.arange(S), labels=stations_sorted, fontsize=12)
                    plt.xlabel('Generated Station', fontsize=13)
                    plt.ylabel('Real Station', fontsize=13)
                    plt.title(f'Real vs TimesNet-Gen f0 Histogram JS Similarity (score={gen_score:.3f})', fontsize=16, fontweight='bold', pad=20)
                    # annotate numbers inside boxes
                    try:
                        vmax = np.nanmax(M_gen) if np.isfinite(M_gen).any() else 1.0
                        for i in range(S):
                            for j in range(S):
                                val = M_gen[i, j]
                                txt_color = 'white' if val > vmax / 2.0 else 'black'
                                plt.text(j, i, f'{val:.3f}', ha='center', va='center', color=txt_color, fontsize=8)
                    except Exception:
                        pass
                    plt.tight_layout()
                    plt.savefig(os.path.join(args.out, 'gen_result_matrix.png'), dpi=300, bbox_inches='tight')
                    plt.close()
                    # 2Sx2S heatmap with numeric annotations (like transfer script)
                    # Ideal 2Sx2S for [R, Gen] interleaved: ones on same-station R-R, G-G, R-G, G-R
                    I_ext_gen = np.zeros_like(M_ext_gen)
                    for i in range(S):
                        I_ext_gen[ridx(i), ridx(i)] = 1.0
                        I_ext_gen[gidx(i), gidx(i)] = 1.0
                        I_ext_gen[ridx(i), gidx(i)] = 1.0
                        I_ext_gen[gidx(i), ridx(i)] = 1.0
                    gen_ext_score = _normalized_corr2(M_ext_gen, I_ext_gen)
                    plt.figure(figsize=(max(10, 0.7 * 2 * S), max(10, 0.7 * 2 * S)))
                    im = plt.imshow(M_ext_gen, cmap='gray_r', aspect='auto', vmin=0.0, vmax=1.0)
                    plt.colorbar(im, fraction=0.046, pad=0.04, label='Jensen-Shannon Similarity')
                    plt.xticks(ticks=np.arange(2 * S), labels=col_labels, rotation=45, ha='right', fontsize=10)
                    plt.yticks(ticks=np.arange(2 * S), labels=row_labels, fontsize=10)
                    plt.xlabel('Column Type × Station', fontsize=12)
                    plt.ylabel('Row Type × Station', fontsize=12)
                    plt.title(f'Extended f0 Histogram JS Similarity [RR, RGen; GenR, GenGen] (score={gen_ext_score:.3f})', fontsize=16, fontweight='bold', pad=20)
                    # annotate numbers
                    try:
                        vmax = np.nanmax(M_ext_gen) if np.isfinite(M_ext_gen).any() else 1.0
                        font_size = max(6, min(10, 20 // max(1, S)))
                        for i in range(2 * S):
                            for j in range(2 * S):
                                val = M_ext_gen[i, j]
                                txt_color = 'white' if val > vmax / 2.0 else 'black'
                                plt.text(j, i, f'{val:.2f}', ha='center', va='center', color=txt_color, fontsize=font_size)
                    except Exception:
                        pass
                    plt.tight_layout()
                    plt.savefig(os.path.join(args.out, 'gen_result_matrix_extended.png'), dpi=300, bbox_inches='tight')
                    plt.close()
                except Exception:
                    pass
        except Exception:
            pass
        # ================= 15x15 Extended Matrices: TimesNet trio and VAE trio =================
        try:
            stations_sorted = sorted(list(set(recon_map.keys()) | set(gen_map.keys())))
            if stations_sorted:
                S = len(stations_sorted)
                f0_bins = np.linspace(1.0, 20.0, 21)
                def _hist_pdf(vals, bins):
                    v = np.asarray(vals, dtype=float).ravel()
                    v = v[np.isfinite(v)]
                    if v.size == 0:
                        return np.zeros(bins.size - 1, dtype=float)
                    h, _ = np.histogram(v, bins=bins)
                    h = h.astype(float)
                    s = h.sum()
                    return (h / s) if s > 0 else h
                # Collect per-station f0 vectors
                real_f0 = {}
                for st in stations_sorted:
                    vals = None
                    if st in recon_map and 'f0_real' in recon_map[st]:
                        vals = recon_map[st]['f0_real']
                    elif st in gen_map and 'f0_real' in gen_map[st]:
                        vals = gen_map[st]['f0_real']
                    real_f0[st] = np.asarray(vals) if vals is not None else np.array([])
                tn_recon_f0 = {st: (recon_map[st]['f0_timesnet'] if (st in recon_map and 'f0_timesnet' in recon_map[st]) else np.array([])) for st in stations_sorted}
                tn_gen_f0   = {st: (gen_map[st]['f0_timesnet'] if (st in gen_map and 'f0_timesnet' in gen_map[st]) else np.array([])) for st in stations_sorted}
                vae_recon_f0 = {st: (recon_map[st]['f0_vae'] if (st in recon_map and 'f0_vae' in recon_map[st]) else np.array([])) for st in stations_sorted}
                # Build VAE-Gen f0 from ASC
                vae_gen_f0 = {st: np.array([]) for st in stations_sorted}
                if vae_gen_base_dir and os.path.isdir(vae_gen_base_dir):
                    vae_gen_map = {
                        '0205': '0205_generated_5_class_trial_condition_append',
                        '1716': '1716_generated_5_class_trial_condition_append',
                        '2020': '2020_generated_5_class_trial_condition_append',
                        '3130': '3130_generated_5_class_trial_condition_append',
                        '4628': '4628_generated_5_class_trial_condition_append',
                    }
                    for st in stations_sorted:
                        dir_st = None
                        mapped = vae_gen_map.get(st)
                        if mapped:
                            cand = os.path.join(vae_gen_base_dir, mapped)
                            if os.path.isdir(cand):
                                dir_st = cand
                        if dir_st is None:
                            pats = [
                                os.path.join(vae_gen_base_dir, f"{st}_generated*condition_append*"),
                                os.path.join(vae_gen_base_dir, '**', f"{st}_generated*condition_append*"),
                            ]
                            cands = []
                            for p in pats:
                                cands.extend(glob.glob(p, recursive=True))
                            for d in cands:
                                if os.path.isdir(d):
                                    dir_st = d; break
                        if dir_st is None:
                            continue
                        try:
                            e_files = sorted([f for f in os.listdir(dir_st) if f.endswith('_Channel_E.asc')])
                            f0_list = []
                            max_samples = min(200, len(e_files))
                            for ef in e_files[:max_samples]:
                                base = ef[:-len('_Channel_E.asc')]
                                e_path = os.path.join(dir_st, ef)
                                n_path = os.path.join(dir_st, base + '_Channel_N.asc')
                                u_path = os.path.join(dir_st, base + '_Channel_U.asc')
                                if not (os.path.exists(e_path) and os.path.exists(n_path) and os.path.exists(u_path)):
                                    continue
                                try:
                                    e = np.loadtxt(e_path, dtype=np.float64)
                                    n = np.loadtxt(n_path, dtype=np.float64)
                                    u = np.loadtxt(u_path, dtype=np.float64)
                                    T = int(min(len(e), len(n), len(u)))
                                    if T < 16:
                                        continue
                                    sig = np.stack([e[:T], n[:T], u[:T]], axis=1)
                                    fva, hva = _compute_hvsr(sig, fs=100.0)
                                    if fva is None or hva is None:
                                        continue
                                    idx = int(np.argmax(hva))
                                    f0 = float(fva[idx])
                                    if np.isfinite(f0):
                                        f0_list.append(f0)
                                except Exception:
                                    continue
                            vae_gen_f0[st] = np.asarray(f0_list, dtype=float)
                        except Exception:
                            continue
                # Jensen-Shannon similarity helper
                # (Using module-level _jensen_shannon_similarity and _normalized_corr2)
                # TimesNet trio 15x15
                H_r = {st: _hist_pdf(real_f0.get(st, np.array([])), f0_bins) for st in stations_sorted}
                H_q = {st: _hist_pdf(tn_recon_f0.get(st, np.array([])), f0_bins) for st in stations_sorted}
                H_g = {st: _hist_pdf(tn_gen_f0.get(st, np.array([])), f0_bins) for st in stations_sorted}
                N = 3 * S
                M_ext_tn = np.zeros((N, N), dtype=float)
                row_labels = []
                col_labels = []
                for s in stations_sorted:
                    row_labels.extend([f'R:{s}', f'Recon:{s}', f'Gen:{s}'])
                    col_labels.extend([f'R:{s}', f'Recon:{s}', f'Gen:{s}'])
                def ridx(k):
                    return 3 * k
                def qidx(k):
                    return 3 * k + 1
                def gidx(k):
                    return 3 * k + 2
                for i, si in enumerate(stations_sorted):
                    for j, sj in enumerate(stations_sorted):
                        M_ext_tn[ridx(i), ridx(j)] = _jensen_shannon_similarity(H_r[si], H_r[sj])
                        M_ext_tn[ridx(i), qidx(j)] = _jensen_shannon_similarity(H_r[si], H_q[sj])
                        M_ext_tn[ridx(i), gidx(j)] = _jensen_shannon_similarity(H_r[si], H_g[sj])
                        M_ext_tn[qidx(i), ridx(j)] = _jensen_shannon_similarity(H_q[si], H_r[sj])
                        M_ext_tn[qidx(i), qidx(j)] = _jensen_shannon_similarity(H_q[si], H_q[sj])
                        M_ext_tn[qidx(i), gidx(j)] = _jensen_shannon_similarity(H_q[si], H_g[sj])
                        M_ext_tn[gidx(i), ridx(j)] = _jensen_shannon_similarity(H_g[si], H_r[sj])
                        M_ext_tn[gidx(i), qidx(j)] = _jensen_shannon_similarity(H_g[si], H_q[sj])
                        M_ext_tn[gidx(i), gidx(j)] = _jensen_shannon_similarity(H_g[si], H_g[sj])
                try:
                    with open(os.path.join(args.out, 'extended_timesnet_15x15.csv'), 'w') as fcsv:
                        fcsv.write(',' + ','.join(col_labels) + '\n')
                        for i in range(N):
                            row = [f"{M_ext_tn[i, j]:.6f}" for j in range(N)]
                            fcsv.write(row_labels[i] + ',' + ','.join(row) + '\n')
                except Exception:
                    pass
                try:
                    # Ideal 15x15 (3S x 3S): for each station, 3x3 all-ones block across (R, Recon, Gen)
                    I_ext_tn = np.zeros_like(M_ext_tn)
                    for i in range(S):
                        idxs = [ridx(i), qidx(i), gidx(i)]
                        for a in idxs:
                            for b in idxs:
                                I_ext_tn[a, b] = 1.0
                    tn_score = _normalized_corr2(M_ext_tn, I_ext_tn)
                    # Save generic ideal 15x15 as separate figure and CSV (shared for TimesNet/VAE)
                    try:
                        # CSV
                        with open(os.path.join(args.out, 'extended_ideal_15x15.csv'), 'w') as fcsv:
                            fcsv.write(',' + ','.join(col_labels) + '\n')
                            for i in range(N):
                                row = [f"{I_ext_tn[i, j]:.1f}" for j in range(N)]
                                fcsv.write(row_labels[i] + ',' + ','.join(row) + '\n')
                        # PNG
                        plt.figure(figsize=(max(12, 0.7 * N), max(12, 0.7 * N)))
                        im = plt.imshow(I_ext_tn, cmap='gray_r', aspect='auto', vmin=0.0, vmax=1.0)
                        plt.colorbar(im, fraction=0.046, pad=0.04, label='Ideal (1=same station block)')
                        plt.xticks(ticks=np.arange(N), labels=col_labels, rotation=45, ha='right', fontsize=9)
                        plt.yticks(ticks=np.arange(N), labels=row_labels, fontsize=9)
                        plt.xlabel('Column Type × Station', fontsize=12)
                        plt.ylabel('Row Type × Station', fontsize=12)
                        plt.title('Extended Ideal Matrix (3×3 blocks per station)', fontsize=16, fontweight='bold', pad=20)
                        try:
                            font_size = max(5, min(9, 100 // max(1, N)))
                            for i in range(N):
                                for j in range(N):
                                    val = I_ext_tn[i, j]
                                    txt_color = 'white' if val > 0.5 else 'black'
                                    plt.text(j, i, f'{val:.0f}', ha='center', va='center', color=txt_color, fontsize=font_size)
                        except Exception:
                            pass
                        plt.tight_layout()
                        plt.savefig(os.path.join(args.out, 'extended_ideal_15x15.png'), dpi=300, bbox_inches='tight')
                        plt.close()
                    except Exception:
                        pass
                    plt.figure(figsize=(max(12, 0.7 * N), max(12, 0.7 * N)))
                    im = plt.imshow(M_ext_tn, cmap='gray_r', aspect='auto', vmin=0.0, vmax=1.0)
                    plt.colorbar(im, fraction=0.046, pad=0.04, label='Jensen-Shannon Similarity')
                    plt.xticks(ticks=np.arange(N), labels=col_labels, rotation=45, ha='right', fontsize=9)
                    plt.yticks(ticks=np.arange(N), labels=row_labels, fontsize=9)
                    plt.xlabel('Column Type × Station', fontsize=12)
                    plt.ylabel('Row Type × Station', fontsize=12)
                    plt.title(f'TimesNet Extended Matrix: Real vs Recon vs Gen\n(f0 Histogram JS Similarity, score={tn_score:.3f})', fontsize=16, fontweight='bold', pad=20)
                    # Annotate each station block with fixed target f0 values (Hz) instead of per-cell JS numbers
                    try:
                        # Hard-coded target f0 values per station (Hz)
                        target_f0 = {
                            '2020': 5.1,
                            '4628': 1.8,
                            '0205': 2.6,
                            '1716': 6.4,
                            '3130': 12.8,
                        }
                        ann_color = 'magenta'
                        for idx, st in enumerate(stations_sorted):
                            if st not in target_f0:
                                continue
                            # 3x3 block indices for this station
                            r0 = ridx(idx)
                            c0 = ridx(idx)
                            x_center = c0 + 1
                            y_center = r0 + 1
                            f0_val = target_f0[st]
                            plt.text(
                                x_center, y_center,
                                f"{f0_val:.1f} Hz",
                                ha='center', va='center',
                                color=ann_color, fontsize=16, fontweight='bold',
                                bbox=dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.95, edgecolor='black', linewidth=1.2)
                            )
                    except Exception:
                        pass
                    plt.tight_layout()
                    plt.savefig(os.path.join(args.out, 'extended_timesnet_15x15.png'), dpi=300, bbox_inches='tight')
                    plt.close()
                except Exception:
                    pass
                # VAE trio 15x15
                H_vr = {st: _hist_pdf(real_f0.get(st, np.array([])), f0_bins) for st in stations_sorted}
                H_vq = {st: _hist_pdf(vae_recon_f0.get(st, np.array([])), f0_bins) for st in stations_sorted}
                H_vg = {st: _hist_pdf(vae_gen_f0.get(st, np.array([])), f0_bins) for st in stations_sorted}
                M_ext_vae = np.zeros((N, N), dtype=float)
                row_labels_v = []
                col_labels_v = []
                for s in stations_sorted:
                    row_labels_v.extend([f'R:{s}', f'VAE-Recon:{s}', f'VAE-Gen:{s}'])
                    col_labels_v.extend([f'R:{s}', f'VAE-Recon:{s}', f'VAE-Gen:{s}'])
                def vridx(k):
                    return 3 * k
                def vqidx(k):
                    return 3 * k + 1
                def vgidx(k):
                    return 3 * k + 2
                for i, si in enumerate(stations_sorted):
                    for j, sj in enumerate(stations_sorted):
                        M_ext_vae[vridx(i), vridx(j)] = _jensen_shannon_similarity(H_vr[si], H_vr[sj])
                        M_ext_vae[vridx(i), vqidx(j)] = _jensen_shannon_similarity(H_vr[si], H_vq[sj])
                        M_ext_vae[vridx(i), vgidx(j)] = _jensen_shannon_similarity(H_vr[si], H_vg[sj])
                        M_ext_vae[vqidx(i), vridx(j)] = _jensen_shannon_similarity(H_vq[si], H_vr[sj])
                        M_ext_vae[vqidx(i), vqidx(j)] = _jensen_shannon_similarity(H_vq[si], H_vq[sj])
                        M_ext_vae[vqidx(i), vgidx(j)] = _jensen_shannon_similarity(H_vq[si], H_vg[sj])
                        M_ext_vae[vgidx(i), vridx(j)] = _jensen_shannon_similarity(H_vg[si], H_vr[sj])
                        M_ext_vae[vgidx(i), vqidx(j)] = _jensen_shannon_similarity(H_vg[si], H_vq[sj])
                        M_ext_vae[vgidx(i), vgidx(j)] = _jensen_shannon_similarity(H_vg[si], H_vg[sj])
                try:
                    with open(os.path.join(args.out, 'extended_vae_15x15.csv'), 'w') as fcsv:
                        fcsv.write(',' + ','.join(col_labels_v) + '\n')
                        for i in range(N):
                            row = [f"{M_ext_vae[i, j]:.6f}" for j in range(N)]
                            fcsv.write(row_labels_v[i] + ',' + ','.join(row) + '\n')
                except Exception:
                    pass
                try:
                    # Ideal 15x15 for VAE trio: for each station, 3x3 all-ones block across (R, VAE-Recon, VAE-Gen)
                    I_ext_vae = np.zeros_like(M_ext_vae)
                    for i in range(S):
                        idxs = [vridx(i), vqidx(i), vgidx(i)]
                        for a in idxs:
                            for b in idxs:
                                I_ext_vae[a, b] = 1.0
                    vae_score = _normalized_corr2(M_ext_vae, I_ext_vae)
                    # Do not duplicate ideal save for VAE; reuse the generic ideal saved above
                    plt.figure(figsize=(max(12, 0.7 * N), max(12, 0.7 * N)))
                    im = plt.imshow(M_ext_vae, cmap='gray_r', aspect='auto', vmin=0.0, vmax=1.0)
                    plt.colorbar(im, fraction=0.046, pad=0.04, label='Jensen-Shannon Similarity')
                    plt.xticks(ticks=np.arange(N), labels=col_labels_v, rotation=45, ha='right')
                    plt.yticks(ticks=np.arange(N), labels=row_labels_v)
                    plt.title(f'VAE Extended Matrix: Real vs Recon vs Gen\n(f0 Histogram JS Similarity, score={vae_score:.3f})', fontsize=16, fontweight='bold', pad=20)
                    # Annotate each station block with fixed target f0 values (Hz) instead of per-cell JS numbers
                    try:
                        target_f0 = {
                            '2020': 5.1,
                            '4628': 1.8,
                            '0205': 2.6,
                            '1716': 6.4,
                            '3130': 12.8,
                        }
                        ann_color = 'magenta'
                        for idx, st in enumerate(stations_sorted):
                            if st not in target_f0:
                                continue
                            r0 = vridx(idx)
                            c0 = vridx(idx)
                            x_center = c0 + 1
                            y_center = r0 + 1
                            f0_val = target_f0[st]
                            plt.text(
                                x_center, y_center,
                                f"{f0_val:.1f} Hz",
                                ha='center', va='center',
                                color=ann_color, fontsize=16, fontweight='bold',
                                bbox=dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.95, edgecolor='black', linewidth=1.2)
                            )
                    except Exception:
                        pass
                    plt.tight_layout()
                    plt.savefig(os.path.join(args.out, 'extended_vae_15x15.png'), dpi=300, bbox_inches='tight')
                    plt.close()
                except Exception:
                    pass
        except Exception:
            pass
    finally:
        pass  # No cleanup needed


if __name__ == '__main__':
    main()


