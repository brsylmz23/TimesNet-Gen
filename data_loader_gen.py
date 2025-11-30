import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io as sio
import glob

def _iter_np_arrays(obj):
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

def _load_signal_from_mat_any(path):
    """Load 3-channel signal from a MATLAB .mat file using scipy only (no HDF5)."""
    try:
        mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        arrays = list(_iter_np_arrays(mat))
        # Common direct keys first
        for key in ['signal', 'data', 'sig', 'x', 'X', 'signal3c', 'acc', 'NS', 'EW', 'UD']:
            if key in mat and isinstance(mat[key], np.ndarray):
                arrays.insert(0, mat[key])
        cand = _find_3ch_from_arrays(arrays)
        if cand is not None:
            return cand
    except NotImplementedError:
        # Likely a v7.3 file; user requested not to use HDF5 reader
        return None
    except Exception:
        return None
    return None


class GenMatDataset(Dataset):
    """Generative dataset: reads MAT signals listed in split CSV (name, station).
    Returns (x: (T,3) float32, station_id: long, name: str)
    """
    def __init__(self, root_dir: str, csv_file: str, seq_len: int = 6000):
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.items = []
        self.station_to_id = {}
        self.id_to_station = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    name, station = row[0], row[1]
                    self.items.append((name, station))
                    if station not in self.station_to_id:
                        self.station_to_id[station] = len(self.id_to_station)
                        self.id_to_station.append(station)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        name, station = self.items[idx]
        path = os.path.join(self.root_dir, name)
        # Ensure .mat extension
        if not path.lower().endswith('.mat'):
            path = path + '.mat'
        # Try direct, then split subfolders, then recursive search
        if not os.path.exists(path):
            for sub in ['train', 'val', 'test']:
                alt = os.path.join(self.root_dir, sub, os.path.basename(path))
                if os.path.exists(alt):
                    path = alt
                    break
        if not os.path.exists(path):
            candidates = glob.glob(os.path.join(self.root_dir, '**', os.path.basename(path)), recursive=True)
            if candidates:
                path = candidates[0]
        
        # Load MAT via scipy (mimic data_loader.py logic)
        sig = None
        try:
            mat = sio.loadmat(path)
            # Primary: 'signal' key (direct signal array)
            if 'signal' in mat:
                signal = mat['signal']
                if signal.shape == (3, self.seq_len):
                    signal = signal.T  # (T, 3)
                elif signal.shape == (self.seq_len, 3):
                    pass
                else:
                    # Try crop/pad if length doesn't match but channels=3
                    if signal.shape[0] == 3:
                        signal = signal.T
                    # Now signal should be (T, 3); adjust T
                    T = signal.shape[0]
                    if T < self.seq_len:
                        pad = np.zeros((self.seq_len - T, signal.shape[1]), dtype=signal.dtype)
                        signal = np.concatenate([signal, pad], axis=0)
                    elif T > self.seq_len:
                        signal = signal[:self.seq_len]
                sig = signal
            # Fallback: 'EQ' nested structure (as in PSMSegLoader)
            elif 'EQ' in mat:
                try:
                    dataset = mat['EQ'][0][0]['anEQ']
                    accel = dataset['Accel'][0][0]  # (N, 3)
                    if accel.ndim != 2 or accel.shape[1] != 3:
                        print(f"[DEBUG] Unexpected Accel shape in {path}: {accel.shape}")
                    else:
                        sig = accel
                except Exception as e:
                    print(f"[DEBUG] Could not parse EQ structure in {path}: {e}")
        except NotImplementedError:
            print(f"[DEBUG] File {path} is v7.3 (HDF5), skipping as per user constraint.")
        except Exception as e:
            print(f"[DEBUG] Error loading {path}: {e}")
        
        if sig is None:
            # Print available keys for debugging
            try:
                mat_debug = sio.loadmat(path, squeeze_me=False, struct_as_record=False)
                keys = [k for k in mat_debug.keys() if not k.startswith('__')]
                print(f"[DEBUG] Available keys in {path}: {keys}")
                for k in keys[:5]:  # show shapes of first 5 keys
                    v = mat_debug[k]
                    print(f"  {k}: type={type(v)}, shape={getattr(v, 'shape', 'N/A')}, dtype={getattr(v, 'dtype', 'N/A')}")
            except Exception:
                pass
            raise RuntimeError(f'signal not found in {path}')
        if sig.ndim == 2 and sig.shape[0] == 3:
            sig = sig.transpose(1, 0)
        sig = sig.astype(np.float32)
        T = sig.shape[0]
        if T >= self.seq_len:
            sig = sig[:self.seq_len]
        else:
            pad = np.zeros((self.seq_len - T, sig.shape[1]), dtype=np.float32)
            sig = np.concatenate([sig, pad], axis=0)
        x = torch.from_numpy(sig)
        sid = torch.tensor(self.station_to_id[station], dtype=torch.long)
        return x, sid, name


