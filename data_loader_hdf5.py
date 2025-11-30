#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

class AugmentedHDF5Dataset(torch.utils.data.Dataset):
    """HDF5 data için augmentation - P ve S anlarını da kaydırır"""
    def __init__(self, base_dataset, sampling_rate=100):
        self.base_dataset = base_dataset
        self.sampling_rate = sampling_rate
        self.total_len = len(base_dataset) * 3  # orijinal + 2 shift

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        base_idx = idx // 3
        shift_idx = idx % 3
        signal, labels, classification_labels, name, distance = self.base_dataset[base_idx]
        
        if isinstance(signal, np.ndarray):
            signal = torch.from_numpy(signal).float()
        elif isinstance(signal, torch.Tensor):
            signal = signal.float()
        
        window_len = signal.shape[0]
        
        if shift_idx == 0:
            # Orijinal - hiçbir değişiklik yok
            return signal, labels, classification_labels, name, distance
        elif shift_idx == 1:
            # Orta shift: window_len // 3 kadar kaydır
            shift_amount = window_len // 3
            signal_shifted = torch.roll(signal, shifts=shift_amount, dims=0)
            
            # P ve S anlarını da kaydır (saniye cinsinden)
            p_time, s_time = labels[0], labels[1]
            if p_time > 0:  # P zamanı varsa
                p_time_shifted = (p_time + shift_amount / self.sampling_rate) % (window_len / self.sampling_rate)
            else:
                p_time_shifted = 0.0
                
            if s_time > 0:  # S zamanı varsa
                s_time_shifted = (s_time + shift_amount / self.sampling_rate) % (window_len / self.sampling_rate)
            else:
                s_time_shifted = 0.0
            
            labels_shifted = torch.tensor([p_time_shifted, s_time_shifted], dtype=torch.float32)
            return signal_shifted, labels_shifted, classification_labels, name, distance
            
        else:
            # Son shift: 2 * window_len // 3 kadar kaydır
            shift_amount = 2 * window_len // 3
            signal_shifted = torch.roll(signal, shifts=shift_amount, dims=0)
            
            # P ve S anlarını da kaydır (saniye cinsinden)
            p_time, s_time = labels[0], labels[1]
            if p_time > 0:  # P zamanı varsa
                p_time_shifted = (p_time + shift_amount / self.sampling_rate) % (window_len / self.sampling_rate)
            else:
                p_time_shifted = 0.0
                
            if s_time > 0:  # S zamanı varsa
                s_time_shifted = (s_time + shift_amount / self.sampling_rate) % (window_len / self.sampling_rate)
            else:
                s_time_shifted = 0.0
            
            labels_shifted = torch.tensor([p_time_shifted, s_time_shifted], dtype=torch.float32)
            return signal_shifted, labels_shifted, classification_labels, name, distance

def plot_augmented_hdf5_examples(train_data, folder_path, sampling_rate=100):
    """HDF5 augmentation örneklerini çiz"""
    plt.figure(figsize=(15, 10))
    
    for i in range(3):
        signal, labels, classification_labels, name, distance = train_data[i]
        if isinstance(signal, torch.Tensor):
            signal_np = signal.cpu().numpy()
        else:
            signal_np = signal
        
        p_time, s_time = labels[0], labels[1]
        
        plt.subplot(3, 1, i+1)
        plt.plot(signal_np[:, 0], color='blue', alpha=0.7, label='Z Channel')
        
        # P ve S anlarını işaretle
        if p_time > 0:
            p_sample = int(p_time * sampling_rate)
            plt.axvline(p_sample, color='red', linestyle='--', label=f'P @ {p_time:.2f}s')
        
        if s_time > 0:
            s_sample = int(s_time * sampling_rate)
            plt.axvline(s_sample, color='green', linestyle='--', label=f'S @ {s_time:.2f}s')
        
        shift_type = ['Orijinal', 'Orta Shift', 'Son Shift'][i]
        plt.title(f'{shift_type}: {name} - P: {p_time:.2f}s, S: {s_time:.2f}s')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(folder_path, 'hdf5_augmented_examples.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"HDF5 augmented examples plot saved to: {save_path}")

class HDF5PSDataset(Dataset):
    """
    HDF5 dosyasından P-S wave detection için data yükleyen dataset
    """
    
    def __init__(self, hdf5_path, flag='train', seq_len=6000, sampling_rate=100):
        """
        Args:
            hdf5_path: HDF5 dosya yolu
            flag: 'train', 'val', 'test'
            seq_len: Sinyal uzunluğu (default: 6000 sample)
            sampling_rate: Sampling rate (Hz) - default: 100 Hz
        """
        self.hdf5_path = hdf5_path
        self.flag = flag
        self.seq_len = seq_len
        self.sampling_rate = sampling_rate
        
        # Data ve label'ları yükle
        self.data, self.labels, self.classification_labels, self.names, self.distances = self._load_data()
        
        print(f"{flag} dataset loaded:")
        print(f"  Total samples: {len(self.data)}")
        print(f"  Samples with P arrival: {np.sum([1 for label in self.labels if label[0] > 0])}")
        print(f"  Samples with S arrival: {np.sum([1 for label in self.labels if label[1] > 0])}")
        print(f"  Signal shape: {self.data[0].shape if len(self.data) > 0 else 'No data'}")
    
    def _load_data(self):
        """HDF5 dosyasından data yükle"""
        data = []
        labels = []
        classification_labels = []
        names = []
        distances = []
        
        try:
            with h5py.File(self.hdf5_path, 'r') as f:
                # Ana 'data' group'unu bul
                if 'data' in f:
                    data_group = f['data']
                    # Tüm sample'ları listele
                    sample_names = list(data_group.keys())
                    
                    for sample_name in sample_names:
                        try:
                            # Sample data'yı al
                            sample_data = data_group[sample_name]
                            
                            if isinstance(sample_data, h5py.Dataset):
                                # Signal data
                                signal = sample_data[:]  # (3, T) veya (T, 3)
                                
                                # Shape kontrol et ve düzelt
                                if signal.shape[0] == 3:  # (3, T)
                                    signal = signal.T  # (T, 3) yap
                                
                                # P ve S arrival time'ları al önce
                                p_time, s_time = self._extract_arrival_times(sample_data.attrs)
                                
                                # Sequence length kontrolü - sadece eşit olanları al
                                if signal.shape[0] != self.seq_len:
                                    continue
                                
                                # Classification label'ları oluştur
                                p_exists, s_exists = self._extract_classification_labels(p_time, s_time)
                                
                                # Distance bilgisini al
                                distance = self._extract_distance(sample_data.attrs)
                                
                                # Data'ya ekle
                                data.append(signal.astype(np.float32))
                                labels.append([p_time, s_time])
                                classification_labels.append([p_exists, s_exists])
                                names.append(sample_name)
                                distances.append(distance)
                                
                        except Exception as e:
                            print(f"Error loading sample {sample_name}: {e}")
                            continue
                else:
                    print("'data' group not found in HDF5 file")
                    return [], [], [], [], []
                        
        except Exception as e:
            print(f"Error opening HDF5 file: {e}")
            return [], [], [], [], []
        
        # Train/val/test split
        if len(data) > 0:
            data, labels, classification_labels, names, distances = self._split_data(data, labels, classification_labels, names, distances)
        
        return data, labels, classification_labels, names, distances
    
    def _extract_arrival_times(self, attrs):
        """
        HDF5 attributes'dan P ve S arrival time'ları çıkar
        
        Returns:
            tuple: (p_time, s_time) - saniye cinsinden, -1 yerine 0 kullan
        """
        p_time = 0.0  # Default: P yok
        s_time = 0.0  # Default: S yok
        
        # P arrival time (sample cinsinden mi saniye cinsinden mi kontrol et)
        if 'p_arrival_sample' in attrs:
            p_arrival_sample = attrs['p_arrival_sample']
            if p_arrival_sample is not None and str(p_arrival_sample) != 'None':
                try:
                    # Eğer sample cinsinden ise saniyeye çevir, değilse olduğu gibi kullan
                    if isinstance(p_arrival_sample, (int, float)) and p_arrival_sample > 1000:
                        # Büyük değer muhtemelen sample cinsinden
                        p_time = float(p_arrival_sample) / self.sampling_rate
                    else:
                        # Küçük değer muhtemelen saniye cinsinden
                        p_time = float(p_arrival_sample)
                    
                    if p_time < 0:  # Negatif değer varsa 0 yap
                        p_time = 0.0
                except Exception as e:
                    p_time = 0.0  # Hata durumunda 0
        
        # S arrival time (zaten saniye cinsinden)
        if 's_arrival' in attrs:
            s_arrival = attrs['s_arrival']
            if s_arrival is not None and str(s_arrival) != 'None':
                try:
                    s_time = float(s_arrival)
                    if s_time < 0:  # Negatif değer varsa 0 yap
                        s_time = 0.0
                except:
                    s_time = 0.0  # Hata durumunda 0
        
        return p_time, s_time
    
    def _handle_sequence_length(self, signal, p_time, s_time):
        """Signal'i target seq_len'e resize et ve P/S zamanlarını scaling et"""
        current_length = signal.shape[0]
        target_length = self.seq_len
        
        if current_length == target_length:
            # Aynı uzunluk, hiçbir şey yapma
            return signal, p_time, s_time
        
        if current_length > target_length:
            # Downsample: Merkez kısmını al
            start_idx = (current_length - target_length) // 2
            end_idx = start_idx + target_length
            resized_signal = signal[start_idx:end_idx, :]
            
            # P/S zamanlarını yeni koordinat sistemine çevir
            if p_time > 0:
                p_sample_idx = p_time * 100  # saniye -> sample
                if start_idx <= p_sample_idx < end_idx:
                    p_time = (p_sample_idx - start_idx) / 100.0  # Yeni zaman
                else:
                    p_time = 0.0  # Kesilen bölgedeyse sıfırla
            
            if s_time > 0:
                s_sample_idx = s_time * 100
                if start_idx <= s_sample_idx < end_idx:
                    s_time = (s_sample_idx - start_idx) / 100.0
                else:
                    s_time = 0.0
                    
        else:
            # Upsample: Zero padding ile uzat
            padding_needed = target_length - current_length
            # Sonu pad et
            padding = np.zeros((padding_needed, signal.shape[1]), dtype=signal.dtype)
            resized_signal = np.concatenate([signal, padding], axis=0)
            
            # P/S zamanları aynı kalır çünkü başlangıç korunuyor
            
        return resized_signal, p_time, s_time
    
    def _extract_classification_labels(self, p_time, s_time):
        """
        P ve S classification label'ları oluştur (var/yok)
        
        Args:
            p_time: P arrival time (0.0 = yok, >0 = var)
            s_time: S arrival time (0.0 = yok, >0 = var)
            
        Returns:
            tuple: (p_exists, s_exists) - 1.0 = var, 0.0 = yok
        """
        p_exists = 1.0 if p_time > 0.0 else 0.0
        s_exists = 1.0 if s_time > 0.0 else 0.0
        return p_exists, s_exists
    

    def _extract_distance(self, attrs):
        """Distance bilgisini çıkar ve normalize et"""
        if 'source_distance_km' in attrs:
            distance = attrs['source_distance_km']
            if distance is not None and str(distance) != 'None':
                try:
                    distance_val = float(distance)
                    if distance_val < 0:  # Negatif değer varsa 0 yap
                        distance_val = 0.0
                    return distance_val
                except:
                    return 0.0  # Hata durumunda 0
        return 0.0  # Distance bilgisi yoksa 0
    
    def _split_data(self, data, labels, classification_labels, names, distances):
        """Data'yı train/val/test olarak böl"""
        if self.flag == 'train':
            # Train için %80
            indices = list(range(len(data)))
            train_indices, temp_indices = train_test_split(
                indices, test_size=0.2, random_state=0
            )
            return ([data[i] for i in train_indices],
                   [labels[i] for i in train_indices],
                   [classification_labels[i] for i in train_indices],
                   [names[i] for i in train_indices],
                   [distances[i] for i in train_indices])
        
        elif self.flag == 'val':
            # Val için %10
            indices = list(range(len(data)))
            temp_indices, val_indices = train_test_split(
                indices, test_size=0.2, random_state=0
            )
            val_indices, _ = train_test_split(
                val_indices, test_size=0.5, random_state=0
            )
            return ([data[i] for i in val_indices],
                   [labels[i] for i in val_indices],
                   [classification_labels[i] for i in val_indices],
                   [names[i] for i in val_indices],
                   [distances[i] for i in val_indices])
        
        else:  # test
            # Test için %10
            indices = list(range(len(data)))
            temp_indices, test_indices = train_test_split(
                indices, test_size=0.2, random_state=0
            )
            _, test_indices = train_test_split(
                test_indices, test_size=0.5, random_state=0
            )
            return ([data[i] for i in test_indices],
                   [labels[i] for i in test_indices],
                   [classification_labels[i] for i in test_indices],
                   [names[i] for i in test_indices],
                   [distances[i] for i in test_indices])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return (
            torch.tensor(self.data[index], dtype=torch.float32),  # Signal data (seq_len, 3)
            torch.tensor(self.labels[index], dtype=torch.float32),  # [P_time, S_time] - Regression
            torch.tensor(self.classification_labels[index], dtype=torch.float32),  # [P_exists, S_exists] - Classification
            self.names[index],          # Sample name
            self.distances[index]       # Distance
        )

def create_hdf5_data_loaders(hdf5_path, batch_size=32, seq_len=6000, sampling_rate=100, num_workers=0):
    """
    HDF5 data loader'ları oluştur
    
    Args:
        hdf5_path: HDF5 dosya yolu
        batch_size: Batch size
        seq_len: Sinyal uzunluğu
        sampling_rate: Sampling rate (Hz)
        num_workers: DataLoader worker sayısı
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    
    # Datasets oluştur
    train_dataset = HDF5PSDataset(hdf5_path, 'train', seq_len, sampling_rate)
    val_dataset = HDF5PSDataset(hdf5_path, 'val', seq_len, sampling_rate)
    test_dataset = HDF5PSDataset(hdf5_path, 'test', seq_len, sampling_rate)
    
    # Augment datasets
    train_dataset = AugmentedHDF5Dataset(train_dataset)
    val_dataset = AugmentedHDF5Dataset(val_dataset)
    test_dataset = AugmentedHDF5Dataset(test_dataset)

    # DataLoaders oluştur
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test için
    hdf5_path = "/Applications/Projects/DeepEQ/Datasets/Afad to HDF5.hdf5"
    
    if os.path.exists(hdf5_path):
        print("Testing HDF5 data loader...")
        
        # Dataset'i test et
        dataset = HDF5PSDataset(hdf5_path, 'train', seq_len=6000)
        
        if len(dataset) > 0:
            print(f"\nFirst sample:")
            signal, label, name, distance = dataset[0]
            print(f"  Signal shape: {signal.shape}")
            print(f"  Label: {label}")
            print(f"  Name: {name}")
            print(f"  Distance: {distance}")
            
            # DataLoader'ı test et
            train_loader, val_loader, test_loader = create_hdf5_data_loaders(
                hdf5_path, batch_size=4, seq_len=6000
            )
            
            print(f"\nDataLoader sizes:")
            print(f"  Train batches: {len(train_loader)}")
            print(f"  Val batches: {len(val_loader)}")
            print(f"  Test batches: {len(test_loader)}")
            
            # İlk batch'i test et
            for batch_x, batch_y, names, distances in train_loader:
                print(f"\nFirst batch:")
                print(f"  Batch X shape: {batch_x.shape}")
                print(f"  Batch Y shape: {batch_y.shape}")
                print(f"  Names: {names}")
                print(f"  Distances: {distances}")
                break
        else:
            print("No data loaded!")
    else:
        print(f"HDF5 file not found: {hdf5_path}")
