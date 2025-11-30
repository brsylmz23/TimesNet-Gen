import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
import shutil
from datetime import datetime
from utils.augmentation import run_augmentation_single
import scipy.io as sio

warnings.filterwarnings('ignore')
def extract_station_id_from_mat_name(name: str):
    """Extract a 4-digit station id from typical AFAD-style names.
    Tries pattern 'AFAD.XXXX.' first; falls back to any 4 consecutive digits."""
    m = re.search(r'AFAD\.(\d{4})\.', str(name))
    if m:
        return m.group(1)
    m2 = re.search(r'(?<!\d)(\d{4})(?!\d)', str(name))
    return m2.group(1) if m2 else str(name)

def generate_and_save_station_splits_mat(root_dir: str,
                                          selected_stations: list,
                                          output_dir: str,
                                          seq_len: int = 6000,
                                          train_ratio: float = 0.8,
                                          val_ratio: float = 0.1,
                                          seed: int = 0,
                                          copy_files: bool = True):
    """
    Split generator for MAT files with station control.
    
    Behavior:
    - If selected_stations is provided (list of station IDs as strings), then:
        • All files of stations in selected_stations go to TEST.
        • All files of stations NOT in selected_stations go to TRAIN.
        • VAL is left empty.
    - If selected_stations is None, fallback to legacy behavior:
        • Per-station split with at least 1 test file per station (80/10/10 approx).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Tüm MAT dosyalarını tara
    mat_files = [f for f in os.listdir(root_dir) if f.lower().endswith('.mat')]
    all_items = []
    
    for fname in mat_files:
        st_id = extract_station_id_from_mat_name(fname)
        # Optional quick length check by opening mat (best-effort)
        try:
            path = os.path.join(root_dir, fname)
            mat = sio.loadmat(path)
            if 'signal' in mat:
                sig = mat['signal']
                T = sig.shape[1] if sig.shape[0] == 3 else sig.shape[0]
                if T != seq_len:
                    continue
        except Exception:
            pass
        all_items.append((fname, st_id))

    if not all_items:
        print(f"[splits] No files found under {root_dir} with seq_len={seq_len}")
        return

    # İstasyon ID'lerini çıkar ve grupla
    station_groups = {}
    for fname, st_id in all_items:
        if st_id not in station_groups:
            station_groups[st_id] = []
        station_groups[st_id].append(fname)
    
    print(f"[splits] Found {len(all_items)} MAT files from {len(station_groups)} stations")
    
    # Her istasyon için split yap
    train_list = []
    val_list = []
    test_list = []
    test_station_info = {}  # Test setindeki sinyalleri kaydet
    
    rng = np.random.RandomState(seed)
    
    # YENİ: Eğer selected_stations verildiyse, bu istasyonların TÜM dosyaları TEST'e, diğerleri TRAIN'e
    if selected_stations:
        selected_set = set(str(s) for s in selected_stations)
        print(f"[splits] Forcing stations to TEST: {sorted(selected_set)}")
        for station_id, station_files in station_groups.items():
            num_files = len(station_files)
            print(f"[splits] Processing station {station_id}: {num_files} files")
            shuffled_files = station_files.copy()
            rng.shuffle(shuffled_files)
            if station_id in selected_set:
                test_list.extend(shuffled_files)
                test_station_info[station_id] = list(shuffled_files)
                print(f"  {station_id}: 0 train, 0 val, {len(shuffled_files)} test (ALL to test)")
            else:
                train_list.extend(shuffled_files)
                print(f"  {station_id}: {len(shuffled_files)} train, 0 val, 0 test (ALL to train)")
    else:
        # Eski davranış: her istasyondan en az 1 test dosyası olacak şekilde 80/10/10 böl
        for station_id, station_files in station_groups.items():
            num_files = len(station_files)
            print(f"[splits] Processing station {station_id}: {num_files} files")
            # Dosyaları karıştır
            shuffled_files = station_files.copy()
            rng.shuffle(shuffled_files)
            if num_files < 3:
                print(f"[WARN] Station {station_id} has only {num_files} files, putting all in train")
                train_list.extend(shuffled_files)
                continue
            if num_files == 3:
                train_list.append(shuffled_files[0])
                val_list.append(shuffled_files[1])
                test_list.append(shuffled_files[2])
                test_station_info[station_id] = [shuffled_files[2]]
            else:
                min_test_files = max(1, int(num_files * 0.1))
                min_test_files = min(min_test_files, num_files - 2)
                test_files = shuffled_files[:min_test_files]
                remaining_files = shuffled_files[min_test_files:]
                remaining_count = len(remaining_files)
                train_count = int(remaining_count * train_ratio / (train_ratio + val_ratio))
                val_count = remaining_count - train_count
                train_files = remaining_files[:train_count]
                val_files = remaining_files[train_count:]
                train_list.extend(train_files)
                val_list.extend(val_files)
                test_list.extend(test_files)
                test_station_info[station_id] = test_files
                print(f"  {station_id}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
    
    # CSV dosyalarını kaydet
    def save_csv(name, files):
        out_csv = os.path.join(output_dir, f"{name}_list.csv")
        with open(out_csv, 'w') as f:
            f.write('name,station\n')
            for fname in files:
                st_id = extract_station_id_from_mat_name(fname)
                f.write(f"{fname},{st_id}\n")
        print(f"[splits] Saved {name} list → {out_csv}")
    
    save_csv('train', train_list)
    save_csv('val', val_list)
    save_csv('test', test_list)
    
    # YENİ: Test setindeki sinyalleri detaylı olarak kaydet
    test_details_csv = os.path.join(output_dir, 'test_station_details.csv')
    test_details_data = []
    for station_id, test_files in test_station_info.items():
        for test_file in test_files:
            test_details_data.append({
                'station_id': station_id,
                'file_name': test_file,
                'full_path': os.path.join(root_dir, test_file)
            })
    
    test_details_df = pd.DataFrame(test_details_data)
    test_details_df.to_csv(test_details_csv, index=False)
    
    print(f"[splits] Saved splits:")
    print(f"  Train: {len(train_list)} files")
    print(f"  Val: {len(val_list)} files")
    print(f"  Test: {len(test_list)} files")
    print(f"  Test stations: {len(test_station_info)} stations")
    print(f"  Test details saved to: {test_details_csv}")
    
    # Dosyaları kopyala (opsiyonel)
    if copy_files:
        for split_name, files in [('train', train_list), ('val', val_list), ('test', test_list)]:
            split_dir = os.path.join(output_dir, 'data', split_name)
            os.makedirs(split_dir, exist_ok=True)
            for fname in files:
                src = os.path.join(root_dir, fname)
                dst = os.path.join(split_dir, fname)
                try:
                    shutil.copy2(src, dst)
                except Exception as e:
                    print(f"[splits] copy error {src} → {dst}: {e}")
    
    return test_details_csv


class Dataset_ETT_hour(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0) 

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_M4(Dataset):
    def __init__(self, args, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly'):
        # size [seq_len, label_len, pred_len]
        # init
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           max(0, cut_point - self.label_len):min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask


# class PSMSegLoader(Dataset):
#     def __init__(self, args, root_path, win_size, step=1, flag="train"):
#         self.flag = flag
#         self.step = step
#         self.win_size = win_size
#         self.scaler = StandardScaler()
#         data = pd.read_csv(os.path.join(root_path, 'train.csv'))
#         data = data.values[:, 1:]
#         data = np.nan_to_num(data)
#         self.scaler.fit(data)
#         data = self.scaler.transform(data)
#         test_data = pd.read_csv(os.path.join(root_path, 'test.csv'))
#         test_data = test_data.values[:, 1:]
#         test_data = np.nan_to_num(test_data)
#         self.test = self.scaler.transform(test_data)
#         self.train = data
#         data_len = len(self.train)
#         self.val = self.train[(int)(data_len * 0.8):]
#         self.test_labels = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]
#         print("test:", self.test.shape)
#         print("train:", self.train.shape)

#     def __len__(self):
#         if self.flag == "train":
#             return (self.train.shape[0] - self.win_size) // self.step + 1
#         elif (self.flag == 'val'):
#             return (self.val.shape[0] - self.win_size) // self.step + 1
#         elif (self.flag == 'test'):
#             return (self.test.shape[0] - self.win_size) // self.step + 1
#         else:
#             return (self.test.shape[0] - self.win_size) // self.win_size + 1

#     def __getitem__(self, index):
#         index = index * self.step
#         if self.flag == "train":
#             return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
#         elif (self.flag == 'val'):
#             return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
#         elif (self.flag == 'test'):
#             return np.float32(self.test[index:index + self.win_size]), np.float32(
#                 self.test_labels[index:index + self.win_size])
#         else:
#             return np.float32(self.test[
#                               index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
#                 self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

# class PSMSegLoader(Dataset):
#     def __init__(self, args, root_path, win_size, step=None, flag="train"):
#         self.flag = flag
#         self.win_size = win_size
#         self.step = step if step is not None else win_size  
#         self.scaler = StandardScaler()

#         data = pd.read_csv(os.path.join(root_path, 'train.csv')).values[:, 1:]
#         data = np.nan_to_num(data)
#         self.scaler.fit(data)
#         self.train = self.scaler.transform(data)

#         test_data = pd.read_csv(os.path.join(root_path, 'test.csv')).values[:, 1:]
#         test_data = np.nan_to_num(test_data)
#         self.test = self.scaler.transform(test_data)

#         self.val = self.train[int(len(self.train) * 0.8):]
#         self.test_labels = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]

#         print("test:", self.test.shape)
#         print("train:", self.train.shape)

#     def __len__(self):
#         if self.flag == "train":
#             return self.train.shape[0] // self.win_size
#         elif self.flag == "val":
#             return self.val.shape[0] // self.win_size
#         elif self.flag == "test":
#             return (self.test.shape[0] - self.win_size) // self.step + 1
#         else:
#             return (self.test.shape[0] - self.win_size) // self.win_size + 1

#     def __getitem__(self, index):
#         if self.flag == "train":
#             start = index * self.win_size
#             end = start + self.win_size
#             return np.float32(self.train[start:end]), np.float32(self.test_labels[0:self.win_size])
#         elif self.flag == "val":
#             start = index * self.win_size
#             end = start + self.win_size
#             return np.float32(self.val[start:end]), np.float32(self.test_labels[0:self.win_size])
#         elif self.flag == "test":
#             start = index * self.win_size
#             end = start + self.win_size
#             return np.float32(self.test[start:end]), np.float32(self.test_labels[start:end])
#         else:
#             start = index * self.win_size
#             end = start + self.win_size
#             return np.float32(self.test[start:end]), np.float32(self.test_labels[start:end])

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import Dataset
from math import radians, cos, sin, asin, sqrt


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r

import os
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def haversine(lon1, lat1, lon2, lat2):
    from math import radians, sin, cos, sqrt, atan2
    R = 6371.0
    dlon = radians(lon2 - lon1)
    dlat = radians(lat2 - lat1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

class PSMSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=None, flag="train"):
        self.flag = flag
        self.win_size = win_size
        self.step = step if step is not None else win_size
        self.scaler = StandardScaler()

        self.data = []
        self.labels = []
        self.names = []
        self.distances = []

        # Yeni mat dosyalarının olduğu dizin (sadece test için)
        mat_dir = r"D:/Baris/PSNtest"

        if flag == "test":
            self._load_test_mat_files(mat_dir)
        else:
            # Tüm train+val verisini oku
            all_data, all_labels, all_names, all_distances = self._load_trainval_from_old_source(root_path)
            # Yüzdesel olarak böl
            train_data, val_data, train_labels, val_labels, train_names, val_names, train_distances, val_distances = train_test_split(
                all_data, all_labels, all_names, all_distances, test_size=0.1, random_state=42
            )
            if flag == "train":
                self.data = train_data
                self.labels = train_labels
                self.names = train_names
                self.distances = train_distances
            elif flag == "val":
                self.data = val_data
                self.labels = val_labels
                self.names = val_names
                self.distances = val_distances

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

        # Normalizasyon (tüm veri üzerinden fit)
        if len(self.data) > 0:
            flat_data = self.data.reshape(-1, 3)
            self.scaler.fit(flat_data)
            self.data = self.scaler.transform(flat_data).reshape(-1, 6000, 3)

    def _load_test_mat_files(self, directory):
        for file in os.listdir(directory):
            if file.endswith('.mat'):
                path = os.path.join(directory, file)
                mat = sio.loadmat(path)
                try:
                    signal = mat['signal']
                    if signal.shape == (3, 6000):
                        signal = signal.T
                    elif signal.shape == (6000, 3):
                        pass
                    else:
                        continue
                    # p_arrival_sample'ı oku
                    p_arrival_sample = None
                    if 'p_arrival_sample' in mat:
                        val = mat['p_arrival_sample']
                        if isinstance(val, np.ndarray):
                            val = val.squeeze()
                            # String ise
                            if hasattr(val, 'dtype') and val.dtype.kind in {'U', 'S'}:
                                if str(val) == 'None':
                                    p_arrival_sample = None
                                else:
                                    try:
                                        p_arrival_sample = float(val)
                                    except Exception:
                                        p_arrival_sample = None
                            # Float ise
                            elif hasattr(val, 'dtype') and val.dtype.kind == 'f':
                                p_arrival_sample = float(val)
                            else:
                                try:
                                    p_arrival_sample = float(val)
                                except Exception:
                                    p_arrival_sample = None
                        else:
                            try:
                                p_arrival_sample = float(val)
                            except Exception:
                                p_arrival_sample = None
                    label = np.zeros(6000)
                    if p_arrival_sample is not None and not np.isnan(p_arrival_sample):
                        p_index = int(round(p_arrival_sample * 100))
                        if 25 <= p_index <= 5975:
                            label[p_index - 25: p_index + 25] = 1
                    # Eğer p_arrival_sample None veya NaN ise, label zaten sıfır kalacak
                    name = file[:-4]
                    distance = -1
                    self.data.append(signal)
                    self.labels.append(label.reshape(-1, 1))
                    self.names.append(name)
                    self.distances.append(distance)
                except Exception as e:
                    print(f"Hata: {file} - {e}")
                    continue

    def _load_trainval_from_old_source(self, root_path):
        # Tüm train+val verisini oku ve döndür
        all_data = []
        all_labels = []
        all_names = []
        all_distances = []
        import re
        old_dir = root_path  # örn: r"D:/Baris/PS_afad"
        for file in os.listdir(old_dir):
            if file.endswith('.mat'):
                path = os.path.join(old_dir, file)
                mat = sio.loadmat(path)
                try:
                    dataset = mat['EQ'][0][0]['anEQ']
                    signal = dataset['Accel'][0][0]  # (N, 3)
                    ptime = dataset['Ptime'][0][0][0][0]
                    if ptime < 2:
                        continue
                    if signal.shape[0] < 6000:
                        continue
                    signal = signal[:6000]
                    label = np.zeros(6000)
                    p_index = int(ptime * 100)
                    if p_index < 25 or p_index > 5975:
                        continue
                    label[p_index - 25: p_index + 25] = 1
                    name = file[:-4]
                    try:
                        stationLat = dataset['statco'][0][0][0][0]
                        stationLon = dataset['statco'][0][0][0][1]
                        epicenterLat = dataset['epicenter'][0][0][0][0]
                        epicenterLon = dataset['epicenter'][0][0][0][1]
                        distance = haversine(stationLon, stationLat, epicenterLon, epicenterLat)
                    except:
                        distance = -1
                    all_data.append(signal)
                    all_labels.append(label.reshape(-1, 1))
                    all_names.append(name)
                    all_distances.append(distance)
                except Exception as e:
                    print(f"Hata: {file} - {e}")
                    continue
        return all_data, all_labels, all_names, all_distances

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (
            np.float32(self.data[index]),
            np.float32(self.labels[index]),
            self.names[index],
            self.distances[index]
        )




class MSLSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SWATSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
        test_data = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class UEAloader(Dataset):
    """
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
        self.args = args
        self.root_path = root_path
        self.flag = flag
        self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)
        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # pre_process
        normalizer = Normalizer()
        self.feature_df = normalizer.normalize(self.feature_df)
        print(len(self.all_IDs))

    def load_all(self, root_path, file_list=None, flag=None):
        """
        Loads datasets from ts files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .ts files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """
        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_path, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_path, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_path, '*')))
        if flag is not None:
            data_paths = list(filter(lambda x: re.search(flag, x), data_paths))
        input_paths = [p for p in data_paths if os.path.isfile(p) and p.endswith('.ts')]
        if len(input_paths) == 0:
            pattern='*.ts'
            raise Exception("No .ts files found using pattern: '{}'".format(pattern))

        all_df, labels_df = self.load_single(input_paths[0])  # a single file contains dataset

        return all_df, labels_df

    def load_single(self, filepath):
        df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                             replace_missing_vals_with='NaN')
        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(labels.cat.codes,
                                 dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        lengths = df.applymap(
            lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series

        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            df = df.applymap(subsample)

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)

        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df

    def instance_norm(self, case):
        if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        batch_x = self.feature_df.loc[self.all_IDs[ind]].values
        labels = self.labels_df.loc[self.all_IDs[ind]].values
        if self.flag == "TRAIN" and self.args.augmentation_ratio > 0:
            num_samples = len(self.all_IDs)
            num_columns = self.feature_df.shape[1]
            seq_len = int(self.feature_df.shape[0] / num_samples)
            batch_x = batch_x.reshape((1, seq_len, num_columns))
            batch_x, labels, augmentation_tags = run_augmentation_single(batch_x, labels, self.args)

            batch_x = batch_x.reshape((1 * seq_len, num_columns))

        return self.instance_norm(torch.from_numpy(batch_x)), \
               torch.from_numpy(labels)

    def __len__(self):
        return len(self.all_IDs)
