import os
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")


def build_window_index(meta, total_len, win_size, step):
    if meta is None or "segment_id" not in meta.columns:
        starts = range(0, max(total_len - win_size + 1, 0), step)
        return list(starts)

    starts = []
    for _, group in meta.groupby("segment_id", sort=False):
        indices = group.index.to_numpy()
        if len(indices) < win_size:
            continue
        for offset in range(0, len(indices) - win_size + 1, step):
            starts.append(int(indices[offset]))
    return starts


class Dataset_ALFA(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train_data = pd.read_csv(os.path.join(root_path, "train.csv"))
        test_data = pd.read_csv(os.path.join(root_path, "test.csv"))
        train_meta_path = os.path.join(root_path, "train_meta.csv")
        test_meta_path = os.path.join(root_path, "test_meta.csv")
        train_meta = pd.read_csv(train_meta_path) if os.path.exists(train_meta_path) else None
        test_meta = pd.read_csv(test_meta_path) if os.path.exists(test_meta_path) else None

        if "label" not in train_data.columns or "label" not in test_data.columns:
            raise ValueError("ALFA csv files must contain a final 'label' column.")

        normal_index = train_data[train_data["label"] == 0].index
        train_data = train_data.loc[normal_index].copy().reset_index(drop=True)
        if train_meta is not None:
            train_meta = train_meta.loc[normal_index].copy().reset_index(drop=True)
        train_data = train_data.bfill().ffill()
        test_data = test_data.bfill().ffill()

        self.train_mark = train_data.iloc[:, 0].to_numpy()
        self.test_mark = test_data.iloc[:, 0].to_numpy()
        self.test_labels = test_data.iloc[:, -1].to_numpy()

        train_values = train_data.iloc[:, 1:-1].to_numpy(dtype=np.float32)
        test_values = test_data.iloc[:, 1:-1].to_numpy(dtype=np.float32)

        self.scaler.fit(train_values)
        train_values = self.scaler.transform(train_values)
        self.test = self.scaler.transform(test_values)

        split = int(len(train_values) * 0.9)
        self.train = train_values[:split]
        self.val = train_values[split:]
        self.val_mark = self.train_mark[split:]
        self.train_mark = self.train_mark[:split]
        self.test_labels = self.test_labels.astype(np.float32)

        self.train_meta = train_meta.iloc[:split].reset_index(drop=True) if train_meta is not None else None
        self.val_meta = train_meta.iloc[split:].reset_index(drop=True) if train_meta is not None else None
        self.test_meta = test_meta.reset_index(drop=True) if test_meta is not None else None
        self.train_windows = build_window_index(
            self.train_meta, len(self.train), self.win_size, self.step
        )
        self.val_windows = build_window_index(
            self.val_meta, len(self.val), self.win_size, self.step
        )
        self.test_windows = build_window_index(
            self.test_meta, len(self.test), self.win_size, self.step
        )
        self.thre_windows = build_window_index(
            self.test_meta, len(self.test), self.win_size, self.win_size
        )

        print("test:", self.test.shape)
        print("train:", self.train.shape)
        print("val:", self.val.shape)
        if self.flag == "test":
            self.anomaly_ratio = np.count_nonzero(self.test_labels) / len(
                self.test_labels
            )
            print("anomaly_ratio:", self.anomaly_ratio)

    def __len__(self):
        if self.flag == "train":
            return len(self.train_windows)
        if self.flag == "val":
            return len(self.val_windows)
        if self.flag == "test":
            return len(self.test_windows)
        return len(self.thre_windows)

    def __getitem__(self, index):
        if self.flag == "thre":
            start = self.thre_windows[index]
            end = start + self.win_size
            return (
                np.float32(self.test[start:end]),
                np.float32(self.test_labels[start:end]),
                np.float32(self.test_mark[start:end]),
            )

        if self.flag == "train":
            start = self.train_windows[index]
        elif self.flag == "val":
            start = self.val_windows[index]
        else:
            start = self.test_windows[index]
        end = start + self.win_size
        if self.flag == "train":
            labels = np.zeros(self.win_size, dtype=np.float32)
            return (
                np.float32(self.train[start:end]),
                labels,
                np.float32(self.train_mark[start:end]),
            )
        if self.flag == "val":
            labels = np.zeros(self.win_size, dtype=np.float32)
            return (
                np.float32(self.val[start:end]),
                labels,
                np.float32(self.val_mark[start:end]),
            )
        return (
            np.float32(self.test[start:end]),
            np.float32(self.test_labels[start:end]),
            np.float32(self.test_mark[start:end]),
        )


class DatasetFD(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        train_data = pd.read_csv(os.path.join(root_path, "train_X.csv"))
        test_data = pd.read_csv(os.path.join(root_path, "test_X.csv"))
        train_labels = pd.read_csv(os.path.join(root_path, "train_y.csv"))
        test_labels = pd.read_csv(os.path.join(root_path, "test_y.csv"))

        fault_index = train_labels[train_labels["fault"] == 1].index
        train_data = train_data.drop(fault_index, axis=0)
        train_labels = train_labels.drop(fault_index, axis=0)

        train_data = train_data.to_numpy(dtype=np.float32)
        test_data = test_data.to_numpy(dtype=np.float32)
        train_labels = train_labels.to_numpy(dtype=np.float32)
        test_labels = test_labels.to_numpy(dtype=np.float32)

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)

        data_len = len(train_data)
        split = int(data_len * 0.8)
        self.train = train_data[:split]
        self.val = train_data[split:]
        self.train_labels = train_labels[:split]
        self.val_labels = train_labels[split:]
        self.test = test_data
        self.test_labels = test_labels
        self.train_mark = train_data
        self.test_mark = test_data
        self.val_mark = self.val

    def __len__(self):
        if self.flag == "train":
            return max((self.train.shape[0] - self.win_size) // self.step + 1, 0)
        if self.flag == "val":
            return max((self.val.shape[0] - self.win_size) // self.step + 1, 0)
        if self.flag == "test":
            return max((self.test.shape[0] - self.win_size) // self.step + 1, 0)
        return max((self.test.shape[0] - self.win_size) // self.win_size + 1, 0)

    def __getitem__(self, index):
        if self.flag == "thre":
            start = index * self.win_size
            end = start + self.win_size
            return (
                np.float32(self.test[start:end]),
                np.float32(self.test_labels[start:end]),
                np.float32(self.test_mark[start:end]),
            )

        start = index * self.step
        end = start + self.win_size
        if self.flag == "train":
            return (
                np.float32(self.train[start:end]),
                np.float32(self.train_labels[start:end]),
                np.float32(self.train_mark[start:end]),
            )
        if self.flag == "val":
            return (
                np.float32(self.val[start:end]),
                np.float32(self.val_labels[start:end]),
                np.float32(self.val_mark[start:end]),
            )
        return (
            np.float32(self.test[start:end]),
            np.float32(self.test_labels[start:end]),
            np.float32(self.test_mark[start:end]),
        )
