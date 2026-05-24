import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# 控制向量维度: [IOB, COB, exercise_intensity, effective_basal_rate, recent_bolus_dose, heart_rate, ISF]
CONTROL_DIM = 7
CONTEXT_DIM = 5
NUM_PATIENTS = 12


class GlucoseDataset(Dataset):
    def __init__(self, data_dir='data', seq_len=24, context_dim=CONTEXT_DIM, test_ratio=0.2):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.context_dim = context_dim
        self.test_ratio = test_ratio

        self.load_data()
        self.split_dataset()

    def load_data(self):
        """加载特征工程后的数据（已包含所有特征）"""
        features_path = os.path.join(self.data_dir, 'glucose_features_simplified.csv')
        self.glucose_df = pd.read_csv(features_path)
        self.glucose_df['timestamp'] = pd.to_datetime(self.glucose_df['timestamp'])

        # 构建患者ID到索引的映射
        unique_pids = sorted(self.glucose_df['patient_id'].unique())
        self.patient_id_map = {int(pid): idx for idx, pid in enumerate(unique_pids)}

        # 按患者+时间排序，保证每个患者的时间序列连续
        self.glucose_df = self.glucose_df.sort_values(['patient_id', 'timestamp']).reset_index(drop=True)

        # 填充可能的NaN
        fill_defaults = {
            'IOB': 0, 'COB': 0, 'exercise_intensity': 0,
            'effective_basal_rate': 0, 'recent_bolus_dose': 0,
            'heart_rate': 70, 'ISF': 1.0,
            'basal_rate': 0, 'temp_basal_rate': 0,
            'basal_insulin_accumulated': 0, 'bolus_dose_3h': 0,
            'heart_rate_delta': 0, 'finger_stick_glucose': 0,
            'sleep_quality': 3, 'delta_glucose': 0,
            'glucose_acceleration': 0, 'prev_glucose': 0,
            'glucose_x_IOB': 0, 'glucose_x_bolus': 0,
            'IOB_x_ISF': 0, 'bolus_x_exercise': 0,
            'basal_x_bolus': 0, 'COB_x_IOB': 0,
            'heart_rate_x_IOB': 0,
        }
        for col, default in fill_defaults.items():
            if col in self.glucose_df.columns:
                self.glucose_df[col] = self.glucose_df[col].fillna(default)

    def get_control_vector(self, row):
        """提取7维控制向量: [bolus_event, meal_event, exercise_intensity, effective_basal_rate, heart_rate, ISF, delta_glucose]"""
        return [
            row.get('bolus_event', 0),
            row.get('meal_event', 0),
            row.get('exercise_intensity', 0),
            row.get('effective_basal_rate', 0),
            row.get('heart_rate', 70),
            row.get('ISF', 1.0),
            row.get('delta_glucose', 0),
        ]

    def get_context_features(self, row):
        """获取5维上下文特征: [hour_sin, hour_cos, sleep_quality, is_dawn, glucose_zone]"""
        hour = row['timestamp'].hour
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        sleep_quality = row.get('sleep_quality', 3)
        is_dawn = 1 if 4 <= hour <= 6 else 0
        glucose_zone = row.get('glucose_zone', 1)
        return np.array([hour_sin, hour_cos, sleep_quality, is_dawn, glucose_zone], dtype=np.float32)

    def split_dataset(self):
        """按患者+时间顺序划分数据集，确保窗口不跨越患者边界"""
        # 为每个患者单独按时间划分训练/测试集
        train_parts, test_parts = [], []
        for pid, group in self.glucose_df.groupby('patient_id'):
            group = group.sort_values('timestamp').reset_index(drop=True)
            split_idx = int(len(group) * (1 - self.test_ratio))
            train_parts.append(group.iloc[:split_idx])
            test_parts.append(group.iloc[split_idx:])

        self.train_data = pd.concat(train_parts, ignore_index=True)
        self.test_data = pd.concat(test_parts, ignore_index=True)

        # 预计算患者边界索引，防止滑窗跨患者
        self.train_patient_bounds = self._compute_patient_bounds(self.train_data)

    def _compute_patient_bounds(self, df):
        """计算每个患者的行范围，用于防止窗口跨患者"""
        bounds = []
        for pid, group in df.groupby('patient_id'):
            indices = group.index
            bounds.append((indices[0], indices[-1]))
        return bounds

    def _is_valid_window(self, start_idx):
        """检查从 start_idx 开始的窗口是否完整（不跨患者、不跨时间间隔>30min）"""
        for lo, hi in self.train_patient_bounds:
            if lo <= start_idx <= hi:
                end_idx = start_idx + self.seq_len
                # 窗口末尾不能超出该患者范围
                if end_idx > hi + 1:
                    return False
                # 检查窗口内时间间隔不超过30分钟（5分钟采样，允许最大30分钟间隔）
                timestamps = self.train_data.loc[start_idx:end_idx - 1, 'timestamp'].values
                diffs = np.diff(timestamps) / np.timedelta64(1, 'm')
                if np.any(diffs > 30):
                    return False
                return True
        return False

    def __len__(self):
        # 只计算有效窗口数（不跨患者、时间连续）
        if not hasattr(self, '_valid_indices'):
            self._valid_indices = [
                i for i in range(len(self.train_data) - self.seq_len)
                if self._is_valid_window(i)
            ]
        return len(self._valid_indices)

    def __getitem__(self, idx):
        # 映射到实际数据行（跳过跨患者的无效窗口）
        actual_idx = self._valid_indices[idx]
        current_row = self.train_data.iloc[actual_idx]
        initial_glucose = current_row['glucose_level']
        patient_id = self.patient_id_map[int(current_row['patient_id'])]

        control_sequence = []
        targets = []

        for i in range(self.seq_len):
            # 控制信号来自 t+i 时刻(与 y[i] 同行)，驱动 t+i → t+i+1 的变化
            ctrl_idx = actual_idx + i
            target_idx = actual_idx + i + 1
            if target_idx >= len(self.train_data):
                break
            control = self.get_control_vector(self.train_data.iloc[ctrl_idx])
            control_sequence.append(control)
            targets.append(self.train_data.iloc[target_idx]['glucose_level'])

        control_sequence = torch.tensor(control_sequence, dtype=torch.float32)  # (seq_len, 7)
        targets = torch.tensor(targets, dtype=torch.float32)
        initial_glucose = torch.tensor(initial_glucose, dtype=torch.float32).unsqueeze(0)

        context = self.get_context_features(current_row)
        context = torch.tensor(context, dtype=torch.float32)

        return {
            'initial_glucose': initial_glucose,
            'control_sequence': control_sequence,
            'context': context,
            'target': targets,
            'patient_id': torch.tensor(patient_id, dtype=torch.long)
        }

    def get_test_dataset(self):
        return TestGlucoseDataset(self.test_data, self.seq_len, self.context_dim, self.patient_id_map)


class TestGlucoseDataset(Dataset):
    def __init__(self, test_data, seq_len, context_dim, patient_id_map):
        self.test_data = test_data.reset_index(drop=True)
        self.seq_len = seq_len
        self.context_dim = context_dim
        self.patient_id_map = patient_id_map

        # 预计算患者边界和有效窗口
        self.patient_bounds = self._compute_patient_bounds(self.test_data)
        self._valid_indices = [
            i for i in range(len(self.test_data) - self.seq_len)
            if self._is_valid_window(i)
        ]

    def _compute_patient_bounds(self, df):
        bounds = []
        for _, group in df.groupby('patient_id'):
            indices = group.index
            bounds.append((indices[0], indices[-1]))
        return bounds

    def _is_valid_window(self, start_idx):
        for lo, hi in self.patient_bounds:
            if lo <= start_idx <= hi:
                end_idx = start_idx + self.seq_len
                if end_idx > hi + 1:
                    return False
                timestamps = self.test_data.loc[start_idx:end_idx - 1, 'timestamp'].values
                diffs = np.diff(timestamps) / np.timedelta64(1, 'm')
                if np.any(diffs > 30):
                    return False
                return True
        return False

    def __len__(self):
        return len(self._valid_indices)

    def __getitem__(self, idx):
        actual_idx = self._valid_indices[idx]
        current_row = self.test_data.iloc[actual_idx]
        initial_glucose = current_row['glucose_level']
        patient_id = self.patient_id_map[int(current_row['patient_id'])]

        control_sequence = []
        targets = []

        for i in range(self.seq_len):
            # 控制信号来自 t+i 时刻(与 y[i] 同行)，驱动 t+i → t+i+1 的变化
            ctrl_idx = actual_idx + i
            target_idx = actual_idx + i + 1
            if target_idx >= len(self.test_data):
                break
            control = self.get_control_vector(self.test_data.iloc[ctrl_idx])
            control_sequence.append(control)
            targets.append(self.test_data.iloc[target_idx]['glucose_level'])

        control_sequence = torch.tensor(control_sequence, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)
        initial_glucose = torch.tensor(initial_glucose, dtype=torch.float32).unsqueeze(0)

        context = self.get_context_features(current_row)
        context = torch.tensor(context, dtype=torch.float32)

        return {
            'initial_glucose': initial_glucose,
            'control_sequence': control_sequence,
            'context': context,
            'target': targets,
            'patient_id': torch.tensor(patient_id, dtype=torch.long)
        }

    def get_control_vector(self, row):
        return [
            row.get('IOB', 0),
            row.get('COB', 0),
            row.get('exercise_intensity', 0),
            row.get('effective_basal_rate', 0),
            row.get('recent_bolus_dose', 0),
            row.get('heart_rate', 70),
            row.get('ISF', 1.0),
        ]

    def get_context_features(self, row):
        hour = row['timestamp'].hour
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        sleep_quality = row.get('sleep_quality', 3)
        is_dawn = 1 if 4 <= hour <= 6 else 0
        glucose_zone = row.get('glucose_zone', 1)
        return np.array([hour_sin, hour_cos, sleep_quality, is_dawn, glucose_zone], dtype=np.float32)

    def get_test_dataloader(self):
        return DataLoader(self, batch_size=32, shuffle=False)


if __name__ == "__main__":
    dataset = GlucoseDataset(data_dir='data', seq_len=24, context_dim=CONTEXT_DIM)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    batch = next(iter(dataloader))
    print(f"initial_glucose: {batch['initial_glucose'].shape}")
    print(f"control_sequence: {batch['control_sequence'].shape}")
    print(f"context: {batch['context'].shape}")
    print(f"target: {batch['target'].shape}")
