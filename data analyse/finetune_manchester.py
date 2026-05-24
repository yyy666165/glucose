"""
Manchester 数据微调 Ohio 模型
================================
1. 加载 Ohio 预训练模型 (best_model_v2.pt)
2. 扩展患者嵌入层 (12→29) 支持 17 名 Manchester 患者
3. 使用 Manchester 数据继续训练 (低学习率微调)
4. 在测试集上评估性能
"""

import sys
import os
import functools
print = functools.partial(print, flush=True)

import warnings
warnings.filterwarnings("ignore")

# 添加 Ohio 项目根目录到路径
OHIO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, OHIO_ROOT)

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import time
from datetime import timedelta

from model.neural_ode_glucose import NeuralODEGlucosePredictor

# 常量（原定义在 glucose_dataset.py）
CONTROL_DIM = 7   # [bolus_event, meal_event, exercise_intensity, effective_basal_rate, heart_rate, ISF, delta_glucose]
CONTEXT_DIM = 5

# === 路径配置 ===
DATA_DIR = os.path.join(OHIO_ROOT, "data")
CHECKPOINT_DIR = os.path.join(OHIO_ROOT, "checkpoints")
MANCHESTER_TRAIN = os.path.join(DATA_DIR, "manchester_train.csv")
MANCHESTER_VAL = os.path.join(DATA_DIR, "manchester_val.csv")
MANCHESTER_TEST = os.path.join(DATA_DIR, "manchester_test.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))

# === 患者映射 ===
# Manchester PID → 新索引 (Ohio 用了 0-11, Manchester 从 12 开始)
MANCHESTER_PIDS = sorted([2301, 2302, 2303, 2304, 2305, 2306, 2307, 2308,
                           2309, 2310, 2313, 2314, 2320, 2401, 2403, 2404, 2405])
PID_TO_IDX = {pid: 12 + i for i, pid in enumerate(MANCHESTER_PIDS)}
NUM_MANCHESTER_PATIENTS = len(MANCHESTER_PIDS)
NEW_NUM_PATIENTS = 12 + NUM_MANCHESTER_PATIENTS  # 29


def remap_checkpoint_keys(state_dict):
    """
    将旧版 checkpoint 的 key 映射到新版模型:
    - ode_func.dynamics_net.0 → ode_func.dynamics_layer1
    - ode_func.dynamics_net.2 → ode_func.dynamics_layer2
    - ode_func.dynamics_net.4 → ode_func.dynamics_layer3
    - extreme_ode_func.correction_net.0 → extreme_ode_func.correction_layer1
    - extreme_ode_func.correction_net.2 → extreme_ode_func.correction_layer2
    - extreme_ode_func.correction_net.4 → extreme_ode_func.correction_layer3
    """
    mapping = {
        "ode_func.dynamics_net.0": "ode_func.dynamics_layer1",
        "ode_func.dynamics_net.2": "ode_func.dynamics_layer2",
        "ode_func.dynamics_net.4": "ode_func.dynamics_layer3",
        "extreme_ode_func.correction_net.0": "extreme_ode_func.correction_layer1",
        "extreme_ode_func.correction_net.2": "extreme_ode_func.correction_layer2",
        "extreme_ode_func.correction_net.4": "extreme_ode_func.correction_layer3",
    }

    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in mapping.items():
            if key == old_prefix + ".weight" or key == old_prefix + ".bias":
                new_key = new_prefix + key[len(old_prefix):]
                break
            if key.startswith(old_prefix + "."):
                new_key = new_prefix + key[len(old_prefix):]
                break
        remapped[new_key] = value
    return remapped


def extend_checkpoint(checkpoint_path):
    """
    加载 checkpoint 并兼容处理：
    - 重命名旧版 key → 新版 key
    - 不支持患者个性化时添加警告
    """
    print(f"加载 checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    state = remap_checkpoint_keys(checkpoint["model_state_dict"])

    # 检查是否有患者个性化参数
    has_patient = any("patient" in k for k in state)
    if not has_patient:
        print("  checkpoint 不含患者个性化参数 (将在新模型中随机初始化)")
    else:
        embed_key = "patient_personalization.patient_embedding.weight"
        if embed_key in state:
            old_embed = state[embed_key]
            embed_mean = old_embed.mean(dim=0, keepdim=True)
            embed_std = old_embed.std(dim=0, keepdim=True)
            new_rows = embed_mean + torch.randn(17, 16) * embed_std * 0.1
            new_embed = torch.cat([old_embed, new_rows], dim=0)
            state[embed_key] = new_embed
            print(f"  患者嵌入已扩展: {old_embed.shape} → {new_embed.shape}")

    return state, checkpoint.get("hidden_dim", 64), checkpoint.get("control_dim", 7)


def _migrate_control_dim(state_dict, old_dim=7, new_dim=8):
    """将旧版 7-dim 控制权重重映射到 8-dim（新增 delta_glucose 列）"""
    migrated = {}
    for key, value in state_dict.items():
        # dynamics/correction 第一层: weight shape [H, 1+dim+H] → 需在 pos 1+old_dim 处插一列
        if key in ("ode_func.dynamics_layer1.weight",
                    "extreme_ode_func.correction_layer1.weight"):
            # value shape: [hidden, 1 + old_dim + hidden]
            h_dim = value.shape[0]
            new_weight = value.new_zeros(h_dim, 1 + new_dim + h_dim)
            # 复制旧权重列
            new_weight[:, :1 + old_dim] = value[:, :1 + old_dim]
            new_weight[:, 1 + new_dim:] = value[:, 1 + old_dim:]  # theta 部分
            # 新列 (index 1+old_dim) 初始化为 0，不干扰现有行为
            migrated[key] = new_weight
            print(f"    迁移 {key}: {list(value.shape)} → {list(new_weight.shape)} (插 delta_glucose 列)")
        elif key.endswith("trend_gate_net.0.weight"):
            # trend_gate_net 只接收 control，shape [H, old_dim] → [H, new_dim]
            h_dim = value.shape[0]
            new_weight = value.new_zeros(h_dim, new_dim)
            new_weight[:, :old_dim] = value
            migrated[key] = new_weight
            print(f"    迁移 {key}: {list(value.shape)} → {list(new_weight.shape)}")
        else:
            migrated[key] = value
    return migrated


def create_model(state_dict, context_dim=5, hidden_dim=64,
                  control_dim=8, num_patients=29, patient_embed_dim=16):
    """从加载的状态创建模型（自动迁移 7→8 维控制向量）"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"创建设备: {device}")

    # 检测是否是旧版 7-dim checkpoint，自动迁移
    old_key = "ode_func.dynamics_layer1.weight"
    if old_key in state_dict:
        old_cols = state_dict[old_key].shape[1]
        expected_cols = 1 + control_dim + hidden_dim  # 1+8+64=73
        if old_cols < expected_cols:
            old_dim = old_cols - 1 - hidden_dim  # 72-1-64=7
            print(f"  检测到旧版 {old_dim}→{control_dim} 维控制向量，自动迁移权重...")
            state_dict = _migrate_control_dim(state_dict, old_dim, control_dim)

    model = NeuralODEGlucosePredictor(
        context_dim=context_dim, hidden_dim=hidden_dim, control_dim=control_dim,
        num_patients=num_patients, patient_embed_dim=patient_embed_dim
    ).to(device).float()

    # 加载权重 (允许患者个性化参数缺失)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        non_patient_missing = [k for k in missing if "patient" not in k.lower()]
        if non_patient_missing:
            print(f"  缺失参数: {non_patient_missing[:5]}...")
        if len(missing) > len(non_patient_missing):
            print(f"  患者个性化参数将随机初始化 ({len(missing) - len(non_patient_missing)} 个)")
    if unexpected:
        print(f"  意外键: {unexpected[:3]}...")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {total_params}")
    return model, device


# === Manchester 数据集 ===
class ManchesterGlucoseDataset(Dataset):
    """Manchester 数据的 Dataset 封装，兼容 Ohio 训练代码"""

    def __init__(self, data_path, seq_len=24, split="train", patient_id_map=None):
        self.data_path = data_path
        self.seq_len = seq_len
        self.split = split
        self.patient_id_map = patient_id_map or PID_TO_IDX

        self.load_data()
        self._compute_patient_bounds()
        self._precompute_valid_indices()

    def load_data(self):
        df = pd.read_csv(self.data_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        # 确保按患者+时间排序
        df = df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)

        # 填充 NaN，与 Ohio 原始逻辑一致
        fill_defaults = {
            "IOB": 0, "COB": 0, "exercise_intensity": 0,
            "effective_basal_rate": 0, "recent_bolus_dose": 0,
            "heart_rate": 70, "ISF": 1.0,
            "basal_rate": 0, "temp_basal_rate": 0,
            "basal_insulin_accumulated": 0, "bolus_dose_3h": 0,
            "heart_rate_delta": 0, "finger_stick_glucose": 0,
            "sleep_quality": 3, "delta_glucose": 0,
            "glucose_acceleration": 0, "prev_glucose": 0,
            "glucose_x_IOB": 0, "glucose_x_bolus": 0,
            "IOB_x_ISF": 0, "bolus_x_exercise": 0,
            "basal_x_bolus": 0, "COB_x_IOB": 0,
            "heart_rate_x_IOB": 0,
        }
        for col, default in fill_defaults.items():
            if col in df.columns:
                df[col] = df[col].fillna(default)

        self.data = df
        print(f"[{self.split}] 加载 {len(df)} 行, {df['patient_id'].nunique()} 名患者")

    def _compute_patient_bounds(self):
        """计算每位患者在 DataFrame 中的行范围"""
        bounds = []
        for pid, group in self.data.groupby("patient_id"):
            indices = group.index
            bounds.append((indices[0], indices[-1]))
        self.patient_bounds = bounds

    def _is_valid_window(self, start_idx):
        """检查窗口：不跨患者、时间间隔 ≤30min"""
        for lo, hi in self.patient_bounds:
            if lo <= start_idx <= hi:
                end_idx = start_idx + self.seq_len
                if end_idx > hi + 1:
                    return False
                timestamps = self.data.loc[start_idx:end_idx - 1, "timestamp"].values
                diffs = np.diff(timestamps) / np.timedelta64(1, "m")
                if np.any(diffs > 30):
                    return False
                return True
        return False

    def _precompute_valid_indices(self):
        self._valid_indices = [
            i for i in range(len(self.data) - self.seq_len)
            if self._is_valid_window(i)
        ]
        print(f"[{self.split}] 有效窗口数: {len(self._valid_indices)}")

    def __len__(self):
        return len(self._valid_indices)

    def get_control_vector(self, row):
        """提取7维控制向量: [bolus_event, meal_event, exercise_intensity, effective_basal_rate, heart_rate, ISF, delta_glucose]
           模型内部会用 PK 动力学计算 IOB/COB (不再传预计算的 IOB/COB)"""
        return [
            row.get("bolus_event", 0),         # 原始注射事件（模型自算IOB）
            row.get("meal_event", 0),           # 原始进餐事件（模型自算COB）
            row.get("exercise_intensity", 0),
            row.get("effective_basal_rate", 0),
            row.get("heart_rate", 70),
            row.get("ISF", 1.0),
            row.get("delta_glucose", 0),
        ]

    def get_context_features(self, row):
        hour = row["timestamp"].hour
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        sleep_quality = row.get("sleep_quality", 3)
        is_dawn = 1 if 4 <= hour <= 6 else 0
        glucose_zone = row.get("glucose_zone", 1)
        return np.array([hour_sin, hour_cos, sleep_quality, is_dawn, glucose_zone], dtype=np.float32)

    def __getitem__(self, idx):
        actual_idx = self._valid_indices[idx]
        current_row = self.data.iloc[actual_idx]
        initial_glucose = current_row["glucose_level"]
        raw_pid = int(current_row["patient_id"])
        patient_id = self.patient_id_map.get(raw_pid, 12)  # 默认映射到 12

        control_sequence = []
        targets = []

        # 当前时刻 delta_glucose (唯一不泄漏的值，未来步都复用此值)
        initial_delta = current_row.get("delta_glucose", 0)

        for i in range(self.seq_len):
            ctrl_idx = actual_idx + i
            target_idx = actual_idx + i + 1
            if target_idx >= len(self.data):
                break
            control = self.get_control_vector(self.data.iloc[ctrl_idx])
            # delta_glucose 泄漏修复: 未来步全用当前时刻值
            if i > 0:
                control[6] = initial_delta
            control_sequence.append(control)
            targets.append(self.data.iloc[target_idx]["glucose_level"])

        control_sequence = torch.tensor(control_sequence, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)
        initial_glucose = torch.tensor(initial_glucose, dtype=torch.float32).unsqueeze(0)
        context = torch.tensor(self.get_context_features(current_row), dtype=torch.float32)

        return {
            "initial_glucose": initial_glucose,
            "control_sequence": control_sequence,
            "context": context,
            "target": targets,
            "patient_id": torch.tensor(patient_id, dtype=torch.long),
        }


# === 损失函数（与原始训练保持一致）===
def custom_loss(y_pred, y_true, control_sequence=None):
    y_pred = torch.clamp(y_pred, min=0, max=600)
    y_true = torch.clamp(y_true, min=0, max=600)

    per_sample_mse = (y_pred - y_true) ** 2
    per_sample_error = torch.abs(y_pred - y_true)

    # HH Loss
    hh_weight = torch.ones_like(y_true)
    hh_weight = torch.where(y_true < 70, hh_weight * 3.0, hh_weight)
    hh_weight = torch.where(y_true > 250, hh_weight * 1.5, hh_weight)

    # 自适应 Focal
    median_err = per_sample_error.detach().median().clamp(min=1.0)
    normalized_error = per_sample_error / median_err
    focal_weight = 1.0 + 0.5 * torch.clamp(normalized_error - 1.0, min=0.0, max=5.0)

    weighted_loss = hh_weight * focal_weight * per_sample_mse
    loss = weighted_loss.mean()

    # 生理边界惩罚
    below = torch.clamp(40 - y_pred, min=0) ** 2
    above = torch.clamp(y_pred - 400, min=0) ** 2
    boundary_penalty = (below.mean() + above.mean()) * 0.1
    loss = loss + boundary_penalty

    return loss


# === 评估函数 ===
def evaluate_model(model, dataloader, device):
    """在 dataloader 上评估模型性能，返回多种指标（含纯MSE对比）"""
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0
    total_plain_mse = 0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            initial_glucose = batch["initial_glucose"].to(device).float()
            control_sequence = batch["control_sequence"].to(device).float()
            context = batch["context"].to(device).float()
            target = batch["target"].to(device).float()
            patient_ids = batch["patient_id"].to(device)

            outputs = model(initial_glucose, control_sequence, context, patient_ids)
            loss = custom_loss(outputs, target, control_sequence)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            total_loss += loss.item()
            total_plain_mse += ((outputs - target) ** 2).mean().item()
            n_batches += 1
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)
    avg_plain_mse = total_plain_mse / max(n_batches, 1)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # 计算指标
    mse = np.mean((all_preds - all_targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(all_preds - all_targets))
    mard = np.mean(np.abs(all_preds - all_targets) / np.clip(all_targets, 40, None)) * 100
    within_20 = np.mean(np.abs(all_preds - all_targets) / np.clip(all_targets, 40, None) <= 0.20) * 100

    # 分段 RMSE（按预测步长）
    step_rmse = []
    for step in range(all_preds.shape[1]):
        step_rmse.append(np.sqrt(np.mean((all_preds[:, step] - all_targets[:, step]) ** 2)))

    return {
        "loss": avg_loss,
        "plain_mse": avg_plain_mse,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mard": mard,
        "within_20_pct": within_20,
        "step_rmse": step_rmse,
    }


def print_metrics(prefix, metrics):
    print(f"{prefix} | Loss: {metrics['loss']:.4f} | "
          f"RMSE: {metrics['rmse']:.2f} | MAE: {metrics['mae']:.2f} | "
          f"MARD: {metrics['mard']:.2f}% | Within20%: {metrics['within_20_pct']:.1f}%")


# === 主微调函数 ===
def finetune():
    # 超参数（微调使用更低学习率）
    seq_len = 24
    batch_size = 32
    num_epochs = 15
    learning_rate = 3e-4        # 原始 1e-3 的 1/3
    min_lr = 1e-5
    warmup_epochs = 2
    hidden_dim = 64
    control_dim = CONTROL_DIM  # 8
    context_dim = 5
    patient_embed_dim = 16
    weight_decay = 5e-5

    # 使用最新的 Ohio 底座 (v4 = 单分支, 8轮, 有患者个性化)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model_v4.pt")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model_v3.pt")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model_v2.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"未找到 checkpoint: {checkpoint_path}")

    print("=" * 60)
    print("Manchester 微调 Ohio 模型")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"超参: lr={learning_rate}, epochs={num_epochs}, batch={batch_size}, warmup={warmup_epochs}")
    print(f"患者嵌入扩展: 12 → {NEW_NUM_PATIENTS}")
    print()

    # 1. 加载并兼容处理 checkpoint
    state_dict, loaded_hidden, loaded_control = extend_checkpoint(checkpoint_path)
    hidden_dim = loaded_hidden or hidden_dim
    control_dim = CONTROL_DIM  # Manchester 总是 8 维 (含 delta_glucose)
    print(f"  模型配置: hidden_dim={hidden_dim}, control_dim={control_dim}")

    # 2. 创建模型
    model, device = create_model(
        state_dict, context_dim=context_dim, hidden_dim=hidden_dim,
        control_dim=control_dim, num_patients=NEW_NUM_PATIENTS,
        patient_embed_dim=patient_embed_dim,
    )

    # 3. 加载数据
    print("\n加载 Manchester 数据...")
    train_dataset = ManchesterGlucoseDataset(MANCHESTER_TRAIN, seq_len=seq_len, split="train")
    val_dataset = ManchesterGlucoseDataset(MANCHESTER_VAL, seq_len=seq_len, split="val")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    print(f"\n训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")

    # 4. 优化器
    # 患者嵌入参数用稍高学习率（如果有的话）
    def get_param_groups(model, base_lr):
        groups = [
            {"params": [p for n, p in model.named_parameters()
                        if "patient_embedding" not in n], "lr": base_lr},
        ]
        embed_params = [p for n, p in model.named_parameters() if "patient_embedding" in n]
        if embed_params:
            groups.append({"params": embed_params, "lr": base_lr * 3})
        return groups

    param_groups = get_param_groups(model, learning_rate)
    optimizer = optim.AdamW(param_groups, lr=learning_rate, weight_decay=weight_decay)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(num_epochs - warmup_epochs, 1)
        return max(min_lr / learning_rate, 0.5 * (1.0 + np.cos(np.pi * progress)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # 5. 训练循环
    best_val_loss = float("inf")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    # 自动编号: 不覆盖上一次训练的最佳参数
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    existing = [f for f in os.listdir(OUTPUT_DIR)
                if f.startswith("best_model_manchester_finetune_v") and f.endswith(".pt")]
    version = max((int(f.split("_v")[1].split(".pt")[0]) for f in existing), default=0) + 1
    best_model_path = os.path.join(OUTPUT_DIR, f"best_model_manchester_finetune_v{version}.pt")
    print(f"本次最佳模型将保存为: {best_model_path}")

    print(f"\n{'='*60}")
    print(f"开始微调 ({num_epochs} 轮)")
    print(f"{'='*60}")

    for epoch in range(num_epochs):
        t0 = time.time()

        # --- 训练 ---
        model.train()
        train_loss = 0.0
        train_plain_mse = 0.0
        train_batches = 0

        for batch in train_loader:
            initial_glucose = batch["initial_glucose"].to(device).float()
            control_sequence = batch["control_sequence"].to(device).float()
            context = batch["context"].to(device).float()
            target = batch["target"].to(device).float()
            patient_ids = batch["patient_id"].to(device)

            optimizer.zero_grad()
            outputs = model(initial_glucose, control_sequence, context, patient_ids)
            loss = custom_loss(outputs, target, control_sequence)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  警告: NaN/Inf 损失，跳过 batch")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                optimizer.zero_grad()
                continue

            optimizer.step()

            with torch.no_grad():
                plain_mse = ((outputs - target) ** 2).mean().item()

            train_loss += loss.item()
            train_plain_mse += plain_mse
            train_batches += 1

        avg_train_loss = train_loss / max(train_batches, 1)
        avg_train_plain_mse = train_plain_mse / max(train_batches, 1)

        # --- 验证 ---
        val_metrics = evaluate_model(model, val_loader, device)
        avg_val_loss = val_metrics["loss"]
        elapsed = time.time() - t0

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch+1}/{num_epochs} ({elapsed:.1f}s) | "
              f"LR: {current_lr:.1e} | "
              f"Train: composite={avg_train_loss:.1f} plainMSE={avg_train_plain_mse:.1f} | "
              f"Val: composite={avg_val_loss:.1f} plainMSE={val_metrics['plain_mse']:.1f} "
              f"RMSE={val_metrics['rmse']:.1f} MAE={val_metrics['mae']:.1f}")

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "val_rmse": val_metrics["rmse"],
                "val_mae": val_metrics["mae"],
                "num_patients": NEW_NUM_PATIENTS,
                "control_dim": control_dim,
                "context_dim": context_dim,
                "hidden_dim": hidden_dim,
                "patient_embed_dim": patient_embed_dim,
                "source_checkpoint": os.path.basename(checkpoint_path),
            }, best_model_path)
            print(f"  → 保存最佳模型 (val_loss={best_val_loss:.4f})")

    print(f"\n微调完成! 最佳验证损失: {best_val_loss:.4f}")
    print(f"模型保存至: {best_model_path}")

    # === 最终测试集评估 ===
    print(f"\n{'='*60}")
    print("最终测试集评估")
    print(f"{'='*60}")

    # 加载最佳模型
    try:
        best_ckpt = torch.load(best_model_path, map_location="cpu", weights_only=True)
    except Exception:
        best_ckpt = torch.load(best_model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.to(device)

    test_dataset = ManchesterGlucoseDataset(MANCHESTER_TEST, seq_len=seq_len, split="test")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_metrics = evaluate_model(model, test_loader, device)

    print_metrics("  测试集", test_metrics)

    # 分段输出
    print(f"\n  分步 RMSE (每步 = 5分钟, 共{seq_len}步 = {seq_len*5}分钟):")
    for step, rmse in enumerate(test_metrics["step_rmse"]):
        minute = (step + 1) * 5
        print(f"    {minute}min: RMSE = {rmse:.2f} mg/dL", end="")
        if (step + 1) % 6 == 0:
            print()

    print(f"\n  平均: RMSE={test_metrics['rmse']:.2f}, MAE={test_metrics['mae']:.2f}, "
          f"MARD={test_metrics['mard']:.2f}%, Within20%={test_metrics['within_20_pct']:.1f}%")

    # === 绘制预测曲线对比图 ===
    try:
        plot_prediction_curves(model, test_dataset, device, seq_len, OUTPUT_DIR)
    except Exception as e:
        print(f"  绘图跳过 (需要 matplotlib): {e}")

    return model, test_metrics


def plot_prediction_curves(model, dataset, device, seq_len, save_dir, n_samples=4):
    """绘制预测曲线 vs 真实值曲线对比图"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    model.eval()
    indices = np.linspace(0, len(dataset) - 1, n_samples, dtype=int)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Manchester 微调: 预测曲线 vs 真实值曲线", fontsize=14, fontweight='bold')
    axes = axes.flatten()

    all_steps_rmse = []

    for plot_idx, data_idx in enumerate(indices):
        sample = dataset[data_idx]
        pid = int(sample["patient_id"])

        with torch.no_grad():
            g = sample["initial_glucose"].unsqueeze(0).to(device)
            c = sample["control_sequence"].unsqueeze(0).to(device)
            ctx = sample["context"].unsqueeze(0).to(device)
            pid_t = sample["patient_id"].unsqueeze(0).to(device)
            pred = model(g, c, ctx, pid_t)[0].cpu().numpy()

        true = sample["target"].numpy()
        t_axis = np.arange(1, seq_len + 1) * 5  # 分钟

        ax = axes[plot_idx]
        ax.plot(t_axis, true, 'b-o', label='真实值', markersize=3, linewidth=1.5)
        ax.plot(t_axis, pred, 'r--x', label='预测值', markersize=3, linewidth=1.5)
        ax.fill_between(t_axis, true, pred, alpha=0.15, color='gray')
        ax.axhline(y=70, color='green', linestyle=':', alpha=0.5, label='低血糖 (70)')
        ax.axhline(y=180, color='orange', linestyle=':', alpha=0.5, label='高血糖 (180)')
        ax.set_xlabel('时间 (分钟)')
        ax.set_ylabel('血糖 (mg/dL)')
        ax.set_title(f'患者 {dataset.data.iloc[data_idx]["patient_id"]}  (PID映射={pid})', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, seq_len * 5 + 5)
        ax.set_ylim(min(40, min(true.min(), pred.min()) - 10),
                    max(400, max(true.max(), pred.max()) + 10))

        step_rmse = np.sqrt(np.mean((pred - true) ** 2, axis=0))
        ax.text(0.02, 0.98, f'RMSE={step_rmse:.1f}', transform=ax.transAxes,
                va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        all_steps_rmse.append(pred - true)

    plt.tight_layout()
    path = os.path.join(save_dir, "prediction_curves.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n预测曲线对比图已保存: {path}")


if __name__ == "__main__":
    model, test_metrics = finetune()
