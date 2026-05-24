"""
从零训练结构化生理模型 (Ohio + Manchester)
------------------------------------------
- 7 维控制: [bolus_event, meal_event, exercise, basal, HR, ISF, delta_G]
- 7 维 ODE 状态: [G, IOB_abs, IOB, COB_abs, COB, EX_fast, EX_slow]
- 包含: 两室 PK + 运动效应 + 自调节 + 患者个性化
"""
import sys, functools, os
print = functools.partial(print, flush=True)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.glucose_dataset import GlucoseDataset, CONTEXT_DIM
from model.neural_ode_glucose import NeuralODEGlucosePredictor

# Manchester 配置
OHIO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(OHIO_ROOT, 'data analyse'))
from finetune_manchester import ManchesterGlucoseDataset


def custom_loss(y_pred, y_true):
    """HH Loss + 自适应Focal + 生理边界惩罚 (去掉胰岛素正则, control是raw events)"""
    y_pred = torch.clamp(y_pred, min=0, max=600)
    y_true = torch.clamp(y_true, min=0, max=600)

    per_sample_mse = (y_pred - y_true) ** 2
    per_sample_error = torch.abs(y_pred - y_true)

    # HH Loss
    hh_weight = torch.ones_like(y_true)
    hh_weight = torch.where(y_true < 70, hh_weight * 3.0, hh_weight)
    hh_weight = torch.where(y_true > 250, hh_weight * 1.5, hh_weight)

    # 自适应Focal
    median_err = per_sample_error.detach().median().clamp(min=1.0)
    normalized_error = per_sample_error / median_err
    focal_weight = 1.0 + 0.5 * torch.clamp(normalized_error - 1.0, min=0.0, max=5.0)

    weighted_loss = hh_weight * focal_weight * per_sample_mse
    loss = weighted_loss.mean()

    # 生理边界惩罚
    below = torch.clamp(40 - y_pred, min=0) ** 2
    above = torch.clamp(y_pred - 400, min=0) ** 2
    boundary_penalty = (below.mean() + above.mean()) * 0.1

    return loss + boundary_penalty


def train_model():
    seq_len = 24
    control_dim = 7
    context_dim = CONTEXT_DIM
    hidden_dim = 64
    batch_size = 64
    num_epochs = 30
    learning_rate = 1e-3
    warmup_epochs = 3
    min_lr = 1e-5
    weight_decay = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"设备: {device}")
    print(f"结构化生理模型 | control_dim={control_dim}, hidden={hidden_dim}")
    print(f"超参: lr={learning_rate}, epochs={num_epochs}, batch={batch_size}, warmup={warmup_epochs}")

    # ─── 1. 加载数据 ───
    print("\n加载 Ohio 数据...")
    ohio_dataset = GlucoseDataset(data_dir=os.path.join(OHIO_ROOT, 'data'), seq_len=seq_len)
    ohio_train_size = len(ohio_dataset)
    ohio_test_dataset = ohio_dataset.get_test_dataset()
    print(f"  Ohio 训练窗口: {ohio_train_size}")

    print("\n加载 Manchester 数据...")
    man_train = ManchesterGlucoseDataset(
        os.path.join(OHIO_ROOT, 'data/manchester_train.csv'), seq_len=seq_len, split='train'
    )
    man_val = ManchesterGlucoseDataset(
        os.path.join(OHIO_ROOT, 'data/manchester_val.csv'), seq_len=seq_len, split='val'
    )
    print(f"  Manchester 训练窗口: {len(man_train)}, 验证窗口: {len(man_val)}")

    # 合并训练集 (shuffle=True 的 DataLoader 会自动混洗)
    combined_train = ConcatDataset([ohio_dataset, man_train])
    train_loader = DataLoader(combined_train, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(man_val, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"\n总训练样本: {len(combined_train):,}, 验证样本: {len(man_val):,}")
    print(f"每 epoch batch 数: ~{len(combined_train)//batch_size}")

    # ─── 2. 创建模型 ───
    print("\n从零初始化模型...")
    model = NeuralODEGlucosePredictor(
        context_dim=context_dim, hidden_dim=hidden_dim, control_dim=control_dim,
        num_patients=12 + 17, patient_embed_dim=16  # Ohio 12 + Manchester 17
    ).to(device).float()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total_params:,}")

    # ─── 3. 优化器 ───
    # 患者嵌入用较高学习率
    def get_param_groups(m, base_lr):
        groups = [
            {"params": [p for n, p in m.named_parameters()
                        if "patient_embedding" not in n], "lr": base_lr},
        ]
        embed_params = [p for n, p in m.named_parameters() if "patient_embedding" in n]
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

    # ─── 4. 训练循环 ───
    best_val_loss = float('inf')
    os.makedirs('checkpoints', exist_ok=True)

    # 自动编号 (用 structured 前缀不覆盖旧模型)
    existing = [f for f in os.listdir('checkpoints')
                if f.startswith('best_model_structured_v') and f.endswith('.pt')]
    version = max((int(f.split('_v')[1].split('.pt')[0]) for f in existing), default=0) + 1
    best_model_path = f'checkpoints/best_model_structured_v{version}.pt'
    print(f"模型保存: {best_model_path}")

    for epoch in range(num_epochs):
        t0 = time.time()

        # ── 训练 ──
        model.train()
        train_loss = 0.0
        train_plain_mse = 0.0
        train_batches = 0

        for batch in train_loader:
            g0 = batch['initial_glucose'].to(device).float()
            ctrl = batch['control_sequence'].to(device).float()
            ctx = batch['context'].to(device).float()
            target = batch['target'].to(device).float()
            pids = batch['patient_id'].to(device)

            optimizer.zero_grad()
            outputs = model(g0, ctrl, ctx, pids)
            loss = custom_loss(outputs, target)

            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                optimizer.zero_grad()
                continue
            optimizer.step()

            train_loss += loss.item()
            train_plain_mse += ((outputs - target) ** 2).mean().item()
            train_batches += 1

        avg_train_loss = train_loss / max(train_batches, 1)
        avg_train_mse = train_plain_mse / max(train_batches, 1)

        # ── 验证 ──
        model.eval()
        val_loss = 0.0
        val_plain_mse = 0.0
        val_mae = 0.0
        val_within20 = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                g0 = batch['initial_glucose'].to(device).float()
                ctrl = batch['control_sequence'].to(device).float()
                ctx = batch['context'].to(device).float()
                target = batch['target'].to(device).float()
                pids = batch['patient_id'].to(device)

                outputs = model(g0, ctrl, ctx, pids)
                loss = custom_loss(outputs, target)

                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                val_loss += loss.item()
                val_plain_mse += ((outputs - target) ** 2).mean().item()
                val_mae += torch.abs(outputs - target).mean().item()

                err_ratio = torch.abs(outputs - target) / torch.clamp(target, 40, None)
                val_within20 += (err_ratio <= 0.20).float().mean().item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        avg_val_mse = val_plain_mse / max(val_batches, 1)
        avg_val_rmse = np.sqrt(avg_val_mse)
        avg_val_mae = val_mae / max(val_batches, 1)
        avg_within20 = val_within20 / max(val_batches, 1) * 100
        elapsed = time.time() - t0

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # 打印新模型特有关键参数
        p_vals = model.ode_func
        print(f'Epoch {epoch+1}/{num_epochs} ({elapsed:.0f}s) | '
              f'LR: {current_lr:.1e} | '
              f'Train: composite={avg_train_loss:.0f} plainMSE={avg_train_mse:.0f} | '
              f'Val: composite={avg_val_loss:.0f} RMSE={avg_val_rmse:.1f} '
              f'MAE={avg_val_mae:.1f} Within20%={avg_within20:.1f}%')
        print(f'  Param: p_eff={p_vals.glucose_effectiveness.item():.4f} '
              f'G_b={p_vals.G_baseline.item():.1f} '
              f'ex_up={p_vals.exercise_uptake.item():.3f} '
              f'isf_b={p_vals.isf_boost_scale.item():.3f} '
              f'ex_f={p_vals.ex_fast_decay.item():.3f} '
              f'DA={p_vals.dynamics_amplitude.item():.2f} '
              f'IS={p_vals.insulin_scale.item():.2f}')

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_rmse': avg_val_rmse,
                'val_mae': avg_val_mae,
                'val_within20': avg_within20,
                'control_dim': control_dim,
                'context_dim': context_dim,
                'hidden_dim': hidden_dim,
                'num_patients': 29,
                'patient_embed_dim': 16,
            }, best_model_path)
            print(f'  → saved (val_loss={best_val_loss:.1f}, RMSE={avg_val_rmse:.2f})')

    print(f"\n训练完成! 最佳验证损失: {best_val_loss:.1f}")
    print(f"模型保存: {best_model_path}")


if __name__ == "__main__":
    train_model()
