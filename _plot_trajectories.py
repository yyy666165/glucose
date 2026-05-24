"""
绘制患者 2301, 2310, 2304 的预测 vs 真实折线图
"""
import sys, os, functools
print = functools.partial(print, flush=True)

OHIO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, OHIO_ROOT)
sys.path.insert(0, os.path.join(OHIO_ROOT, 'data analyse'))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

from model.neural_ode_glucose import NeuralODEGlucosePredictor
from finetune_manchester import ManchesterGlucoseDataset

CHECKPOINT = 'checkpoints/structured_final_v1_epoch6_RMSE33.7.pt'
SAVE_DIR = 'results/structured_model'
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# 加载模型
ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
model = NeuralODEGlucosePredictor(
    context_dim=ckpt['context_dim'], hidden_dim=ckpt['hidden_dim'],
    control_dim=ckpt['control_dim'], num_patients=ckpt['num_patients'],
    patient_embed_dim=ckpt['patient_embed_dim']
).to(device).float()
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# 加载测试集
test_dataset = ManchesterGlucoseDataset(
    os.path.join(OHIO_ROOT, 'data/manchester_test.csv'), seq_len=24, split='test'
)

# 目标患者
target_pids = [2301, 2310, 2304]
t_axis = np.arange(24) * 5  # 0, 5, 10, ..., 115 分钟

fig, axes = plt.subplots(3, 1, figsize=(16, 12))

for plot_idx, target_pid in enumerate(target_pids):
    ax = axes[plot_idx]
    idx_map = {v: k for k, v in test_dataset.patient_id_map.items()}
    mapped_pid = test_dataset.patient_id_map.get(target_pid)

    # 收集该患者的样本
    all_true = []
    all_pred = []
    all_init = []

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            actual_idx = test_dataset._valid_indices[idx]
            raw_pid = int(test_dataset.data.iloc[actual_idx]['patient_id'])
            if raw_pid != target_pid:
                continue

            sample = test_dataset[idx]
            g0 = sample['initial_glucose'].unsqueeze(0).to(device).float()
            ctrl = sample['control_sequence'].unsqueeze(0).to(device).float()
            ctx = sample['context'].unsqueeze(0).to(device).float()
            pid_t = sample['patient_id'].unsqueeze(0).to(device)
            target = sample['target'].numpy()

            pred = model(g0, ctrl, ctx, pid_t)[0].cpu().numpy()

            all_true.append(target)
            all_pred.append(pred)
            all_init.append(sample['initial_glucose'].item())

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    n_samples = len(all_true)

    # 计算均值和标准差
    mean_true = np.mean(all_true, axis=0)
    mean_pred = np.mean(all_pred, axis=0)
    std_true = np.std(all_true, axis=0)
    std_pred = np.std(all_pred, axis=0)

    # 整体 RMSE
    rmse = np.sqrt(np.mean((all_pred - all_true)**2))
    mae = np.mean(np.abs(all_pred - all_true))

    # 绘制
    ax.plot(t_axis, mean_true, 'b-o', linewidth=2, markersize=4, label='Mean True', alpha=0.8)
    ax.fill_between(t_axis, mean_true - std_true, mean_true + std_true, alpha=0.15, color='blue', label='True ±1σ')
    ax.plot(t_axis, mean_pred, 'r--s', linewidth=2, markersize=4, label='Mean Pred', alpha=0.8)
    ax.fill_between(t_axis, mean_pred - std_pred, mean_pred + std_pred, alpha=0.12, color='red', label='Pred ±1σ')

    # 低血糖/高血糖阈值
    ax.axhline(y=70, color='green', linestyle=':', alpha=0.5, linewidth=1)
    ax.axhline(y=180, color='orange', linestyle=':', alpha=0.5, linewidth=1)
    ax.axhspan(70, 180, alpha=0.05, color='green')

    ax.set_xlabel('Time (minutes)', fontsize=12)
    ax.set_ylabel('Glucose (mg/dL)', fontsize=12)
    ax.set_title(f'Participant {target_pid}  —  n={n_samples} windows  |  RMSE={rmse:.1f}  MAE={mae:.1f}', fontsize=13)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, 115)
    ax.set_xticks(np.arange(0, 120, 15))

    # 分步 RMSE 标注
    step_rmse = np.sqrt(np.mean((all_pred - all_true)**2, axis=0))
    ax2 = ax.twinx()
    ax2.bar(t_axis, step_rmse, width=4, alpha=0.2, color='gray', label='Step RMSE')
    ax2.set_ylabel('Step RMSE (mg/dL)', fontsize=10, color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.set_ylim(0, max(step_rmse) * 3)

plt.suptitle('Structured Model: Predicted vs True Glucose Trajectories (Mean ± 1σ)', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
path = os.path.join(SAVE_DIR, 'trajectories_2301_2310_2304.png')
fig.savefig(path, dpi=150, bbox_inches='tight')
plt.close()

print(f"Saved: {path}")
print(f"\nPer-patient metrics:")
for pid in target_pids:
    idx_map = {v: k for k, v in test_dataset.patient_id_map.items()}
    # Collected stats already calculated above, just print from the loop
    # Re-collect to print
    all_t = []
    all_p = []
    with torch.no_grad():
        for idx in range(len(test_dataset)):
            actual_idx = test_dataset._valid_indices[idx]
            raw_pid = int(test_dataset.data.iloc[actual_idx]['patient_id'])
            if raw_pid != pid:
                continue
            sample = test_dataset[idx]
            g0 = sample['initial_glucose'].unsqueeze(0).to(device).float()
            ctrl = sample['control_sequence'].unsqueeze(0).to(device).float()
            ctx = sample['context'].unsqueeze(0).to(device).float()
            pid_t = sample['patient_id'].unsqueeze(0).to(device)
            pred = model(g0, ctrl, ctx, pid_t)[0].cpu().numpy()
            all_t.append(sample['target'].numpy())
            all_p.append(pred)
    all_t = np.array(all_t)
    all_p = np.array(all_p)
    print(f"  Patient {pid}: n={len(all_t)}, RMSE={np.sqrt(np.mean((all_p-all_t)**2)):.1f}, "
          f"MAE={np.mean(np.abs(all_p-all_t)):.1f}")
