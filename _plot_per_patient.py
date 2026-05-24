"""
逐患者预测曲线对比
- 30min 为一个图 (每步=15min, 30min=第2步)
- 60min 和 120min 为一个图
"""
import sys, os, functools
print = functools.partial(print, flush=True)

OHIO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, OHIO_ROOT)
sys.path.insert(0, os.path.join(OHIO_ROOT, 'data analyse'))

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model.neural_ode_glucose import NeuralODEGlucosePredictor
from finetune_manchester import ManchesterGlucoseDataset, PID_TO_IDX

CHECKPOINT = 'checkpoints/structured_final_v1_epoch6_RMSE33.7.pt'
SAVE_DIR = 'results/structured_model/per_patient'
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")

# 1. 加载模型
print("加载模型...")
ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
model = NeuralODEGlucosePredictor(
    context_dim=ckpt['context_dim'], hidden_dim=ckpt['hidden_dim'],
    control_dim=ckpt['control_dim'], num_patients=ckpt['num_patients'],
    patient_embed_dim=ckpt['patient_embed_dim']
).to(device).float()
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"  RMSE={ckpt.get('val_rmse', '?'):}")

# 2. 加载测试集
print("加载测试集...")
test_dataset = ManchesterGlucoseDataset(
    os.path.join(OHIO_ROOT, 'data/manchester_test.csv'), seq_len=24, split='test'
)
print(f"  测试样本: {len(test_dataset)}")

# 3. 按患者分组预测
print("逐样本预测...")
patient_preds = {}  # pid -> {'times': [...], 'true': [...], 'pred_30min': [...], 'pred_60min': [...], 'pred_120min': [...]}

with torch.no_grad():
    for idx in range(len(test_dataset)):
        sample = test_dataset[idx]
        # 获取原始患者ID
        actual_idx = test_dataset._valid_indices[idx]
        raw_pid = int(test_dataset.data.iloc[actual_idx]['patient_id'])

        g0 = sample['initial_glucose'].unsqueeze(0).to(device).float()
        ctrl = sample['control_sequence'].unsqueeze(0).to(device).float()
        ctx = sample['context'].unsqueeze(0).to(device).float()
        pid_t = sample['patient_id'].unsqueeze(0).to(device)
        target = sample['target'].numpy()

        pred = model(g0, ctrl, ctx, pid_t)[0].cpu().numpy()

        if raw_pid not in patient_preds:
            patient_preds[raw_pid] = {
                'true_30': [], 'pred_30': [],
                'true_60': [], 'pred_60': [],
                'true_120': [], 'pred_120': [],
                'n': 0,
            }

        # 30min = 第 2 步 (索引 1), 60min = 第 4 步 (索引 3), 120min = 第 8 步 (索引 7)
        # 但通常: 30min=step2(index1), 60min=step4(index3), 120min=step8(index7)
        # 实际上 1步=5min, 所以:
        #   30min = index 5 (第6步)
        #   60min = index 11 (第12步)
        #   120min = index 23 (第24步)
        pred_30 = pred[5]   # 5步×5min=25min, 约30min
        true_30 = target[5]
        pred_60 = pred[11]  # 11步×5min=55min, 约60min
        true_60 = target[11]
        pred_120 = pred[23]  # 23步×5min=115min, 约120min
        true_120 = target[23]

        patient_preds[raw_pid]['true_30'].append(true_30)
        patient_preds[raw_pid]['pred_30'].append(pred_30)
        patient_preds[raw_pid]['true_60'].append(true_60)
        patient_preds[raw_pid]['pred_60'].append(pred_60)
        patient_preds[raw_pid]['true_120'].append(true_120)
        patient_preds[raw_pid]['pred_120'].append(pred_120)
        patient_preds[raw_pid]['n'] += 1

        if (idx + 1) % 1000 == 0:
            print(f"  {idx+1}/{len(test_dataset)}")

# 4. 绘图
print("\n生成逐患者图表...")

all_pids = sorted(patient_preds.keys())
n_patients = len(all_pids)

# ---- 图1: 30min 预测 ----
fig_30, axes_30 = plt.subplots(5, 4, figsize=(20, 20))
axes_30 = axes_30.flatten()
for i, pid in enumerate(all_pids):
    if i >= 20:
        break
    ax = axes_30[i]
    d = patient_preds[pid]
    true_arr = np.array(d['true_30'])
    pred_arr = np.array(d['pred_30'])

    rmse = np.sqrt(np.mean((pred_arr - true_arr)**2))
    mae = np.mean(np.abs(pred_arr - true_arr))

    ax.scatter(true_arr, pred_arr, alpha=0.3, s=4, c='#2196F3', edgecolors='none')
    lims = [40, 400]
    ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=1)
    ax.axhline(y=70, color='r', linestyle=':', alpha=0.3)
    ax.axhline(y=180, color='orange', linestyle=':', alpha=0.3)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title(f'Participant {pid} (n={d["n"]})\nRMSE={rmse:.1f} MAE={mae:.1f}', fontsize=10)
    ax.set_xlabel('True', fontsize=8)
    ax.set_ylabel('Pred', fontsize=8)
    ax.grid(True, alpha=0.15)

for i in range(n_patients, 20):
    axes_30[i].axis('off')

fig_30.suptitle('Structured Model - 30min Prediction per Patient', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
path_30 = os.path.join(SAVE_DIR, 'per_patient_30min.png')
fig_30.savefig(path_30, dpi=150, bbox_inches='tight')
plt.close()
print(f"  30min: {path_30}")

# ---- 图2: 60min + 120min 对比 ----
n_cols = 4
n_rows = (n_patients + n_cols - 1) // n_cols
fig_60_120, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
axes = axes.flatten()

for i, pid in enumerate(all_pids):
    ax = axes[i]
    d = patient_preds[pid]
    true_60 = np.array(d['true_60'])
    pred_60 = np.array(d['pred_60'])
    true_120 = np.array(d['true_120'])
    pred_120 = np.array(d['pred_120'])

    rmse_60 = np.sqrt(np.mean((pred_60 - true_60)**2))
    rmse_120 = np.sqrt(np.mean((pred_120 - true_120)**2))

    ax.scatter(true_60, pred_60, alpha=0.25, s=3, c='#FF9800', edgecolors='none', label=f'60min')
    ax.scatter(true_120, pred_120, alpha=0.25, s=3, c='#F44336', edgecolors='none', label=f'120min')
    lims = [40, 400]
    ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=1)
    ax.axhline(y=70, color='g', linestyle=':', alpha=0.2)
    ax.axhline(y=180, color='orange', linestyle=':', alpha=0.2)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title(f'Participant {pid} (n={d["n"]})\n60min RMSE={rmse_60:.1f}  120min RMSE={rmse_120:.1f}', fontsize=10)
    ax.set_xlabel('True', fontsize=8)
    ax.set_ylabel('Pred', fontsize=8)
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.15)

for i in range(n_patients, len(axes)):
    axes[i].axis('off')

fig_60_120.suptitle('Structured Model - 60min vs 120min Prediction per Patient', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
path_60_120 = os.path.join(SAVE_DIR, 'per_patient_60min_120min.png')
fig_60_120.savefig(path_60_120, dpi=150, bbox_inches='tight')
plt.close()
print(f"  60+120min: {path_60_120}")

# ---- 图3: 每患者RMSE对比条形图 ----
fig_bar, ax = plt.subplots(figsize=(14, 6))
pids_display = []
rmse_30_list = []
rmse_60_list = []
rmse_120_list = []

for pid in all_pids:
    d = patient_preds[pid]
    if d['n'] < 10:
        continue
    rmse_30_list.append(np.sqrt(np.mean((np.array(d['pred_30']) - np.array(d['true_30']))**2)))
    rmse_60_list.append(np.sqrt(np.mean((np.array(d['pred_60']) - np.array(d['true_60']))**2)))
    rmse_120_list.append(np.sqrt(np.mean((np.array(d['pred_120']) - np.array(d['true_120']))**2)))
    pids_display.append(str(pid))

x = np.arange(len(pids_display))
w = 0.25
ax.bar(x - w, rmse_30_list, w, label='30min', alpha=0.8, color='#4CAF50')
ax.bar(x, rmse_60_list, w, label='60min', alpha=0.8, color='#FF9800')
ax.bar(x + w, rmse_120_list, w, label='120min', alpha=0.8, color='#F44336')
ax.set_xticks(x)
ax.set_xticklabels(pids_display, fontsize=9)
ax.set_ylabel('RMSE (mg/dL)')
ax.set_title('Per-Participant RMSE by Prediction Horizon')
ax.legend()
ax.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
path_bar = os.path.join(SAVE_DIR, 'per_patient_rmse_bar.png')
fig_bar.savefig(path_bar, dpi=150, bbox_inches='tight')
plt.close()
print(f"  RMSE柱状图: {path_bar}")

# 汇总打印
print(f"\n{'='*60}")
print(f"逐患者 RMSE 汇总 ({n_patients} 人)")
print(f"{'='*60}")
print(f"{'Patient':>10} {'n':>6} {'30min':>8} {'60min':>8} {'120min':>8}")
print('-' * 42)
for pid in all_pids:
    d = patient_preds[pid]
    r30 = np.sqrt(np.mean((np.array(d['pred_30']) - np.array(d['true_30']))**2))
    r60 = np.sqrt(np.mean((np.array(d['pred_60']) - np.array(d['true_60']))**2))
    r120 = np.sqrt(np.mean((np.array(d['pred_120']) - np.array(d['true_120']))**2))
    print(f"{pid:>10} {d['n']:>6} {r30:>7.1f}  {r60:>7.1f}  {r120:>7.1f}")

print(f"\n图表已保存: {SAVE_DIR}/")
