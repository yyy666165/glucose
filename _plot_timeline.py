"""
多天连续血糖预测对比图
- 真实 CGM 曲线
- 30min / 60min / 120min 预测曲线叠加
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
import matplotlib.dates as mdates
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

from model.neural_ode_glucose import NeuralODEGlucosePredictor
from finetune_manchester import PID_TO_IDX

CHECKPOINT = 'checkpoints/best_model_structured_v2.pt'
SAVE_DIR = 'results/structured_model'
os.makedirs(SAVE_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {device}")
print("Loading model...")
ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
model = NeuralODEGlucosePredictor(
    context_dim=ckpt['context_dim'], hidden_dim=ckpt['hidden_dim'],
    control_dim=ckpt['control_dim'], num_patients=ckpt['num_patients'],
    patient_embed_dim=ckpt['patient_embed_dim']
).to(device).float()
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# 加载原始CSV（不是滑窗dataset，是原始时间线）
SEQ = 24
STEP = 6  # 每30分钟预测一次

# 对目标患者做预测
TARGET_PIDS = [2301, 2310, 2304]

fig, axes = plt.subplots(len(TARGET_PIDS), 1, figsize=(24, 5 * len(TARGET_PIDS)))

for plot_idx, pid in enumerate(TARGET_PIDS):
    ax = axes[plot_idx]
    print(f"\nProcessing participant {pid}...")

    csv_path = os.path.join(OHIO_ROOT, 'data/manchester_test.csv')
    raw = pd.read_csv(csv_path)
    raw['timestamp'] = pd.to_datetime(raw['timestamp'])
    pdf = raw[raw['patient_id'] == pid].sort_values('timestamp').reset_index(drop=True)

    ts = pdf['timestamp'].values
    glu = pdf['glucose_level'].values
    n = len(pdf)
    print(f"  {n} rows, {pd.Timestamp(ts[0]).date()} ~ {pd.Timestamp(ts[-1]).date()}")

    pid_idx = PID_TO_IDX.get(pid, 12)

    def get_ctrl(row):
        return [
            row.get('bolus_event', 0), row.get('meal_event', 0),
            row.get('exercise_intensity', 0), row.get('effective_basal_rate', 0),
            row.get('heart_rate', 70), row.get('ISF', 1.0),
            row.get('delta_glucose', 0),
        ]

    def get_ctx(row):
        h = row['timestamp'].hour
        return np.array([
            np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
            row.get('sleep_quality', 3), 1 if 4 <= h <= 6 else 0,
            row.get('glucose_zone', 1),
        ], dtype=np.float32)

    # 初始化预测数组
    pred_30 = np.full(n, np.nan)
    pred_60 = np.full(n, np.nan)
    pred_120 = np.full(n, np.nan)

    with torch.no_grad():
        for start in range(0, n - SEQ, STEP):
            i = start
            g0 = torch.tensor([[pdf.at[i, 'glucose_level']]], dtype=torch.float32).to(device)
            ctx = torch.tensor([get_ctx(pdf.iloc[i])], dtype=torch.float32).to(device)
            pid_t = torch.tensor([pid_idx], dtype=torch.long).to(device)

            ctrl = np.zeros((1, SEQ, 7))
            delta_g0 = pdf.at[i, 'delta_glucose'] if pd.notna(pdf.at[i, 'delta_glucose']) else 0
            for j in range(SEQ):
                row = pdf.iloc[min(i + j, n - 1)]
                vec = get_ctrl(row)
                vec[6] = delta_g0  # 防泄漏
                ctrl[0, j] = vec

            ctrl_t = torch.tensor(ctrl, dtype=torch.float32).to(device)
            preds = model(g0, ctrl_t, ctx, pid_t)[0].cpu().numpy()

            # 30min=第6步, 60min=第12步, 120min=第24步
            p30 = i + 6  # index 5
            p60 = i + 12  # index 11
            p120 = i + 24  # index 23
            if p30 < n: pred_30[p30] = preds[5]
            if p60 < n: pred_60[p60] = preds[11]
            if p120 < n: pred_120[p120] = preds[23]

    # --- 绘制 ---
    time_h = pd.to_datetime(ts)
    valid_30 = ~np.isnan(pred_30)
    valid_60 = ~np.isnan(pred_60)
    valid_120 = ~np.isnan(pred_120)

    r30 = np.sqrt(np.mean((pred_30[valid_30] - glu[valid_30])**2))
    r60 = np.sqrt(np.mean((pred_60[valid_60] - glu[valid_60])**2))
    r120 = np.sqrt(np.mean((pred_120[valid_120] - glu[valid_120])**2))

    ax.plot(time_h, glu, 'k-', linewidth=1.0, alpha=0.7, label='True CGM')
    ax.plot(time_h[valid_30], pred_30[valid_30], linewidth=1.0, alpha=0.9, label=f'30min pred (RMSE={r30:.1f})', color='#2196F3')
    ax.plot(time_h[valid_60], pred_60[valid_60], linewidth=1.0, alpha=0.9, label=f'60min pred (RMSE={r60:.1f})', color='#FF9800')
    ax.plot(time_h[valid_120], pred_120[valid_120], linewidth=1.2, alpha=0.9, label=f'120min pred (RMSE={r120:.1f})', color='#F44336')

    ax.axhline(70, color='green', ls=':', alpha=0.4)
    ax.axhline(180, color='orange', ls=':', alpha=0.4)
    ax.fill_between([time_h[0], time_h[-1]], 70, 180, alpha=0.05, color='green')

    ax.set_ylabel('Glucose (mg/dL)', fontsize=11)
    ax.set_title(f'Participant {pid} ({n} points, {int(n*5/60/24)} days)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, ncol=5)
    ax.grid(alpha=0.15)
    ax.set_ylim(40, 400)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=8)

plt.suptitle('Blood Glucose Prediction: True CGM vs Multi-Horizon Forecasts', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
path = os.path.join(SAVE_DIR, 'timeline_prediction_v2.png')
fig.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {path} ({os.path.getsize(path)/1024:.0f} KB)")
