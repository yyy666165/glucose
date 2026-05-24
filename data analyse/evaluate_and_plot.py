"""
多步长预测曲线: 真实时间线 + 30min/60min/120min 预测
- 单 x 轴（真实时间戳）
- 修复: delta_glucose 不泄漏未来值（仅用当前时刻的变化率）
- 修复: 索引对齐 (preds[k]→i+k+1)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.cuda.is_available = lambda: False

from finetune_manchester import (create_model, CONTROL_DIM, PID_TO_IDX)

# 加载模型
ckpt = torch.load('d:/ohio/data analyse/best_model_manchester_finetune_v1.pt',
                   map_location='cpu', weights_only=False)
state = ckpt['model_state_dict']
hidden_dim = ckpt.get('hidden_dim', 64)
model, _ = create_model(state, hidden_dim=hidden_dim, control_dim=CONTROL_DIM)
model.load_state_dict(state, strict=False)
model.eval()

# 测试集
test_csv = 'd:/ohio/data/manchester_test.csv'
raw_df = pd.read_csv(test_csv)
raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])

SEQ = 24
patients = [2301, 2401]

fig, axes = plt.subplots(len(patients), 1, figsize=(16, 4.5*len(patients)))
if len(patients) == 1:
    axes = [axes]

for pi, pid in enumerate(patients):
    ax = axes[pi]
    pdf = raw_df[raw_df['patient_id']==pid].sort_values('timestamp').reset_index(drop=True)
    n = len(pdf)
    print(f"Patient {pid}: {n} rows")
    if n < SEQ + 10:
        ax.set_title(f'Patient {pid} (insufficient data)'); continue

    ts = pdf['timestamp'].values
    glu = pdf['glucose_level'].values
    pid_idx = PID_TO_IDX.get(pid, 12)

    pred_30 = np.full(n, np.nan)
    pred_60 = np.full(n, np.nan)
    pred_120 = np.full(n, np.nan)

    def get_ctrl(row):
        return [row.get(c, 0) for c in
                ['IOB','COB','exercise_intensity','effective_basal_rate',
                 'recent_bolus_dose','heart_rate','ISF','delta_glucose']]

    def get_ctx(row):
        h = row['timestamp'].hour
        return np.array([
            np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24),
            row.get('sleep_quality',3), 1 if 4<=h<=6 else 0,
            row.get('glucose_zone',1)
        ], dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n - SEQ + 1, 6):  # step=6 平衡速度与密度
            i = start
            g0 = torch.tensor([[pdf.at[i, 'glucose_level']]], dtype=torch.float32)
            ctx = torch.tensor([get_ctx(pdf.iloc[i])], dtype=torch.float32)
            pid_t = torch.tensor([pid_idx], dtype=torch.long)

            # 当前时刻的 delta_glucose（唯一不泄漏的值）
            delta_g0 = pdf.at[i, 'delta_glucose'] if pd.notna(pdf.at[i, 'delta_glucose']) else 0

            ctrl = np.zeros((1, SEQ, 8))
            for j in range(SEQ):
                row = pdf.iloc[min(i + j, n - 1)]
                vec = get_ctrl(row)
                # 🔴 泄漏修复: 未来步的 delta_glucose 用当前时刻的值代替
                if j == 0:
                    vec[7] = delta_g0      # 当前步: 真实 delta_glucose
                else:
                    vec[7] = delta_g0      # 未来步: 用当前值近似，不泄漏
                ctrl[0, j] = vec

            ctrl_t = torch.tensor(ctrl, dtype=torch.float32)
            preds = model(g0, ctrl_t, ctx, pid_t).numpy()[0]  # (24,)

            # 正确索引: preds[k] 对应未来 k+1 步 = 时间点 i+k+1
            pos_30 = i + 6   # preds[5] → 30min → i+6
            pos_60 = i + 12  # preds[11] → 60min → i+12
            pos_120 = i + 24 # preds[23] → 120min → i+24

            if pos_30 < n:  pred_30[pos_30]  = preds[5]
            if pos_60 < n:  pred_60[pos_60]  = preds[11]
            if pos_120 < n: pred_120[pos_120] = preds[23]

    # ---- 绘图 (x轴: 小时, 从0开始) ----
    time_h = np.arange(n) * 5 / 60

    # 填充 NaN 使曲线连续
    pred_30_f = pd.Series(pred_30).interpolate(limit_area='inside').values

    ax.plot(time_h, glu, 'k-', linewidth=1.2, label='True CGM', alpha=0.85)
    ax.plot(time_h, pred_30_f, linewidth=1.5, alpha=0.9, color='#2196F3', label='Pred 30min')

    ax.axhline(70, color='green', ls=':', alpha=0.4)
    ax.axhline(180, color='orange', ls=':', alpha=0.4)

    v30 = ~np.isnan(pred_30)
    rmse30 = np.sqrt(np.mean((pred_30[v30]-glu[v30])**2)) if v30.any() else 0

    ax.text(0.98, 0.97, f'RMSE 30min: {rmse30:.1f} mg/dL',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel('Time (hours)')
    ax.set_title(f'Patient {pid} — Multi-horizon Prediction', fontsize=12)
    ax.set_ylabel('Glucose (mg/dL)')
    ax.legend(fontsize=8, ncol=4)
    ax.grid(alpha=0.2)
    ax.set_ylim(30, 450)

plt.tight_layout()
path = 'd:/ohio/data analyse/multi_horizon_prediction.png'
fig.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {path} ({os.path.getsize(path)/1024:.0f} KB)')
