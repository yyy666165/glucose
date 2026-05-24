"""验证 demo.ipynb 中的模型加载和预测代码"""
import importlib
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 强制重新导入模块
import model.neural_ode_glucose
importlib.reload(model.neural_ode_glucose)
from model.neural_ode_glucose import NeuralODEGlucosePredictor

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 从checkpoint加载最佳模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('checkpoints/best_model.pt', map_location=device, weights_only=False)
state_dict = checkpoint['model_state_dict']

# 从权重形状推断维度
hidden_dim = state_dict['context_encoder.0.weight'].shape[0]
context_dim = state_dict['context_encoder.0.weight'].shape[1]
control_dim = state_dict['ode_func.dynamics_net.0.weight'].shape[1] - 1 - hidden_dim

print(f'从state_dict推断: hidden_dim={hidden_dim}, context_dim={context_dim}, control_dim={control_dim}')

model = NeuralODEGlucosePredictor(
    context_dim=context_dim,
    hidden_dim=hidden_dim,
    control_dim=control_dim,
).to(device)
model.load_state_dict(state_dict)
model.eval()
print(f'模型加载完成 (epoch {checkpoint["epoch"]+1}, val_loss={checkpoint["val_loss"]:.2f})')

# === 工具函数 ===
SEQ_LEN = 8

def make_context(hour=8, sleep_quality=3, glucose_zone=1):
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    is_dawn = 1 if 4 <= hour <= 6 else 0
    return torch.tensor([[hour_sin, hour_cos, sleep_quality, is_dawn, glucose_zone]],
                        dtype=torch.float32, device=device)

def make_control(seq_len, iob_list, cob_list, exercise_list,
                 basal_rate=0.8, bolus_dose=0.0, heart_rate=70, isf=1.0):
    ctrl = torch.zeros(1, seq_len, 7, dtype=torch.float32, device=device)
    ctrl[0, :, 0] = torch.tensor(iob_list[:seq_len], dtype=torch.float32)
    ctrl[0, :, 1] = torch.tensor(cob_list[:seq_len], dtype=torch.float32)
    ctrl[0, :, 2] = torch.tensor(exercise_list[:seq_len], dtype=torch.float32)
    ctrl[0, :, 3] = basal_rate
    ctrl[0, :, 4] = bolus_dose
    ctrl[0, :, 5] = heart_rate
    ctrl[0, :, 6] = isf
    return ctrl

def predict(glucose_0, control, context):
    g0 = torch.tensor([[glucose_0]], dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = model(g0, control, context)
    return pred[0].cpu().numpy()

# === 场景1 ===
glucose_0 = 110
context_s1 = make_context(hour=8, sleep_quality=3, glucose_zone=1)
iob_s1 = [4.0 * np.exp(-i * 0.25 / 2.0) for i in range(SEQ_LEN)]
cob_s1 = [60 * (0.7 * np.exp(-i * 0.25 / 0.5) + 0.3 * np.exp(-i * 0.25 / 3.0)) for i in range(SEQ_LEN)]
exercise_s1 = [0.0] * SEQ_LEN
control_s1 = make_control(SEQ_LEN, iob_s1, cob_s1, exercise_s1,
                           basal_rate=0.8, bolus_dose=4.0, heart_rate=72, isf=1.2)
pred_s1 = predict(glucose_0, control_s1, context_s1)
print(f'\n场景1: 正常工作日 — 餐后2小时血糖从 {glucose_0} 升至 {pred_s1[-1]:.1f} mg/dL')

# === 场景2 ===
exercise_s2 = [0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 0.0, 0.0]
control_s2 = make_control(SEQ_LEN, iob_s1.copy(), cob_s1.copy(), exercise_s2,
                           basal_rate=0.8, bolus_dose=4.0, heart_rate=72, isf=1.2)
pred_s2 = predict(glucose_0, control_s2, context_s1)
delta = pred_s2 - pred_s1
print(f'场景2: 运动日 — 运动使血糖最多降低 {np.min(delta):.1f} mg/dL, 餐后2小时血糖为 {pred_s2[-1]:.1f} mg/dL')

# === 场景3 ===
control_no_insulin = control_s1.clone()
control_no_insulin[0, :, 0] = 0.0
control_no_insulin[0, :, 3] = 0.0
control_no_insulin[0, :, 4] = 0.0
g0 = torch.tensor([[glucose_0]], dtype=torch.float32, device=device)
with torch.no_grad():
    pred_with = model(g0, control_s1, context_s1)[0].cpu().numpy()
    pred_no = model.counterfactual(g0, control_s1, control_no_insulin, context_s1)[0].cpu().numpy()
diff = pred_no - pred_with
print(f'场景3: 反事实 — 不打胰岛素时餐后2小时血糖为 {pred_no[-1]:.1f} mg/dL, 比打胰岛素高 {diff[-1]:.1f} mg/dL')

print('\n所有场景预测成功!')
