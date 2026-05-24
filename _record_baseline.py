"""
基线指标记录 - 改结构前先跑一遍
记录 Ohio + Manchester 的完整评估指标
"""
import sys, os, torch, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data analyse'))

from model.neural_ode_glucose import NeuralODEGlucosePredictor
from data.glucose_dataset import GlucoseDataset
from finetune_manchester import ManchesterGlucoseDataset, PID_TO_IDX

OHIO_ROOT = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cpu')

results = {}

# ============================================================
# 1. Ohio 模型评估
# ============================================================
print('=' * 60)
print('(1) Ohio 测试集 (best_model_v4.pt)')
print('=' * 60)

ckpt = torch.load(os.path.join(OHIO_ROOT, 'checkpoints/best_model_v4.pt'),
                   map_location='cpu', weights_only=False)
model = NeuralODEGlucosePredictor(
    context_dim=ckpt['context_dim'], hidden_dim=ckpt['hidden_dim'],
    control_dim=ckpt['control_dim'], num_patients=ckpt['num_patients'],
    patient_embed_dim=ckpt['patient_embed_dim']
).to(device).float()
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

dataset = GlucoseDataset(data_dir=os.path.join(OHIO_ROOT, 'data'), seq_len=24)
test_dataset = dataset.get_test_dataset()
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

all_pred, all_true, all_pids = [], [], []
with torch.no_grad():
    for batch in test_loader:
        g0 = batch['initial_glucose'].to(device).float()
        ctrl = batch['control_sequence'].to(device).float()
        ctx = batch['context'].to(device).float()
        pid = batch['patient_id'].to(device)
        target = batch['target'].to(device).float()
        pred = model(g0, ctrl, ctx, pid)
        all_pred.append(pred.cpu())
        all_true.append(target.cpu())
        all_pids.append(pid.cpu())

preds = torch.cat(all_pred).numpy()
targets = torch.cat(all_true).numpy()
pids = torch.cat(all_pids).numpy()

rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
mae = float(np.mean(np.abs(preds - targets)))
mard = float(np.mean(np.abs(preds - targets) / np.clip(targets, 40, None)) * 100)
within20 = float(np.mean(np.abs(preds - targets) / np.clip(targets, 40, None) <= 0.20) * 100)
step_rmse = np.sqrt(np.mean((preds - targets) ** 2, axis=0)).tolist()

results['ohio'] = {
    'samples': len(preds),
    'rmse': round(rmse, 2),
    'mae': round(mae, 2),
    'mard': round(mard, 2),
    'within20': round(within20, 1),
    'step_rmse_15min': round(step_rmse[0], 1),
    'step_rmse_30min': round(step_rmse[2], 1),
    'step_rmse_60min': round(step_rmse[5], 1),
    'step_rmse_120min': round(step_rmse[11], 1),
}

print(f"  样本数: {len(preds):,}")
print(f"  RMSE:   {rmse:.2f} mg/dL")
print(f"  MAE:    {mae:.2f} mg/dL")
print(f"  MARD:   {mard:.2f}%")
print(f"  Within20%: {within20:.1f}%")
print(f"  分步 RMSE: 15min={step_rmse[0]:.1f}, 30min={step_rmse[2]:.1f}, "
      f"60min={step_rmse[5]:.1f}, 120min={step_rmse[11]:.1f}")

# 分血糖区间
print(f"\n  分血糖区间 RMSE:")
for lo, hi, label in [(0, 70, '低血糖 <70'), (70, 180, '正常 70-180'), (180, 600, '高血糖 >180')]:
    mask = (targets >= lo) & (targets < hi)
    if mask.sum() > 0:
        r = float(np.sqrt(np.mean((preds[mask] - targets[mask]) ** 2)))
        print(f"    {label:<20}: RMSE={r:.2f} (n={mask.sum():,})")
        results['ohio'][f'rmse_{label.split()[0]}'] = round(r, 2)

# 分患者
print(f"\n  各患者指标:")
pid_to_real = {v: k for k, v in dataset.patient_id_map.items()}
patient_metrics = {}
for p_idx in sorted(set(int(p) for p in pids)):
    mask = pids == p_idx
    if mask.sum() < 10:
        continue
    r = float(np.sqrt(np.mean((preds[mask] - targets[mask]) ** 2)))
    m = float(np.mean(np.abs(preds[mask] - targets[mask])))
    real_id = pid_to_real[p_idx]
    patient_metrics[f'patient_{real_id}'] = {'rmse': round(r, 2), 'mae': round(m, 2), 'n': int(mask.sum())}
    print(f"    患者 {real_id}: n={mask.sum():,}, RMSE={r:.2f}, MAE={m:.2f}")
results['ohio']['per_patient'] = patient_metrics

# ============================================================
# 2. Manchester 微调模型评估
# ============================================================
print('\n' + '=' * 60)
print('(2) Manchester 测试集 (finetune_v1.pt)')
print('=' * 60)

ckpt2 = torch.load(os.path.join(OHIO_ROOT, 'data analyse/best_model_manchester_finetune_v1.pt'),
                    map_location='cpu', weights_only=False)
state2 = ckpt2['model_state_dict']
model2 = NeuralODEGlucosePredictor(
    context_dim=ckpt2.get('context_dim', 5),
    hidden_dim=ckpt2.get('hidden_dim', 64),
    control_dim=ckpt2.get('control_dim', 8),
    num_patients=ckpt2.get('num_patients', 29),
    patient_embed_dim=ckpt2.get('patient_embed_dim', 16),
).to(device).float()
model2.load_state_dict(state2, strict=False)
model2.eval()

test_ds = ManchesterGlucoseDataset(
    os.path.join(OHIO_ROOT, 'data/manchester_test.csv'),
    seq_len=24, split='test'
)
test_loader2 = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)

all_pred2, all_true2, all_pids2 = [], [], []
with torch.no_grad():
    for batch in test_loader2:
        g0 = batch['initial_glucose'].to(device).float()
        ctrl = batch['control_sequence'].to(device).float()
        ctx = batch['context'].to(device).float()
        pid = batch['patient_id'].to(device)
        target = batch['target'].to(device).float()
        pred = model2(g0, ctrl, ctx, pid)
        all_pred2.append(pred.cpu())
        all_true2.append(target.cpu())
        all_pids2.append(pid.cpu())

preds2 = torch.cat(all_pred2).numpy()
targets2 = torch.cat(all_true2).numpy()
pids2 = torch.cat(all_pids2).numpy()

rmse2 = float(np.sqrt(np.mean((preds2 - targets2) ** 2)))
mae2 = float(np.mean(np.abs(preds2 - targets2)))
mard2 = float(np.mean(np.abs(preds2 - targets2) / np.clip(targets2, 40, None)) * 100)
within20_2 = float(np.mean(np.abs(preds2 - targets2) / np.clip(targets2, 40, None) <= 0.20) * 100)
step_rmse2 = np.sqrt(np.mean((preds2 - targets2) ** 2, axis=0)).tolist()

results['manchester'] = {
    'samples': len(preds2),
    'rmse': round(rmse2, 2),
    'mae': round(mae2, 2),
    'mard': round(mard2, 2),
    'within20': round(within20_2, 1),
    'step_rmse_15min': round(step_rmse2[0], 1),
    'step_rmse_30min': round(step_rmse2[2], 1),
    'step_rmse_60min': round(step_rmse2[5], 1),
    'step_rmse_120min': round(step_rmse2[11], 1),
}

print(f"  样本数: {len(preds2):,}")
print(f"  RMSE:   {rmse2:.2f} mg/dL")
print(f"  MAE:    {mae2:.2f} mg/dL")
print(f"  MARD:   {mard2:.2f}%")
print(f"  Within20%: {within20_2:.1f}%")
print(f"  分步 RMSE: 15min={step_rmse2[0]:.1f}, 30min={step_rmse2[2]:.1f}, "
      f"60min={step_rmse2[5]:.1f}, 120min={step_rmse2[11]:.1f}")

for lo, hi, label in [(0, 70, '低血糖 <70'), (70, 180, '正常 70-180'), (180, 600, '高血糖 >180')]:
    mask = (targets2 >= lo) & (targets2 < hi)
    if mask.sum() > 0:
        r = float(np.sqrt(np.mean((preds2[mask] - targets2[mask]) ** 2)))
        print(f"  {label:<20}: RMSE={r:.2f} (n={mask.sum():,})")
        results['manchester'][f'rmse_{label.split()[0]}'] = round(r, 2)

# 分患者
print(f"\n  各患者指标:")
pid_to_real2 = {v: k for k, v in PID_TO_IDX.items()}
patient_metrics2 = {}
for p_idx in sorted(set(int(p) for p in pids2)):
    mask = pids2 == p_idx
    if mask.sum() < 10:
        continue
    r = float(np.sqrt(np.mean((preds2[mask] - targets2[mask]) ** 2)))
    m = float(np.mean(np.abs(preds2[mask] - targets2[mask])))
    real_id = pid_to_real2.get(p_idx, f'unknown_{p_idx}')
    patient_metrics2[f'participant_{real_id}'] = {'rmse': round(r, 2), 'mae': round(m, 2), 'n': int(mask.sum())}
    print(f"    参与者 {real_id}: n={mask.sum():,}, RMSE={r:.2f}, MAE={m:.2f}")
results['manchester']['per_patient'] = patient_metrics2

# ============================================================
# 3. 行为指标 - 反事实场景
# ============================================================
print('\n' + '=' * 60)
print('(3) 行为指标 - 反事实场景')
print('=' * 60)

# 场景1: 不吃不喝不运动，基线血糖变化
# 场景2: 60g碳水 + 4U胰岛素
# 场景3: 60g碳水 + 4U胰岛素 + 运动

# 用 Ohio 模型
model = model  # already loaded
model.eval()

def predict_scenario(model, initial_glucose, hour, carbs_g, bolus_u, exercise_intensity):
    """模拟一个场景，返回 24 步预测"""
    import math
    seq_len = 24

    # 简单 IOB/COB 计算
    def iob_cob(dose, peak, duration, t_hours):
        if t_hours <= 0:
            return 0.0
        val = dose * (math.exp(-t_hours / duration) - math.exp(-t_hours / peak))
        return max(0, val)

    g0 = torch.tensor([[initial_glucose]], dtype=torch.float32)
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)
    ctx = torch.tensor([[hour_sin, hour_cos, 3.0, 1.0 if 4 <= hour <= 6 else 0.0, 1.0]], dtype=torch.float32)

    control_dim = 7  # Ohio 模型
    ctrl = np.zeros((1, seq_len, control_dim))
    for step in range(seq_len):
        t = step * 5 / 60  # hours
        iob_val = iob_cob(bolus_u, 1.5, 6.0, t)
        cob_val = iob_cob(carbs_g, 1.0, 4.0, t)
        ctrl[0, step] = [iob_val, cob_val, exercise_intensity, 0.5, bolus_u if step == 0 else 0, 75, 1.0]

    ctrl_t = torch.tensor(ctrl, dtype=torch.float32)
    pid = torch.tensor([0], dtype=torch.long)

    with torch.no_grad():
        pred = model(g0, ctrl_t, ctx, pid)[0].numpy()

    return pred

scenarios = [
    ('场景A: 不吃不喝，无运动', 150, 8, 0, 0, 0),
    ('场景B: 60g碳水 + 4U胰岛素', 150, 8, 60, 4.0, 0),
    ('场景C: 60g碳水 + 4U胰岛素 + 运动', 150, 8, 60, 4.0, 5),
    ('场景D: 60g碳水 + 无胰岛素', 150, 8, 60, 0, 0),
]

behavior = {}
for label, init_g, hour, carbs, bolus, ex in scenarios:
    pred = predict_scenario(model, init_g, hour, carbs, bolus, ex)
    behavior[label] = {
        '30min_pred': round(float(pred[5]), 1),
        '60min_pred': round(float(pred[11]), 1),
        '120min_pred': round(float(pred[23]), 1),
        'min': round(float(pred.min()), 1),
        'max': round(float(pred.max()), 1),
    }
    print(f"  {label}")
    print(f"    初始={init_g}, 30min={pred[5]:.1f}, 60min={pred[11]:.1f}, 120min={pred[23]:.1f}")
    print(f"    区间: [{pred.min():.1f}, {pred.max():.1f}]")

results['behavior'] = behavior

# ============================================================
# 4. 模型参数量统计
# ============================================================
print('\n' + '=' * 60)
print('(4) 模型参数')
print('=' * 60)
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  总参数量: {total:,}")
print(f"  可训练参数: {trainable:,}")
results['model_params'] = {'total': total, 'trainable': trainable}

# ============================================================
# 保存结果
# ============================================================
import json
save_path = os.path.join(OHIO_ROOT, 'results', 'baseline_metrics.json')
os.makedirs(os.path.join(OHIO_ROOT, 'results'), exist_ok=True)
with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f'\n基线指标已保存: {save_path}')
print('=' * 60)
