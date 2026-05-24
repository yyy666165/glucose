"""使用训练好的模型进行血糖预测"""
import sys
import os
import functools
print = functools.partial(print, flush=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.glucose_dataset import GlucoseDataset, CONTROL_DIM, CONTEXT_DIM, NUM_PATIENTS
from model.neural_ode_glucose import NeuralODEGlucosePredictor
# 尝试导入 Manchester 数据集 (路径兼容)
try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data analyse'))
    from finetune_manchester import ManchesterGlucoseDataset, PID_TO_IDX
except ImportError:
    ManchesterGlucoseDataset = None
    PID_TO_IDX = None


def load_model(checkpoint_path, device):
    """从checkpoint加载模型（兼容旧版 8-dim 和新版 7-dim 结构化模型）"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    hidden_dim = checkpoint['hidden_dim']
    control_dim = checkpoint['control_dim']
    context_dim = checkpoint['context_dim']
    num_patients = checkpoint.get('num_patients', NUM_PATIENTS)
    patient_embed_dim = checkpoint.get('patient_embed_dim', 16)

    print(f"Checkpoint: epoch={checkpoint['epoch']+1}, "
          f"val_loss={checkpoint['val_loss']:.1f}, "
          f"RMSE={checkpoint.get('val_rmse', 0):.1f}, "
          f"hidden={hidden_dim}, control={control_dim}, "
          f"patients={num_patients}")

    model = NeuralODEGlucosePredictor(
        context_dim=context_dim,
        hidden_dim=hidden_dim,
        control_dim=control_dim,
        num_patients=num_patients,
        patient_embed_dim=patient_embed_dim,
    ).to(device).float()

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, context_dim, control_dim


def predict_on_test(model, dataset, device, max_samples=None):
    """在测试集上运行预测"""
    test_dataset = dataset.get_test_dataset()
    all_predictions = []
    all_targets = []
    all_initials = []

    n = len(test_dataset)
    if max_samples:
        n = min(n, max_samples)

    print(f"测试样本数: {n}")

    with torch.no_grad():
        for i in range(n):
            sample = test_dataset[i]
            initial_glucose = sample['initial_glucose'].unsqueeze(0).to(device).float()
            control_sequence = sample['control_sequence'].unsqueeze(0).to(device).float()
            context = sample['context'].unsqueeze(0).to(device).float()
            patient_ids = sample['patient_id'].unsqueeze(0).to(device)
            target = sample['target']

            pred = model(initial_glucose, control_sequence, context, patient_ids)
            all_predictions.append(pred[0].cpu().numpy())
            all_targets.append(target.numpy())
            all_initials.append(initial_glucose[0, 0].item())

            if (i + 1) % 200 == 0:
                print(f"  已预测 {i+1}/{n}")

    return np.array(all_predictions), np.array(all_targets), np.array(all_initials)


def compute_metrics(predictions, targets):
    """计算评估指标"""
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))

    # 逐时间步指标
    step_rmse = np.sqrt(np.mean((predictions - targets) ** 2, axis=0))
    step_mae = np.mean(np.abs(predictions - targets), axis=0)

    # Clarke误差网格分析 (简化版)
    within_20pct = np.abs(predictions - targets) / np.maximum(targets, 1) < 0.20
    pct_within_20 = np.mean(within_20pct) * 100

    within_15_15 = (np.abs(predictions - targets) <= 15) | (
        np.abs(predictions - targets) / np.maximum(targets, 1) <= 0.15
    )
    pct_zone_ab = np.mean(within_15_15) * 100

    return {
        'MSE': mse, 'RMSE': rmse, 'MAE': mae,
        '% within 20%': pct_within_20,
        '% Clarke A+B': pct_zone_ab,
        'step_RMSE': step_rmse,
        'step_MAE': step_mae,
    }


def plot_results(predictions, targets, initials, metrics, save_dir='results'):
    """绘制预测结果"""
    os.makedirs(save_dir, exist_ok=True)

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. 预测 vs 真实 散点图
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(targets.flatten(), predictions.flatten(), alpha=0.1, s=5, c='#2196F3')
    lims = [40, 400]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='理想预测')
    # Clarke误差网格参考线
    ax.fill_between([40, 70], [40, 70], [40, 56], alpha=0.1, color='green')
    ax.axhline(y=70, color='red', linestyle='--', alpha=0.3, label='低血糖阈值(70)')
    ax.axhline(y=180, color='orange', linestyle='--', alpha=0.3, label='高血糖阈值(180)')
    ax.set_xlabel('真实血糖 (mg/dL)', fontsize=12)
    ax.set_ylabel('预测血糖 (mg/dL)', fontsize=12)
    ax.set_title(f'预测 vs 真实  (RMSE={metrics["RMSE"]:.1f}, MAE={metrics["MAE"]:.1f})', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/scatter_pred_vs_true.png', dpi=150)
    plt.close()
    print(f"  保存: {save_dir}/scatter_pred_vs_true.png")

    # 2. 逐时间步RMSE
    seq_len = len(metrics['step_RMSE'])
    time_min = np.arange(seq_len) * 15  # 每15分钟

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(time_min, metrics['step_RMSE'], width=10, color='#FF5722', alpha=0.7, label='RMSE')
    ax1.set_xlabel('预测时间 (分钟)', fontsize=12)
    ax1.set_ylabel('RMSE (mg/dL)', fontsize=12, color='#FF5722')
    ax1.tick_params(axis='y', labelcolor='#FF5722')

    ax2 = ax1.twinx()
    ax2.plot(time_min, metrics['step_MAE'], 'o-', color='#2196F3', linewidth=2, label='MAE')
    ax2.set_ylabel('MAE (mg/dL)', fontsize=12, color='#2196F3')
    ax2.tick_params(axis='y', labelcolor='#2196F3')

    ax1.set_title('逐时间步预测误差', fontsize=14)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/stepwise_error.png', dpi=150)
    plt.close()
    print(f"  保存: {save_dir}/stepwise_error.png")

    # 3. 随机选取几条样本轨迹
    n_samples = min(6, len(predictions))
    indices = np.random.choice(len(predictions), n_samples, replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for i, idx in enumerate(indices):
        ax = axes[i]
        time = np.arange(len(predictions[idx])) * 15
        ax.plot(time, targets[idx], 'o-', color='#2196F3', label='真实', linewidth=2)
        ax.plot(time, predictions[idx], 's-', color='#FF5722', label='预测', linewidth=2)
        ax.axhline(y=70, color='red', linestyle='--', alpha=0.3)
        ax.axhline(y=180, color='orange', linestyle='--', alpha=0.3)
        ax.axhspan(70, 180, alpha=0.05, color='green')
        ax.set_title(f'样本#{idx} (初始血糖={initials[idx]:.0f})', fontsize=11)
        ax.set_xlabel('时间 (分钟)')
        ax.set_ylabel('血糖 (mg/dL)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.suptitle('预测轨迹示例', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/sample_trajectories.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {save_dir}/sample_trajectories.png")

    # 4. Clarke误差网格 (简化)
    fig, ax = plt.subplots(figsize=(8, 8))
    # Zone A
    ax.fill_between([40, 400], [40, 400], alpha=0.1, color='green')
    # Zone boundaries
    ax.plot([40, 400], [40, 400], 'k-', alpha=0.5)
    ax.plot([40, 400], [40 * 1.2, 400 * 1.2], 'g--', alpha=0.3)
    ax.plot([40, 400], [40 / 1.2, 400 / 1.2], 'g--', alpha=0.3)
    ax.scatter(targets.flatten(), predictions.flatten(), alpha=0.05, s=3, c='#2196F3')
    ax.set_xlim(40, 400)
    ax.set_ylim(40, 400)
    ax.set_xlabel('真实血糖 (mg/dL)', fontsize=12)
    ax.set_ylabel('预测血糖 (mg/dL)', fontsize=12)
    ax.set_title(f'Clarke误差网格  (A+B区域: {metrics["% Clarke A+B"]:.1f}%)', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/clarke_grid.png', dpi=150)
    plt.close()
    print(f"  保存: {save_dir}/clarke_grid.png")


def main():
    # 使用结构化模型 checkpoint
    checkpoint_path = 'checkpoints/structured_final_v1_epoch6_RMSE33.7.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 1. 加载模型
    print("\n=== 加载结构化模型 ===")
    model, context_dim, control_dim = load_model(checkpoint_path, device)
    print(f"模型类型: 结构化生理模型 (7维控制 + 7维ODE状态)")

    # 输出学习者关键生理参数
    p = model.ode_func
    print(f"\n  关键生理参数:")
    print(f"    自调节(p1):      {p.glucose_effectiveness.item():.4f}")
    print(f"    血糖稳态(Gb):    {p.G_baseline.item():.1f} mg/dL")
    print(f"    运动耗糖(gamma): {p.exercise_uptake.item():.4f}")
    print(f"    ISF提升(beta):   {p.isf_boost_scale.item():.4f}")
    print(f"    NN幅度(DA):      {p.dynamics_amplitude.item():.2f}")
    print(f"    胰岛素缩放(IS):  {p.insulin_scale.item():.2f}")

    # 2. 加载测试数据 (Manchester)
    print("\n=== 加载 Manchester 测试集 ===")
    test_dataset = ManchesterGlucoseDataset(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/manchester_test.csv'),
        seq_len=24, split='test'
    )
    print(f"测试样本数: {len(test_dataset)}")

    # 3. 预测
    print("\n=== 开始预测 ===")
    max_samples = min(2000, len(test_dataset))
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    all_preds = []
    all_targets = []
    all_initials = []

    with torch.no_grad():
        for batch in test_loader:
            g0 = batch['initial_glucose'].to(device).float()
            ctrl = batch['control_sequence'].to(device).float()
            ctx = batch['context'].to(device).float()
            pids = batch['patient_id'].to(device)
            target = batch['target'].to(device).float()

            pred = model(g0, ctrl, ctx, pids)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            all_initials.append(g0[:, 0].cpu().numpy())

    predictions = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    initials = np.concatenate(all_initials)

    # 4. 评估
    print("\n=== 评估指标 ===")
    metrics = compute_metrics(predictions, targets)
    print(f"  RMSE:  {metrics['RMSE']:.2f} mg/dL")
    print(f"  MAE:   {metrics['MAE']:.2f} mg/dL")
    print(f"  Within 20%: {metrics['% within 20%']:.1f}%")
    print(f"  Clarke A+B: {metrics['% Clarke A+B']:.1f}%")
    print(f"  分步 RMSE (15/30/60/120min): "
          f"{metrics['step_RMSE'][0]:.1f} / {metrics['step_RMSE'][2]:.1f} / "
          f"{metrics['step_RMSE'][5]:.1f} / {metrics['step_RMSE'][11]:.1f}")

    # 分血糖区间
    for lo, hi, label in [(0, 70, '低血糖 <70'), (70, 180, '正常'), (180, 600, '高血糖 >180')]:
        mask = (targets >= lo) & (targets < hi)
        if mask.sum() > 0:
            r = np.sqrt(np.mean((predictions[mask] - targets[mask])**2))
            print(f"    {label:<15}: RMSE={r:.2f} (n={mask.sum():,})")

    # 5. 可视化
    save_dir = 'results/structured_model'
    print(f"\n=== 生成可视化 ({save_dir}/) ===")
    plot_results(predictions, targets, initials, metrics, save_dir)

    print(f"\n{'='*50}")
    print(f"结构化模型预测完成")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
