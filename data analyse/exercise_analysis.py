"""
运动对血糖预测的影响分析
======================
三种方法：
1. 反事实对比: 同一场景，运动 vs 不运动，预测轨迹差异
2. 剂量-响应: 运动强度 0→10 的连续变化对血糖预测的影响
3. 消融实验: 去掉运动特征后，模型精度下降多少
"""
import sys, os, functools
print = functools.partial(print, flush=True)

# 添加项目根目录
OHIO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, OHIO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from model.neural_ode_glucose import NeuralODEGlucosePredictor
from finetune_manchester import create_model, CONTROL_DIM, PID_TO_IDX, ManchesterGlucoseDataset

# ─── 配置 ───
CHECKPOINT = os.path.join(OHIO_ROOT, 'data analyse/best_model_manchester_finetune_v1.pt')
DATA_DIR = os.path.join(OHIO_ROOT, 'data')
SAVE_DIR = os.path.join(OHIO_ROOT, 'results')
os.makedirs(SAVE_DIR, exist_ok=True)

SEQ_LEN = 24
DEVICE = torch.device('cpu')


# ─── 1. 加载模型 ───
def load_model():
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    state = ckpt['model_state_dict']
    hidden_dim = ckpt.get('hidden_dim', 64)
    control_dim = ckpt.get('control_dim', CONTROL_DIM)
    num_patients = ckpt.get('num_patients', 29)
    patient_embed_dim = ckpt.get('patient_embed_dim', 16)
    context_dim = ckpt.get('context_dim', 5)

    model = NeuralODEGlucosePredictor(
        context_dim=context_dim, hidden_dim=hidden_dim, control_dim=control_dim,
        num_patients=num_patients, patient_embed_dim=patient_embed_dim,
    ).to(DEVICE).float()
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"模型加载完成 | patients={num_patients}, control_dim={control_dim}, hidden={hidden_dim}")
    return model, control_dim, context_dim, num_patients


# ─── 2. 获取测试样本 ───
def get_test_samples(n_samples=6):
    """从 Manchester 测试集随机抽取样本"""
    test_dataset = ManchesterGlucoseDataset(
        os.path.join(DATA_DIR, 'manchester_test.csv'),
        seq_len=SEQ_LEN, split='test'
    )
    indices = np.random.choice(len(test_dataset), min(n_samples, len(test_dataset)), replace=False)
    samples = []
    for idx in indices:
        sample = test_dataset[idx]
        samples.append({
            'initial_glucose': sample['initial_glucose'].unsqueeze(0),
            'control_sequence': sample['control_sequence'].unsqueeze(0),
            'context': sample['context'].unsqueeze(0),
            'patient_id': sample['patient_id'].unsqueeze(0),
            'target': sample['target'].unsqueeze(0),
            'raw_idx': idx,
        })
    print(f"抽取 {len(samples)} 个测试样本")
    return samples, test_dataset


# ─── 3. 分析1: 反事实对比 ───
def counterfactual_analysis(model, samples, control_dim):
    """同一场景，运动强度设为 0 / 5 / 10，对比预测轨迹"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    t_axis = np.arange(1, SEQ_LEN + 1) * 5  # 分钟

    for plot_i, sample in enumerate(samples[:6]):
        ax = axes[plot_i]
        g0 = sample['initial_glucose'].to(DEVICE).float()
        ctx = sample['context'].to(DEVICE).float()
        pid = sample['patient_id'].to(DEVICE)
        target = sample['target'].numpy()[0]

        # 画真实值
        ax.plot(t_axis, target, 'k-', linewidth=2, alpha=0.8, label='真实值')

        exercise_levels = [0, 3, 7, 10]
        colors = ['#4CAF50', '#FF9800', '#F44336', '#9C27B0']
        linestyles = ['--', '-.', ':', '-']

        for ex_i, ex_val in enumerate(exercise_levels):
            ctrl = sample['control_sequence'].clone()
            ctrl[:, :, 2] = ex_val  # exercise_intensity 是第 3 维 (index=2)

            with torch.no_grad():
                if control_dim == 8 and ctrl.shape[2] == 7:
                    # 补 delta_glucose 列
                    pad = torch.zeros(ctrl.shape[0], ctrl.shape[1], 1)
                    ctrl = torch.cat([ctrl, pad], dim=2)

                pred = model(g0, ctrl.to(DEVICE).float(), ctx, pid)[0].cpu().numpy()

            label = f'运动={ex_val}' if ex_i == 0 else f'{ex_val}'
            ax.plot(t_axis, pred, color=colors[ex_i % len(colors)],
                    linestyle=linestyles[ex_i % len(linestyles)],
                    linewidth=1.5, label=label)

        ax.axhline(y=70, color='green', linestyle=':', alpha=0.4)
        ax.axhline(y=180, color='orange', linestyle=':', alpha=0.4)
        ax.fill_between([0, 24*5], 70, 180, alpha=0.05, color='green')
        pid_real = list(PID_TO_IDX.keys())[list(PID_TO_IDX.values()).index(pid.item())]
        ax.set_title(f'样本 #{sample["raw_idx"]} (患者 {pid_real})', fontsize=11)
        ax.set_xlabel('时间 (分钟)')
        ax.set_ylabel('血糖 (mg/dL)')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(0, SEQ_LEN * 5 + 5)

    plt.suptitle('反事实分析: 不同运动强度下的血糖预测轨迹', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'exercise_counterfactual.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"反事实对比图: {path}")


# ─── 4. 分析2: 运动剂量-响应曲线 ───
def dose_response_analysis(model, samples, control_dim):
    """固定样本，运动强度 0→10 逐步增加，看不同预测视野的血糖变化"""
    n_ex_levels = 21
    ex_range = np.linspace(0, 10, n_ex_levels)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for plot_i, sample in enumerate(samples[:6]):
        ax = axes[plot_i]
        g0 = sample['initial_glucose'].to(DEVICE).float()
        ctx = sample['context'].to(DEVICE).float()
        pid = sample['patient_id'].to(DEVICE)
        target = sample['target'].numpy()[0]

        horizons = [5, 11, 23]  # indices for 30min, 60min, 120min
        horizon_labels = ['30min', '60min', '120min']
        colors = ['#2196F3', '#FF9800', '#F44336']

        responses = {h: [] for h in horizons}

        for ex_val in ex_range:
            ctrl = sample['control_sequence'].clone()
            ctrl[:, :, 2] = ex_val

            with torch.no_grad():
                if control_dim == 8 and ctrl.shape[2] == 7:
                    pad = torch.zeros(ctrl.shape[0], ctrl.shape[1], 1)
                    ctrl = torch.cat([ctrl, pad], dim=2)
                pred = model(g0, ctrl.to(DEVICE).float(), ctx, pid)[0].cpu().numpy()

            for h in horizons:
                responses[h].append(pred[h])

        for h_idx, h in enumerate(horizons):
            ax.plot(ex_range, responses[h], 'o-', color=colors[h_idx],
                    linewidth=2, markersize=4, label=f'{horizon_labels[h_idx]} 预测')

        # 标注真实值作为参考
        for h_idx, h in enumerate(horizons):
            ax.axhline(y=target[h], color=colors[h_idx], linestyle=':', alpha=0.3)

        pid_real = list(PID_TO_IDX.keys())[list(PID_TO_IDX.values()).index(pid.item())]
        ax.set_title(f'样本 #{sample["raw_idx"]} (患者 {pid_real})', fontsize=11)
        ax.set_xlabel('运动强度')
        ax.set_ylabel('预测血糖 (mg/dL)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    plt.suptitle('运动剂量-响应曲线: 运动强度对血糖预测的影响', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'exercise_dose_response.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"剂量-响应图: {path}")


# ─── 5. 分析3: 消融实验 ───
def ablation_analysis(model, control_dim, context_dim, num_patients):
    """在整个测试集上去掉运动特征，看精度下降多少"""
    print("\n=== 消融实验: 运动特征对精度的贡献 ===")
    test_dataset = ManchesterGlucoseDataset(
        os.path.join(DATA_DIR, 'manchester_test.csv'),
        seq_len=SEQ_LEN, split='test'
    )
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    def evaluate(exercise_zero=False):
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in loader:
                g0 = batch['initial_glucose'].to(DEVICE).float()
                ctrl = batch['control_sequence'].to(DEVICE).float()
                ctx = batch['context'].to(DEVICE).float()
                pid = batch['patient_id'].to(DEVICE)
                target = batch['target'].to(DEVICE).float()

                if exercise_zero:
                    ctrl[:, :, 2] = 0  # exercise_intensity 置零

                pred = model(g0, ctrl, ctx, pid)
                all_preds.append(pred.cpu().numpy())
                all_targets.append(target.cpu().numpy())

        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        rmse = np.sqrt(np.mean((preds - targets) ** 2))
        mae = np.mean(np.abs(preds - targets))
        mard = np.mean(np.abs(preds - targets) / np.clip(targets, 40, None)) * 100
        within_20 = np.mean(np.abs(preds - targets) / np.clip(targets, 40, None) <= 0.20) * 100
        return {'RMSE': rmse, 'MAE': mae, 'MARD': mard, 'Within20%': within_20}

    baseline = evaluate(exercise_zero=False)
    ablated = evaluate(exercise_zero=True)

    print(f"{'指标':<12} {'原始':>10} {'去运动':>10} {'变化':>10}")
    print("-" * 44)
    for metric in ['RMSE', 'MAE', 'MARD', 'Within20%']:
        change = ablated[metric] - baseline[metric]
        arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
        print(f"{metric:<12} {baseline[metric]:>8.2f}  {ablated[metric]:>8.2f}  {arrow}{abs(change):>7.2f}")

    # 分步RMSE对比（分运动/不运动子集）
    print("\n--- 分步 RMSE 对比 (原始 vs 去运动) ---")
    def evaluate_stepwise(exercise_zero=False):
        model.eval()
        all_preds, all_targets, all_ex = [], [], []
        with torch.no_grad():
            for batch in loader:
                g0 = batch['initial_glucose'].to(DEVICE).float()
                ctrl = batch['control_sequence'].to(DEVICE).float()
                ctx = batch['context'].to(DEVICE).float()
                pid = batch['patient_id'].to(DEVICE)
                target = batch['target'].to(DEVICE).float()
                orig_ex = ctrl[:, :, 2].clone()
                if exercise_zero:
                    ctrl[:, :, 2] = 0
                pred = model(g0, ctrl, ctx, pid)
                all_preds.append(pred.cpu().numpy())
                all_targets.append(target.cpu().numpy())
                all_ex.append(orig_ex.cpu().numpy())
        return np.concatenate(all_preds), np.concatenate(all_targets), np.concatenate(all_ex)

    preds_base, targets, ex_data = evaluate_stepwise(exercise_zero=False)
    preds_ablated, _, _ = evaluate_stepwise(exercise_zero=True)

    # 按运动强度分组
    avg_ex = ex_data.mean(axis=1)  # 每个样本的平均运动强度
    for lo, hi, label in [(0, 0.1, '无运动'), (0.1, 3, '轻度运动'), (3, 7, '中度运动'), (7, 11, '高强度运动')]:
        mask = (avg_ex >= lo) & (avg_ex < hi)
        if mask.sum() < 5:
            continue
        rmse_base = np.sqrt(np.mean((preds_base[mask] - targets[mask]) ** 2))
        rmse_abl = np.sqrt(np.mean((preds_ablated[mask] - targets[mask]) ** 2))
        print(f"  {label:<10} ({mask.sum():>4}样本): RMSE原始={rmse_base:.2f}, 去运动={rmse_abl:.2f}, 上升={rmse_abl-rmse_base:.2f}")

    return baseline, ablated


# ─── 6. 汇总可视化 ───
def summary_plot(baseline, ablated, sample_results):
    """汇总消融结果"""
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics = ['RMSE', 'MAE', 'MARD']
    x = np.arange(len(metrics))
    w = 0.35

    base_vals = [baseline[m] for m in metrics]
    ablate_vals = [ablated[m] for m in metrics]
    bars1 = ax.bar(x - w/2, base_vals, w, label='原始模型', color='#2196F3', alpha=0.8)
    bars2 = ax.bar(x + w/2, ablate_vals, w, label='去掉运动特征', color='#F44336', alpha=0.8)

    for bar, val in zip(bars1, base_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, ablate_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('指标值')
    ax.set_title('消融实验: 运动特征对模型精度的影响')
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'exercise_ablation_summary.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"消融汇总图: {path}")


# ─── 7. 主函数 ───
def main():
    print("=" * 60)
    print("运动对血糖预测影响分析")
    print("=" * 60)

    # 加载模型
    model, control_dim, context_dim, num_patients = load_model()

    # 抽取样本
    samples, test_dataset = get_test_samples(n_samples=6)

    # 分析1: 反事实对比
    print("\n=== 分析1: 反事实对比 ===")
    counterfactual_analysis(model, samples, control_dim)

    # 分析2: 剂量-响应
    print("\n=== 分析2: 剂量-响应曲线 ===")
    dose_response_analysis(model, samples, control_dim)

    # 分析3: 消融实验
    print("\n=== 分析3: 消融实验 ===")
    baseline, ablated = ablation_analysis(model, control_dim, context_dim, num_patients)

    # 汇总
    summary_plot(baseline, ablated, None)

    print(f"\n所有图表已保存到 {SAVE_DIR}/")
    print("  - exercise_counterfactual.png  (反事实轨迹对比)")
    print("  - exercise_dose_response.png   (剂量-响应曲线)")
    print("  - exercise_ablation_summary.png (消融汇总)")


if __name__ == '__main__':
    main()
