"""所有评估绘图函数"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import functools
print = functools.partial(print, flush=True)


def plot_results(predictions, targets, timestamps, initials, metrics, patient_id, save_dir='results'):
    """基础评估图表: 散点图、逐时间步误差、样本轨迹、时间线"""
    os.makedirs(save_dir, exist_ok=True)
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. Scatter: predicted vs true
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(targets.flatten(), predictions.flatten(), alpha=0.1, s=5, c='#2196F3')
    lims = [40, 400]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='Perfect prediction')
    ax.axhline(y=70, color='red', linestyle='--', alpha=0.3, label='Hypo threshold (70)')
    ax.axhline(y=180, color='orange', linestyle='--', alpha=0.3, label='Hyper threshold (180)')
    ax.set_xlabel('True Glucose (mg/dL)', fontsize=12)
    ax.set_ylabel('Predicted Glucose (mg/dL)', fontsize=12)
    ax.set_title(f'Patient {patient_id}: Pred vs True  (RMSE={metrics["RMSE"]:.1f}, MAE={metrics["MAE"]:.1f})', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/patient_{patient_id}_scatter.png', dpi=150)
    plt.close()

    # 2. Step-wise RMSE/MAE
    seq_len = len(metrics['step_RMSE'])
    time_min = np.arange(seq_len) * 15
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(time_min, metrics['step_RMSE'], width=10, color='#FF5722', alpha=0.7, label='RMSE')
    ax1.set_xlabel('Prediction Horizon (min)', fontsize=12)
    ax1.set_ylabel('RMSE (mg/dL)', fontsize=12, color='#FF5722')
    ax2 = ax1.twinx()
    ax2.plot(time_min, metrics['step_MAE'], 'o-', color='#2196F3', linewidth=2, label='MAE')
    ax2.set_ylabel('MAE (mg/dL)', fontsize=12, color='#2196F3')
    ax1.set_title(f'Patient {patient_id}: Error by Prediction Horizon', fontsize=13)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/patient_{patient_id}_stepwise_error.png', dpi=150)
    plt.close()

    # 3. Sample trajectories
    n_samples = min(6, len(predictions))
    np.random.seed(42)
    indices = np.linspace(0, len(predictions) - 1, n_samples, dtype=int)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for i, idx in enumerate(indices):
        ax = axes[i]
        time = np.arange(len(predictions[idx])) * 15
        ax.plot(time, targets[idx], 'o-', color='#2196F3', label='True', linewidth=2)
        ax.plot(time, predictions[idx], 's-', color='#FF5722', label='Pred', linewidth=2)
        ax.axhline(y=70, color='red', linestyle='--', alpha=0.3)
        ax.axhline(y=180, color='orange', linestyle='--', alpha=0.3)
        ax.axhspan(70, 180, alpha=0.05, color='green')
        ts0 = pd.Timestamp(timestamps[idx][0])
        ax.set_title(f'Start: {ts0.strftime("%m/%d %H:%M")} (G0={initials[idx]:.0f})', fontsize=11)
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Glucose (mg/dL)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.suptitle(f'Patient {patient_id}: Prediction Trajectories', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/patient_{patient_id}_trajectories.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Full timeline
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(timestamps[0], targets[0], '-', color='#2196F3', alpha=0.5, linewidth=1, label='True')
    for step_idx, color, label in [(1, '#FF9800', 'Pred 15min'), (7, '#4CAF50', 'Pred 120min')]:
        pred_ts = [ts[step_idx] for ts in timestamps if len(ts) > step_idx]
        pred_vals = [predictions[i][step_idx] for i in range(len(predictions)) if len(predictions[i]) > step_idx]
        ax.plot(pred_ts, pred_vals, 'o', color=color, markersize=3, alpha=0.6, label=label)
    ax.axhline(y=70, color='red', linestyle='--', alpha=0.3)
    ax.axhline(y=180, color='orange', linestyle='--', alpha=0.3)
    ax.axhspan(70, 180, alpha=0.05, color='green')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Glucose (mg/dL)', fontsize=12)
    ax.set_title(f'Patient {patient_id}: Full Timeline', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/patient_{patient_id}_timeline.png', dpi=150)
    plt.close()


def plot_event_window(result, meal_df, bolus_df, exercise_df, patient_id, save_dir):
    """绘制事件窗口：预测 vs 真实，标注饮食/胰岛素/运动事件"""
    ts = result['timestamps']
    pred = result['pred']
    true_g = result['true']
    g0 = result['initial_glucose']
    t_start, t_end = ts[0], ts[-1]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(ts, true_g, 'o-', color='#2196F3', linewidth=2, markersize=4, label='真实血糖 (CGM)')
    ax.plot(ts, pred, 's-', color='#FF5722', linewidth=2, markersize=4, label='模型预测')
    ax.axhline(y=g0, color='gray', linestyle=':', alpha=0.4)
    ax.axhspan(70, 180, alpha=0.08, color='green')
    ax.axhline(y=70, color='red', linestyle='--', alpha=0.4, label='低血糖阈值 (70)')
    ax.axhline(y=180, color='orange', linestyle='--', alpha=0.4, label='高血糖阈值 (180)')

    for _, m in meal_df[(meal_df['timestamp'] >= t_start - pd.Timedelta(minutes=15)) &
                        (meal_df['timestamp'] <= t_end + pd.Timedelta(minutes=15))].iterrows():
        ax.axvline(x=m['timestamp'], color='#4CAF50', linestyle='-', alpha=0.6, linewidth=1.5)
        ax.annotate(f"饮食 {m['carbs']:.0f}g", xy=(m['timestamp'], 280),
                    fontsize=9, color='#4CAF50', fontweight='bold', ha='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#4CAF50', alpha=0.8))

    for _, b in bolus_df[(bolus_df['timestamp'] >= t_start - pd.Timedelta(minutes=15)) &
                         (bolus_df['timestamp'] <= t_end + pd.Timedelta(minutes=15))].iterrows():
        ax.axvline(x=b['timestamp'], color='#9C27B0', linestyle='-', alpha=0.6, linewidth=1.5)
        ax.annotate(f"胰岛素 {b['bolus_dose']:.1f}U", xy=(b['timestamp'], 260),
                    fontsize=9, color='#9C27B0', fontweight='bold', ha='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#9C27B0', alpha=0.8))

    for _, e in exercise_df[(exercise_df['timestamp'] >= t_start - pd.Timedelta(hours=1)) &
                            (exercise_df['timestamp'] <= t_end + pd.Timedelta(hours=1))].iterrows():
        ax.axvline(x=e['timestamp'], color='#FF9800', linestyle='-', alpha=0.6, linewidth=1.5)
        ax.annotate(f"运动 {e['duration']:.0f}min", xy=(e['timestamp'], 240),
                    fontsize=9, color='#FF9800', fontweight='bold', ha='center',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#FF9800', alpha=0.8))

    rmse = np.sqrt(np.mean((pred - true_g) ** 2))
    mae = np.mean(np.abs(pred - true_g))
    mard = np.mean(np.abs(pred - true_g) / np.maximum(true_g, 1)) * 100
    ts0 = pd.Timestamp(ts[0])
    ax.set_xlabel('时间', fontsize=12)
    ax.set_ylabel('血糖 (mg/dL)', fontsize=12)
    ax.set_title(f'患者 {patient_id}: 预测 vs 真实 | RMSE={rmse:.1f}, MAE={mae:.1f}, MARD={mard:.1f}%\n'
                 f'起始: {ts0.strftime("%m/%d %H:%M")} (G0={g0:.0f} mg/dL)', fontsize=13)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(40, 300)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    start_idx = result['start_idx']
    plt.savefig(f'{save_dir}/patient_{patient_id}_event_window_{start_idx}.png', dpi=150)
    plt.close()
    print(f"  保存: {save_dir}/patient_{patient_id}_event_window_{start_idx}.png")

    # 误差条形图
    fig2, ax2 = plt.subplots(figsize=(14, 3))
    time_min = np.arange(len(pred)) * 15
    error = pred - true_g
    colors = ['#4CAF50' if e >= 0 else '#FF5722' for e in error]
    ax2.bar(time_min, error, width=12, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.set_xlabel('预测时间 (分钟)', fontsize=12)
    ax2.set_ylabel('预测误差 (mg/dL)', fontsize=12)
    ax2.set_title('逐时间步预测误差 (正值=高估, 负值=低估)', fontsize=13)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/patient_{patient_id}_error_{start_idx}.png', dpi=150)
    plt.close()


def _add_event_markers(ax, meal_df, bolus_df, exercise_df, t_start, t_end, y_top=290):
    """在时间轴内添加事件标记"""
    meals = meal_df[(meal_df['timestamp'] >= t_start) & (meal_df['timestamp'] <= t_end)]
    for _, m in meals.iterrows():
        ax.axvline(x=m['timestamp'], color='#4CAF50', linestyle='--', alpha=0.4, linewidth=1)
        ax.text(m['timestamp'], y_top, f"饮食{m['carbs']:.0f}g", fontsize=6,
                color='#4CAF50', ha='center', va='bottom', rotation=45)

    boluses = bolus_df[(bolus_df['timestamp'] >= t_start) & (bolus_df['timestamp'] <= t_end)]
    for _, b in boluses.iterrows():
        ax.axvline(x=b['timestamp'], color='#9C27B0', linestyle='--', alpha=0.4, linewidth=1)
        ax.text(b['timestamp'], y_top - 12, f"胰岛素{b['bolus_dose']:.1f}U", fontsize=6,
                color='#9C27B0', ha='center', va='bottom', rotation=45)

    exercises = exercise_df[(exercise_df['timestamp'] >= t_start) & (exercise_df['timestamp'] <= t_end)]
    for _, e in exercises.iterrows():
        ax.axvline(x=e['timestamp'], color='#FF9800', linestyle='--', alpha=0.4, linewidth=1)
        ax.text(e['timestamp'], y_top - 24, f"运动{e['duration']:.0f}min", fontsize=6,
                color='#FF9800', ha='center', va='bottom', rotation=45)


def plot_full_patient_timeline(predictions, targets, timestamps, initials,
                               features_df, meal_df, bolus_df, exercise_df,
                               patient_id, save_dir='results',
                               horizons_min=(30, 60, 120)):
    """绘制患者全时段血糖预测对比图，超过24小时按天拆子图"""
    os.makedirs(save_dir, exist_ok=True)
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    true_ts = features_df['timestamp'].values
    true_glucose = features_df['glucose_level'].values
    t_min = pd.Timestamp(true_ts.min())
    t_max = pd.Timestamp(true_ts.max())
    total_hours = (t_max - t_min).total_seconds() / 3600
    print(f"  数据时间范围: {t_min} ~ {t_max} ({total_hours:.1f}h, {len(true_ts)}个CGM点)")

    horizon_colors = {30: '#FF9800', 60: '#E91E63', 120: '#9C27B0',
                      180: '#3F51B5', 240: '#009688'}
    horizon_data = {}
    for h_min in horizons_min:
        step_idx = h_min // 15 - 1
        if step_idx < 0 or step_idx >= predictions.shape[1]:
            print(f"  PH={h_min}min 超出预测范围，跳过")
            continue
        p_times, p_vals, t_vals = [], [], []
        for i in range(len(predictions)):
            if len(timestamps[i]) > step_idx:
                p_times.append(pd.Timestamp(timestamps[i][step_idx]))
                p_vals.append(predictions[i][step_idx])
                t_vals.append(targets[i][step_idx])
        if not p_times:
            continue
        order = np.argsort(p_times)
        p_times = np.array(p_times)[order]
        p_vals = np.array(p_vals)[order]
        t_vals = np.array(t_vals)[order]
        rmse = np.sqrt(np.mean((p_vals - t_vals) ** 2))
        mae = np.mean(np.abs(p_vals - t_vals))
        mard = np.mean(np.abs(p_vals - t_vals) / np.maximum(t_vals, 1)) * 100
        horizon_data[h_min] = {
            'times': p_times, 'pred': p_vals, 'true': t_vals,
            'RMSE': rmse, 'MAE': mae, 'MARD': mard,
        }
        print(f"  PH={h_min}min: {len(p_times)}个预测点, "
              f"RMSE={rmse:.1f}, MAE={mae:.1f}, MARD={mard:.1f}%")

    if not horizon_data:
        print("  没有可用的预测时间窗，跳过全时段图")
        return

    n_days = (t_max.normalize() - t_min.normalize()).days + 1
    split_daily = n_days > 1 and total_hours > 24

    if split_daily:
        day_starts = pd.date_range(t_min.normalize(), t_max.normalize(), freq='D')
        n_sub = len(day_starts)
        fig, axes = plt.subplots(n_sub, 1, figsize=(18, 4 * n_sub),
                                 sharex=False, sharey=True)
        if n_sub == 1:
            axes = [axes]
    else:
        fig, axes = plt.subplots(1, 1, figsize=(18, 6))
        axes = [axes]
        day_starts = [t_min.normalize()]

    for ax_i, day_start in enumerate(day_starts):
        ax = axes[ax_i]
        day_end = day_start + pd.Timedelta(days=1)

        if split_daily:
            mask = (true_ts >= np.datetime64(day_start)) & (true_ts < np.datetime64(day_end))
            sub_ts = true_ts[mask]
            sub_g = true_glucose[mask]
        else:
            sub_ts = true_ts
            sub_g = true_glucose
            day_end = t_max + pd.Timedelta(minutes=30)

        ax.axhspan(70, 180, alpha=0.06, color='green')
        ax.axhline(y=70, color='red', linestyle='--', alpha=0.4, label='低血糖阈值 (70)')
        ax.axhline(y=180, color='orange', linestyle='--', alpha=0.4, label='高血糖阈值 (180)')
        ax.plot(sub_ts, sub_g, '-', color='#2196F3', linewidth=1.2, alpha=0.8, label='真实血糖 (CGM)')

        for h_min, hd in sorted(horizon_data.items()):
            color = horizon_colors.get(h_min, '#666666')
            mask_h = (hd['times'] >= np.datetime64(day_start)) & (hd['times'] < np.datetime64(day_end))
            h_ts = hd['times'][mask_h]
            h_pv = hd['pred'][mask_h]
            if len(h_ts) > 0:
                ax.plot(h_ts, h_pv, '-', color=color, linewidth=1.0, alpha=0.7,
                        label=f'PH={h_min}min (RMSE={hd["RMSE"]:.1f})')

        _add_event_markers(ax, meal_df, bolus_df, exercise_df, day_start, day_end)

        ax.set_ylim(40, 300)
        ax.set_ylabel('血糖 (mg/dL)', fontsize=11)
        if split_daily:
            ax.set_title(day_start.strftime('%Y-%m-%d'), fontsize=12)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        else:
            if total_hours > 24:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
            else:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8, ncol=2)

    axes[-1].set_xlabel('时间', fontsize=12)
    metrics_parts = [f"PH={h}min: RMSE={d['RMSE']:.1f}, MAE={d['MAE']:.1f}, MARD={d['MARD']:.1f}%"
                     for h, d in sorted(horizon_data.items())]
    fig.suptitle(f'患者 {patient_id}: 全时段血糖预测对比\n' + '\n'.join(metrics_parts),
                 fontsize=13, y=1.0 if not split_daily else 0.98)

    plt.tight_layout()
    suffix = '_daily' if split_daily else ''
    out_path = f'{save_dir}/patient_{patient_id}_full_timeline{suffix}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {out_path}")
