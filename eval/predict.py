"""滑窗预测 + 事件窗口预测 + 交互式预测"""
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import functools
print = functools.partial(print, flush=True)

CONTROL_DIM = 7
CONTEXT_DIM = 5


def get_control_vector(row):
    return [
        row.get('IOB', 0),
        row.get('COB', 0),
        row.get('exercise_intensity', 0),
        row.get('effective_basal_rate', 0),
        row.get('recent_bolus_dose', 0),
        row.get('heart_rate', 70),
        row.get('ISF', 1.0),
    ]


def get_context_features(row):
    hour = row['timestamp'].hour
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    sleep_quality = row.get('sleep_quality', 3)
    is_dawn = 1 if 4 <= hour <= 6 else 0
    glucose_zone = row.get('glucose_zone', 1)
    return np.array([hour_sin, hour_cos, sleep_quality, is_dawn, glucose_zone], dtype=np.float32)


def sliding_window_predict(model, features_df, seq_len=24, step=3, device='cpu', patient_id_idx=0):
    """滑动窗口预测，每隔 step 个时间步预测一次"""
    model.eval()
    all_preds = []
    all_targets = []
    all_timestamps = []
    all_initials = []

    n = len(features_df)
    start_indices = list(range(0, n - seq_len, step))
    print(f"  Sliding window: {len(start_indices)} windows (seq_len={seq_len}, step={step})")

    with torch.no_grad():
        for idx in start_indices:
            current_row = features_df.iloc[idx]
            initial_glucose = torch.tensor([[current_row['glucose_level']]],
                                           dtype=torch.float32, device=device)
            context = torch.tensor(get_context_features(current_row),
                                   dtype=torch.float32, device=device).unsqueeze(0)
            patient_ids = torch.tensor([patient_id_idx], dtype=torch.long, device=device)

            control_sequence = []
            targets = []
            for i in range(seq_len):
                ctrl_idx = idx + i
                target_idx = idx + i + 1
                if target_idx >= n:
                    break
                control_sequence.append(get_control_vector(features_df.iloc[ctrl_idx]))
                targets.append(features_df.iloc[target_idx]['glucose_level'])

            if len(control_sequence) < seq_len:
                continue

            control_seq = torch.tensor(control_sequence,
                                       dtype=torch.float32, device=device).unsqueeze(0)
            pred = model(initial_glucose, control_seq, context, patient_ids)

            all_preds.append(pred[0].cpu().numpy())
            all_targets.append(np.array(targets))
            all_timestamps.append(features_df.iloc[idx + 1: idx + 1 + seq_len]['timestamp'].values)
            all_initials.append(current_row['glucose_level'])

    return (np.array(all_preds), np.array(all_targets),
            all_timestamps, np.array(all_initials))


def find_event_windows(features_df, meal_df, bolus_df, exercise_df,
                       seq_len=24, n_windows=3, step=6):
    """找到包含最多事件（饮食+胰岛素+运动）的窗口"""
    window_scores = []
    for start_idx in range(0, len(features_df) - seq_len - 1, step):
        window = features_df.iloc[start_idx:start_idx + seq_len + 1]
        t_start = window['timestamp'].iloc[0]
        t_end = window['timestamp'].iloc[-1]
        n_meals = len(meal_df[(meal_df['timestamp'] >= t_start) &
                              (meal_df['timestamp'] <= t_end)])
        n_bolus = len(bolus_df[(bolus_df['timestamp'] >= t_start) &
                               (bolus_df['timestamp'] <= t_end)])
        n_exercise = len(exercise_df[(exercise_df['timestamp'] >= t_start) &
                                     (exercise_df['timestamp'] <= t_end)])
        score = n_meals + n_bolus + n_exercise
        window_scores.append((start_idx, score, n_meals, n_bolus, n_exercise))

    window_scores.sort(key=lambda x: x[1], reverse=True)
    selected = []
    for idx, score, nm, nb, ne in window_scores:
        if len(selected) >= n_windows:
            break
        if all(abs(idx - s) > seq_len for s in selected):
            selected.append(idx)
            print(f"  窗口 idx={idx}: 事件数={score} (饮食={nm}, 胰岛素={nb}, 运动={ne})")
    return selected


def predict_window(model, features_df, start_idx, seq_len, device, patient_id_idx=0):
    """从 start_idx 开始预测 seq_len 步血糖"""
    n = len(features_df)
    if start_idx + seq_len + 1 > n:
        return None

    current_row = features_df.iloc[start_idx]
    initial_glucose = torch.tensor([[current_row['glucose_level']]],
                                   dtype=torch.float32, device=device)
    context = torch.tensor(get_context_features(current_row),
                           dtype=torch.float32, device=device).unsqueeze(0)
    patient_ids = torch.tensor([patient_id_idx], dtype=torch.long, device=device)

    control_sequence = []
    true_glucose = []
    timestamps = []
    for i in range(seq_len):
        ctrl_idx = start_idx + i
        target_idx = start_idx + i + 1
        if target_idx >= n:
            break
        control_sequence.append(get_control_vector(features_df.iloc[ctrl_idx]))
        true_glucose.append(features_df.iloc[target_idx]['glucose_level'])
        timestamps.append(features_df.iloc[target_idx]['timestamp'])

    if len(control_sequence) < seq_len:
        return None

    control_seq = torch.tensor(control_sequence,
                               dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        pred = model(initial_glucose, control_seq, context, patient_ids)

    return {
        'pred': pred[0].cpu().numpy(),
        'true': np.array(true_glucose),
        'timestamps': timestamps,
        'initial_glucose': current_row['glucose_level'],
        'start_idx': start_idx,
    }


def interactive_predict(model, device,
                       initial_glucose=120, hour=8, sleep_quality=3,
                       carbs=60, bolus_dose=4.0, basal_rate=0.8,
                       exercise_intensity=0, heart_rate=72, isf=1.0,
                       seq_len=24, save_path=None, patient_id_idx=0):
    """交互式血糖预测：输入运动、饮食、胰岛素，预测血糖变化"""
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    is_dawn = 1 if 4 <= hour <= 6 else 0
    glucose_zone = 0 if initial_glucose < 70 else (
        1 if initial_glucose < 140 else (2 if initial_glucose < 180 else 3))
    context = torch.tensor([[hour_sin, hour_cos, sleep_quality, is_dawn, glucose_zone]],
                           dtype=torch.float32, device=device)

    iob_list = [bolus_dose * np.exp(-i * 0.25 / 2.0) for i in range(seq_len)]
    cob_list = [carbs * (0.7 * np.exp(-i * 0.25 / 0.5) + 0.3 * np.exp(-i * 0.25 / 3.0))
                for i in range(seq_len)]
    if exercise_intensity > 0:
        ex_list = [exercise_intensity if i < 4
                   else exercise_intensity * np.exp(-(i - 4) * 0.5)
                   for i in range(seq_len)]
    else:
        ex_list = [0.0] * seq_len

    control = torch.zeros(1, seq_len, 7, dtype=torch.float32, device=device)
    control[0, :, 0] = torch.tensor(iob_list)
    control[0, :, 1] = torch.tensor(cob_list)
    control[0, :, 2] = torch.tensor(ex_list)
    control[0, :, 3] = basal_rate
    control[0, :, 4] = bolus_dose
    control[0, :, 5] = heart_rate
    control[0, :, 6] = isf

    g0 = torch.tensor([[initial_glucose]], dtype=torch.float32, device=device)
    patient_ids = torch.tensor([patient_id_idx], dtype=torch.long, device=device)
    with torch.no_grad():
        pred = model(g0, control, context, patient_ids)
    pred_np = pred[0].cpu().numpy()

    time_min = np.arange(seq_len) * 15
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(time_min, pred_np, 'o-', color='#FF5722', linewidth=2.5, markersize=5, label='预测血糖')
    ax.axhline(y=initial_glucose, color='gray', linestyle=':', alpha=0.5,
               label=f'初始血糖 ({initial_glucose:.0f})')
    ax.axhspan(70, 180, alpha=0.08, color='green', label='目标范围 (70-180)')
    ax.axhline(y=70, color='red', linestyle='--', alpha=0.4, label='低血糖阈值')
    ax.axhline(y=180, color='orange', linestyle='--', alpha=0.4, label='高血糖阈值')

    input_text = (f'输入: 碳水={carbs}g, 胰岛素={bolus_dose}U, '
                  f'基础率={basal_rate}U/h, 运动={exercise_intensity}')
    ax.text(0.02, 0.98, input_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    peak_idx = np.argmax(np.abs(pred_np - initial_glucose))
    peak_val = pred_np[peak_idx]
    peak_time = time_min[peak_idx]
    direction = '↑' if peak_val > initial_glucose else '↓'
    ax.annotate(f'{direction} {peak_val:.0f} mg/dL\n({peak_time}min后)',
                xy=(peak_time, peak_val), xytext=(peak_time + 30, peak_val + 15),
                fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red'),
                bbox=dict(boxstyle='round', facecolor='lightyellow'))

    ax.set_xlabel('时间 (分钟)', fontsize=12)
    ax.set_ylabel('血糖 (mg/dL)', fontsize=12)
    ax.set_title(f'交互式血糖预测 (起始 {hour}:00, 血糖 {initial_glucose} mg/dL)', fontsize=13)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(40, 300)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  保存: {save_path}")
    plt.close()

    print(f"\n预测结果:")
    print(f"  起始血糖:   {initial_glucose:.0f} mg/dL")
    print(f"  30分钟后:   {pred_np[1]:.1f} mg/dL")
    print(f"  1小时后:    {pred_np[3]:.1f} mg/dL")
    print(f"  2小时后:    {pred_np[7]:.1f} mg/dL")
    print(f"  4小时后:    {pred_np[15]:.1f} mg/dL")
    print(f"  6小时后:    {pred_np[-1]:.1f} mg/dL")
    if pred_np.max() > 180:
        print(f"  !! 高血糖风险: 峰值 {pred_np.max():.1f} mg/dL")
    if pred_np.min() < 70:
        print(f"  !! 低血糖风险: 谷值 {pred_np.min():.1f} mg/dL")
    return pred_np
