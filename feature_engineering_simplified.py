import pandas as pd
import numpy as np
from datetime import datetime

# 加载数据
glucose_df = pd.read_csv('data/glucose_data.csv')
meal_df = pd.read_csv('data/meal_data.csv')
exercise_df = pd.read_csv('data/exercise_data.csv')
sleep_df = pd.read_csv('data/sleep_data.csv')
bolus_df = pd.read_csv('data/bolus_data.csv')
basal_df = pd.read_csv('data/basal_data.csv')
temp_basal_df = pd.read_csv('data/temp_basal_data.csv')
finger_df = pd.read_csv('data/finger_stick_data.csv')
heart_rate_df = pd.read_csv('data/heart_rate_data.csv')

# 将timestamp转换为datetime
glucose_df['timestamp'] = pd.to_datetime(glucose_df['timestamp'])
meal_df['timestamp'] = pd.to_datetime(meal_df['timestamp'])
exercise_df['timestamp'] = pd.to_datetime(exercise_df['timestamp'])
sleep_df['sleep_start'] = pd.to_datetime(sleep_df['sleep_start'])
sleep_df['sleep_end'] = pd.to_datetime(sleep_df['sleep_end'])
bolus_df['timestamp'] = pd.to_datetime(bolus_df['timestamp'])
basal_df['timestamp'] = pd.to_datetime(basal_df['timestamp'])
temp_basal_df['temp_basal_start'] = pd.to_datetime(temp_basal_df['temp_basal_start'])
temp_basal_df['temp_basal_end'] = pd.to_datetime(temp_basal_df['temp_basal_end'])
finger_df['timestamp'] = pd.to_datetime(finger_df['timestamp'])
heart_rate_df['timestamp'] = pd.to_datetime(heart_rate_df['timestamp'])

# 排序
glucose_df = glucose_df.sort_values(['patient_id', 'timestamp']).reset_index(drop=True)


def merge_asof_by_patient(left, right, on, by='patient_id', direction='backward', tolerance=None):
    """按patient_id分组执行merge_asof，避免全局排序问题"""
    result_parts = []
    right_cols = [c for c in right.columns if c != by]  # 右表去掉by列避免重复
    for pid in left[by].unique():
        l_part = left[left[by] == pid].sort_values(on)
        r_part = right[right[by] == pid][right_cols].sort_values(on)
        if r_part.empty:
            result_parts.append(l_part)
            continue
        merged = pd.merge_asof(l_part, r_part, on=on, direction=direction, tolerance=tolerance)
        result_parts.append(merged)
    return pd.concat(result_parts, ignore_index=True)


def build_features_vectorized(glucose_df, meal_df, exercise_df, sleep_df,
                               bolus_df, basal_df, temp_basal_df, finger_df, heart_rate_df):
    """向量化构建特征"""
    features_df = glucose_df.copy()
    patient_ids = features_df['patient_id'].unique()

    # ===== 1. Bolus IOB (向量化) =====
    print("  计算IOB (bolus)...")
    insulin_peak = 1.5
    insulin_duration = 6.0
    iob_values = np.zeros(len(features_df))

    for pid in patient_ids:
        g_mask = features_df['patient_id'] == pid
        g_indices = np.where(g_mask)[0]
        g_times = features_df.loc[g_mask, 'timestamp'].values
        b_data = bolus_df[bolus_df['patient_id'] == pid]

        for _, b_row in b_data.iterrows():
            b_time = b_row['timestamp']
            b_dose = b_row['bolus_dose']
            time_since = (g_times - np.datetime64(b_time)) / np.timedelta64(1, 'h')
            valid = (time_since > 0) & (time_since <= insulin_duration)
            if not np.any(valid):
                continue
            ts = time_since[valid]
            indices = g_indices[valid]
            iob_values[indices] += b_dose * (np.exp(-ts / insulin_duration) - np.exp(-ts / insulin_peak))

    features_df['IOB'] = np.maximum(0, iob_values)

    # ===== 2. Basal rate (按patient merge_asof) =====
    print("  匹配basal rate...")
    basal_sub = basal_df[['patient_id', 'timestamp', 'basal_rate']].sort_values(['patient_id', 'timestamp'])
    features_df = merge_asof_by_patient(features_df, basal_sub, on='timestamp')
    features_df['basal_rate'] = features_df['basal_rate'].fillna(0)

    # ===== 3. Temp basal rate =====
    print("  匹配temp_basal rate...")
    features_df['temp_basal_rate'] = 0.0
    for _, tb in temp_basal_df.iterrows():
        mask = (
            (features_df['patient_id'] == tb['patient_id']) &
            (features_df['timestamp'] >= tb['temp_basal_start']) &
            (features_df['timestamp'] <= tb['temp_basal_end'])
        )
        features_df.loc[mask, 'temp_basal_rate'] = tb['temp_basal_rate']

    features_df['effective_basal_rate'] = np.where(
        features_df['temp_basal_rate'] > 0,
        features_df['temp_basal_rate'],
        features_df['basal_rate']
    )
    features_df['basal_insulin_accumulated'] = features_df['effective_basal_rate'] * 2.0

    # ===== 4. Recent bolus dose =====
    print("  计算recent bolus dose...")
    features_df['recent_bolus_dose'] = 0.0
    features_df['bolus_dose_3h'] = 0.0

    for _, b in bolus_df.iterrows():
        mask_1h = (
            (features_df['patient_id'] == b['patient_id']) &
            (features_df['timestamp'] >= b['timestamp']) &
            (features_df['timestamp'] <= b['timestamp'] + pd.Timedelta(hours=1))
        )
        features_df.loc[mask_1h, 'recent_bolus_dose'] += b['bolus_dose']

        mask_3h = (
            (features_df['patient_id'] == b['patient_id']) &
            (features_df['timestamp'] >= b['timestamp']) &
            (features_df['timestamp'] <= b['timestamp'] + pd.Timedelta(hours=3))
        )
        features_df.loc[mask_3h, 'bolus_dose_3h'] += b['bolus_dose']

    # ===== 5. COB (向量化) =====
    print("  计算COB...")
    carb_peak = 1.0
    carb_duration = 4.0
    cob_values = np.zeros(len(features_df))

    for pid in patient_ids:
        g_mask = features_df['patient_id'] == pid
        g_indices = np.where(g_mask)[0]
        g_times = features_df.loc[g_mask, 'timestamp'].values
        m_data = meal_df[meal_df['patient_id'] == pid]

        for _, m in m_data.iterrows():
            m_time = m['timestamp']
            m_carbs = m['carbs']
            time_since = (g_times - np.datetime64(m_time)) / np.timedelta64(1, 'h')
            valid = (time_since > 0) & (time_since <= carb_duration)
            if not np.any(valid):
                continue
            ts = time_since[valid]
            indices = g_indices[valid]
            cob_values[indices] += m_carbs * (np.exp(-ts / carb_duration) - np.exp(-ts / carb_peak))

    features_df['COB'] = np.maximum(0, cob_values)

    # ===== 5b. Raw events for model-internal PK =====
    print("  计算原始事件(bolus_event, meal_event)...")
    bolus_event = np.zeros(len(features_df))
    timestamps = features_df['timestamp'].values
    for _, b in bolus_df.iterrows():
        b_time = np.datetime64(b['timestamp'])
        time_since = (timestamps - b_time) / np.timedelta64(1, 'm')
        closest_idx = np.argmin(np.abs(time_since))
        if np.abs(time_since[closest_idx]) <= 2.5:
            bolus_event[closest_idx] += b['bolus_dose']
    features_df['bolus_event'] = bolus_event

    meal_event = np.zeros(len(features_df))
    for _, m in meal_df.iterrows():
        m_time = np.datetime64(m['timestamp'])
        time_since = (timestamps - m_time) / np.timedelta64(1, 'm')
        closest_idx = np.argmin(np.abs(time_since))
        if np.abs(time_since[closest_idx]) <= 2.5:
            meal_event[closest_idx] += m['carbs']
    features_df['meal_event'] = meal_event

    # ===== 6. Heart rate (按patient merge_asof) =====
    print("  匹配heart rate...")
    hr_sub = heart_rate_df[['patient_id', 'timestamp', 'heart_rate']].sort_values(['patient_id', 'timestamp'])
    features_df = merge_asof_by_patient(features_df, hr_sub, on='timestamp', tolerance=pd.Timedelta('30min'))
    features_df['heart_rate'] = features_df['heart_rate'].fillna(70)
    features_df['heart_rate_delta'] = features_df.groupby('patient_id')['heart_rate'].diff().fillna(0)

    # ===== 7. Finger stick (按patient merge_asof) =====
    print("  匹配finger stick...")
    finger_sub = finger_df[['patient_id', 'timestamp', 'finger_stick_value']].sort_values(['patient_id', 'timestamp'])
    features_df = merge_asof_by_patient(features_df, finger_sub, on='timestamp', tolerance=pd.Timedelta('15min'))
    features_df['finger_stick_glucose'] = features_df['finger_stick_value'].fillna(0)
    features_df = features_df.drop(columns=['finger_stick_value'], errors='ignore')

    # ===== 8. Exercise intensity =====
    print("  计算exercise intensity...")
    features_df['exercise_intensity'] = 0.0
    for _, ex in exercise_df.iterrows():
        mask = (
            (features_df['patient_id'] == ex['patient_id']) &
            (features_df['timestamp'] >= ex['timestamp'] - pd.Timedelta(hours=1)) &
            (features_df['timestamp'] <= ex['timestamp'] + pd.Timedelta(hours=1))
        )
        intensity = ex['intensity'] * min(ex['duration'] / 60, 1.0) * 2
        features_df.loc[mask, 'exercise_intensity'] = np.maximum(
            features_df.loc[mask, 'exercise_intensity'], intensity
        )
    features_df['exercise_intensity'] = features_df['exercise_intensity'].clip(0, 10)

    # ===== 9. Sleep quality =====
    print("  匹配sleep quality...")
    features_df['sleep_quality'] = 3.0
    for _, sl in sleep_df.iterrows():
        mask = (
            (features_df['patient_id'] == sl['patient_id']) &
            (features_df['timestamp'] >= sl['sleep_start']) &
            (features_df['timestamp'] <= sl['sleep_end'] + pd.Timedelta(hours=12))
        )
        features_df.loc[mask, 'sleep_quality'] = sl['sleep_quality']

    # ===== 10. ISF =====
    print("  计算ISF...")
    features_df['hour'] = features_df['timestamp'].dt.hour
    circadian = 1.0 + 0.2 * np.sin(2 * np.pi * (features_df['hour'] - 6) / 24)
    sleep_debt = 5 - features_df['sleep_quality']
    features_df['ISF'] = (1 + sleep_debt * 0.1) * (1 - features_df['exercise_intensity'] * 0.05) * circadian

    # ===== 11. 血糖动态特征 =====
    features_df['prev_glucose'] = features_df.groupby('patient_id')['glucose_level'].shift(1)
    features_df['delta_glucose'] = features_df['glucose_level'] - features_df['prev_glucose']
    features_df['glucose_acceleration'] = features_df.groupby('patient_id')['delta_glucose'].diff().fillna(0)

    # ===== 12. 交互特征（增强胰岛素交互） =====
    features_df['glucose_x_IOB'] = features_df['glucose_level'] * features_df['IOB']
    features_df['glucose_x_bolus'] = features_df['glucose_level'] * features_df['recent_bolus_dose']
    features_df['IOB_x_ISF'] = features_df['IOB'] * features_df['ISF']
    features_df['bolus_x_exercise'] = features_df['recent_bolus_dose'] * features_df['exercise_intensity']
    features_df['basal_x_bolus'] = features_df['effective_basal_rate'] * features_df['recent_bolus_dose']
    features_df['COB_x_IOB'] = features_df['COB'] * features_df['IOB']
    features_df['heart_rate_x_IOB'] = features_df['heart_rate'] * features_df['IOB']

    # ===== 13. 上下文特征 =====
    features_df['hour_sin'] = np.sin(2 * np.pi * features_df['hour'] / 24)
    features_df['hour_cos'] = np.cos(2 * np.pi * features_df['hour'] / 24)
    features_df['is_dawn'] = features_df['hour'].between(4, 6).astype(int)
    features_df['is_weekend'] = (features_df['timestamp'].dt.dayofweek >= 5).astype(int)

    features_df['glucose_zone'] = pd.cut(features_df['glucose_level'],
                                     bins=[0, 70, 140, 180, float('inf')],
                                     labels=['low', 'normal', 'high', 'very_high'])
    features_df['glucose_zone'] = features_df['glucose_zone'].cat.codes

    # 清除缺失值
    features_df = features_df.dropna(subset=['delta_glucose'])

    return features_df


# 主程序
if __name__ == "__main__":
    print("开始特征工程...")
    features_df = build_features_vectorized(
        glucose_df, meal_df, exercise_df, sleep_df,
        bolus_df, basal_df, temp_basal_df, finger_df, heart_rate_df
    )

    features_df.to_csv('data/glucose_features_simplified.csv', index=False)
    print("特征工程完成，已保存")
    print(f"特征列: {list(features_df.columns)}")
    print(f"数据行数: {len(features_df)}")

    for col in ['IOB', 'recent_bolus_dose', 'effective_basal_rate', 'bolus_dose_3h']:
        if col in features_df.columns:
            print(f"  {col}: mean={features_df[col].mean():.4f}, max={features_df[col].max():.4f}, "
                  f"non-zero%={(features_df[col] > 0).mean()*100:.1f}%")
