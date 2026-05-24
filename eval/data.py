"""XML 解析 + 特征工程"""
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import functools
print = functools.partial(print, flush=True)


def parse_xml(xml_path):
    """解析 OhioT1DM XML 文件，返回各类型 DataFrame"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    patient_id = root.attrib['id']

    def parse_events(elem, ts_key='ts'):
        rows = []
        if elem is None:
            return pd.DataFrame()
        for e in elem.findall('event'):
            row = dict(e.attrib)
            row['patient_id'] = patient_id
            rows.append(row)
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    df_glucose = parse_events(root.find('glucose_level'))
    if not df_glucose.empty:
        df_glucose['timestamp'] = pd.to_datetime(df_glucose['ts'], format='%d-%m-%Y %H:%M:%S')
        df_glucose['glucose_level'] = df_glucose['value'].astype(float)
        df_glucose = df_glucose[['timestamp', 'glucose_level', 'patient_id']]

    df_meal = parse_events(root.find('meal'))
    if not df_meal.empty:
        df_meal['timestamp'] = pd.to_datetime(df_meal['ts'], format='%d-%m-%Y %H:%M:%S')
        df_meal['carbs'] = df_meal['carbs'].astype(float)
        df_meal = df_meal[['timestamp', 'carbs', 'patient_id']]

    df_exercise = parse_events(root.find('exercise'))
    if not df_exercise.empty:
        df_exercise['timestamp'] = pd.to_datetime(df_exercise['ts'], format='%d-%m-%Y %H:%M:%S')
        df_exercise['intensity'] = df_exercise['intensity'].astype(float)
        df_exercise['duration'] = df_exercise['duration'].astype(float)
        df_exercise = df_exercise[['timestamp', 'intensity', 'duration', 'patient_id']]

    df_hr = parse_events(root.find('basis_heart_rate'))
    if not df_hr.empty:
        df_hr['timestamp'] = pd.to_datetime(df_hr['ts'], format='%d-%m-%Y %H:%M:%S')
        df_hr['heart_rate'] = df_hr['value'].astype(float)
        df_hr = df_hr[['timestamp', 'heart_rate', 'patient_id']]

    df_sleep = parse_events(root.find('sleep'))
    if not df_sleep.empty:
        df_sleep['sleep_start'] = pd.to_datetime(df_sleep['ts_begin'], format='%d-%m-%Y %H:%M:%S')
        df_sleep['sleep_end'] = pd.to_datetime(df_sleep['ts_end'], format='%d-%m-%Y %H:%M:%S')
        df_sleep['sleep_quality'] = df_sleep['quality'].astype(float)
        df_sleep = df_sleep[['sleep_start', 'sleep_end', 'sleep_quality', 'patient_id']]

    df_basal = parse_events(root.find('basal'))
    if not df_basal.empty:
        df_basal['timestamp'] = pd.to_datetime(df_basal['ts'], format='%d-%m-%Y %H:%M:%S')
        df_basal['basal_rate'] = df_basal['value'].astype(float)
        df_basal = df_basal[['timestamp', 'basal_rate', 'patient_id']]

    df_temp_basal = parse_events(root.find('temp_basal'))
    if not df_temp_basal.empty:
        df_temp_basal['temp_basal_start'] = pd.to_datetime(df_temp_basal['ts_begin'], format='%d-%m-%Y %H:%M:%S')
        df_temp_basal['temp_basal_end'] = pd.to_datetime(df_temp_basal['ts_end'], format='%d-%m-%Y %H:%M:%S')
        df_temp_basal['temp_basal_rate'] = df_temp_basal['value'].astype(float)
        df_temp_basal = df_temp_basal[['temp_basal_start', 'temp_basal_end', 'temp_basal_rate', 'patient_id']]

    df_bolus = parse_events(root.find('bolus'))
    if not df_bolus.empty:
        df_bolus['timestamp'] = pd.to_datetime(df_bolus['ts_begin'], format='%d-%m-%Y %H:%M:%S')
        df_bolus['bolus_dose'] = df_bolus['dose'].astype(float)
        df_bolus['bwz_carb_input'] = df_bolus.get('bwz_carb_input', pd.Series(0, index=df_bolus.index))
        if 'bwz_carb_input' in df_bolus.columns:
            df_bolus['bwz_carb_input'] = df_bolus['bwz_carb_input'].astype(float)
        df_bolus = df_bolus[['timestamp', 'bolus_dose', 'bwz_carb_input', 'patient_id']]

    df_finger = parse_events(root.find('finger_stick'))
    if not df_finger.empty:
        df_finger['timestamp'] = pd.to_datetime(df_finger['ts'], format='%d-%m-%Y %H:%M:%S')
        df_finger['finger_stick_value'] = df_finger['value'].astype(float)
        df_finger = df_finger[['timestamp', 'finger_stick_value', 'patient_id']]

    return {
        'glucose': df_glucose,
        'meal': df_meal,
        'exercise': df_exercise,
        'heart_rate': df_hr,
        'sleep': df_sleep,
        'basal': df_basal,
        'temp_basal': df_temp_basal,
        'bolus': df_bolus,
        'finger_stick': df_finger,
        'patient_id': patient_id,
    }


def merge_asof_by_patient(left, right, on, by='patient_id', direction='backward', tolerance=None):
    result_parts = []
    right_cols = [c for c in right.columns if c != by]
    for pid in left[by].unique():
        l_part = left[left[by] == pid].sort_values(on)
        r_part = right[right[by] == pid][right_cols].sort_values(on)
        if r_part.empty:
            result_parts.append(l_part)
            continue
        merged = pd.merge_asof(l_part, r_part, on=on, direction=direction, tolerance=tolerance)
        result_parts.append(merged)
    return pd.concat(result_parts, ignore_index=True)


def build_features(glucose_df, meal_df, exercise_df, sleep_df,
                   bolus_df, basal_df, temp_basal_df, finger_df, heart_rate_df):
    """与 feature_engineering_simplified.py 一致的向量化特征构建"""
    features_df = glucose_df.copy()
    patient_ids = features_df['patient_id'].unique()

    # 1. IOB
    print("  Computing IOB...")
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

    # 2. Basal rate
    print("  Matching basal rate...")
    basal_sub = basal_df[['patient_id', 'timestamp', 'basal_rate']].sort_values(['patient_id', 'timestamp'])
    features_df = merge_asof_by_patient(features_df, basal_sub, on='timestamp')
    features_df['basal_rate'] = features_df['basal_rate'].fillna(0)

    # 3. Temp basal rate
    print("  Matching temp_basal rate...")
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

    # 4. Recent bolus dose
    print("  Computing recent bolus dose...")
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

    # 5. COB
    print("  Computing COB...")
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

    # 6. Heart rate
    print("  Matching heart rate...")
    hr_sub = heart_rate_df[['patient_id', 'timestamp', 'heart_rate']].sort_values(['patient_id', 'timestamp'])
    features_df = merge_asof_by_patient(features_df, hr_sub, on='timestamp', tolerance=pd.Timedelta('30min'))
    features_df['heart_rate'] = features_df['heart_rate'].fillna(70)
    features_df['heart_rate_delta'] = features_df.groupby('patient_id')['heart_rate'].diff().fillna(0)

    # 7. Finger stick
    print("  Matching finger stick...")
    finger_sub = finger_df[['patient_id', 'timestamp', 'finger_stick_value']].sort_values(['patient_id', 'timestamp'])
    features_df = merge_asof_by_patient(features_df, finger_sub, on='timestamp', tolerance=pd.Timedelta('15min'))
    features_df['finger_stick_glucose'] = features_df['finger_stick_value'].fillna(0)
    features_df = features_df.drop(columns=['finger_stick_value'], errors='ignore')

    # 8. Exercise intensity
    print("  Computing exercise intensity...")
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

    # 9. Sleep quality
    print("  Matching sleep quality...")
    features_df['sleep_quality'] = 3.0
    for _, sl in sleep_df.iterrows():
        mask = (
            (features_df['patient_id'] == sl['patient_id']) &
            (features_df['timestamp'] >= sl['sleep_start']) &
            (features_df['timestamp'] <= sl['sleep_end'] + pd.Timedelta(hours=12))
        )
        features_df.loc[mask, 'sleep_quality'] = sl['sleep_quality']

    # 10. ISF
    print("  Computing ISF...")
    features_df['hour'] = features_df['timestamp'].dt.hour
    circadian = 1.0 + 0.2 * np.sin(2 * np.pi * (features_df['hour'] - 6) / 24)
    sleep_debt = 5 - features_df['sleep_quality']
    features_df['ISF'] = (1 + sleep_debt * 0.1) * (1 - features_df['exercise_intensity'] * 0.05) * circadian

    # 11. Blood glucose dynamics
    features_df['prev_glucose'] = features_df.groupby('patient_id')['glucose_level'].shift(1)
    features_df['delta_glucose'] = features_df['glucose_level'] - features_df['prev_glucose']
    features_df['glucose_acceleration'] = features_df.groupby('patient_id')['delta_glucose'].diff().fillna(0)

    # 12. Interaction features
    features_df['glucose_x_IOB'] = features_df['glucose_level'] * features_df['IOB']
    features_df['glucose_x_bolus'] = features_df['glucose_level'] * features_df['recent_bolus_dose']
    features_df['IOB_x_ISF'] = features_df['IOB'] * features_df['ISF']
    features_df['bolus_x_exercise'] = features_df['recent_bolus_dose'] * features_df['exercise_intensity']
    features_df['basal_x_bolus'] = features_df['effective_basal_rate'] * features_df['recent_bolus_dose']
    features_df['COB_x_IOB'] = features_df['COB'] * features_df['IOB']
    features_df['heart_rate_x_IOB'] = features_df['heart_rate'] * features_df['IOB']

    # 13. Context features
    features_df['hour_sin'] = np.sin(2 * np.pi * features_df['hour'] / 24)
    features_df['hour_cos'] = np.cos(2 * np.pi * features_df['hour'] / 24)
    features_df['is_dawn'] = features_df['hour'].between(4, 6).astype(int)
    features_df['is_weekend'] = (features_df['timestamp'].dt.dayofweek >= 5).astype(int)
    features_df['glucose_zone'] = pd.cut(features_df['glucose_level'],
                                         bins=[0, 70, 140, 180, float('inf')],
                                         labels=['low', 'normal', 'high', 'very_high'])
    features_df['glucose_zone'] = features_df['glucose_zone'].cat.codes

    features_df = features_df.dropna(subset=['delta_glucose'])
    return features_df
