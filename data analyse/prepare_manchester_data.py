"""
Manchester → Ohio 特征映射与数据准备
============================================
将 ManchesterCSCoordinatedDiabetesStudy 数据转换为
Ohio 模型兼容的特征格式，并按 60/30/10 划分训练/验证/测试集。

输出:
- data/glucose_features_manchester.csv  (全部数据)
- data/manchester_train.csv             (60%)
- data/manchester_val.csv               (30%)
- data/manchester_test.csv              (10%)
"""

import os
import warnings
import numpy as np
import pandas as pd
from datetime import timedelta

warnings.filterwarnings("ignore")

BASE_DIR = r"D:\ohio\ManchesterCSCoordinatedDiabetesStudy-main"
ACTIVITY_DIR = os.path.join(BASE_DIR, "Activity Data")
GLUCOSE_DIR = os.path.join(BASE_DIR, "Glucose Data")
BASAL_DIR = os.path.join(BASE_DIR, "Insulin Data", "Basal Data")
BOLUS_DIR = os.path.join(BASE_DIR, "Insulin Data", "Bolus Data")
NUTRITION_DIR = os.path.join(BASE_DIR, "Nutrition Data")
SLEEP_DIR = os.path.join(BASE_DIR, "Sleep Data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 常量 ===
MMOL_TO_MGDL = 18.0182        # 血糖单位转换
INSULIN_PEAK = 1.5            # IOB 峰值时间 (h)
INSULIN_DURATION = 6.0        # IOB 持续时间 (h)
CARB_PEAK = 1.0               # COB 峰值时间 (h)
CARB_DURATION = 4.0           # COB 持续时间 (h)

# === 工具函数 ===
def load_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig")


def parse_timestamp(series):
    return pd.to_datetime(series.str.strip().str.lstrip("﻿"), dayfirst=True, format="mixed")


def get_participant_files(directory, prefix):
    files = {}
    if not os.path.isdir(directory):
        return files
    for f in sorted(os.listdir(directory)):
        if f.startswith(prefix) and f.endswith(".csv"):
            pid = f.replace(prefix, "").replace(".csv", "")
            # 过滤纯数字PID
            if pid.isdigit():
                files[pid] = os.path.join(directory, f)
            else:
                # 处理 sleep 文件: UoM2301sleeptime.csv
                for p in ["UoM", "sleeptime"]:
                    f2 = f.replace(p, "")
                pid2 = f2.replace(".csv", "")
                if pid2.isdigit():
                    files[pid2] = os.path.join(directory, f)
    return files


# === 1. 加载所有原始数据 ===
def load_all_data():
    """加载所有参与者的所有数据模态，统一到 5-min 血糖网格"""
    print("加载 Manchester 数据...")

    # 获取所有参与者的血糖文件
    glucose_files = {}
    for f in sorted(os.listdir(GLUCOSE_DIR)):
        if f.startswith("UoMGlucose") and f.endswith(".csv"):
            pid = f.replace("UoMGlucose", "").replace(".csv", "")
            glucose_files[pid] = os.path.join(GLUCOSE_DIR, f)

    activity_files = get_participant_files(ACTIVITY_DIR, "UoMActivity")
    basal_files = get_participant_files(BASAL_DIR, "UoMBasal")
    bolus_files = get_participant_files(BOLUS_DIR, "UoMBolus")
    nutrition_files = get_participant_files(NUTRITION_DIR, "UoMNutrition")

    # Sleep-level files (5-min heart rate etc)
    sleep_level_files = get_participant_files(SLEEP_DIR, "UoMsleep")
    # Sleep-summary files
    sleep_summary_files = {}
    for f in sorted(os.listdir(SLEEP_DIR)):
        if "sleeptime" in f and f.endswith(".csv"):
            pid = f.replace("UoM", "").replace("sleeptime.csv", "")
            # 可能还有 UoMsleep 中匹配的，需要去掉前缀 upid
            if pid.isdigit():
                sleep_summary_files[pid] = os.path.join(SLEEP_DIR, f)

    all_features = []

    for pid in sorted(glucose_files.keys()):
        print(f"  处理参与者 {pid}...")

        # --- 血糖基础数据 ---
        gdf = load_csv(glucose_files[pid])
        gdf["bg_ts"] = parse_timestamp(gdf["bg_ts"])
        gdf["value"] = pd.to_numeric(gdf["value"], errors="coerce")
        gdf = gdf.dropna(subset=["value"]).sort_values("bg_ts").reset_index(drop=True)
        gdf["glucose_level"] = gdf["value"] * MMOL_TO_MGDL  # mmol/L → mg/dL

        # 去重时间戳（取均值）
        gdf = gdf.groupby("bg_ts")["glucose_level"].mean().reset_index()

        # 5-min 重采样确保连续
        gdf = gdf.set_index("bg_ts")
        full_range = pd.date_range(start=gdf.index.min(), end=gdf.index.max(), freq="5min")
        gdf = gdf.reindex(full_range)
        gdf = gdf.dropna(subset=["glucose_level"]).reset_index()
        gdf.rename(columns={"index": "timestamp"}, inplace=True)
        gdf["patient_id"] = int(pid)
        gdf = gdf[["timestamp", "glucose_level", "patient_id"]].copy()
        timestamps = gdf["timestamp"].values
        n = len(gdf)

        # --- 1. IOB (从 bolus 计算) ---
        iob = np.zeros(n)
        if pid in bolus_files:
            bdf = load_csv(bolus_files[pid])
            bdf["bolus_ts"] = parse_timestamp(bdf["bolus_ts"])
            bdf["bolus_dose"] = pd.to_numeric(bdf["bolus_dose"], errors="coerce").fillna(0)
            for _, row in bdf.iterrows():
                b_time = row["bolus_ts"].to_datetime64()
                b_dose = row["bolus_dose"]
                if b_dose <= 0:
                    continue
                time_since = (timestamps - b_time) / np.timedelta64(1, "h")
                valid = (time_since > 0) & (time_since <= INSULIN_DURATION)
                if not np.any(valid):
                    continue
                ts = time_since[valid]
                iob[valid] += b_dose * (np.exp(-ts / INSULIN_DURATION) - np.exp(-ts / INSULIN_PEAK))
        gdf["IOB"] = np.maximum(0, iob)

        # --- 1b. Recent bolus dose (1h 窗口) & 3h 窗口 ---
        recent_bolus = np.zeros(n)
        bolus_3h = np.zeros(n)
        if pid in bolus_files:
            bdf = load_csv(bolus_files[pid])
            bdf["bolus_ts"] = parse_timestamp(bdf["bolus_ts"])
            bdf["bolus_dose"] = pd.to_numeric(bdf["bolus_dose"], errors="coerce").fillna(0)
            for _, row in bdf.iterrows():
                b_time = row["bolus_ts"].to_datetime64()
                b_dose = row["bolus_dose"]
                if b_dose <= 0:
                    continue
                time_since = (timestamps - b_time) / np.timedelta64(1, "h")
                valid_1h = (time_since > 0) & (time_since <= 1)
                valid_3h = (time_since > 0) & (time_since <= 3)
                recent_bolus[valid_1h] += b_dose
                bolus_3h[valid_3h] += b_dose
        gdf["recent_bolus_dose"] = recent_bolus
        gdf["bolus_dose_3h"] = bolus_3h

        # --- 2. Basal rate ---
        gdf["basal_rate"] = 0.0
        gdf["temp_basal_rate"] = 0.0
        if pid in basal_files:
            bdf = load_csv(basal_files[pid])
            if len(bdf) > 0:
                bdf["basal_ts"] = parse_timestamp(bdf["basal_ts"])
                bdf["basal_dose"] = pd.to_numeric(bdf["basal_dose"], errors="coerce").fillna(0)

                # 判断胰岛素类型: Lantus(长效)剂量是日总量, 需转为 U/h
                # R(短效)通常来自胰岛素泵, 已是 U/h
                if "insulin_kind" in bdf.columns:
                    l_count = (bdf["insulin_kind"] == "L").sum()
                    r_count = (bdf["insulin_kind"] == "R").sum()
                    is_long_acting = l_count > r_count and l_count > 0
                else:
                    is_long_acting = False

                # 将basal以5min间隔对齐
                bdf = bdf.sort_values("basal_ts").set_index("basal_ts")
                bdf = bdf.resample("5min")["basal_dose"].mean().reset_index()
                # merge_asof 到血糖时间线
                bdf["timestamp"] = bdf["basal_ts"]
                gdf = pd.merge_asof(
                    gdf.sort_values("timestamp"),
                    bdf[["timestamp", "basal_dose"]].sort_values("timestamp"),
                    on="timestamp", direction="backward"
                )
                gdf["basal_rate"] = gdf["basal_dose"].fillna(0)

                # Lantus 日总量→小时速率 (除以24)
                if is_long_acting:
                    gdf["basal_rate"] = gdf["basal_rate"] / 24.0

                gdf.drop(columns=["basal_dose"], inplace=True)

        # Manchester 没有 temp_basal，所以 effective_basal_rate = basal_rate
        gdf["effective_basal_rate"] = gdf["basal_rate"]
        gdf["basal_insulin_accumulated"] = gdf["effective_basal_rate"] * 2.0

        # --- 3. COB (从 nutrition 计算) ---
        cob = np.zeros(n)
        if pid in nutrition_files:
            ndf = load_csv(nutrition_files[pid])
            ndf["meal_ts"] = parse_timestamp(ndf["meal_ts"])
            ndf["carbs_g"] = pd.to_numeric(ndf["carbs_g"], errors="coerce").fillna(0)
            for _, row in ndf.iterrows():
                m_time = row["meal_ts"].to_datetime64()
                m_carbs = row["carbs_g"]
                if m_carbs <= 0:
                    continue
                time_since = (timestamps - m_time) / np.timedelta64(1, "h")
                valid = (time_since > 0) & (time_since <= CARB_DURATION)
                if not np.any(valid):
                    continue
                ts = time_since[valid]
                cob[valid] += m_carbs * (np.exp(-ts / CARB_DURATION) - np.exp(-ts / CARB_PEAK))
        gdf["COB"] = np.maximum(0, cob)

        # --- 3b. Raw events for model-internal PK ---
        bolus_event = np.zeros(n)
        if pid in bolus_files:
            bdf = load_csv(bolus_files[pid])
            bdf["bolus_ts"] = parse_timestamp(bdf["bolus_ts"])
            bdf["bolus_dose"] = pd.to_numeric(bdf["bolus_dose"], errors="coerce").fillna(0)
            for _, row in bdf.iterrows():
                b_time = row["bolus_ts"].to_datetime64()
                b_dose = row["bolus_dose"]
                if b_dose <= 0:
                    continue
                # 找最近的timestamp（2.5min内），标记为原始事件
                time_since = (timestamps - b_time) / np.timedelta64(1, "m")
                closest_idx = np.argmin(np.abs(time_since))
                if np.abs(time_since[closest_idx]) <= 2.5:
                    bolus_event[closest_idx] += b_dose
        gdf["bolus_event"] = bolus_event

        meal_event = np.zeros(n)
        if pid in nutrition_files:
            ndf = load_csv(nutrition_files[pid])
            ndf["meal_ts"] = parse_timestamp(ndf["meal_ts"])
            ndf["carbs_g"] = pd.to_numeric(ndf["carbs_g"], errors="coerce").fillna(0)
            for _, row in ndf.iterrows():
                m_time = row["meal_ts"].to_datetime64()
                m_carbs = row["carbs_g"]
                if m_carbs <= 0:
                    continue
                time_since = (timestamps - m_time) / np.timedelta64(1, "m")
                closest_idx = np.argmin(np.abs(time_since))
                if np.abs(time_since[closest_idx]) <= 2.5:
                    meal_event[closest_idx] += m_carbs
        gdf["meal_event"] = meal_event

        # --- 4. Exercise intensity (从 activity 计算) ---
        gdf["exercise_intensity"] = 0.0
        if pid in activity_files:
            adf = load_csv(activity_files[pid])
            adf["activity_ts"] = parse_timestamp(adf["activity_ts"])
            for _, row in adf.iterrows():
                met = row.get("met", 0)
                step_count = row.get("step_count", 0)
                duration_s = row.get("duration_s", 900)
                if met <= 1 and step_count == 0:
                    continue
                ex_time = row["activity_ts"].to_datetime64()
                # 计算等效 intensity: MET 归一化 + step_count 贡献
                met_factor = min(met / 10.0, 1.0)
                step_factor = min(step_count / 200.0, 1.0)
                intensity = max(met_factor, step_factor) * min(duration_s / 900.0, 2.0)
                intensity = min(intensity * 5.0, 10.0)  # 缩放到 [0, 10]

                # 活动影响窗口: +/- 1h
                time_since = (timestamps - ex_time) / np.timedelta64(1, "h")
                valid = (time_since >= -1) & (time_since <= 1)
                decay = np.maximum(0, 1 - np.abs(time_since[valid]))
                gdf.loc[valid, "exercise_intensity"] = np.maximum(
                    gdf.loc[valid, "exercise_intensity"],
                    intensity * decay
                )
        gdf["exercise_intensity"] = gdf["exercise_intensity"].clip(0, 10)

        # --- 5. Heart rate (从 sleep 数据) ---
        gdf["heart_rate"] = 70  # 默认值
        if pid in sleep_level_files:
            sdf = load_csv(sleep_level_files[pid])
            sdf["sleep_ts"] = parse_timestamp(sdf["sleep_ts"])
            sdf["heart_rate"] = pd.to_numeric(sdf["heart_rate"], errors="coerce")
            sdf = sdf[sdf["heart_rate"] > 0].sort_values("sleep_ts")[["sleep_ts", "heart_rate"]]
            if len(sdf) > 0:
                sdf.rename(columns={"sleep_ts": "timestamp"}, inplace=True)
                gdf = pd.merge_asof(
                    gdf.sort_values("timestamp"),
                    sdf.sort_values("timestamp"),
                    on="timestamp", direction="backward"
                )
                gdf["heart_rate"] = gdf["heart_rate_y"].fillna(70)
                gdf.drop(columns=["heart_rate_y"], inplace=True)

        gdf["heart_rate_delta"] = gdf["heart_rate"].diff().fillna(0)

        # --- 6. Sleep quality (从 sleep summary) ---
        gdf["sleep_quality"] = 3  # 默认中等
        if pid in sleep_summary_files:
            try:
                sdf = load_csv(sleep_summary_files[pid])
                if len(sdf) > 0:
                    sdf["sleep_start_ts"] = parse_timestamp(sdf["sleep_start_ts"])
                    sdf["sleep_end_ts"] = parse_timestamp(sdf["sleep_end_ts"])
                    sdf["deep_sleep_s"] = pd.to_numeric(sdf["deep_sleep_s"], errors="coerce").fillna(0)
                    sdf["light_sleep_s"] = pd.to_numeric(sdf["light_sleep_s"], errors="coerce").fillna(0)
                    sdf["rem_sleep_s"] = pd.to_numeric(sdf["rem_sleep_s"], errors="coerce").fillna(0)
                    sdf["awake_sleep_s"] = pd.to_numeric(sdf["awake_sleep_s"], errors="coerce").fillna(0)

                    for _, sl in sdf.iterrows():
                        total = sl["deep_sleep_s"] + sl["light_sleep_s"] + sl["rem_sleep_s"] + sl["awake_sleep_s"]
                        efficiency = (total - sl["awake_sleep_s"]) / max(total, 1)
                        if efficiency > 0.9:
                            sq = 5
                        elif efficiency > 0.8:
                            sq = 4
                        elif efficiency > 0.7:
                            sq = 3
                        elif efficiency > 0.6:
                            sq = 2
                        else:
                            sq = 1

                        mask = (gdf["timestamp"] >= sl["sleep_start_ts"]) & \
                               (gdf["timestamp"] <= sl["sleep_end_ts"] + timedelta(hours=12))
                        gdf.loc[mask, "sleep_quality"] = max(gdf.loc[mask, "sleep_quality"].max(), sq)

            except Exception:
                pass

        # --- 7. ISF (Circadian + sleep_debt + exercise) ---
        gdf["hour"] = gdf["timestamp"].dt.hour
        circadian = 1.0 + 0.2 * np.sin(2 * np.pi * (gdf["hour"] - 6) / 24)
        sleep_debt = 5 - gdf["sleep_quality"]
        gdf["ISF"] = (1 + sleep_debt * 0.1) * (1 - gdf["exercise_intensity"] * 0.05) * circadian

        # --- 8. Glucose dynamics ---
        gdf["prev_glucose"] = gdf["glucose_level"].shift(1)
        gdf["delta_glucose"] = gdf["glucose_level"] - gdf["prev_glucose"]
        gdf["glucose_acceleration"] = gdf["delta_glucose"].diff().fillna(0)

        # --- 9. 交互特征 ---
        gdf["glucose_x_IOB"] = gdf["glucose_level"] * gdf["IOB"]
        gdf["glucose_x_bolus"] = gdf["glucose_level"] * gdf["recent_bolus_dose"]
        gdf["IOB_x_ISF"] = gdf["IOB"] * gdf["ISF"]
        gdf["bolus_x_exercise"] = gdf["recent_bolus_dose"] * gdf["exercise_intensity"]
        gdf["basal_x_bolus"] = gdf["effective_basal_rate"] * gdf["recent_bolus_dose"]
        gdf["COB_x_IOB"] = gdf["COB"] * gdf["IOB"]
        gdf["heart_rate_x_IOB"] = gdf["heart_rate"] * gdf["IOB"]

        # --- 10. 上下文特征 ---
        gdf["hour_sin"] = np.sin(2 * np.pi * gdf["hour"] / 24)
        gdf["hour_cos"] = np.cos(2 * np.pi * gdf["hour"] / 24)
        gdf["is_dawn"] = gdf["hour"].between(4, 6).astype(int)
        gdf["is_weekend"] = (gdf["timestamp"].dt.dayofweek >= 5).astype(int)

        gdf["glucose_zone"] = pd.cut(
            gdf["glucose_level"],
            bins=[0, 70, 140, 180, float("inf")],
            labels=[0, 1, 2, 3]
        ).astype(float).fillna(1)

        # --- 11. Finger stick (Manchester 没有，设为 0) ---
        gdf["finger_stick_glucose"] = 0

        # 清理 NaN
        fill_defaults = {
            "IOB": 0, "COB": 0, "exercise_intensity": 0,
            "effective_basal_rate": 0, "recent_bolus_dose": 0,
            "heart_rate": 70, "ISF": 1.0,
            "basal_rate": 0, "temp_basal_rate": 0,
            "basal_insulin_accumulated": 0, "bolus_dose_3h": 0,
            "heart_rate_delta": 0, "finger_stick_glucose": 0,
            "sleep_quality": 3, "delta_glucose": 0,
            "glucose_acceleration": 0, "prev_glucose": 0,
            "glucose_x_IOB": 0, "glucose_x_bolus": 0,
            "IOB_x_ISF": 0, "bolus_x_exercise": 0,
            "basal_x_bolus": 0, "COB_x_IOB": 0,
            "heart_rate_x_IOB": 0,
            "bolus_event": 0, "meal_event": 0,
        }
        for col, default in fill_defaults.items():
            if col in gdf.columns:
                gdf[col] = gdf[col].fillna(default)

        gdf = gdf.dropna(subset=["delta_glucose"])
        all_features.append(gdf)

    if not all_features:
        raise ValueError("没有成功处理任何参与者数据!")

    result = pd.concat(all_features, ignore_index=True)

    # 确保列顺序与 Ohio 一致
    ohio_cols = ["timestamp", "glucose_level", "patient_id",
                 "IOB", "basal_rate", "temp_basal_rate", "effective_basal_rate",
                 "basal_insulin_accumulated", "recent_bolus_dose", "bolus_dose_3h",
                 "COB", "heart_rate", "heart_rate_delta", "finger_stick_glucose",
                 "exercise_intensity", "sleep_quality", "hour", "ISF",
                 "prev_glucose", "delta_glucose", "glucose_acceleration",
                 "glucose_x_IOB", "glucose_x_bolus", "IOB_x_ISF",
                 "bolus_x_exercise", "basal_x_bolus", "COB_x_IOB",
                 "heart_rate_x_IOB", "hour_sin", "hour_cos",
                 "is_dawn", "is_weekend", "glucose_zone",
                 "bolus_event", "meal_event"]
    existing_cols = [c for c in ohio_cols if c in result.columns]
    result = result[existing_cols]

    print(f"  总行数: {len(result)}, 参与者数: {result['patient_id'].nunique()}")
    return result


# === 2. 按参与者划分 60/30/10 ===
def split_by_participant(df, train_ratio=0.6, val_ratio=0.3):
    """按参与者+时间顺序划分，不交叉"""
    train_parts, val_parts, test_parts = [], [], []

    for pid, group in df.groupby("patient_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        n = len(group)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_parts.append(group.iloc[:train_end])
        val_parts.append(group.iloc[train_end:val_end])
        test_parts.append(group.iloc[val_end:])

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    return train_df, val_df, test_df


# === 3. 保存 ===
def save_data(df, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  保存 {path}  ({len(df)} 行)")


def print_data_stats(train, val, test):
    print("\n=== 数据划分统计 ===")
    print(f"训练集:   {len(train)} 行, {train['patient_id'].nunique()} 名参与者")
    print(f"验证集:   {len(val)} 行, {val['patient_id'].nunique()} 名参与者")
    print(f"测试集:   {len(test)} 行, {test['patient_id'].nunique()} 名参与者")
    print(f"总计:     {len(train) + len(val) + test} 行")

    # 按参与者列出比例
    print("\n--- 按参与者划分详情 ---")
    all_pids = sorted(set(train["patient_id"].unique()) |
                      set(val["patient_id"].unique()) |
                      set(test["patient_id"].unique()))
    for pid in all_pids:
        t_n = len(train[train["patient_id"] == pid]) if pid in train["patient_id"].values else 0
        v_n = len(val[val["patient_id"] == pid]) if pid in val["patient_id"].values else 0
        e_n = len(test[test["patient_id"] == pid]) if pid in test["patient_id"].values else 0
        total = t_n + v_n + e_n
        print(f"  PID {pid}: {t_n:>6} ({t_n/total*100:5.1f}%) + "
              f"{v_n:>6} ({v_n/total*100:5.1f}%) + "
              f"{e_n:>6} ({e_n/total*100:5.1f}%) = {total}")


def print_feature_stats(df):
    """打印特征统计"""
    key_feats = ["IOB", "COB", "exercise_intensity", "effective_basal_rate",
                 "recent_bolus_dose", "heart_rate", "ISF"]
    print("\n=== 关键特征统计 ===")
    stats = df[key_feats].describe().T[["mean", "std", "min", "max"]]
    print(stats.to_string(float_format="%.4f"))

    glu_stats = df["glucose_level"].describe()
    print(f"\n血糖 (mg/dL): mean={glu_stats['mean']:.1f}, std={glu_stats['std']:.1f}, "
          f"min={glu_stats['min']:.1f}, max={glu_stats['max']:.1f}")
    print(f"血糖 (mmol/L): mean={glu_stats['mean']/MMOL_TO_MGDL:.1f}, "
          f"min={glu_stats['min']/MMOL_TO_MGDL:.1f}, "
          f"max={glu_stats['max']/MMOL_TO_MGDL:.1f}")


def main():
    print("=" * 60)
    print("Manchester → Ohio 特征映射与数据划分")
    print("=" * 60)

    # 加载并转换数据
    df = load_all_data()

    # 划分
    train, val, test = split_by_participant(df)

    # 保存
    save_data(df, "glucose_features_manchester.csv")
    save_data(train, "manchester_train.csv")
    save_data(val, "manchester_val.csv")
    save_data(test, "manchester_test.csv")

    # 统计
    print_data_stats(train, val, test)
    print_feature_stats(df)

    return df, train, val, test


if __name__ == "__main__":
    df, train, val, test = main()
