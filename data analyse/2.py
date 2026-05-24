"""
T1D-UOM Dataset Feature Extraction
====================================
从 ManchesterCSCoordinatedDiabetesStudy 数据集中提取特征，涵盖：
- 人口统计学特征 (Demographics)
- 血糖特征 (Glucose)
- 活动特征 (Activity)
- 睡眠特征 (Sleep)
- 胰岛素特征 (Insulin)
- 营养特征 (Nutrition)
- 跨模态关联特征 (Cross-modal)
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = r"D:\ohio\ManchesterCSCoordinatedDiabetesStudy-main"
DEMOGRAPHICS_DIR = os.path.join(BASE_DIR, "Demographics")
ACTIVITY_DIR = os.path.join(BASE_DIR, "Activity Data")
GLUCOSE_DIR = os.path.join(BASE_DIR, "Glucose Data")
BASAL_DIR = os.path.join(BASE_DIR, "Insulin Data", "Basal Data")
BOLUS_DIR = os.path.join(BASE_DIR, "Insulin Data", "Bolus Data")
NUTRITION_DIR = os.path.join(BASE_DIR, "Nutrition Data")
SLEEP_DIR = os.path.join(BASE_DIR, "Sleep Data")

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# 工具函数
# ============================================================
def parse_timestamp(series, dayfirst=True):
    """安全解析时间戳列，兼容BOM和格式差异"""
    cleaned = series.str.strip().str.lstrip("﻿")
    return pd.to_datetime(cleaned, dayfirst=dayfirst, format="mixed")


def extract_participant_id(filename, prefix, suffix=".csv"):
    """从文件名提取参与者ID，如 UoMActivity2301.csv -> 2301"""
    name = os.path.basename(filename)
    return name.replace(prefix, "").replace(suffix, "")


def load_csv(path):
    """加载CSV，处理BOM"""
    return pd.read_csv(path, encoding="utf-8-sig")


def get_participant_files(directory, prefix):
    """获取目录中所有匹配前缀的CSV文件"""
    files = {}
    for f in sorted(os.listdir(directory)):
        if f.startswith(prefix) and f.endswith(".csv"):
            pid = extract_participant_id(f, prefix)
            files[pid] = os.path.join(directory, f)
    return files


# ============================================================
# 1. 人口统计学特征
# ============================================================
def extract_demographics():
    """提取人口统计学特征：体重、身高、BMI"""
    path = os.path.join(DEMOGRAPHICS_DIR, "UoMBMI.csv")
    df = load_csv(path)
    df["participant_id"] = df["participant_id"].str.strip().str.lstrip("﻿")
    df["pid"] = df["participant_id"].str.extract(r"(\d+)")
    print(f"[Demographics] 加载 {len(df)} 条记录")
    return df[["pid", "weight_kg", "height_m", "bmi"]]


# ============================================================
# 2. 血糖特征（核心）
# ============================================================
def extract_glucose_features():
    """
    提取血糖特征，包括：
    - 基础统计：均值、标准差、最小值、最大值、中位数
    - 变异指标：CV、MAGE、TIR（Time in Range）、TOR、TBR、TAR
    - 低血糖/高血糖事件计数
    - 日内/日间变异
    """
    files = get_participant_files(GLUCOSE_DIR, "UoMGlucose")
    all_features = []

    for pid, path in files.items():
        df = load_csv(path)
        df["bg_ts"] = parse_timestamp(df["bg_ts"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        df["date"] = df["bg_ts"].dt.date

        vals = df["value"]

        # --- 基础统计 ---
        mean_g = vals.mean()
        std_g = vals.std()
        median_g = vals.median()
        min_g = vals.min()
        max_g = vals.max()

        # --- 变异指标 ---
        cv = std_g / mean_g * 100 if mean_g != 0 else np.nan  # Coefficient of Variation (%)

        # MAGE (Mean Amplitude of Glycemic Excursions)
        sorted_vals = vals.sort_values().values
        diffs = np.abs(np.diff(sorted_vals))
        mage = np.mean(diffs[diffs > std_g]) if len(diffs[diffs > std_g]) > 0 else 0

        # --- TIR 指标 (mmol/L) ---
        # 目标范围: 3.9-10.0 mmol/L
        # 低血糖: <3.9, 严重低血糖: <3.0
        # 高血糖: >10.0, 严重高血糖: >13.9
        total = len(vals)
        tir = (vals.between(3.9, 10.0).sum() / total) * 100  # Time in Range
        tbr = (vals < 3.9).sum() / total * 100                  # Time Below Range
        tbr_severe = (vals < 3.0).sum() / total * 100           # Severe TBR
        tar = (vals > 10.0).sum() / total * 100                  # Time Above Range
        tar_severe = (vals > 13.9).sum() / total * 100          # Severe TAR

        # --- 低血糖/高血糖事件 ---
        # 连续>=15min（3个5min读数）低于3.9视为一次低血糖事件
        below = (vals < 3.9).astype(int).values
        hypo_events = 0
        in_event = False
        count = 0
        for b in below:
            if b:
                count += 1
                if count >= 3 and not in_event:
                    hypo_events += 1
                    in_event = True
            else:
                in_event = False
                count = 0

        # --- 日间/日内变异 ---
        daily_stats = df.groupby("date")["value"].agg(["mean", "std"])
        inter_day_cv = daily_stats["mean"].std() / daily_stats["mean"].mean() * 100 if daily_stats["mean"].mean() != 0 else np.nan
        intra_day_cv = daily_stats["std"].mean() / mean_g * 100 if mean_g != 0 else np.nan

        # --- 血糖模式 ---
        df["hour"] = df["bg_ts"].dt.hour
        hourly_mean = df.groupby("hour")["value"].mean()
        peak_hour = hourly_mean.idxmax()
        trough_hour = hourly_mean.idxmin()
        dawn_phenomenon = hourly_mean.get(5, np.nan) - hourly_mean.get(3, np.nan)  # 黎明现象

        # --- 读数数量与覆盖天数 ---
        n_readings = len(vals)
        n_days = df["date"].nunique()

        all_features.append({
            "pid": pid,
            "glucose_mean": mean_g,
            "glucose_std": std_g,
            "glucose_median": median_g,
            "glucose_min": min_g,
            "glucose_max": max_g,
            "glucose_cv_percent": cv,
            "glucose_mage": mage,
            "tir_percent": tir,
            "tbr_percent": tbr,
            "tbr_severe_percent": tbr_severe,
            "tar_percent": tar,
            "tar_severe_percent": tar_severe,
            "hypo_events": hypo_events,
            "inter_day_cv_percent": inter_day_cv,
            "intra_day_cv_percent": intra_day_cv,
            "peak_glucose_hour": peak_hour,
            "trough_glucose_hour": trough_hour,
            "dawn_phenomenon_rise": dawn_phenomenon,
            "n_glucose_readings": n_readings,
            "n_days_covered": n_days,
        })

    result = pd.DataFrame(all_features)
    print(f"[Glucose] 提取 {len(result)} 名参与者特征")
    return result


# ============================================================
# 3. 活动特征
# ============================================================
def extract_activity_features():
    """
    提取活动特征：
    - 步数统计
    - 卡路里消耗
    - 活动类型分布
    - 活动强度分布
    - MET统计
    - 久坐时间比例
    """
    files = get_participant_files(ACTIVITY_DIR, "UoMActivity")
    all_features = []

    for pid, path in files.items():
        df = load_csv(path)
        df["activity_ts"] = parse_timestamp(df["activity_ts"])
        df["date"] = df["activity_ts"].dt.date

        # --- 步数 ---
        total_steps = df["step_count"].sum()
        daily_steps = df.groupby("date")["step_count"].sum()
        mean_daily_steps = daily_steps.mean()
        std_daily_steps = daily_steps.std()

        # --- 卡路里 ---
        total_kcal = df["active_Kcal"].sum()
        mean_daily_kcal = df.groupby("date")["active_Kcal"].sum().mean()

        # --- 活动类型分布 ---
        type_counts = df["activity_type"].value_counts(normalize=True)
        sedentary_ratio = type_counts.get("SEDENTARY", 0)
        walking_ratio = type_counts.get("WALKING", 0)
        running_ratio = type_counts.get("RUNNING", 0)

        # --- 活动强度分布 ---
        intensity_counts = df["intensity"].value_counts(normalize=True)
        active_ratio = intensity_counts.get("ACTIVE", 0)
        highly_active_ratio = intensity_counts.get("HIGHLY_ACTIVE", 0)

        # --- MET ---
        mean_met = df["met"].mean()
        max_met = df["met"].max()

        # --- 运动强度 ---
        mean_motion = df["motion_intensity_mean"].mean()
        max_motion = df["motion_intensity_max"].max()

        # --- 距离 ---
        total_distance = df["distance_m"].sum()
        mean_daily_distance = df.groupby("date")["distance_m"].sum().mean()

        # --- 活动时间 ---
        total_active_time_s = df["active_time_s"].sum()

        n_days = df["date"].nunique()

        all_features.append({
            "pid": pid,
            "total_steps": total_steps,
            "mean_daily_steps": mean_daily_steps,
            "std_daily_steps": std_daily_steps,
            "total_active_kcal": total_kcal,
            "mean_daily_kcal": mean_daily_kcal,
            "sedentary_ratio": sedentary_ratio,
            "walking_ratio": walking_ratio,
            "running_ratio": running_ratio,
            "active_ratio": active_ratio,
            "highly_active_ratio": highly_active_ratio,
            "mean_met": mean_met,
            "max_met": max_met,
            "mean_motion_intensity": mean_motion,
            "max_motion_intensity": max_motion,
            "total_distance_m": total_distance,
            "mean_daily_distance_m": mean_daily_distance,
            "total_active_time_h": total_active_time_s / 3600,
            "n_activity_days": n_days,
        })

    result = pd.DataFrame(all_features)
    print(f"[Activity] 提取 {len(result)} 名参与者特征")
    return result


# ============================================================
# 4. 睡眠特征
# ============================================================
def extract_sleep_features():
    """
    提取睡眠特征：
    - 睡眠时长统计
    - 睡眠阶段分布（深睡、浅睡、REM、清醒）
    - 睡眠效率
    - 心率与压力
    """
    # Sleep-level files
    level_files = get_participant_files(SLEEP_DIR, "UoMsleep")
    # Sleep-summary files
    summary_files = {}
    for f in sorted(os.listdir(SLEEP_DIR)):
        if "sleeptime" in f and f.endswith(".csv"):
            pid = f.replace("UoM", "").replace("sleeptime.csv", "")
            summary_files[pid] = os.path.join(SLEEP_DIR, f)

    all_features = []

    # --- 从summary文件提取 ---
    for pid, path in summary_files.items():
        try:
            df = load_csv(path)
            # 跳过空文件
            if len(df) == 0:
                continue

            # 睡眠阶段时长（秒->小时）
            deep_h = df["deep_sleep_s"].mean() / 3600
            light_h = df["light_sleep_s"].mean() / 3600
            rem_h = df["rem_sleep_s"].mean() / 3600
            awake_h = df["awake_sleep_s"].mean() / 3600

            total_sleep_h = deep_h + light_h + rem_h
            total_bed_h = total_sleep_h + awake_h
            sleep_efficiency = (total_sleep_h / total_bed_h * 100) if total_bed_h > 0 else np.nan

            # 阶段比例
            deep_pct = (deep_h / total_sleep_h * 100) if total_sleep_h > 0 else np.nan
            light_pct = (light_h / total_sleep_h * 100) if total_sleep_h > 0 else np.nan
            rem_pct = (rem_h / total_sleep_h * 100) if total_sleep_h > 0 else np.nan

            # 睡眠时长变异
            df["total_sleep_h"] = (df["deep_sleep_s"] + df["light_sleep_s"] + df["rem_sleep_s"]) / 3600
            sleep_duration_std = df["total_sleep_h"].std()

            n_nights = len(df)

            all_features.append({
                "pid": pid,
                "mean_deep_sleep_h": deep_h,
                "mean_light_sleep_h": light_h,
                "mean_rem_sleep_h": rem_h,
                "mean_awake_h": awake_h,
                "mean_total_sleep_h": total_sleep_h,
                "sleep_efficiency_percent": sleep_efficiency,
                "deep_sleep_pct": deep_pct,
                "light_sleep_pct": light_pct,
                "rem_sleep_pct": rem_pct,
                "sleep_duration_std_h": sleep_duration_std,
                "n_nights": n_nights,
            })
        except Exception:
            continue

    # --- 从level文件提取心率/压力 ---
    level_features = []
    for pid, path in level_files.items():
        try:
            df = load_csv(path)
            if len(df) == 0:
                continue

            # 仅睡眠期间的数据
            sleep_df = df[df["sleep_level"] == 1]
            hr_sleep = sleep_df["heart_rate"]
            hr_sleep = hr_sleep[hr_sleep > 0]  # 排除无效值

            all_hr = df["heart_rate"]
            all_hr = all_hr[all_hr > 0]
            all_stress = df["stress_level_value"]

            level_features.append({
                "pid": pid,
                "mean_hr_sleep": hr_sleep.mean() if len(hr_sleep) > 0 else np.nan,
                "mean_hr_awake": all_hr[df["sleep_level"] == 0].mean() if (df["sleep_level"] == 0).any() and len(all_hr[df["sleep_level"] == 0]) > 0 else np.nan,
                "mean_stress_sleep": all_stress[df["sleep_level"] == 1].mean() if (df["sleep_level"] == 1).any() else np.nan,
                "mean_stress_awake": all_stress[df["sleep_level"] == 0].mean() if (df["sleep_level"] == 0).any() else np.nan,
            })
        except Exception:
            continue

    df_summary = pd.DataFrame(all_features)
    df_level = pd.DataFrame(level_features)

    if len(df_summary) > 0 and len(df_level) > 0:
        result = df_summary.merge(df_level, on="pid", how="outer")
    elif len(df_summary) > 0:
        result = df_summary
    else:
        result = df_level

    print(f"[Sleep] 提取 {len(result)} 名参与者特征")
    return result


# ============================================================
# 5. 胰岛素特征
# ============================================================
def extract_insulin_features():
    """
    提取胰岛素特征：
    - 基础胰岛素(Basal)：日均剂量、R/L比例
    - 餐时胰岛素(Bolus)：日均剂量、次数、每餐平均剂量
    - 总胰岛素量与基础/餐时比
    """
    basal_files = get_participant_files(BASAL_DIR, "UoMBasal")
    bolus_files = get_participant_files(BOLUS_DIR, "UoMBolus")
    all_features = []

    # 收集所有参与者ID
    all_pids = set(basal_files.keys()) | set(bolus_files.keys())

    for pid in all_pids:
        feat = {"pid": pid}

        # --- Basal ---
        if pid in basal_files:
            df = load_csv(basal_files[pid])
            if len(df) > 0:
                df["basal_ts"] = parse_timestamp(df["basal_ts"])
                df["date"] = df["basal_ts"].dt.date

                # 按类型统计
                r_df = df[df["insulin_kind"] == "R"]
                l_df = df[df["insulin_kind"] == "L"]

                feat["basal_mean_daily_dose"] = df.groupby("date")["basal_dose"].sum().mean()
                feat["basal_r_mean_daily"] = r_df.groupby("date")["basal_dose"].sum().mean() if len(r_df) > 0 else 0
                feat["basal_l_mean_daily"] = l_df.groupby("date")["basal_dose"].sum().mean() if len(l_df) > 0 else 0
                feat["basal_r_ratio"] = len(r_df) / len(df) if len(df) > 0 else 0
            else:
                feat["basal_mean_daily_dose"] = np.nan
                feat["basal_r_mean_daily"] = np.nan
                feat["basal_l_mean_daily"] = np.nan
                feat["basal_r_ratio"] = np.nan
        else:
            feat["basal_mean_daily_dose"] = np.nan
            feat["basal_r_mean_daily"] = np.nan
            feat["basal_l_mean_daily"] = np.nan
            feat["basal_r_ratio"] = np.nan

        # --- Bolus ---
        if pid in bolus_files:
            df = load_csv(bolus_files[pid])
            if len(df) > 0:
                df["bolus_ts"] = parse_timestamp(df["bolus_ts"])
                df["date"] = df["bolus_ts"].dt.date

                daily_bolus = df.groupby("date")["bolus_dose"].agg(["sum", "count"])
                feat["bolus_mean_daily_dose"] = daily_bolus["sum"].mean()
                feat["bolus_mean_daily_count"] = daily_bolus["count"].mean()
                feat["bolus_mean_per_event"] = df["bolus_dose"].mean()
                feat["bolus_total_events"] = len(df)
            else:
                feat["bolus_mean_daily_dose"] = np.nan
                feat["bolus_mean_daily_count"] = np.nan
                feat["bolus_mean_per_event"] = np.nan
                feat["bolus_total_events"] = 0
        else:
            feat["bolus_mean_daily_dose"] = np.nan
            feat["bolus_mean_daily_count"] = np.nan
            feat["bolus_mean_per_event"] = np.nan
            feat["bolus_total_events"] = 0

        # --- 基础/餐时比 ---
        basal_d = feat.get("basal_mean_daily_dose", np.nan)
        bolus_d = feat.get("bolus_mean_daily_dose", np.nan)
        if pd.notna(basal_d) and pd.notna(bolus_d) and bolus_d > 0:
            feat["basal_bolus_ratio"] = basal_d / bolus_d
        else:
            feat["basal_bolus_ratio"] = np.nan

        all_features.append(feat)

    result = pd.DataFrame(all_features)
    print(f"[Insulin] 提取 {len(result)} 名参与者特征")
    return result


# ============================================================
# 6. 营养特征
# ============================================================
def extract_nutrition_features():
    """
    提取营养特征：
    - 宏量营养素日均摄入
    - 各餐类型分布
    - 碳水化合物变异
    - 餐次数
    """
    files = get_participant_files(NUTRITION_DIR, "UoMNutrition")
    all_features = []

    for pid, path in files.items():
        df = load_csv(path)
        if len(df) == 0:
            continue

        df["meal_ts"] = parse_timestamp(df["meal_ts"])
        df["date"] = df["meal_ts"].dt.date

        # 宏量营养素
        for col in ["carbs_g", "prot_g", "fat_g", "fibre_g"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        daily = df.groupby("date").agg({
            "carbs_g": "sum",
            "prot_g": "sum",
            "fat_g": "sum",
            "fibre_g": "sum",
            "meal_ts": "count",
        }).rename(columns={"meal_ts": "meal_count"})

        # 餐类型分布
        meal_type_dist = df["meal_type"].value_counts(normalize=True)

        # 碳水变异
        carbs_cv = daily["carbs_g"].std() / daily["carbs_g"].mean() * 100 if daily["carbs_g"].mean() > 0 else np.nan

        n_days = len(daily)

        all_features.append({
            "pid": pid,
            "mean_daily_carbs_g": daily["carbs_g"].mean(),
            "mean_daily_prot_g": daily["prot_g"].mean(),
            "mean_daily_fat_g": daily["fat_g"].mean(),
            "mean_daily_fibre_g": daily["fibre_g"].mean(),
            "mean_daily_meal_count": daily["meal_count"].mean(),
            "std_daily_carbs_g": daily["carbs_g"].std(),
            "carbs_cv_percent": carbs_cv,
            "breakfast_ratio": meal_type_dist.get("Breakfast", 0),
            "lunch_ratio": meal_type_dist.get("Lunch", 0),
            "dinner_ratio": meal_type_dist.get("Dinner", 0),
            "snack_ratio": meal_type_dist.get("Snack", 0),
            "n_nutrition_days": n_days,
        })

    result = pd.DataFrame(all_features)
    print(f"[Nutrition] 提取 {len(result)} 名参与者特征")
    return result


# ============================================================
# 7. 跨模态关联特征（日级别对齐）
# ============================================================
def extract_daily_crossmodal_features():
    """
    在日级别对齐多模态数据，提取跨模态关联特征：
    - 运动当天的血糖变化
    - 餐后血糖响应（PPGR）
    - 睡眠质量与次日空腹血糖
    - 胰岛素剂量与血糖控制
    """
    glucose_files = get_participant_files(GLUCOSE_DIR, "UoMGlucose")
    activity_files = get_participant_files(ACTIVITY_DIR, "UoMActivity")
    nutrition_files = get_participant_files(NUTRITION_DIR, "UoMNutrition")
    bolus_files = get_participant_files(BOLUS_DIR, "UoMBolus")
    sleep_summary_files = {}
    for f in sorted(os.listdir(SLEEP_DIR)):
        if "sleeptime" in f and f.endswith(".csv"):
            pid = f.replace("UoM", "").replace("sleeptime.csv", "")
            sleep_summary_files[pid] = os.path.join(SLEEP_DIR, f)

    all_features = []

    common_pids = set(glucose_files.keys())

    for pid in common_pids:
        # --- 加载血糖数据 ---
        gdf = load_csv(glucose_files[pid])
        gdf["bg_ts"] = parse_timestamp(gdf["bg_ts"])
        gdf["value"] = pd.to_numeric(gdf["value"], errors="coerce")
        gdf["date"] = gdf["bg_ts"].dt.date

        # 日级血糖统计
        daily_glu = gdf.groupby("date")["value"].agg(["mean", "std", "min", "max"]).reset_index()
        daily_glu.columns = ["date", "glucose_mean", "glucose_std", "glucose_min", "glucose_max"]

        # 次日空腹血糖（取6:00-8:00均值）
        gdf["hour"] = gdf["bg_ts"].dt.hour
        fasting = gdf[gdf["hour"].between(6, 8)].groupby("date")["value"].mean().reset_index()
        fasting.columns = ["date", "fasting_glucose"]
        # 将空腹血糖映射到"前一天"（因为是起床时测量）
        fasting["date"] = fasting["date"] + pd.Timedelta(days=-1)
        daily_glu = daily_glu.merge(fasting, on="date", how="left")

        # --- 加载活动数据 ---
        if pid in activity_files:
            adf = load_csv(activity_files[pid])
            adf["activity_ts"] = parse_timestamp(adf["activity_ts"])
            adf["date"] = adf["activity_ts"].dt.date
            daily_act = adf.groupby("date").agg({
                "step_count": "sum",
                "active_Kcal": "sum",
                "active_time_s": "sum",
            }).reset_index()
            daily_act["active_time_h"] = daily_act["active_time_s"] / 3600
            daily_glu = daily_glu.merge(
                daily_act[["date", "step_count", "active_Kcal", "active_time_h"]],
                on="date", how="left"
            )
        else:
            daily_glu["step_count"] = np.nan
            daily_glu["active_Kcal"] = np.nan
            daily_glu["active_time_h"] = np.nan

        # --- 加载营养数据 ---
        if pid in nutrition_files:
            ndf = load_csv(nutrition_files[pid])
            ndf["meal_ts"] = parse_timestamp(ndf["meal_ts"])
            ndf["date"] = ndf["meal_ts"].dt.date
            for col in ["carbs_g", "prot_g", "fat_g"]:
                ndf[col] = pd.to_numeric(ndf[col], errors="coerce").fillna(0)
            daily_nutr = ndf.groupby("date").agg({
                "carbs_g": "sum",
                "prot_g": "sum",
                "fat_g": "sum",
            }).reset_index()
            daily_glu = daily_glu.merge(daily_nutr, on="date", how="left")
        else:
            daily_glu["carbs_g"] = np.nan
            daily_glu["prot_g"] = np.nan
            daily_glu["fat_g"] = np.nan

        # --- 加载Bolus数据 ---
        if pid in bolus_files:
            bdf = load_csv(bolus_files[pid])
            bdf["bolus_ts"] = parse_timestamp(bdf["bolus_ts"])
            bdf["date"] = bdf["bolus_ts"].dt.date
            daily_bol = bdf.groupby("date")["bolus_dose"].agg(["sum", "count"]).reset_index()
            daily_bol.columns = ["date", "bolus_total", "bolus_count"]
            daily_glu = daily_glu.merge(daily_bol, on="date", how="left")
        else:
            daily_glu["bolus_total"] = np.nan
            daily_glu["bolus_count"] = np.nan

        # --- 加载睡眠数据 ---
        if pid in sleep_summary_files:
            try:
                sdf = load_csv(sleep_summary_files[pid])
                if len(sdf) > 0:
                    sdf["calendar_date"] = parse_timestamp(sdf["calendar_date"])
                    sdf["date"] = sdf["calendar_date"].dt.date
                    sdf["total_sleep_h"] = (sdf["deep_sleep_s"] + sdf["light_sleep_s"] + sdf["rem_sleep_s"]) / 3600
                    daily_sleep = sdf.groupby("date").agg({
                        "total_sleep_h": "mean",
                        "deep_sleep_s": "mean",
                    }).reset_index()
                    daily_sleep["deep_sleep_h"] = daily_sleep["deep_sleep_s"] / 3600
                    daily_glu = daily_glu.merge(
                        daily_sleep[["date", "total_sleep_h", "deep_sleep_h"]],
                        on="date", how="left"
                    )
                else:
                    daily_glu["total_sleep_h"] = np.nan
                    daily_glu["deep_sleep_h"] = np.nan
            except Exception:
                daily_glu["total_sleep_h"] = np.nan
                daily_glu["deep_sleep_h"] = np.nan
        else:
            daily_glu["total_sleep_h"] = np.nan
            daily_glu["deep_sleep_h"] = np.nan

        daily_glu["pid"] = pid
        all_features.append(daily_glu)

    if len(all_features) == 0:
        return pd.DataFrame()

    result = pd.concat(all_features, ignore_index=True)
    print(f"[Cross-modal] 提取 {result['pid'].nunique()} 名参与者的日级对齐特征, 共 {len(result)} 天记录")

    # --- 计算跨模态相关性（参与者级别汇总）---
    corr_features = []
    for pid, grp in result.groupby("pid"):
        feat = {"pid": pid}

        # 运动-血糖相关性
        valid = grp.dropna(subset=["step_count", "glucose_mean"])
        if len(valid) > 5:
            feat["steps_glucose_corr"] = valid["step_count"].corr(valid["glucose_mean"])
            feat["active_time_glucose_corr"] = valid["active_time_h"].corr(valid["glucose_mean"]) if "active_time_h" in valid.columns else np.nan
        else:
            feat["steps_glucose_corr"] = np.nan
            feat["active_time_glucose_corr"] = np.nan

        # 碳水-血糖相关性
        valid = grp.dropna(subset=["carbs_g", "glucose_mean"])
        if len(valid) > 5:
            feat["carbs_glucose_corr"] = valid["carbs_g"].corr(valid["glucose_mean"])
        else:
            feat["carbs_glucose_corr"] = np.nan

        # 睡眠-次日空腹血糖相关性
        valid = grp.dropna(subset=["total_sleep_h", "fasting_glucose"])
        if len(valid) > 5:
            feat["sleep_fasting_glucose_corr"] = valid["total_sleep_h"].corr(valid["fasting_glucose"])
        else:
            feat["sleep_fasting_glucose_corr"] = np.nan

        # 高运动量日vs低运动量日的血糖差异
        valid = grp.dropna(subset=["step_count", "glucose_mean"])
        if len(valid) > 10:
            median_steps = valid["step_count"].median()
            high_act = valid[valid["step_count"] >= median_steps]["glucose_mean"].mean()
            low_act = valid[valid["step_count"] < median_steps]["glucose_mean"].mean()
            feat["high_vs_low_activity_glucose_diff"] = low_act - high_act
        else:
            feat["high_vs_low_activity_glucose_diff"] = np.nan

        corr_features.append(feat)

    corr_df = pd.DataFrame(corr_features)
    print(f"[Cross-modal] 提取 {len(corr_df)} 名参与者跨模态关联特征")

    return result, corr_df


# ============================================================
# 8. 主函数：整合所有特征
# ============================================================
def main():
    print("=" * 60)
    print("T1D-UOM 数据集特征提取")
    print("=" * 60)

    # 1. 人口统计学
    demo = extract_demographics()

    # 2. 血糖特征
    glucose = extract_glucose_features()

    # 3. 活动特征
    activity = extract_activity_features()

    # 4. 睡眠特征
    sleep = extract_sleep_features()

    # 5. 胰岛素特征
    insulin = extract_insulin_features()

    # 6. 营养特征
    nutrition = extract_nutrition_features()

    # 7. 跨模态关联特征
    daily_crossmodal, crossmodal_corr = extract_daily_crossmodal_features()

    # --- 合并参与者级特征 ---
    features = demo.copy()
    for df in [glucose, activity, sleep, insulin, nutrition, crossmodal_corr]:
        if len(df) > 0 and "pid" in df.columns:
            features = features.merge(df, on="pid", how="outer")

    # --- 输出 ---
    print("\n" + "=" * 60)
    print("特征提取完成")
    print("=" * 60)
    print(f"参与者级特征矩阵: {features.shape[0]} 名参与者 × {features.shape[1]} 个特征")
    print(f"日级跨模态数据: {daily_crossmodal.shape[0]} 天 × {daily_crossmodal.shape[1]} 列")

    # 保存
    features_path = os.path.join(OUTPUT_DIR, "participant_features.csv")
    features.to_csv(features_path, index=False, encoding="utf-8-sig")
    print(f"\n参与者级特征已保存: {features_path}")

    daily_path = os.path.join(OUTPUT_DIR, "daily_crossmodal_features.csv")
    daily_crossmodal.to_csv(daily_path, index=False, encoding="utf-8-sig")
    print(f"日级跨模态特征已保存: {daily_path}")

    # --- 特征摘要 ---
    print("\n--- 特征摘要 ---")
    print(f"人口统计学特征: 3 个")
    print(f"血糖特征: {len(glucose.columns) - 1} 个")
    print(f"活动特征: {len(activity.columns) - 1} 个")
    print(f"睡眠特征: {len(sleep.columns) - 1} 个")
    print(f"胰岛素特征: {len(insulin.columns) - 1} 个")
    print(f"营养特征: {len(nutrition.columns) - 1} 个")
    print(f"跨模态关联特征: {len(crossmodal_corr.columns) - 1} 个")
    print(f"总计: {features.shape[1] - 1} 个特征（不含pid列）")

    return features, daily_crossmodal


if __name__ == "__main__":
    features, daily_crossmodal = main()
