"""
Manchester 非血糖数据详情
"""
import pandas as pd, numpy as np, os

BASE = r'D:\ohio\ManchesterCSCoordinatedDiabetesStudy-main'
GI = os.path.join(BASE, 'Glucose Data')
ACT = os.path.join(BASE, 'Activity Data')
BASAL = os.path.join(BASE, 'Insulin Data', 'Basal Data')
BOLUS = os.path.join(BASE, 'Insulin Data', 'Bolus Data')
NUTR = os.path.join(BASE, 'Nutrition Data')
SLEEP = os.path.join(BASE, 'Sleep Data')
DEMO = os.path.join(BASE, 'Demographics')

def load_csv(path):
    df = pd.read_csv(path, encoding='utf-8-sig')
    # strip BOM from string columns
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].str.replace('﻿', '', regex=False)
    return df

def collect_files(directory, prefix, pid_extract=None):
    """Collect files matching prefix, return dict of pid->path"""
    files = {}
    for f in sorted(os.listdir(directory)):
        if f.startswith(prefix) and f.endswith('.csv'):
            if pid_extract:
                pid = pid_extract(f)
            else:
                pid = f.replace(prefix, '').replace('.csv', '')
            if pid.isdigit():
                files[pid] = os.path.join(directory, f)
    return files

def gather_data(file_dict):
    """Load and concatenate all files in dict, adding pid column"""
    parts = []
    for pid, path in file_dict.items():
        df = load_csv(path)
        df['pid'] = pid
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)

def num(s):
    try:
        return pd.to_numeric(s, errors='coerce')
    except:
        return s

print('=' * 70)
print('Manchester 非血糖数据详情')
print('=' * 70)

# ===== 1. 人口统计学 =====
print('\n一、人口统计学 (Demographics)')
demo_path = os.path.join(DEMO, 'UoMBMI.csv')
if os.path.exists(demo_path):
    demo = load_csv(demo_path)
    print(f'  记录数: {len(demo)}')
    print(f'  列: {list(demo.columns)}')
    for _, row in demo.iterrows():
        print(f'    参与者 {row.get("participant_id", "?"):>10}: '
              f'体重={row.get("weight_kg", "?"):>5} kg, '
              f'身高={row.get("height_m", "?"):>4} m, '
              f'BMI={row.get("bmi", "?"):>5}')

# ===== 2. 活动数据 =====
print('\n二、活动数据 (Activity Data)')
act_files = collect_files(ACT, 'UoMActivity')
print(f'  文件数: {len(act_files)}, 参与者: {", ".join(sorted(act_files.keys(), key=int))}')
act = gather_data(act_files)
if len(act) > 0:
    print(f'  总行数: {len(act):,}')
    print(f'  列: {list(act.columns)}')
    ts_col = act.columns[0]
    act['ts'] = pd.to_datetime(act[ts_col], dayfirst=True, format='mixed')
    print(f'  时间范围: {act["ts"].min()} ~ {act["ts"].max()}')

    for col in ['activity_type', 'intensity']:
        if col in act.columns:
            print(f'  {col} 分布:')
            vc = act[col].value_counts()
            for k, v in vc.items():
                print(f'    {k}: {v} ({v/len(act)*100:.1f}%)')

    for col in ['step_count', 'active_Kcal', 'active_time_s', 'met', 'distance_m', 'motion_intensity_mean']:
        if col in act.columns:
            v = num(act[col])
            print(f'  {col}: mean={v.mean():.1f}, median={v.median():.1f}, '
                  f'min={v.min():.1f}, max={v.max():.1f}, zero%={(v==0).mean()*100:.1f}%')

    # 每参与者日均步数
    if 'step_count' in act.columns:
        act['date'] = act['ts'].dt.date
        daily = act.groupby(['pid', 'date'])['step_count'].sum().reset_index()
        print(f'\n  每参与者日均步数:')
        for pid in sorted(act['pid'].unique(), key=int):
            pdf = daily[daily['pid'] == pid]
            print(f'    参与者 {pid}: {len(pdf)} 天, 均值={pdf["step_count"].mean():.0f} 步/天, '
                  f'最大={pdf["step_count"].max():,.0f} 步')

# ===== 3. 基础胰岛素 =====
print('\n三、基础胰岛素 (Basal Data)')
basal_files = collect_files(BASAL, 'UoMBasal')
print(f'  文件数: {len(basal_files)}, 参与者: {", ".join(sorted(basal_files.keys(), key=int))}')
basal = gather_data(basal_files)
if len(basal) > 0:
    print(f'  总行数: {len(basal):,}')
    print(f'  列: {list(basal.columns)}')
    if 'insulin_kind' in basal.columns:
        print(f'  胰岛素类型分布:')
        for k, v in basal['insulin_kind'].value_counts().items():
            print(f'    {k}: {v} ({v/len(basal)*100:.1f}%)')
    if 'basal_dose' in basal.columns:
        v = num(basal['basal_dose'])
        print(f'  basal_dose (U/h): mean={v.mean():.4f}, median={v.median():.4f}, max={v.max():.4f}')

    print(f'  每参与者统计:')
    for pid in sorted(basal['pid'].unique(), key=int):
        pdf = basal[basal['pid'] == pid]
        doses = num(pdf['basal_dose'])
        print(f'    参与者 {pid}: {len(pdf)} 条, 均剂量={doses.mean():.3f} U/h')

# ===== 4. 餐时胰岛素 =====
print('\n四、餐时胰岛素 (Bolus Data)')
bolus_files = collect_files(BOLUS, 'UoMBolus')
print(f'  文件数: {len(bolus_files)}, 参与者: {", ".join(sorted(bolus_files.keys(), key=int))}')
bolus = gather_data(bolus_files)
if len(bolus) > 0:
    print(f'  总行数: {len(bolus):,}')
    print(f'  列: {list(bolus.columns)}')
    if 'bolus_dose' in bolus.columns:
        v = num(bolus['bolus_dose'])
        print(f'  bolus_dose (U): mean={v.mean():.2f}, median={v.median():.2f}, max={v.max():.2f}')

    print(f'  每参与者:')
    for pid in sorted(bolus['pid'].unique(), key=int):
        pdf = bolus[bolus['pid'] == pid]
        doses = num(pdf['bolus_dose'])
        print(f'    参与者 {pid}: {len(pdf)} 次注射, 均剂量={doses.mean():.2f}U, '
              f'总剂量={doses.sum():.0f}U')

# ===== 5. 营养数据 =====
print('\n五、营养数据 (Nutrition Data)')
nutr_files = collect_files(NUTR, 'UoMNutrition')
print(f'  文件数: {len(nutr_files)}, 参与者: {", ".join(sorted(nutr_files.keys(), key=int))}')
nutr = gather_data(nutr_files)
if len(nutr) > 0:
    print(f'  总行数: {len(nutr):,}')
    print(f'  列: {list(nutr.columns)}')
    ts_col = nutr.columns[0]
    nutr['ts'] = pd.to_datetime(nutr[ts_col], dayfirst=True, format='mixed')

    for col in ['meal_type', 'meal']:
        if col in nutr.columns and col != 'meal_ts':
            print(f'  {col} 分布:')
            vc = nutr[col].value_counts()
            for k, v in vc.items():
                print(f'    {k}: {v} ({v/len(nutr)*100:.1f}%)')

    for col in ['carbs_g', 'prot_g', 'fat_g', 'fibre_g', 'energy_kcal']:
        if col in nutr.columns:
            v = num(nutr[col])
            print(f'  {col}: mean={v.mean():.1f}, median={v.median():.1f}, max={v.max():.1f}')

    nutr['date'] = nutr['ts'].dt.date
    print(f'  每参与者日均营养摄入:')
    for pid in sorted(nutr['pid'].unique(), key=int):
        pdf = nutr[nutr['pid'] == pid]
        daily = pdf.groupby('date').agg({
            'carbs_g': lambda x: num(x).sum(),
            'prot_g': lambda x: num(x).sum(),
            'fat_g': lambda x: num(x).sum(),
        })
        print(f'    参与者 {pid}: {len(pdf)} 餐, {len(daily)} 天, '
              f'日均碳水={daily["carbs_g"].mean():.0f}g, '
              f'蛋白={daily["prot_g"].mean():.0f}g, '
              f'脂肪={daily["fat_g"].mean():.0f}g')

# ===== 6. 睡眠数据 =====
print('\n六、睡眠数据 (Sleep Data)')

# 6a. 睡眠详情（含心率、压力）
sl_files = collect_files(SLEEP, 'UoMsleep')
print(f'  睡眠详情文件: {len(sl_files)}, 参与者: {", ".join(sorted(sl_files.keys(), key=int))}')
sleepl = gather_data(sl_files)
if len(sleepl) > 0:
    print(f'  总行数: {len(sleepl):,}')
    print(f'  列: {list(sleepl.columns)}')
    ts_col = sleepl.columns[0]
    sleepl['ts'] = pd.to_datetime(sleepl[ts_col], dayfirst=True, format='mixed')
    for col in ['heart_rate', 'stress_level_value', 'sleep_level']:
        if col in sleepl.columns:
            v = num(sleepl[col])
            print(f'  {col}: mean={v.mean():.1f}, min={v.min():.1f}, max={v.max():.1f}')

    # 睡眠 vs 清醒时段的心率
    if 'sleep_level' in sleepl.columns:
        asleep = sleepl[num(sleepl['sleep_level']) == 1]
        awake = sleepl[num(sleepl['sleep_level']) == 0]
        if len(asleep) > 0 and 'heart_rate' in asleep.columns:
            hr_asleep = num(asleep['heart_rate'])
            hr_awake = num(awake['heart_rate']) if len(awake) > 0 else pd.Series()
            print(f'  睡眠时心率: mean={hr_asleep.mean():.1f}, 清醒时心率: mean={hr_awake.mean() if len(hr_awake)>0 else 0:.1f}')

# 6b. 睡眠汇总（阶段、时长）
ss_files = {}
for f in os.listdir(SLEEP):
    if 'sleeptime' in f.lower() and f.endswith('.csv'):
        pid = f.replace('UoM', '').replace('sleeptime.csv', '')
        if pid.isdigit():
            ss_files[pid] = os.path.join(SLEEP, f)
print(f'  睡眠汇总文件(sleeptime): {len(ss_files)}, 参与者: {", ".join(sorted(ss_files.keys(), key=int))}')
sleeps = gather_data(ss_files)
if len(sleeps) > 0:
    print(f'  总行数: {len(sleeps):,}')
    print(f'  列: {list(sleeps.columns)}')

    for col in ['deep_sleep_s', 'light_sleep_s', 'rem_sleep_s', 'awake_sleep_s']:
        if col in sleeps.columns:
            v = num(sleeps[col])
            print(f'  {col}: mean={v.mean()/3600:.2f}h, min={v.min()/3600:.2f}h, max={v.max()/3600:.2f}h')

    # 睡眠效率
    if all(c in sleeps.columns for c in ['deep_sleep_s', 'light_sleep_s', 'rem_sleep_s', 'awake_sleep_s']):
        deep = num(sleeps['deep_sleep_s'])
        light = num(sleeps['light_sleep_s'])
        rem = num(sleeps['rem_sleep_s'])
        awake = num(sleeps['awake_sleep_s'])
        total_sleep = deep + light + rem
        total_bed = total_sleep + awake
        eff = (total_sleep / total_bed * 100)
        print(f'  总睡眠时长: mean={total_sleep.mean()/3600:.2f}h')
        print(f'  睡眠效率: mean={eff.mean():.1f}%, min={eff.min():.1f}%, max={eff.max():.1f}%')

        # 睡眠阶段比例
        print(f'  睡眠阶段平均比例:')
        print(f'    深睡: {(deep/total_sleep*100).mean():.1f}%')
        print(f'    浅睡: {(light/total_sleep*100).mean():.1f}%')
        print(f'    REM: {(rem/total_sleep*100).mean():.1f}%')
        print(f'    清醒: {(awake/total_bed*100).mean():.1f}%')

# ===== 汇总 =====
print('\n' + '=' * 70)
print('汇总')
print('=' * 70)
rows = [
    ('血糖 (Glucose)', len(act_files), '?', '?'),
    ('活动 (Activity)', len(act_files), f'{len(act):,}', f'{act["pid"].nunique()}' if len(act) > 0 else '0'),
]
print(f'{"数据类型":<22} {"文件数":<6} {"总记录数":<12} {"覆盖参与者":<10}')
print('-' * 52)
if len(act) > 0:
    print(f'{"活动 (Activity)":<22} {len(act_files):<6} {len(act):<12,} {act["pid"].nunique():<10}')
if len(basal) > 0:
    print(f'{"基础胰岛素 (Basal)":<22} {len(basal_files):<6} {len(basal):<12,} {basal["pid"].nunique():<10}')
if len(bolus) > 0:
    print(f'{"餐时胰岛素 (Bolus)":<22} {len(bolus_files):<6} {len(bolus):<12,} {bolus["pid"].nunique():<10}')
if len(nutr) > 0:
    print(f'{"营养 (Nutrition)":<22} {len(nutr_files):<6} {len(nutr):<12,} {nutr["pid"].nunique():<10}')
if len(sleepl) > 0:
    print(f'{"睡眠详情 (Sleep)":<22} {len(sl_files):<6} {len(sleepl):<12,} {sleepl["pid"].nunique():<10}')
if len(sleeps) > 0:
    print(f'{"睡眠汇总 (Sleeptime)":<22} {len(ss_files):<6} {len(sleeps):<12,} {sleeps["pid"].nunique():<10}')
