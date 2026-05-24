import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# 加载数据
glucose_df = pd.read_csv('glucose_data.csv')
meal_df = pd.read_csv('meal_data.csv')
exercise_df = pd.read_csv('exercise_data.csv')
sleep_df = pd.read_csv('sleep_data.csv')

# 将timestamp转换为datetime
glucose_df['timestamp'] = pd.to_datetime(glucose_df['timestamp'])
meal_df['timestamp'] = pd.to_datetime(meal_df['timestamp'])
exercise_df['timestamp'] = pd.to_datetime(exercise_df['timestamp'])
sleep_df['sleep_start'] = pd.to_datetime(sleep_df['sleep_start'])
sleep_df['sleep_end'] = pd.to_datetime(sleep_df['sleep_end'])

# 计算血糖变化（ΔG）
glucose_df = glucose_df.sort_values(['patient_id', 'timestamp'])
glucose_df['prev_glucose'] = glucose_df.groupby('patient_id')['glucose_level'].shift(1)
glucose_df['delta_glucose'] = glucose_df['glucose_level'] - glucose_df['prev_glucose']
glucose_df = glucose_df.dropna(subset=['delta_glucose'])

# 1. 分析运动+高IOB时的低血糖概率
print("=== 交互关系探索 ===")

# 定义低血糖阈值
LOW_BLOOD_SUGAR_THRESHOLD = 70  # mg/dL

# 合并运动和血糖数据
exercise_glucose = pd.merge_asof(
    glucose_df.sort_values('timestamp'),
    exercise_df.sort_values('timestamp'),
    by='patient_id',
    on='timestamp',
    direction='backward',
    tolerance=pd.Timedelta(hours=1)
)

# 标记运动前后1小时内的血糖读数
exercise_glucose['is_exercise_period'] = ~exercise_glucose['intensity'].isna()

# 计算运动期间的低血糖概率
exercise_low_sugar = exercise_glucose[
    (exercise_glucose['is_exercise_period']) &
    (exercise_glucose['glucose_level'] < LOW_BLOOD_SUGAR_THRESHOLD)
]

# 计算非运动期间的低血糖概率
non_exercise_low_sugar = exercise_glucose[
    (~exercise_glucose['is_exercise_period']) &
    (exercise_glucose['glucose_level'] < LOW_BLOOD_SUGAR_THRESHOLD)
]

print(f"\n运动期间低血糖概率: {len(exercise_low_sugar) / len(exercise_glucose[exercise_glucose['is_exercise_period']]):.2%}")
print(f"非运动期间低血糖概率: {len(non_exercise_low_sugar) / len(exercise_glucose[~exercise_glucose['is_exercise_period']]):.2%}")

# 2. 分析睡眠质量对胰岛素敏感性的影响
# 计算睡眠债（基于睡眠质量）
sleep_df['sleep_debt'] = 5 - sleep_df['sleep_quality']  # 睡眠质量越低，睡眠债越高

# 简单的时间窗口分析，避免merge_asof的排序问题

# 分析运动期间的血糖数据
exercise_glucose = glucose_df.copy()
exercise_glucose['is_exercise_period'] = False

for _, exercise_row in exercise_df.iterrows():
    # 标记运动前后1小时内的血糖读数
    mask = (
        (exercise_glucose['patient_id'] == exercise_row['patient_id']) &
        (exercise_glucose['timestamp'] >= exercise_row['timestamp'] - pd.Timedelta(hours=1)) &
        (exercise_glucose['timestamp'] <= exercise_row['timestamp'] + pd.Timedelta(hours=1))
    )
    exercise_glucose.loc[mask, 'is_exercise_period'] = True

# 分析睡眠质量对血糖变化的影响
sleep_glucose = glucose_df.copy()
sleep_glucose['sleep_debt'] = 0  # 默认睡眠债为0

for _, sleep_row in sleep_df.iterrows():
    # 为睡眠期间和睡眠后12小时内的血糖读数分配睡眠债
    mask = (
        (sleep_glucose['patient_id'] == sleep_row['patient_id']) &
        (sleep_glucose['timestamp'] >= sleep_row['sleep_start']) &
        (sleep_glucose['timestamp'] <= sleep_row['sleep_end'] + pd.Timedelta(hours=12))
    )
    sleep_glucose.loc[mask, 'sleep_debt'] = 5 - sleep_row['sleep_quality']

# 添加睡眠质量信息到睡眠分析中
sleep_glucose['sleep_quality'] = 0  # 默认睡眠质量为0

for _, sleep_row in sleep_df.iterrows():
    # 为睡眠期间和睡眠后12小时内的血糖读数分配睡眠质量
    mask = (
        (sleep_glucose['patient_id'] == sleep_row['patient_id']) &
        (sleep_glucose['timestamp'] >= sleep_row['sleep_start']) &
        (sleep_glucose['timestamp'] <= sleep_row['sleep_end'] + pd.Timedelta(hours=12))
    )
    sleep_glucose.loc[mask, 'sleep_quality'] = sleep_row['sleep_quality']

# 添加运动强度信息到睡眠分析中
sleep_glucose['intensity'] = 0  # 默认运动强度为0

for _, exercise_row in exercise_df.iterrows():
    # 为运动前后1小时内的血糖读数分配运动强度
    mask = (
        (sleep_glucose['patient_id'] == exercise_row['patient_id']) &
        (sleep_glucose['timestamp'] >= exercise_row['timestamp'] - pd.Timedelta(hours=1)) &
        (sleep_glucose['timestamp'] <= exercise_row['timestamp'] + pd.Timedelta(hours=1))
    )
    sleep_glucose.loc[mask, 'intensity'] = exercise_row['intensity']

# 计算运动强度与睡眠债的交互效应
sleep_glucose['exercise_intensity_sleep_debt'] = sleep_glucose['intensity'] * sleep_glucose['sleep_debt']

# 分析睡眠债对血糖变化的影响
print("\n睡眠债对血糖变化的影响:")
sleep_debt_effect = sleep_glucose.groupby('sleep_debt')['delta_glucose'].mean()
print(sleep_debt_effect)

# 3. 分析进餐时间与运动时间的交互效应
# 简单的时间窗口分析，避免merge_asof的排序问题

# 分析餐后运动的血糖变化
meal_exercise = []
for _, meal_row in meal_df.iterrows():
    # 查找餐后2小时内的运动
    exercise_mask = (
        (exercise_df['patient_id'] == meal_row['patient_id']) &
        (exercise_df['timestamp'] >= meal_row['timestamp']) &
        (exercise_df['timestamp'] <= meal_row['timestamp'] + pd.Timedelta(hours=2))
    )
    exercise_rows = exercise_df[exercise_mask]

    if not exercise_rows.empty:
        # 如果有餐后运动，记录相关信息
        for _, exercise_row in exercise_rows.iterrows():
            meal_exercise.append({
                'patient_id': meal_row['patient_id'],
                'timestamp': meal_row['timestamp'],
                'carbs': meal_row['carbs'],
                'meal_type': meal_row['meal_type'],
                'intensity': exercise_row['intensity'],
                'duration': exercise_row['duration'],
                'is_post_meal_exercise': True
            })
    else:
        # 如果没有餐后运动，记录餐后2小时内的血糖变化
        glucose_mask = (
            (glucose_df['patient_id'] == meal_row['patient_id']) &
            (glucose_df['timestamp'] >= meal_row['timestamp']) &
            (glucose_df['timestamp'] <= meal_row['timestamp'] + pd.Timedelta(hours=2))
        )
        glucose_rows = glucose_df[glucose_mask]

        if not glucose_rows.empty:
            # 计算餐后2小时内的血糖变化
            delta_glucose = glucose_rows['glucose_level'].iloc[-1] - glucose_rows['glucose_level'].iloc[0]
            meal_exercise.append({
                'patient_id': meal_row['patient_id'],
                'timestamp': meal_row['timestamp'],
                'carbs': meal_row['carbs'],
                'meal_type': meal_row['meal_type'],
                'delta_glucose': delta_glucose,
                'is_post_meal_exercise': False
            })

meal_exercise = pd.DataFrame(meal_exercise)

# 分析餐后运动的血糖变化
post_meal_exercise = meal_exercise[meal_exercise['is_post_meal_exercise']]
non_post_meal_exercise = meal_exercise[~meal_exercise['is_post_meal_exercise']]

print(f"\n餐后运动时的平均血糖变化: {post_meal_exercise['carbs'].mean():.1f}g 碳水化合物, ΔG: {post_meal_exercise['delta_glucose'].mean():.1f} mg/dL")
print(f"非餐后运动时的平均血糖变化: {non_post_meal_exercise['carbs'].mean():.1f}g 碳水化合物, ΔG: {non_post_meal_exercise['delta_glucose'].mean():.1f} mg/dL")

# 可视化交互关系
plt.figure(figsize=(12, 8))

# 运动强度与睡眠债的交互效应
plt.subplot(2, 2, 1)
sns.scatterplot(data=sleep_glucose, x='intensity', y='delta_glucose', hue='sleep_debt', size='sleep_debt', palette='viridis')
plt.title('运动强度与睡眠债的交互效应')
plt.xlabel('运动强度')
plt.ylabel('血糖变化 (ΔG)')

# 餐后运动 vs 非餐后运动的血糖变化
plt.subplot(2, 2, 2)
sns.boxplot(data=meal_exercise, x='is_post_meal_exercise', y='delta_glucose')
plt.title('餐后运动 vs 非餐后运动的血糖变化')
plt.xlabel('是否为餐后运动')
plt.ylabel('血糖变化 (ΔG)')

# 睡眠质量对血糖变化的影响
plt.subplot(2, 2, 3)
sns.boxplot(data=sleep_glucose, x='sleep_quality', y='delta_glucose')
plt.title('睡眠质量对血糖变化的影响')
plt.xlabel('睡眠质量 (1-5)')
plt.ylabel('血糖变化 (ΔG)')

# 运动期间 vs 非运动期间的血糖水平
plt.subplot(2, 2, 4)
sns.boxplot(data=exercise_glucose, x='is_exercise_period', y='glucose_level')
plt.title('运动期间 vs 非运动期间的血糖水平')
plt.xlabel('是否为运动期间')
plt.ylabel('血糖水平 (mg/dL)')

plt.tight_layout()
plt.savefig('interaction_effects.png')
plt.close()

# 保存结果
exercise_low_sugar.to_csv('exercise_low_sugar_events.csv', index=False)
sleep_debt_effect.to_csv('sleep_debt_effect.csv', index=False)
meal_exercise.to_csv('meal_exercise_interaction.csv', index=False)

print("\n分析完成，已生成交互关系图和统计结果")