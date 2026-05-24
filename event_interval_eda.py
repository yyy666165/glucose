import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# 加载数据
meal_df = pd.read_csv('meal_data.csv')
exercise_df = pd.read_csv('exercise_data.csv')

# 将timestamp转换为datetime
meal_df['timestamp'] = pd.to_datetime(meal_df['timestamp'])
exercise_df['timestamp'] = pd.to_datetime(exercise_df['timestamp'])

# 按患者分组并按时间排序
meal_df = meal_df.sort_values(['patient_id', 'timestamp'])
exercise_df = exercise_df.sort_values(['patient_id', 'timestamp'])

# 计算进餐时间间隔
meal_df['prev_timestamp'] = meal_df.groupby('patient_id')['timestamp'].shift(1)
meal_df['meal_interval'] = (meal_df['timestamp'] - meal_df['prev_timestamp']).dt.total_seconds() / 3600  # 转换为小时

# 计算运动时间间隔
exercise_df['prev_timestamp'] = exercise_df.groupby('patient_id')['timestamp'].shift(1)
exercise_df['exercise_interval'] = (exercise_df['timestamp'] - exercise_df['prev_timestamp']).dt.total_seconds() / 3600  # 转换为小时

# 移除第一个事件（没有前一个事件）
meal_intervals = meal_df.dropna(subset=['meal_interval'])
exercise_intervals = exercise_df.dropna(subset=['exercise_interval'])

print("=== 事件间隔统计 ===")

# 进餐时间间隔统计
print("\n进餐时间间隔统计:")
print(meal_intervals['meal_interval'].describe())
print(f"平均进餐间隔: {meal_intervals['meal_interval'].mean():.2f} 小时")
print(f"中位进餐间隔: {meal_intervals['meal_interval'].median():.2f} 小时")

# 运动时间间隔统计
print("\n运动时间间隔统计:")
print(exercise_intervals['exercise_interval'].describe())
print(f"平均运动间隔: {exercise_intervals['exercise_interval'].mean():.2f} 小时")
print(f"中位运动间隔: {exercise_intervals['exercise_interval'].median():.2f} 小时")

# 可视化进餐间隔分布
plt.figure(figsize=(12, 6))
sns.histplot(meal_intervals['meal_interval'], bins=50, kde=True)
plt.title('进餐时间间隔分布')
plt.xlabel('时间间隔 (小时)')
plt.ylabel('频次')
plt.axvline(meal_intervals['meal_interval'].mean(), color='r', linestyle='--', label=f'均值: {meal_intervals["meal_interval"].mean():.1f}小时')
plt.axvline(meal_intervals['meal_interval'].median(), color='g', linestyle='--', label=f'中位数: {meal_intervals["meal_interval"].median():.1f}小时')
plt.legend()
plt.savefig('meal_interval_distribution.png')
plt.close()

# 可视化运动间隔分布
plt.figure(figsize=(12, 6))
sns.histplot(exercise_intervals['exercise_interval'], bins=50, kde=True)
plt.title('运动时间间隔分布')
plt.xlabel('时间间隔 (小时)')
plt.ylabel('频次')
plt.axvline(exercise_intervals['exercise_interval'].mean(), color='r', linestyle='--', label=f'均值: {exercise_intervals["exercise_interval"].mean():.1f}小时')
plt.axvline(exercise_intervals['exercise_interval'].median(), color='g', linestyle='--', label=f'中位数: {exercise_intervals["exercise_interval"].median():.1f}小时')
plt.legend()
plt.savefig('exercise_interval_distribution.png')
plt.close()

# 按患者分组的进餐模式
meal_patterns = meal_df.groupby(['patient_id', 'meal_type']).size().unstack(fill_value=0)
print("\n按患者分组的进餐类型分布:")
print(meal_patterns)

# 按患者分组的运动模式
exercise_patterns = exercise_df.groupby('patient_id').agg({
    'intensity': ['mean', 'std', 'min', 'max'],
    'duration': ['mean', 'std', 'min', 'max']
})
print("\n按患者分组的运动模式:")
print(exercise_patterns)

# 保存结果
meal_intervals.to_csv('meal_intervals.csv', index=False)
exercise_intervals.to_csv('exercise_intervals.csv', index=False)

print("\n分析完成，已生成时间间隔分布图和统计结果")