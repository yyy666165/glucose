import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 加载数据
glucose_df = pd.read_csv('glucose_data.csv')
meal_df = pd.read_csv('meal_data.csv')
exercise_df = pd.read_csv('exercise_data.csv')
heart_rate_df = pd.read_csv('heart_rate_data.csv')
skin_temp_df = pd.read_csv('skin_temperature_data.csv')
gsr_df = pd.read_csv('gsr_data.csv')
sleep_df = pd.read_csv('sleep_data.csv')

# 将timestamp转换为datetime
glucose_df['timestamp'] = pd.to_datetime(glucose_df['timestamp'])
meal_df['timestamp'] = pd.to_datetime(meal_df['timestamp'])
exercise_df['timestamp'] = pd.to_datetime(exercise_df['timestamp'])
heart_rate_df['timestamp'] = pd.to_datetime(heart_rate_df['timestamp'])
skin_temp_df['timestamp'] = pd.to_datetime(skin_temp_df['timestamp'])
gsr_df['timestamp'] = pd.to_datetime(gsr_df['timestamp'])
sleep_df['sleep_start'] = pd.to_datetime(sleep_df['sleep_start'])
sleep_df['sleep_end'] = pd.to_datetime(sleep_df['sleep_end'])

# 1. 血糖分布与变异系数分析
print("=== 血糖分布与变异系数分析 ===")
print(f"总数据点数: {len(glucose_df)}")
print(f"患者数量: {glucose_df['patient_id'].nunique()}")

# 基本统计量
print("\n基本统计量:")
print(glucose_df['glucose_level'].describe())

# 变异系数 (CV = 标准差/均值)
mean_glucose = glucose_df['glucose_level'].mean()
std_glucose = glucose_df['glucose_level'].std()
cv_glucose = std_glucose / mean_glucose
print(f"\n血糖变异系数 (CV): {cv_glucose:.2f}")

# 血糖分布直方图
plt.figure(figsize=(12, 6))
sns.histplot(glucose_df['glucose_level'], bins=50, kde=True)
plt.title('血糖水平分布')
plt.xlabel('血糖水平 (mg/dL)')
plt.ylabel('频次')
plt.axvline(mean_glucose, color='r', linestyle='--', label=f'均值: {mean_glucose:.1f}')
plt.axvline(mean_glucose + std_glucose, color='g', linestyle='--', label=f'+1σ: {mean_glucose + std_glucose:.1f}')
plt.axvline(mean_glucose - std_glucose, color='g', linestyle='--', label=f'-1σ: {mean_glucose - std_glucose:.1f}')
plt.legend()
plt.savefig('glucose_distribution.png')
plt.close()

# 按患者分组的血糖统计
patient_stats = glucose_df.groupby('patient_id')['glucose_level'].agg(['mean', 'std', 'count'])
patient_stats['cv'] = patient_stats['std'] / patient_stats['mean']
print("\n按患者分组的血糖统计:")
print(patient_stats)

# 保存患者统计结果
patient_stats.to_csv('patient_glucose_stats.csv')

print("\n分析完成，已生成血糖分布图和统计结果")