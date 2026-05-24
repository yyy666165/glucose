import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# 加载数据
glucose_df = pd.read_csv('glucose_data.csv')
meal_df = pd.read_csv('meal_data.csv')
exercise_df = pd.read_csv('exercise_data.csv')
heart_rate_df = pd.read_csv('heart_rate_data.csv')
skin_temp_df = pd.read_csv('skin_temperature_data.csv')
gsr_df = pd.read_csv('gsr_data.csv')

# 将timestamp转换为datetime
glucose_df['timestamp'] = pd.to_datetime(glucose_df['timestamp'])
meal_df['timestamp'] = pd.to_datetime(meal_df['timestamp'])
exercise_df['timestamp'] = pd.to_datetime(exercise_df['timestamp'])
heart_rate_df['timestamp'] = pd.to_datetime(heart_rate_df['timestamp'])
skin_temp_df['timestamp'] = pd.to_datetime(skin_temp_df['timestamp'])
gsr_df['timestamp'] = pd.to_datetime(gsr_df['timestamp'])

# 计算血糖变化（ΔG）
glucose_df = glucose_df.sort_values(['patient_id', 'timestamp'])
glucose_df['prev_glucose'] = glucose_df.groupby('patient_id')['glucose_level'].shift(1)
glucose_df['delta_glucose'] = glucose_df['glucose_level'] - glucose_df['prev_glucose']
glucose_df = glucose_df.dropna(subset=['delta_glucose'])

# 函数：计算特征与血糖变化的滞后相关性
def calculate_lag_correlation(feature_df, feature_name, glucose_df, max_lag_hours=6, lag_step=0.5):
    """计算特征与血糖变化的滞后相关性"""
    correlations = []

    for lag in np.arange(0, max_lag_hours + lag_step, lag_step):
        lag_seconds = lag * 3600

        # 合并特征和血糖数据
        merged = pd.merge_asof(
            glucose_df.sort_values('timestamp'),
            feature_df.sort_values('timestamp'),
            by='patient_id',
            on='timestamp',
            direction='backward',
            tolerance=pd.Timedelta(seconds=lag_seconds)
        )

        # 计算相关性
        if len(merged) > 0:
            corr = merged['delta_glucose'].corr(merged[feature_name])
            correlations.append({
                'lag_hours': lag,
                'correlation': corr,
                'sample_size': len(merged)
            })

    return pd.DataFrame(correlations)

# 计算各特征的滞后相关性
print("=== 特征与血糖变化的滞后相关性分析 ===")

# 进餐特征的滞后相关性（使用碳水化合物克数）
meal_correlations = calculate_lag_correlation(
    meal_df, 'carbs', glucose_df, max_lag_hours=6, lag_step=0.5
)
print("\n进餐（碳水化合物）与血糖变化的滞后相关性:")
print(meal_correlations.head())

# 运动特征的滞后相关性（使用强度和持续时间）
exercise_correlations_intensity = calculate_lag_correlation(
    exercise_df, 'intensity', glucose_df, max_lag_hours=6, lag_step=0.5
)
print("\n运动强度与血糖变化的滞后相关性:")
print(exercise_correlations_intensity.head())

exercise_correlations_duration = calculate_lag_correlation(
    exercise_df, 'duration', glucose_df, max_lag_hours=6, lag_step=0.5
)
print("\n运动持续时间与血糖变化的滞后相关性:")
print(exercise_correlations_duration.head())

# 心率特征的滞后相关性
heart_rate_correlations = calculate_lag_correlation(
    heart_rate_df, 'heart_rate', glucose_df, max_lag_hours=6, lag_step=0.5
)
print("\n心率与血糖变化的滞后相关性:")
print(heart_rate_correlations.head())

# 皮肤温度特征的滞后相关性
skin_temp_correlations = calculate_lag_correlation(
    skin_temp_df, 'skin_temperature', glucose_df, max_lag_hours=6, lag_step=0.5
)
print("\n皮肤温度与血糖变化的滞后相关性:")
print(skin_temp_correlations.head())

# 皮肤电反应特征的滞后相关性
gsr_correlations = calculate_lag_correlation(
    gsr_df, 'gsr', glucose_df, max_lag_hours=6, lag_step=0.5
)
print("\n皮肤电反应与血糖变化的滞后相关性:")
print(gsr_correlations.head())

# 可视化滞后相关性
plt.figure(figsize=(12, 8))
plt.plot(meal_correlations['lag_hours'], meal_correlations['correlation'], 'o-', label='碳水化合物')
plt.plot(exercise_correlations_intensity['lag_hours'], exercise_correlations_intensity['correlation'], 'o-', label='运动强度')
plt.plot(exercise_correlations_duration['lag_hours'], exercise_correlations_duration['correlation'], 'o-', label='运动持续时间')
plt.plot(heart_rate_correlations['lag_hours'], heart_rate_correlations['correlation'], 'o-', label='心率')
plt.plot(skin_temp_correlations['lag_hours'], skin_temp_correlations['correlation'], 'o-', label='皮肤温度')
plt.plot(gsr_correlations['lag_hours'], gsr_correlations['correlation'], 'o-', label='皮肤电反应')

plt.title('各特征与血糖变化的滞后相关性')
plt.xlabel('滞后时间 (小时)')
plt.ylabel('相关性系数')
plt.legend()
plt.grid(True)
plt.savefig('lag_correlation.png')
plt.close()

# 保存结果
meal_correlations.to_csv('meal_lag_correlation.csv', index=False)
exercise_correlations_intensity.to_csv('exercise_intensity_lag_correlation.csv', index=False)
exercise_correlations_duration.to_csv('exercise_duration_lag_correlation.csv', index=False)
heart_rate_correlations.to_csv('heart_rate_lag_correlation.csv', index=False)
skin_temp_correlations.to_csv('skin_temp_lag_correlation.csv', index=False)
gsr_correlations.to_csv('gsr_lag_correlation.csv', index=False)

print("\n分析完成，已生成滞后相关性图和统计结果")