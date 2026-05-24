import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

# 加载特征数据
features_df = pd.read_csv('glucose_features_simplified.csv')

# 准备数据
X = features_df.drop(['timestamp', 'patient_id', 'glucose_level', 'prev_glucose', 'delta_glucose'], axis=1)
y = features_df['delta_glucose']

# 训练随机森林模型
print("训练随机森林模型...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X, y)

# 计算特征重要性
print("计算特征重要性...")
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# 置换重要性
print("计算置换重要性...")
result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
perm_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': result.importances_mean
}).sort_values('importance', ascending=False)

# 保存结果
importance.to_csv('feature_importance.csv', index=False)
perm_importance.to_csv('permutation_importance.csv', index=False)

# 可视化特征重要性
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
sns.barplot(x='importance', y='feature', data=importance.head(15))
plt.title('特征重要性（随机森林）')
plt.xlabel('重要性分数')
plt.ylabel('特征')

plt.subplot(2, 1, 2)
sns.barplot(x='importance', y='feature', data=perm_importance.head(15))
plt.title('特征重要性（置换重要性）')
plt.xlabel('重要性分数')
plt.ylabel('特征')

plt.tight_layout()
plt.savefig('feature_importance_visualization.png')
plt.close()

print("特征重要性分析完成，已生成可视化结果")

# 可视化典型日曲线
def visualize_typical_day(features_df):
    """可视化典型日曲线"""
    # 选择一个典型患者
    patient_id = features_df['patient_id'].unique()[0]
    patient_data = features_df[features_df['patient_id'] == patient_id]

    # 按小时聚合
    hourly_data = patient_data.groupby('hour').agg({
        'IOB': 'mean',
        'COB': 'mean',
        'ISF': 'mean',
        'glucose_level': 'mean'
    }).reset_index()

    # 可视化
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    plt.plot(hourly_data['hour'], hourly_data['IOB'], 'b-', linewidth=2, label='IOB')
    plt.title('典型日IOB变化')
    plt.xlabel('小时')
    plt.ylabel('IOB')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(hourly_data['hour'], hourly_data['COB'], 'g-', linewidth=2, label='COB')
    plt.title('典型日COB变化')
    plt.xlabel('小时')
    plt.ylabel('COB')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(hourly_data['hour'], hourly_data['ISF'], 'r-', linewidth=2, label='ISF')
    plt.title('典型日ISF变化')
    plt.xlabel('小时')
    plt.ylabel('ISF')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(hourly_data['hour'], hourly_data['glucose_level'], 'm-', linewidth=2, label='血糖')
    plt.title('典型日血糖变化')
    plt.xlabel('小时')
    plt.ylabel('血糖 (mg/dL)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('typical_day_curves.png')
    plt.close()

    # 保存典型日数据
    hourly_data.to_csv('typical_day_data.csv', index=False)

    print("典型日曲线可视化完成")

# 执行典型日曲线可视化
visualize_typical_day(features_df)

print("所有分析完成！")