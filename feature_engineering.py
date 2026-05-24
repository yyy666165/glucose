import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import shap

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

class GlucoseFeatureEngineer:
    def __init__(self, glucose_df, meal_df, exercise_df, sleep_df):
        self.glucose_df = glucose_df.sort_values(['patient_id', 'timestamp'])
        self.meal_df = meal_df.sort_values(['patient_id', 'timestamp'])
        self.exercise_df = exercise_df.sort_values(['patient_id', 'timestamp'])
        self.sleep_df = sleep_df.sort_values(['patient_id', 'sleep_start'])

        # 初始化参数（后续可通过学习优化）
        self.insulin_peak_time = 1.5  # 胰岛素峰值时间（小时）
        self.insulin_duration = 6.0   # 胰岛素作用持续时间（小时）
        self.carb_absorption_peak = 1.0  # 碳水吸收峰值时间（小时）
        self.carb_absorption_duration = 4.0  # 碳水吸收持续时间（小时）

    def calculate_IOB(self, row):
        """计算体内活性胰岛素(IOB)"""
        patient_id = row['patient_id']
        current_time = row['timestamp']

        # 获取该患者的历史胰岛素注射数据（假设从bolus数据中获取）
        # 这里简化处理，实际需要从胰岛素注射数据中获取
        # 使用假设的胰岛素作用曲线
        iob = 0.0

        # 查找过去6小时内的胰岛素注射
        time_window = current_time - timedelta(hours=self.insulin_duration)
        recent_injections = self.meal_df[
            (self.meal_df['patient_id'] == patient_id) &
            (self.meal_df['timestamp'] >= time_window) &
            (self.meal_df['timestamp'] <= current_time)
        ]

        for _, injection in recent_injections.iterrows():
            dose = injection['dose'] if 'dose' in injection else 0
            time_since_injection = (current_time - injection['timestamp']).total_seconds() / 3600

            # 胰岛素作用曲线（简化版）
            if time_since_injection <= 0:
                continue

            # 使用双指数模型模拟胰岛素作用
            absorption = np.exp(-time_since_injection / self.insulin_peak_time)
            elimination = np.exp(-time_since_injection / self.insulin_duration)
            iob += dose * (absorption - elimination)

        return max(0, iob)

    def calculate_COB(self, row):
        """计算活性碳水(COB)"""
        patient_id = row['patient_id']
        current_time = row['timestamp']

        cob = 0.0

        # 查找过去4小时内的进餐
        time_window = current_time - timedelta(hours=self.carb_absorption_duration)
        recent_meals = self.meal_df[
            (self.meal_df['patient_id'] == patient_id) &
            (self.meal_df['timestamp'] >= time_window) &
            (self.meal_df['timestamp'] <= current_time)
        ]

        for _, meal in recent_meals.iterrows():
            carbs = meal['carbs']
            time_since_meal = (current_time - meal['timestamp']).total_seconds() / 3600

            if time_since_meal <= 0:
                continue

            # 碳水吸收曲线（简化版）
            absorption = np.exp(-time_since_meal / self.carb_absorption_peak)
            elimination = np.exp(-time_since_meal / self.carb_absorption_duration)
            cob += carbs * (absorption - elimination)

        return max(0, cob)

    def calculate_ISF(self, row):
        """计算时变胰岛素敏感性(ISF*)"""
        patient_id = row['patient_id']
        current_time = row['timestamp']

        # 基础胰岛素敏感性（假设值，实际需要学习）
        base_isf = 1.0

        # 睡眠债影响
        sleep_debt = 0
        for _, sleep in self.sleep_df.iterrows():
            if (sleep['patient_id'] == patient_id and
                current_time >= sleep['sleep_start'] and
                current_time <= sleep['sleep_end'] + timedelta(hours=12)):
                sleep_debt = 5 - sleep['sleep_quality']
                break

        # 运动影响
        exercise_intensity = 0
        for _, exercise in self.exercise_df.iterrows():
            if (exercise['patient_id'] == patient_id and
                abs((current_time - exercise['timestamp']).total_seconds() / 3600) <= 2):
                exercise_intensity = exercise['intensity']
                break

        # 时间影响（昼夜节律）
        hour = current_time.hour
        circadian_factor = 1.0 + 0.2 * np.sin(2 * np.pi * (hour - 6) / 24)  # 假设早晨胰岛素敏感性较高

        # 计算时变胰岛素敏感性
        isf = base_isf * (1 + sleep_debt * 0.1) * (1 - exercise_intensity * 0.05) * circadian_factor

        return isf

    def build_features(self):
        """构建完整特征集"""
        features_df = self.glucose_df.copy()

        # 合并运动数据
        exercise_glucose = pd.merge_asof(
            features_df.sort_values(['patient_id', 'timestamp']),
            self.exercise_df.sort_values(['patient_id', 'timestamp']),
            by='patient_id',
            on='timestamp',
            direction='backward',
            tolerance=pd.Timedelta(hours=1)
        )
        features_df['exercise_intensity'] = exercise_glucose['intensity'].fillna(0)

        # 计算生理状态特征
        features_df['IOB'] = features_df.apply(self.calculate_IOB, axis=1)
        features_df['COB'] = features_df.apply(self.calculate_COB, axis=1)
        features_df['ISF'] = features_df.apply(self.calculate_ISF, axis=1)

        # 计算血糖变化
        features_df['prev_glucose'] = features_df.groupby('patient_id')['glucose_level'].shift(1)
        features_df['delta_glucose'] = features_df['glucose_level'] - features_df['prev_glucose']
        features_df = features_df.dropna(subset=['delta_glucose'])

        # 添加缺失的列（简化处理）
        features_df['gastric_emptying_factor'] = 1.0  # 胃排空系数
        features_df['dawn_phenomenon'] = 0  # 黎明现象
        features_df['work_stress'] = 0  # 工作压力
        features_df['cortisol_level'] = 1.0  # 皮质醇水平
        features_df['consecutive_hypo_count'] = 0  # 连续低血糖次数
        features_df['rebound_risk'] = 1.0  # 反跳风险

        # 构建交互特征
        features_df['IOB_x_exercise'] = features_df['IOB'] * features_df['exercise_intensity']
        features_df['COB_x_gastric_emptying'] = features_df['COB'] * features_df['gastric_emptying_factor']
        features_df['current_glucose_x_IOB'] = features_df['glucose_level'] * features_df['IOB']
        features_df['sleep_quality_x_dawn_phenomenon'] = features_df['sleep_quality'] * features_df['dawn_phenomenon']
        features_df['work_stress_x_cortisol'] = features_df['work_stress'] * features_df['cortisol_level']
        features_df['consecutive_hypo_x_rebound'] = features_df['consecutive_hypo_count'] * features_df['rebound_risk']

        # 添加上下文特征
        features_df['hour'] = features_df['timestamp'].dt.hour
        features_df['is_dawn'] = features_df['hour'].between(4, 6).astype(int)
        features_df['is_dusk'] = features_df['hour'].between(16, 18).astype(int)
        features_df['is_weekend'] = features_df['timestamp'].dt.dayofweek >= 5

        # 血糖区间和趋势
        features_df['glucose_zone'] = pd.cut(features_df['glucose_level'],
                                         bins=[0, 70, 140, 180, float('inf')],
                                         labels=['low', 'normal', 'high', 'very_high'])
        features_df['glucose_trend'] = features_df.groupby('patient_id')['glucose_level'].diff().fillna(0)

        # 事件时间距离
        features_df['time_since_meal'] = (features_df['timestamp'] -
                                      features_df.groupby('patient_id')['timestamp'].shift(1)).dt.total_seconds() / 3600
        features_df['time_since_exercise'] = (features_df['timestamp'] -
                                          features_df.groupby('patient_id')['timestamp'].shift(1)).dt.total_seconds() / 3600

        # 添加睡眠质量（从睡眠数据中获取）
        features_df['sleep_quality'] = 0
        for _, sleep in self.sleep_df.iterrows():
            mask = (features_df['patient_id'] == sleep['patient_id']) & \
                   (features_df['timestamp'] >= sleep['sleep_start']) & \
                   (features_df['timestamp'] <= sleep['sleep_end'] + pd.Timedelta(hours=12))
            features_df.loc[mask, 'sleep_quality'] = sleep['sleep_quality']

        return features_df

    def analyze_feature_importance(self, features_df):
        """分析特征重要性"""
        # 准备数据
        X = features_df.drop(['timestamp', 'patient_id', 'glucose_level', 'prev_glucose', 'delta_glucose'], axis=1)
        y = features_df['delta_glucose']

        # 训练随机森林模型
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        # 计算特征重要性
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        # 置换重要性
        result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
        perm_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': result.importances_mean
        }).sort_values('importance', ascending=False)

        return importance, perm_importance

    def visualize_typical_day(self, features_df):
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
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(hourly_data['hour'], hourly_data['IOB'], 'b-', label='IOB')
        plt.title('典型日IOB变化')
        plt.xlabel('小时')
        plt.ylabel('IOB')
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(hourly_data['hour'], hourly_data['COB'], 'g-', label='COB')
        plt.title('典型日COB变化')
        plt.xlabel('小时')
        plt.ylabel('COB')
        plt.grid(True)

        plt.subplot(2, 2, 3)
        plt.plot(hourly_data['hour'], hourly_data['ISF'], 'r-', label='ISF')
        plt.title('典型日ISF变化')
        plt.xlabel('小时')
        plt.ylabel('ISF')
        plt.grid(True)

        plt.subplot(2, 2, 4)
        plt.plot(hourly_data['hour'], hourly_data['glucose_level'], 'm-', label='血糖')
        plt.title('典型日血糖变化')
        plt.xlabel('小时')
        plt.ylabel('血糖 (mg/dL)')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('typical_day_curves.png')
        plt.close()

        return hourly_data

# 主程序
if __name__ == "__main__":
    engineer = GlucoseFeatureEngineer(glucose_df, meal_df, exercise_df, sleep_df)
    features_df = engineer.build_features()

    # 保存特征数据
    features_df.to_csv('glucose_features.csv', index=False)

    # 分析特征重要性
    importance, perm_importance = engineer.analyze_feature_importance(features_df)
    importance.to_csv('feature_importance.csv', index=False)
    perm_importance.to_csv('permutation_importance.csv', index=False)

    # 可视化典型日曲线
    hourly_data = engineer.visualize_typical_day(features_df)
    hourly_data.to_csv('typical_day_data.csv', index=False)

    print("特征工程完成，已生成特征文件和可视化结果")