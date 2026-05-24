import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from datetime import datetime
import os

def parse_glucose_xml(file_path):
    """解析单个XML文件并返回DataFrame"""
    tree = ET.parse(file_path)
    root = tree.getroot()

    # 提取患者信息
    patient_id = root.attrib['id']
    weight = float(root.attrib['weight'])
    insulin_type = root.attrib['insulin_type']

    # 解析血糖数据
    glucose_data = []
    for event in root.find('glucose_level').findall('event'):
        ts = datetime.strptime(event.attrib['ts'], '%d-%m-%Y %H:%M:%S')
        value = float(event.attrib['value'])
        glucose_data.append({'timestamp': ts, 'glucose_level': value, 'patient_id': patient_id})

    glucose_df = pd.DataFrame(glucose_data)

    # 解析进餐数据
    meal_data = []
    for event in root.find('meal').findall('event'):
        ts = datetime.strptime(event.attrib['ts'], '%d-%m-%Y %H:%M:%S')
        meal_type = event.attrib['type']
        carbs = float(event.attrib['carbs'])
        meal_data.append({'timestamp': ts, 'meal_type': meal_type, 'carbs': carbs, 'patient_id': patient_id})

    meal_df = pd.DataFrame(meal_data)

    # 解析运动数据
    exercise_data = []
    for event in root.find('exercise').findall('event'):
        ts = datetime.strptime(event.attrib['ts'], '%d-%m-%Y %H:%M:%S')
        intensity = int(event.attrib['intensity'])
        duration = int(event.attrib['duration'])
        exercise_data.append({'timestamp': ts, 'intensity': intensity, 'duration': duration, 'patient_id': patient_id})

    exercise_df = pd.DataFrame(exercise_data)

    # 解析基础心率数据
    hr_data = []
    for event in root.find('basis_heart_rate').findall('event'):
        ts = datetime.strptime(event.attrib['ts'], '%d-%m-%Y %H:%M:%S')
        value = float(event.attrib['value'])
        hr_data.append({'timestamp': ts, 'heart_rate': value, 'patient_id': patient_id})

    hr_df = pd.DataFrame(hr_data)

    # 解析基础皮肤温度数据
    temp_data = []
    for event in root.find('basis_skin_temperature').findall('event'):
        ts = datetime.strptime(event.attrib['ts'], '%d-%m-%Y %H:%M:%S')
        value = float(event.attrib['value'])
        temp_data.append({'timestamp': ts, 'skin_temperature': value, 'patient_id': patient_id})

    temp_df = pd.DataFrame(temp_data)

    # 解析基础皮肤电反应数据
    gsr_data = []
    for event in root.find('basis_gsr').findall('event'):
        ts = datetime.strptime(event.attrib['ts'], '%d-%m-%Y %H:%M:%S')
        value = float(event.attrib['value'])
        gsr_data.append({'timestamp': ts, 'gsr': value, 'patient_id': patient_id})

    gsr_df = pd.DataFrame(gsr_data)

    # 解析睡眠数据
    sleep_data = []
    for event in root.find('sleep').findall('event'):
        ts_begin = datetime.strptime(event.attrib['ts_begin'], '%d-%m-%Y %H:%M:%S')
        ts_end = datetime.strptime(event.attrib['ts_end'], '%d-%m-%Y %H:%M:%S')
        quality = int(event.attrib['quality'])
        sleep_duration = (ts_end - ts_begin).total_seconds() / 3600
        sleep_data.append({'sleep_start': ts_begin, 'sleep_end': ts_end, 'sleep_duration': sleep_duration,
                         'sleep_quality': quality, 'patient_id': patient_id})

    sleep_df = pd.DataFrame(sleep_data)

    # 解析基础胰岛素率数据
    basal_data = []
    basal_elem = root.find('basal')
    if basal_elem is not None and len(basal_elem) > 0:
        for event in basal_elem.findall('event'):
            ts = datetime.strptime(event.attrib['ts'], '%d-%m-%Y %H:%M:%S')
            value = float(event.attrib['value'])
            basal_data.append({'timestamp': ts, 'basal_rate': value, 'patient_id': patient_id})
    basal_df = pd.DataFrame(basal_data) if basal_data else pd.DataFrame(columns=['timestamp', 'basal_rate', 'patient_id'])

    # 解析临时基础胰岛素率数据
    temp_basal_data = []
    temp_basal_elem = root.find('temp_basal')
    if temp_basal_elem is not None and len(temp_basal_elem) > 0:
        for event in temp_basal_elem.findall('event'):
            ts_begin = datetime.strptime(event.attrib['ts_begin'], '%d-%m-%Y %H:%M:%S')
            ts_end = datetime.strptime(event.attrib['ts_end'], '%d-%m-%Y %H:%M:%S')
            value = float(event.attrib['value'])
            temp_basal_data.append({'temp_basal_start': ts_begin, 'temp_basal_end': ts_end,
                                   'temp_basal_rate': value, 'patient_id': patient_id})
    temp_basal_df = pd.DataFrame(temp_basal_data) if temp_basal_data else pd.DataFrame(
        columns=['temp_basal_start', 'temp_basal_end', 'temp_basal_rate', 'patient_id'])

    # 解析大剂量胰岛素数据
    bolus_data = []
    bolus_elem = root.find('bolus')
    if bolus_elem is not None and len(bolus_elem) > 0:
        for event in bolus_elem.findall('event'):
            ts_begin = datetime.strptime(event.attrib['ts_begin'], '%d-%m-%Y %H:%M:%S')
            ts_end = datetime.strptime(event.attrib['ts_end'], '%d-%m-%Y %H:%M:%S')
            bolus_type = event.attrib.get('type', 'normal')
            dose = float(event.attrib['dose'])
            carb_input = float(event.attrib['bwz_carb_input']) if 'bwz_carb_input' in event.attrib else 0.0
            bolus_data.append({'timestamp': ts_begin, 'bolus_end': ts_end, 'bolus_type': bolus_type,
                              'bolus_dose': dose, 'bolus_carb_input': carb_input, 'patient_id': patient_id})
    bolus_df = pd.DataFrame(bolus_data) if bolus_data else pd.DataFrame(
        columns=['timestamp', 'bolus_end', 'bolus_type', 'bolus_dose', 'bolus_carb_input', 'patient_id'])

    # 解析指尖血糖数据
    finger_stick_data = []
    finger_elem = root.find('finger_stick')
    if finger_elem is not None and len(finger_elem) > 0:
        for event in finger_elem.findall('event'):
            ts = datetime.strptime(event.attrib['ts'], '%d-%m-%Y %H:%M:%S')
            value = float(event.attrib['value'])
            finger_stick_data.append({'timestamp': ts, 'finger_stick_value': value, 'patient_id': patient_id})
    finger_df = pd.DataFrame(finger_stick_data) if finger_stick_data else pd.DataFrame(
        columns=['timestamp', 'finger_stick_value', 'patient_id'])

    return {
        'glucose': glucose_df,
        'meal': meal_df,
        'exercise': exercise_df,
        'heart_rate': hr_df,
        'skin_temperature': temp_df,
        'gsr': gsr_df,
        'sleep': sleep_df,
        'basal': basal_df,
        'temp_basal': temp_basal_df,
        'bolus': bolus_df,
        'finger_stick': finger_df,
        'patient_info': {'id': patient_id, 'weight': weight, 'insulin_type': insulin_type}
    }

def load_all_data(data_dir):
    """加载目录中所有XML文件"""
    all_data = []
    xml_files = [f for f in os.listdir(data_dir) if f.endswith('.xml')]

    for xml_file in xml_files:
        file_path = os.path.join(data_dir, xml_file)
        print(f"Processing {xml_file}...")
        patient_data = parse_glucose_xml(file_path)
        all_data.append(patient_data)

    # 合并所有患者数据
    glucose_df = pd.concat([d['glucose'] for d in all_data], ignore_index=True)
    meal_df = pd.concat([d['meal'] for d in all_data], ignore_index=True)
    exercise_df = pd.concat([d['exercise'] for d in all_data], ignore_index=True)
    hr_df = pd.concat([d['heart_rate'] for d in all_data], ignore_index=True)
    temp_df = pd.concat([d['skin_temperature'] for d in all_data], ignore_index=True)
    gsr_df = pd.concat([d['gsr'] for d in all_data], ignore_index=True)
    sleep_df = pd.concat([d['sleep'] for d in all_data], ignore_index=True)
    basal_df = pd.concat([d['basal'] for d in all_data], ignore_index=True)
    temp_basal_df = pd.concat([d['temp_basal'] for d in all_data], ignore_index=True)
    bolus_df = pd.concat([d['bolus'] for d in all_data], ignore_index=True)
    finger_df = pd.concat([d['finger_stick'] for d in all_data], ignore_index=True)

    return {
        'glucose': glucose_df,
        'meal': meal_df,
        'exercise': exercise_df,
        'heart_rate': hr_df,
        'skin_temperature': temp_df,
        'gsr': gsr_df,
        'sleep': sleep_df,
        'basal': basal_df,
        'temp_basal': temp_basal_df,
        'bolus': bolus_df,
        'finger_stick': finger_df
    }

if __name__ == "__main__":
    # 合并2018和2020的训练+测试数据
    data_dirs = [
        "D:\\ohio\\OhioT1DM\\OhioT1DM 2018\\train",
        "D:\\ohio\\OhioT1DM\\OhioT1DM 2018\\test",
        "D:\\ohio\\OhioT1DM\\OhioT1DM 2020\\train",
        "D:\\ohio\\OhioT1DM\\OhioT1DM 2020\\test",
    ]

    all_combined = {}
    for data_dir in data_dirs:
        if os.path.isdir(data_dir):
            print(f"\n=== 加载目录: {data_dir} ===")
            data = load_all_data(data_dir)
            for key, df in data.items():
                if key == 'patient_info':
                    continue
                if key not in all_combined:
                    all_combined[key] = []
                all_combined[key].append(df)

    # 合并所有目录的数据
    for key in all_combined:
        all_combined[key] = pd.concat(all_combined[key], ignore_index=True)

    # 保存为CSV文件
    os.makedirs('data', exist_ok=True)
    all_combined['glucose'].to_csv('data/glucose_data.csv', index=False)
    all_combined['meal'].to_csv('data/meal_data.csv', index=False)
    all_combined['exercise'].to_csv('data/exercise_data.csv', index=False)
    all_combined['heart_rate'].to_csv('data/heart_rate_data.csv', index=False)
    all_combined['skin_temperature'].to_csv('data/skin_temperature_data.csv', index=False)
    all_combined['gsr'].to_csv('data/gsr_data.csv', index=False)
    all_combined['sleep'].to_csv('data/sleep_data.csv', index=False)
    all_combined['basal'].to_csv('data/basal_data.csv', index=False)
    all_combined['temp_basal'].to_csv('data/temp_basal_data.csv', index=False)
    all_combined['bolus'].to_csv('data/bolus_data.csv', index=False)
    all_combined['finger_stick'].to_csv('data/finger_stick_data.csv', index=False)

    print("\n数据加载完成，已保存为CSV文件")
    for key, df in all_combined.items():
        print(f"  {key}: {len(df)} 行")