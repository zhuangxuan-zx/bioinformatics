#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
新样本预测示例脚本
用途：展示如何使用训练好的模型预测新样本的分子亚型
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

# 设置项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
PREDICTION_DIR = PROJECT_ROOT / "results" / "prediction_models"

# 将预测模型目录添加到系统路径
sys.path.append(str(PREDICTION_DIR))

def load_model_and_scaler():
    """加载模型和数据预处理器"""
    model_path = PREDICTION_DIR / "SVM_model.pkl"
    scaler_path = PREDICTION_DIR / "scaler.pkl"
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        return model, scaler
    except Exception as e:
        print(f"加载模型或预处理器时出错: {e}")
        return None, None

def load_feature_list():
    """加载模型所需的特征列表"""
    feature_file = PREDICTION_DIR / "selected_model_features.csv"
    
    try:
        features_df = pd.read_csv(feature_file)
        feature_list = []
        
        for _, row in features_df.iterrows():
            feature_name = f"{row['omics_type']}_{row['feature']}"
            feature_list.append(feature_name)
        
        return feature_list
    except Exception as e:
        print(f"加载特征列表时出错: {e}")
        return None

def load_test_data():
    """加载测试数据（示例：使用部分训练数据）"""
    data_file = PREDICTION_DIR / "model_training_data.csv"
    
    try:
        data = pd.read_csv(data_file)
        
        # 提取特征和标签
        X = data.drop(columns=['subtype'])
        y = data['subtype']
        
        # 仅使用少量样本作为示例
        sample_indices = np.random.choice(len(X), min(5, len(X)), replace=False)
        X_sample = X.iloc[sample_indices]
        y_sample = y.iloc[sample_indices]
        
        return X_sample, y_sample
    except Exception as e:
        print(f"加载测试数据时出错: {e}")
        return None, None

def simulate_new_patient_data(feature_list):
    """模拟新患者数据"""
    # 从训练数据中随机选择样本
    X_sample, _ = load_test_data()
    
    if X_sample is None or len(X_sample) == 0:
        print("无法创建模拟数据")
        return None
    
    # 选择一个样本并添加随机变异
    patient_data = X_sample.iloc[0].copy()
    
    # 模拟一些数据缺失和变异
    for feature in np.random.choice(feature_list, 10, replace=False):
        if np.random.random() < 0.3:
            patient_data[feature] = np.nan  # 缺失值
        else:
            patient_data[feature] *= (1 + np.random.normal(0, 0.1))  # 随机变异
    
    return pd.DataFrame([patient_data])

def predict_patient_subtype(patient_data, model, scaler, feature_list):
    """预测新患者的分子亚型"""
    try:
        # 确保数据包含所有需要的特征
        missing_features = [f for f in feature_list if f not in patient_data.columns]
        for feature in missing_features:
            patient_data[feature] = 0  # 对缺失特征填0
        
        # 处理缺失值
        patient_data = patient_data.fillna(0)
        
        # 确保特征顺序正确
        X = patient_data[feature_list]
        
        # 标准化
        X_scaled = scaler.transform(X)
        
        # 预测
        prediction = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        
        return prediction, probabilities
    except Exception as e:
        print(f"预测时出错: {e}")
        return None, None

def main():
    print("胃癌分子亚型预测示例")
    
    # 1. 加载模型和预处理器
    model, scaler = load_model_and_scaler()
    if model is None or scaler is None:
        return
    
    # 2. 加载特征列表
    feature_list = load_feature_list()
    if feature_list is None:
        return
    
    print(f"模型使用{len(feature_list)}个特征进行预测")
    
    # 3. 模拟新患者数据
    print("\n生成模拟患者数据...")
    patient_data = simulate_new_patient_data(feature_list)
    if patient_data is None:
        return
    
    # 4. 预测亚型
    print("\n预测患者分子亚型...")
    prediction, probabilities = predict_patient_subtype(patient_data, model, scaler, feature_list)
    
    if prediction is not None:
        print(f"\n预测亚型: {prediction[0]}")
        print("\n各亚型概率:")
        for i, prob in enumerate(probabilities[0]):
            print(f"  亚型 {i}: {prob:.4f} ({prob*100:.1f}%)")
        
        # 5. 展示亚型特征
        print("\n患者所属分子亚型的特征:")
        try:
            with open(PROJECT_ROOT / "results" / "molecular_subtypes" / "molecular_subtype_descriptions.txt", "r", encoding='utf-8') as f:
                descriptions = f.read().split("\n\n")
            
            for desc in descriptions:
                if desc.startswith(f"分子亚型 {prediction[0]}"):
                    print(desc)
                    break
        except:
            print("  无法加载亚型描述")
            
        print("\n预测完成！在实际应用中，可以根据预测结果为患者提供个性化治疗建议。")

if __name__ == "__main__":
    main()