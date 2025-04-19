#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TCGA-STAD胃癌临床数据预处理脚本
用途：处理临床数据，提取关键特征，处理缺失值，合并生存数据
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 设置项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures" / "clinical"

# 确保输出目录存在
PROCESSED_DIR.mkdir(exist_ok=True, parents=True)
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

def load_data():
    """加载原始临床数据和生存数据"""
    # 加载临床数据
    clinical_path = RAW_DIR / "clinical" / "TCGA.STAD.sampleMap_STAD_clinicalMatrix"
    clinical_data = pd.read_csv(clinical_path, sep='\t', index_col=0)
    
    # 加载生存数据
    survival_path = RAW_DIR / "survival" / "SURVIVAL_STAD_SURVIVAL.TXT"
    survival_data = pd.read_csv(survival_path, sep='\t', index_col=0)
    
    print(f"成功加载临床数据: {clinical_data.shape[0]}行 x {clinical_data.shape[1]}列")
    print(f"成功加载生存数据: {survival_data.shape[0]}行 x {survival_data.shape[1]}列")
    
    return clinical_data, survival_data

def clean_clinical_data(data):
    """清洗临床数据：选择关键特征，处理缺失值和异常值"""
    # 选择关键临床特征列
    key_features = [
        'CDE_ID_3226963',  # MSI状态
        'age_at_initial_pathologic_diagnosis',  # 确诊年龄
        'gender',  # 性别
        'anatomic_neoplasm_subdivision',  # 肿瘤位置
        'pathologic_stage',  # 病理分期
        'pathologic_T',  # T分期
        'pathologic_N',  # N分期
        'pathologic_M',  # M分期
        'neoplasm_histologic_grade',  # 组织学分级
        'histological_type',  # 组织学类型
        'vital_status',  # 生存状态
        'days_to_death',  # 死亡天数
        'days_to_last_followup',  # 最后随访天数
        'new_tumor_event_after_initial_treatment',  # 初始治疗后是否有新肿瘤事件
        'primary_therapy_outcome_success',  # 一线治疗结果
        'person_neoplasm_cancer_status',  # 肿瘤状态
        'h_pylori_infection'  # 幽门螺杆菌感染
    ]
    
    # 确保所有关键特征都在数据中
    available_features = [f for f in key_features if f in data.columns]
    selected_data = data[available_features].copy()
    
    # 输出缺失值百分比
    na_percent = selected_data.isna().sum() * 100 / len(selected_data)
    print("各特征缺失值百分比:")
    print(na_percent)
    
    # 仅保留肿瘤样本(不包含-11样本)
    tumor_samples = selected_data.index.str.contains('-01')
    selected_data = selected_data[tumor_samples]
    
    # 创建患者ID列(从样本ID中提取)
    selected_data['patient_id'] = selected_data.index.str.extract('(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})')
    
    return selected_data

def transform_data(data):
    """转换数据：编码分类变量，特征工程"""
    transformed_data = data.copy()
    
    # 1. MSI状态编码
    msi_mapping = {'MSS': 0, 'MSI-L': 1, 'MSI-H': 2}
    transformed_data['MSI_status'] = transformed_data['CDE_ID_3226963'].map(msi_mapping)
    
    # 2. 性别编码
    transformed_data['gender_code'] = transformed_data['gender'].map({'MALE': 0, 'FEMALE': 1})
    
    # 3. 生存状态编码
    transformed_data['vital_status_code'] = transformed_data['vital_status'].map({'LIVING': 0, 'DECEASED': 1})
    
    # 4. 肿瘤位置编码
    location_mapping = {
        'Antrum/Distal': 0,
        'Cardia/Proximal': 1, 
        'Fundus/Body': 2,
        'Gastroesophageal Junction': 3,
        'Stomach (NOS)': 4
    }
    transformed_data['tumor_location_code'] = transformed_data['anatomic_neoplasm_subdivision'].map(location_mapping)
    
    # 5. 处理分期数据
    # 提取数字部分并转换为整数
    def extract_stage(stage_str):
        if pd.isna(stage_str):
            return np.nan
        # 处理罗马数字分期
        if 'IV' in stage_str:
            return 4
        elif 'III' in stage_str:
            return 3
        elif 'II' in stage_str:
            return 2
        elif 'I' in stage_str:
            return 1
        else:
            return np.nan
    
    transformed_data['stage_code'] = transformed_data['pathologic_stage'].apply(extract_stage)
    
    # 6. 处理生存时间
    # 使用days_to_death(如果有)或days_to_last_followup作为总生存时间
    transformed_data['overall_survival_time'] = transformed_data['days_to_death']
    mask = transformed_data['overall_survival_time'].isna()
    transformed_data.loc[mask, 'overall_survival_time'] = transformed_data.loc[mask, 'days_to_last_followup']
    
    return transformed_data

def merge_with_survival_data(clinical_data, survival_data):
    """合并临床数据和生存数据"""
    # 确保索引匹配
    survival_data_matched = survival_data.loc[clinical_data.index.intersection(survival_data.index)]
    
    # 如果生存数据中有临床数据中没有的列，则添加到临床数据
    for col in ['OS', 'OS.time', 'DSS', 'DSS.time', 'DFI', 'DFI.time', 'PFI', 'PFI.time']:
        if col in survival_data_matched.columns and col not in clinical_data.columns:
            clinical_data.loc[survival_data_matched.index, col] = survival_data_matched[col]
    
    return clinical_data

def analyze_data(data):
    """基本统计分析和可视化"""
    # 1. MSI状态分布
    msi_counts = data['MSI_status'].value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=msi_counts.index, y=msi_counts.values)
    plt.title('MSI状态分布')
    plt.xlabel('MSI状态 (0=MSS, 1=MSI-L, 2=MSI-H)')
    plt.ylabel('样本数')
    plt.savefig(FIGURES_DIR / 'msi_status_distribution.png', dpi=300)
    
    # 2. 年龄分布
    plt.figure(figsize=(10, 6))
    sns.histplot(data['age_at_initial_pathologic_diagnosis'].dropna(), bins=20)
    plt.title('患者年龄分布')
    plt.xlabel('年龄')
    plt.ylabel('样本数')
    plt.savefig(FIGURES_DIR / 'age_distribution.png', dpi=300)
    
    # 3. 性别分布
    gender_counts = data['gender'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%')
    plt.title('性别分布')
    plt.savefig(FIGURES_DIR / 'gender_distribution.png', dpi=300)
    
    # 4. 肿瘤位置分布
    location_counts = data['anatomic_neoplasm_subdivision'].value_counts()
    plt.figure(figsize=(12, 6))
    sns.barplot(x=location_counts.index, y=location_counts.values)
    plt.title('肿瘤位置分布')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'tumor_location_distribution.png', dpi=300)
    
    # 5. MSI状态与生存关系
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='MSI_status', y='overall_survival_time', data=data)
    plt.title('MSI状态与总生存期关系')
    plt.xlabel('MSI状态 (0=MSS, 1=MSI-L, 2=MSI-H)')
    plt.ylabel('总生存期(天)')
    plt.savefig(FIGURES_DIR / 'msi_vs_survival.png', dpi=300)
    
    return msi_counts, gender_counts, location_counts

def main():
    """主函数"""
    print(f"开始处理: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 加载数据
    clinical_data, survival_data = load_data()
    
    # 2. 清洗临床数据
    clinical_data_cleaned = clean_clinical_data(clinical_data)
    print(f"清洗后的临床数据: {clinical_data_cleaned.shape[0]}行 x {clinical_data_cleaned.shape[1]}列")
    
    # 3. 转换数据
    clinical_data_transformed = transform_data(clinical_data_cleaned)
    
    # 4. 合并生存数据
    merged_data = merge_with_survival_data(clinical_data_transformed, survival_data)
    
    # 5. 分析数据
    msi_counts, gender_counts, location_counts = analyze_data(merged_data)
    
    # 输出基本统计信息
    print("\n基本统计信息:")
    print(f"样本总数: {len(merged_data)}")
    print(f"MSI状态分布: {msi_counts.to_dict()}")
    print(f"性别分布: {gender_counts.to_dict()}")
    print(f"年龄范围: {merged_data['age_at_initial_pathologic_diagnosis'].min()}-"
          f"{merged_data['age_at_initial_pathologic_diagnosis'].max()}岁，"
          f"平均{merged_data['age_at_initial_pathologic_diagnosis'].mean():.1f}岁")
    
    # 6. 保存处理后的数据
    output_path = PROCESSED_DIR / "clinical_processed.csv"
    merged_data.to_csv(output_path)
    print(f"处理后的数据已保存至: {output_path}")
    print(f"处理完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()