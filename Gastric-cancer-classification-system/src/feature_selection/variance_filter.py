#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
方差特征选择脚本
用途：基于方差选择每个组学中最有信息量的特征，过滤低变异特征
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# 设置项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "figures" / "feature_selection"

# 确保输出目录存在
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

def load_all_omics_data():
    """加载所有预处理后的组学数据"""
    omics_data = {}
    
    # 尝试加载每种组学数据
    omics_files = {
        "expression": "expression_filtered.csv",
        "methylation": "methylation_filtered_cg_only.csv",
        "mirna": "mirna_filtered.csv",
        "cnv": "cnv_processed.csv",
        "mutation": "mutation_matrix.csv",
        "protein": "protein_filtered.csv"
    }
    
    for omics_type, filename in omics_files.items():
        file_path = PROCESSED_DIR / filename
        if file_path.exists():
            try:
                data = pd.read_csv(file_path, index_col=0)
                omics_data[omics_type] = data
                print(f"{omics_type}数据加载成功: {data.shape[0]}样本 x {data.shape[1]}特征")
            except Exception as e:
                print(f"警告: 加载{omics_type}数据时出错: {e}")
        else:
            print(f"警告: {omics_type}数据文件不存在，已跳过")
    
    return omics_data

def variance_filtering(omics_data, threshold_percentile=10):
    """基于方差过滤低变异特征"""
    filtered_data = {}
    feature_variances = {}
    
    for omics_type, data in omics_data.items():
        # 计算每个特征的方差
        variances = data.var(axis=0)
        
        # 确定阈值(排除底部10%的低变异特征)
        threshold = np.percentile(variances, threshold_percentile)
        
        # 选择方差大于阈值的特征
        high_var_features = variances[variances > threshold].index
        filtered = data[high_var_features]
        
        print(f"{omics_type}: 从{data.shape[1]}个特征中选择了{filtered.shape[1]}个高变异特征")
        
        filtered_data[omics_type] = filtered
        feature_variances[omics_type] = variances
    
    return filtered_data, feature_variances

def visualize_variance_distribution(feature_variances):
    """可视化各组学数据的方差分布 - 增强版，处理不同范围的数据"""
    plt.figure(figsize=(15, 10))
    
    for i, (omics_type, variances) in enumerate(feature_variances.items(), 1):
        plt.subplot(2, 3, i)
        
        # 检查数据是否适合直方图
        unique_values = len(np.unique(variances))
        
        if unique_values < 10:
            # 数据点太少或者太集中，使用条形图代替
            values_count = pd.Series(variances).value_counts().sort_index()
            plt.bar(values_count.index.astype(str), values_count.values)
            plt.title(f"{omics_type}特征方差分布 (离散值)")
        else:
            # 动态确定bin数量
            bins = min(50, max(10, unique_values // 5))
            
            try:
                # 尝试使用KDE
                sns.histplot(variances, bins=bins, kde=True)
            except ValueError:
                # 如果KDE失败，尝试简单直方图
                try:
                    plt.hist(variances, bins=bins)
                except ValueError:
                    # 如果直方图也失败，使用散点图代替
                    plt.scatter(range(len(variances)), sorted(variances), alpha=0.5, s=10)
                    plt.xlabel("特征索引(排序后)")
            
            plt.title(f"{omics_type}特征方差分布")
        
        plt.xlabel("方差")
        plt.ylabel("频率")
        
        # 添加垂直线标记10%阈值
        threshold = np.percentile(variances, 10)
        plt.axvline(threshold, color='red', linestyle='--', 
                    label=f"10%阈值: {threshold:.4f}")
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "variance_distributions.png", dpi=300)
    print("已保存方差分布可视化结果")
    plt.close()

def main():
    print("开始执行方差特征选择...")
    
    # 1. 加载所有组学数据
    omics_data = load_all_omics_data()
    
    # 2. 基于方差过滤
    filtered_data, feature_variances = variance_filtering(omics_data)
    
    # 3. 可视化方差分布
    visualize_variance_distribution(feature_variances)
    
    # 4. 保存过滤后的数据
    for omics_type, data in filtered_data.items():
        output_path = PROCESSED_DIR / f"{omics_type}_variance_filtered.csv"
        data.to_csv(output_path)
        print(f"已保存{omics_type}方差过滤后的数据: {output_path}")
    
    print("方差特征选择完成！")

if __name__ == "__main__":
    main()