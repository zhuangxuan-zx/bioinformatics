#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MAD(中位数绝对偏差)特征选择脚本
用途：基于MAD选择每个组学中最有信息量的特征，对异常值更稳健
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from scipy import stats

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
    
    # 优先加载方差过滤后的数据，如果不存在则加载原始预处理数据
    omics_types = ["expression", "methylation", "mirna", "cnv", "mutation", "protein"]
    
    for omics_type in omics_types:
        # 尝试加载方差过滤后的数据
        variance_file = PROCESSED_DIR / f"{omics_type}_variance_filtered.csv"
        original_file = PROCESSED_DIR / f"{omics_type}_filtered.csv"
        
        # 针对甲基化数据的特殊处理
        if omics_type == "methylation":
            original_file = PROCESSED_DIR / "methylation_filtered_cg_only.csv"
        elif omics_type == "mutation":
            original_file = PROCESSED_DIR / "mutation_matrix.csv"
            
        # 优先使用方差过滤后的数据
        if variance_file.exists():
            try:
                data = pd.read_csv(variance_file, index_col=0)
                omics_data[omics_type] = data
                print(f"{omics_type}数据加载成功(方差过滤后): {data.shape[0]}样本 x {data.shape[1]}特征")
                continue
            except Exception as e:
                print(f"警告: 加载{omics_type}方差过滤数据时出错: {e}")
        
        # 如果方差过滤数据不存在或加载失败，尝试加载原始预处理数据
        if original_file.exists():
            try:
                data = pd.read_csv(original_file, index_col=0)
                omics_data[omics_type] = data
                print(f"{omics_type}数据加载成功(原始预处理): {data.shape[0]}样本 x {data.shape[1]}特征")
            except Exception as e:
                print(f"警告: 加载{omics_type}原始数据时出错: {e}")
        else:
            print(f"警告: {omics_type}数据文件不存在，已跳过")
    
    return omics_data

def calculate_mad(data):
    """计算每个特征的中位数绝对偏差(MAD)"""
    # 对每个特征计算中位数
    medians = np.median(data, axis=0)
    
    # 计算每个观测值与中位数的绝对偏差
    abs_dev = np.abs(data - medians)
    
    # 计算绝对偏差的中位数
    mad_values = np.median(abs_dev, axis=0)
    
    return pd.Series(mad_values, index=data.columns)

def mad_filtering(omics_data, top_percentile=90):
    """基于MAD选择顶部特征"""
    filtered_data = {}
    feature_mads = {}
    
    for omics_type, data in omics_data.items():
        # 计算每个特征的MAD
        mads = calculate_mad(data)
        
        # 确定阈值(保留顶部10%的高变异特征)
        threshold = np.percentile(mads, 100 - top_percentile)
        
        # 选择MAD大于阈值的特征
        high_mad_features = mads[mads >= threshold].index
        filtered = data[high_mad_features]
        
        print(f"{omics_type}: 从{data.shape[1]}个特征中选择了{filtered.shape[1]}个高MAD特征")
        
        filtered_data[omics_type] = filtered
        feature_mads[omics_type] = mads
    
    return filtered_data, feature_mads

def visualize_mad_distribution(feature_mads):
    """可视化各组学数据的MAD分布"""
    plt.figure(figsize=(15, 10))
    
    for i, (omics_type, mads) in enumerate(feature_mads.items(), 1):
        plt.subplot(2, 3, i)
        
        # 检查数据是否适合直方图
        unique_values = len(np.unique(mads))
        
        if unique_values < 10:
            # 数据点太少或者太集中，使用条形图代替
            values_count = pd.Series(mads).value_counts().sort_index()
            plt.bar(values_count.index.astype(str), values_count.values)
            plt.title(f"{omics_type}特征MAD分布 (离散值)")
        else:
            # 动态确定bin数量
            bins = min(50, max(10, unique_values // 5))
            
            try:
                # 尝试使用KDE
                sns.histplot(mads, bins=bins, kde=True)
            except ValueError:
                # 如果KDE失败，尝试简单直方图
                try:
                    plt.hist(mads, bins=bins)
                except ValueError:
                    # 如果直方图也失败，使用散点图代替
                    plt.scatter(range(len(mads)), sorted(mads), alpha=0.5, s=10)
                    plt.xlabel("特征索引(排序后)")
            
            plt.title(f"{omics_type}特征MAD分布")
        
        plt.xlabel("MAD值")
        plt.ylabel("频率")
        
        # 添加垂直线标记90%阈值
        threshold = np.percentile(mads, 10)
        plt.axvline(threshold, color='red', linestyle='--', 
                    label=f"10%阈值: {threshold:.4f}")
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "mad_distributions.png", dpi=300)
    print("已保存MAD分布可视化结果")
    plt.close()

def compare_variance_mad_selection(omics_type, var_data, mad_data):
    """比较方差和MAD选择的特征差异"""
    var_features = set(var_data.columns)
    mad_features = set(mad_data.columns)
    
    # 计算重叠特征
    common_features = var_features.intersection(mad_features)
    var_only = var_features - mad_features
    mad_only = mad_features - var_features
    
    print(f"\n{omics_type}方差与MAD特征选择比较:")
    print(f"  方差选择的特征数: {len(var_features)}")
    print(f"  MAD选择的特征数: {len(mad_features)}")
    print(f"  共同特征数: {len(common_features)} ({len(common_features)/len(var_features)*100:.1f}%)")
    print(f"  仅方差选择的特征数: {len(var_only)}")
    print(f"  仅MAD选择的特征数: {len(mad_only)}")

def main():
    print("开始执行MAD特征选择...")
    
    # 1. 加载所有组学数据
    omics_data = load_all_omics_data()
    
    # 2. 基于MAD选择特征
    filtered_data, feature_mads = mad_filtering(omics_data, top_percentile=90)
    
    # 3. 可视化MAD分布
    visualize_mad_distribution(feature_mads)
    
    # 4. 比较方差和MAD选择的特征差异
    for omics_type, mad_data in filtered_data.items():
        # 尝试加载方差过滤数据进行比较
        var_file = PROCESSED_DIR / f"{omics_type}_variance_filtered.csv"
        if var_file.exists():
            try:
                var_data = pd.read_csv(var_file, index_col=0)
                compare_variance_mad_selection(omics_type, var_data, mad_data)
            except Exception:
                pass
    
    # 5. 保存MAD过滤后的数据
    for omics_type, data in filtered_data.items():
        output_path = PROCESSED_DIR / f"{omics_type}_mad_filtered.csv"
        data.to_csv(output_path)
        print(f"已保存{omics_type} MAD过滤后的数据: {output_path}")
    
    print("MAD特征选择完成！")

if __name__ == "__main__":
    main()