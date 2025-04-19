#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
高速版相关性特征过滤脚本
用途：快速移除高度相关的冗余特征
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

# 设置项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "figures" / "feature_selection"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

def load_all_omics_data():
    """优先加载已过滤数据"""
    omics_data = {}
    omics_types = ["expression", "methylation", "mirna", "cnv", "mutation", "protein"]
    
    for omics_type in omics_types:
        # 优先加载MAD过滤数据
        for file_prefix in ["mad_filtered", "variance_filtered", "filtered"]:
            if omics_type == "methylation" and file_prefix == "filtered":
                file_path = PROCESSED_DIR / "methylation_filtered_cg_only.csv"
            elif omics_type == "mutation" and file_prefix == "filtered":
                file_path = PROCESSED_DIR / "mutation_matrix.csv"
            else:
                file_path = PROCESSED_DIR / f"{omics_type}_{file_prefix}.csv"
                
            if file_path.exists():
                try:
                    data = pd.read_csv(file_path, index_col=0)
                    omics_data[omics_type] = data
                    print(f"{omics_type}数据加载成功({file_prefix}): {data.shape[0]}样本 x {data.shape[1]}特征")
                    break  # 成功加载后不再尝试其他文件
                except Exception:
                    continue
    
    return omics_data

def fast_filter_correlated_features(data, threshold=0.9, method='pearson', max_features=3000):
    """快速版相关性过滤"""
    start_time = time.time()
    feature_count = data.shape[1]
    
    # 更激进地缩减特征数量
    if feature_count > max_features:
        print(f"  特征数量({feature_count})超过阈值({max_features})，进行激进缩减")
        std_vals = data.std().sort_values(ascending=False)
        selected_features = std_vals.index[:max_features].tolist()
        data = data[selected_features]
        print(f"  已缩减至{data.shape[1]}个特征")
    
    # 二进制数据使用Jaccard相似度
    if method == 'jaccard':
        print("  二进制数据：使用快速Jaccard计算...")
        # 对于二进制数据，使用稀疏存储和快速计算
        features_to_keep = []
        features = list(data.columns)
        
        for i, feature in enumerate(features):
            if i % 10 == 0:
                print(f"  处理特征 {i}/{len(features)}...")
                
            if feature not in features_to_keep:
                features_to_keep.append(feature)
                
                # 只计算该特征与其后特征的相似度
                for j in range(i+1, len(features)):
                    other_feature = features[j]
                    
                    # 快速计算Jaccard
                    v1 = data[feature].values.astype(bool)
                    v2 = data[other_feature].values.astype(bool)
                    intersection = np.sum(v1 & v2)
                    union = np.sum(v1 | v2)
                    
                    # 避免除零
                    similarity = intersection / max(union, 1)
                    
                    # 如果相似度高于阈值，不保留该特征
                    if similarity > threshold:
                        # 保留变异度更高的特征
                        if data[feature].std() < data[other_feature].std():
                            features_to_keep.remove(feature)
                            features_to_keep.append(other_feature)
                        break
    else:
        # 对于连续数据，分块处理
        print("  连续数据：使用分块相关性计算...")
        features_to_keep = []
        features = list(data.columns)
        
        # 分块处理以减少内存使用
        chunk_size = min(500, len(features))
        
        for i in range(0, len(features), chunk_size):
            print(f"  处理特征块 {i}-{min(i+chunk_size, len(features))}/{len(features)}...")
            chunk_features = features[i:min(i+chunk_size, len(features))]
            
            # 计算当前块与所有已保留特征的相关性
            for feature in chunk_features:
                keep = True
                
                for kept_feature in features_to_keep:
                    corr = abs(data[feature].corr(data[kept_feature], method='pearson'))
                    
                    if corr > threshold:
                        # 如果相关性高，保留变异度更大的特征
                        if data[feature].std() > data[kept_feature].std():
                            features_to_keep.remove(kept_feature)
                            features_to_keep.append(feature)
                        keep = False
                        break
                
                if keep:
                    features_to_keep.append(feature)
    
    # 使用保留的特征创建过滤后的数据集
    filtered_data = data[features_to_keep]
    
    print(f"  初始特征数: {feature_count}")
    print(f"  过滤后特征数: {len(features_to_keep)}")
    print(f"  移除了{feature_count - len(features_to_keep)}个高度相关特征")
    print(f"  处理时间: {time.time() - start_time:.1f}秒")
    
    return filtered_data

def main():
    print("开始执行快速相关性特征过滤...")
    
    # 1. 加载数据
    omics_data = load_all_omics_data()
    
    # 2. 设置阈值
    correlation_thresholds = {
        "expression": 0.9, "methylation": 0.95, "mirna": 0.85, 
        "cnv": 0.95, "mutation": 0.75, "protein": 0.85
    }
    
    # 3. 按组学类型过滤
    filtered_data = {}
    
    for omics_type, data in omics_data.items():
        print(f"\n处理{omics_type}数据...")
        threshold = correlation_thresholds.get(omics_type, 0.9)
        method = 'jaccard' if omics_type == "mutation" else 'pearson'
        
        # 设置不同组学类型的最大特征数
        max_features = {
            "expression": 3000, "methylation": 2000, "mirna": 500, 
            "cnv": 3000, "mutation": 100, "protein": 100
        }.get(omics_type, 1000)
        
        # 快速过滤
        filtered = fast_filter_correlated_features(
            data, threshold=threshold, method=method, max_features=max_features)
        filtered_data[omics_type] = filtered
    
    # 4. 保存结果
    for omics_type, data in filtered_data.items():
        output_path = PROCESSED_DIR / f"{omics_type}_correlation_filtered.csv"
        data.to_csv(output_path)
        print(f"已保存{omics_type}相关性过滤后的数据: {output_path}")
    
    print("\n快速相关性特征过滤完成！")

if __name__ == "__main__":
    main()