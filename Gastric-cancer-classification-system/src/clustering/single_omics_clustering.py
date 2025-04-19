#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
单组学聚类分析脚本 - 简化版
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# 设置项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
CLUSTER_DIR = RESULTS_DIR / "clustering"
FIGURES_DIR = CLUSTER_DIR / "figures"

# 确保输出目录存在
CLUSTER_DIR.mkdir(exist_ok=True, parents=True)
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

def load_all_omics_data():
    """加载所有已完成特征选择的组学数据"""
    omics_data = {}
    omics_types = ["expression", "methylation", "mirna", "cnv", "mutation", "protein"]
    
    for omics_type in omics_types:
        # 按优先级加载数据
        for file_prefix in ["correlation_filtered", "mad_filtered", "variance_filtered", "filtered"]:
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
                    break
                except Exception:
                    continue
    
    return omics_data

def load_clinical_data():
    """加载临床数据"""
    clinical_path = PROCESSED_DIR / "clinical_processed.csv"
    if clinical_path.exists():
        clinical_data = pd.read_csv(clinical_path, index_col=0)
        print(f"临床数据加载成功: {clinical_data.shape[0]}行 x {clinical_data.shape[1]}列")
        return clinical_data
    else:
        print("警告: 未找到临床数据文件")
        return None

def determine_optimal_clusters(data, max_clusters=8):
    """使用轮廓系数确定最佳聚类数"""
    # 标准化数据
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # 计算不同聚类数的轮廓系数
    silhouette_scores = []
    
    for n_clusters in range(2, min(max_clusters + 1, data.shape[0])):
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_data)
            score = silhouette_score(scaled_data, cluster_labels)
            silhouette_scores.append(score)
            print(f"  聚类数 {n_clusters}: 轮廓系数 = {score:.4f}")
        except:
            silhouette_scores.append(-1)
    
    # 找到最佳聚类数
    if not silhouette_scores or max(silhouette_scores) <= 0:
        return 2, silhouette_scores
    best_n_clusters = silhouette_scores.index(max(silhouette_scores)) + 2  # +2因为从2个聚类开始
    
    return best_n_clusters, silhouette_scores

def perform_clustering(data, n_clusters):
    """使用K-means进行聚类"""
    # 标准化数据
    scaler = StandardScaler()
    scaled_data_array = scaler.fit_transform(data)
    scaled_data = pd.DataFrame(scaled_data_array, index=data.index, columns=data.columns)
    
    # K-means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_data)
    
    # 创建聚类结果Series
    clusters = pd.Series(cluster_labels, index=data.index, name='cluster')
    
    return clusters, scaled_data

def visualize_clusters(data, clusters, omics_type):
    """可视化聚类结果"""
    # PCA降维可视化
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)
    
    viz_df = pd.DataFrame({
        'PCA1': pca_result[:, 0],
        'PCA2': pca_result[:, 1],
        'Cluster': clusters
    })
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=viz_df, x='PCA1', y='PCA2', hue='Cluster', palette='viridis', s=100, alpha=0.7)
    plt.title(f'{omics_type} 聚类结果 - PCA可视化')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.savefig(FIGURES_DIR / f"{omics_type}_pca_clusters.png", dpi=300)
    plt.close()
    
    # 热图可视化
    try:
        # 选择最有代表性的100个特征
        if data.shape[1] > 100:
            std_vals = data.std().sort_values(ascending=False)
            top_features = std_vals.index[:100].tolist()
            data_subset = data[top_features]
        else:
            data_subset = data
        
        # 按聚类结果排序
        sorted_idx = clusters.sort_values().index
        sorted_data = data_subset.loc[sorted_idx]
        
        # 创建聚类颜色映射 - 修复颜色映射问题
        n_clusters = int(clusters.max()) + 1
        cluster_colors = sns.color_palette("viridis", n_clusters)  # 使用viridis调色板
        lut = dict(zip(range(n_clusters), cluster_colors))  # 创建标签到颜色的查找表
        
        # 将聚类标签映射到颜色
        row_colors = clusters.map(lut)
        
        # 绘制热图
        plt.figure(figsize=(12, 10))
        g = sns.clustermap(
            sorted_data,
            row_cluster=False,
            col_cluster=True,
            cmap='viridis',
            z_score=0,
            row_colors=row_colors.loc[sorted_idx],  # 使用颜色映射
            xticklabels=False,
            yticklabels=False
        )
        plt.savefig(FIGURES_DIR / f"{omics_type}_cluster_heatmap.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"  热图可视化出错: {e}")

def main():
    print("开始单组学聚类分析...")
    
    # 1. 加载数据
    omics_data = load_all_omics_data()
    clinical_data = load_clinical_data()
    
    # 2. 对每个组学类型进行聚类分析
    for omics_type, data in omics_data.items():
        print(f"\n分析{omics_type}数据...")
        
        # 跳过样本过少的组学
        if data.shape[0] < 5:
            print(f"  样本数量过少({data.shape[0]})，跳过聚类分析")
            continue
        
        # 确定最佳聚类数
        best_n_clusters, silhouette_scores = determine_optimal_clusters(data)
        print(f"  最佳聚类数: {best_n_clusters}")
        
        # 可视化轮廓系数
        plt.figure(figsize=(10, 6))
        plt.plot(range(2, 2 + len(silhouette_scores)), silhouette_scores, 'o-')
        plt.axvline(best_n_clusters, color='red', linestyle='--')
        plt.title(f'{omics_type}数据最佳聚类数确定')
        plt.xlabel('聚类数')
        plt.ylabel('轮廓系数')
        plt.grid(alpha=0.3)
        plt.savefig(FIGURES_DIR / f"{omics_type}_optimal_clusters.png", dpi=300)
        plt.close()
        
        # 执行聚类
        clusters, scaled_data = perform_clustering(data, best_n_clusters)
        
        # 可视化聚类结果
        visualize_clusters(scaled_data, clusters, omics_type)
        
        # 保存聚类结果
        clusters.to_csv(CLUSTER_DIR / f"{omics_type}_clusters.csv")
    
    print("\n单组学聚类分析完成！")

if __name__ == "__main__":
    main()