#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简化的多组学整合脚本 - 排除蛋白质组数据
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# 设置项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
CLUSTER_DIR = RESULTS_DIR / "clustering"
INTEGRATION_DIR = RESULTS_DIR / "integration"
FIGURES_DIR = INTEGRATION_DIR / "figures"

# 确保输出目录存在
INTEGRATION_DIR.mkdir(exist_ok=True, parents=True)
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

def load_selected_omics_clusters():
    """加载选定组学的聚类结果"""
    # 我们将使用除protein外的所有组学
    selected_omics = ["expression", "methylation", "mirna", "cnv", "mutation"]
    cluster_results = {}
    
    for omics_type in selected_omics:
        cluster_file = CLUSTER_DIR / f"{omics_type}_clusters.csv"
        if cluster_file.exists():
            try:
                clusters = pd.read_csv(cluster_file, index_col=0)
                cluster_results[omics_type] = clusters
                print(f"加载{omics_type}聚类结果: {len(clusters)}个样本")
            except Exception as e:
                print(f"加载{omics_type}聚类结果出错: {e}")
    
    # 找出所有组学共有的样本
    common_samples = None
    for omics_type, clusters in cluster_results.items():
        if common_samples is None:
            common_samples = set(clusters.index)
        else:
            common_samples &= set(clusters.index)
    
    if not common_samples:
        print("警告: 所选组学之间没有共同样本")
        # 尝试找出具有最多共同样本的组合
        from itertools import combinations
        max_samples = 0
        best_combo = []
        
        for k in range(len(selected_omics), 1, -1):
            for combo in combinations(selected_omics, k):
                combo_samples = set(cluster_results[combo[0]].index)
                for omics in combo[1:]:
                    if omics in cluster_results:
                        combo_samples &= set(cluster_results[omics].index)
                
                if len(combo_samples) > max_samples:
                    max_samples = len(combo_samples)
                    best_combo = combo
                    
                    # 如果找到很好的组合就提前终止
                    if len(combo_samples) >= 300 and len(combo) >= 3:
                        break
            
            if max_samples >= 300 and len(best_combo) >= 3:
                break
        
        if max_samples > 0:
            print(f"推荐使用组学组合: {', '.join(best_combo)}")
            print(f"这些组学共有{max_samples}个样本")
            
            # 更新组学和共同样本
            cluster_results = {k: cluster_results[k] for k in best_combo}
            common_samples = set.intersection(*[set(v.index) for v in cluster_results.values()])
    
    print(f"所选组学之间共有{len(common_samples)}个样本")
    return cluster_results, common_samples

def create_similarity_matrices(cluster_results, common_samples):
    """为每个组学数据创建样本相似性矩阵"""
    similarity_matrices = {}
    common_samples_list = sorted(common_samples)
    
    for omics_type, clusters in cluster_results.items():
        # 过滤为共同样本
        filtered_clusters = clusters.loc[common_samples_list]
        
        # 创建相似性矩阵
        n_samples = len(common_samples_list)
        sim_matrix = np.zeros((n_samples, n_samples))
        
        cluster_values = filtered_clusters.values.flatten()
        
        for i in range(n_samples):
            for j in range(i, n_samples):
                # 如果两个样本在同一聚类中，则相似度为1，否则为0
                if cluster_values[i] == cluster_values[j]:
                    sim_matrix[i, j] = 1
                    sim_matrix[j, i] = 1
        
        similarity_matrices[omics_type] = sim_matrix
        print(f"已创建{omics_type}相似性矩阵，形状: {sim_matrix.shape}")
    
    return similarity_matrices

def snf_integrate(similarity_matrices, k=20, t=20):
    """执行SNF(相似性网络融合)算法"""
    # 获取样本数量
    n_samples = next(iter(similarity_matrices.values())).shape[0]
    
    # 确保k小于样本数
    k = min(k, max(1, n_samples // 2))
    
    # 初始化融合矩阵为均值
    fused_matrix = np.zeros((n_samples, n_samples))
    for sim_matrix in similarity_matrices.values():
        # 确保相似矩阵值在0-1之间
        cleaned_matrix = np.clip(sim_matrix, 0, 1)
        fused_matrix += cleaned_matrix
    fused_matrix /= len(similarity_matrices)
    
    # 构建局部相似性矩阵
    local_similarities = {}
    
    for omics_type, sim_matrix in similarity_matrices.items():
        # 确保相似矩阵值有效
        cleaned_matrix = np.clip(sim_matrix, 0, 1)
        
        # 计算每个点的k近邻索引
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(cleaned_matrix)
        distances, indices = nbrs.kneighbors(cleaned_matrix)
        
        # 构建局部相似性矩阵
        local_sim = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            neighbors = indices[i, 1:]  # 排除自身
            local_sim[i, neighbors] = cleaned_matrix[i, neighbors]
            
            # 行归一化
            row_sum = local_sim[i, :].sum()
            if row_sum > 0:
                local_sim[i, :] /= row_sum
        
        local_similarities[omics_type] = local_sim
    
    # 迭代融合
    for iteration in range(t):
        new_similarities = {}
        
        for omics_type, local_sim in local_similarities.items():
            # 计算其他组学的平均相似性
            other_types = [typ for typ in local_similarities.keys() if typ != omics_type]
            other_sim = np.zeros_like(fused_matrix)
            
            for other_type in other_types:
                other_sim += local_similarities[other_type]
            
            if other_types:
                other_sim /= len(other_types)
            
            # 更新该组学的相似性
            new_sim = np.matmul(np.matmul(local_sim, other_sim), local_sim.T)
            # 清理更新后的相似性矩阵
            new_sim = np.nan_to_num(new_sim, nan=0, posinf=1, neginf=0)
            new_sim = np.clip(new_sim, 0, 1)
            new_similarities[omics_type] = new_sim
        
        # 更新局部相似性矩阵
        local_similarities = new_similarities
        
        # 更新融合矩阵
        fused_matrix = np.zeros((n_samples, n_samples))
        for sim_matrix in local_similarities.values():
            fused_matrix += sim_matrix
        fused_matrix /= len(local_similarities)
        
        if (iteration + 1) % 5 == 0:
            print(f"SNF融合迭代: {iteration + 1}/{t}")
    
    # 最终清理融合矩阵
    fused_matrix = np.nan_to_num(fused_matrix, nan=0, posinf=1, neginf=0)
    fused_matrix = np.clip(fused_matrix, 0, 1)
    
    return fused_matrix

def determine_optimal_clusters(fused_matrix, max_clusters=8):
    """确定融合矩阵的最佳聚类数"""
    # 清理融合矩阵，确保值在0-1之间
    fused_matrix_cleaned = np.clip(fused_matrix, 0, 1)
    
    # 将相似度转为距离
    distance_matrix = 1 - fused_matrix_cleaned
    np.fill_diagonal(distance_matrix, 0)  # 确保对角线为0
    
    # 检查并修复无效值
    if np.any(np.isinf(distance_matrix)) or np.any(np.isnan(distance_matrix)):
        print("警告：距离矩阵中包含无效值，执行修复...")
        distance_matrix = np.nan_to_num(distance_matrix, nan=0, posinf=1, neginf=0)
    
    silhouette_scores = []
    
    for n_clusters in range(2, max_clusters + 1):
        try:
            spectral = SpectralClustering(
                n_clusters=n_clusters, 
                affinity='precomputed',
                assign_labels='discretize',
                random_state=42
            )
            labels = spectral.fit_predict(fused_matrix_cleaned)  # 使用清理后的矩阵
            
            # 计算轮廓系数
            score = silhouette_score(distance_matrix, labels, metric='precomputed')
            silhouette_scores.append(score)
            print(f"聚类数 {n_clusters}: 轮廓系数 = {score:.4f}")
        except Exception as e:
            print(f"聚类数 {n_clusters} 计算失败: {e}")
            silhouette_scores.append(-1)
    
    # 处理结果
    if not silhouette_scores or max(silhouette_scores) <= 0:
        print("无法确定最佳聚类数，默认使用3个聚类")
        return 3, silhouette_scores
    
    # 找到最佳聚类数
    best_n_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
    
    return best_n_clusters, silhouette_scores

def perform_integrated_clustering(fused_matrix, n_clusters, sample_names):
    """对融合矩阵执行谱聚类"""
    spectral = SpectralClustering(
        n_clusters=n_clusters, 
        affinity='precomputed',
        assign_labels='discretize',
        random_state=42
    )
    
    cluster_labels = spectral.fit_predict(fused_matrix)
    
    # 创建聚类结果Series
    integrated_clusters = pd.Series(
        cluster_labels, 
        index=sample_names, 
        name='integrated_cluster'
    )
    
    return integrated_clusters

def visualize_integrated_clusters(fused_matrix, integrated_clusters):
    """可视化整合聚类结果"""
    # 1. 融合矩阵热图
    plt.figure(figsize=(12, 10))
    # 按聚类排序
    sorted_idx = integrated_clusters.sort_values().index
    sorted_matrix = pd.DataFrame(
        fused_matrix, 
        index=integrated_clusters.index, 
        columns=integrated_clusters.index
    ).loc[sorted_idx, sorted_idx]
    
    # 创建聚类颜色条
    n_clusters = int(integrated_clusters.max()) + 1
    cluster_colors = sns.color_palette("viridis", n_clusters)
    lut = dict(zip(range(n_clusters), cluster_colors))
    row_colors = integrated_clusters.map(lut)
    
    g = sns.clustermap(
        sorted_matrix,
        row_cluster=False,
        col_cluster=False,
        cmap='viridis',
        row_colors=row_colors.loc[sorted_idx],
        xticklabels=False,
        yticklabels=False
    )
    plt.suptitle('整合后的相似性网络', y=1.02)
    plt.savefig(FIGURES_DIR / "integrated_similarity_matrix.png", dpi=300)
    plt.close()
    
    # 2. MDS可视化
    from sklearn.manifold import MDS
    
    # 将相似度转换为距离
    distance_matrix = 1 - fused_matrix
    np.fill_diagonal(distance_matrix, 0)
    
    # MDS降维
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    mds_result = mds.fit_transform(distance_matrix)
    
    # 创建可视化数据框
    viz_df = pd.DataFrame({
        'MDS1': mds_result[:, 0],
        'MDS2': mds_result[:, 1],
        'Cluster': integrated_clusters.values
    }, index=integrated_clusters.index)
    
    # 绘制散点图
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=viz_df, 
        x='MDS1', 
        y='MDS2', 
        hue='Cluster', 
        palette='viridis', 
        s=100, 
        alpha=0.7
    )
    plt.title('整合聚类结果 - MDS可视化')
    plt.savefig(FIGURES_DIR / "integrated_clusters_mds.png", dpi=300)
    plt.close()

def main():
    print("开始简化多组学整合分析...")
    
    # 1. 加载选定的组学聚类结果(排除蛋白质组)
    cluster_results, common_samples = load_selected_omics_clusters()
    
    if len(cluster_results) < 2:
        print("错误: 至少需要两种组学的聚类结果才能进行整合分析")
        return
    
    if len(common_samples) < 30:
        print(f"错误: 共同样本数量({len(common_samples)})过少，无法进行有效的整合分析")
        return
        
    # 2. 创建样本相似性矩阵
    print("\n为各组学数据创建相似性矩阵...")
    similarity_matrices = create_similarity_matrices(cluster_results, common_samples)
    
    # 3. SNF融合整合
    print("\n使用SNF方法进行多组学整合...")
    fused_matrix = snf_integrate(similarity_matrices)
    
    # 4. 确定最佳聚类数
    print("\n确定整合后的最佳聚类数...")
    best_n_clusters, silhouette_scores = determine_optimal_clusters(fused_matrix)
    print(f"最佳整合聚类数: {best_n_clusters}")
    
    # 可视化轮廓系数
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, 2 + len(silhouette_scores)), silhouette_scores, 'o-')
    plt.axvline(best_n_clusters, color='red', linestyle='--')
    plt.title('整合数据最佳聚类数确定')
    plt.xlabel('聚类数')
    plt.ylabel('轮廓系数')
    plt.grid(alpha=0.3)
    plt.savefig(FIGURES_DIR / "integrated_optimal_clusters.png", dpi=300)
    plt.close()
    
    # 5. 执行整合聚类
    print("\n执行整合聚类...")
    sample_names = sorted(common_samples)
    integrated_clusters = perform_integrated_clustering(fused_matrix, best_n_clusters, sample_names)
    
    # 6. 可视化整合聚类结果
    print("\n可视化整合聚类结果...")
    visualize_integrated_clusters(fused_matrix, integrated_clusters)
    
    # 7. 保存整合聚类结果
    integrated_clusters.to_csv(INTEGRATION_DIR / "integrated_clusters.csv")
    
    # 统计各个亚型的样本数
    cluster_counts = integrated_clusters.value_counts().sort_index()
    print("\n整合后的分子亚型样本分布:")
    for cluster, count in cluster_counts.items():
        print(f"  亚型 {cluster}: {count}个样本 ({count/len(integrated_clusters)*100:.1f}%)")
    
    print("\n多组学整合分析完成！")
    print(f"整合聚类结果已保存至: {INTEGRATION_DIR / 'integrated_clusters.csv'}")
    print(f"可视化结果已保存至: {FIGURES_DIR}")

if __name__ == "__main__":
    main()