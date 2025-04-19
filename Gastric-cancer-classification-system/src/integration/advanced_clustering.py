#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
高级聚类分析脚本
用途：尝试多种聚类方法获取更多胃癌分子亚型
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
import warnings
warnings.filterwarnings('ignore')

# 设置项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
INTEGRATION_DIR = RESULTS_DIR / "integration"
ADV_CLUSTER_DIR = RESULTS_DIR / "advanced_clustering"
FIGURES_DIR = ADV_CLUSTER_DIR / "figures"

# 创建输出目录
ADV_CLUSTER_DIR.mkdir(exist_ok=True, parents=True)
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

def load_fused_matrix():
    """加载或重建融合相似性矩阵"""
    fused_matrix_path = INTEGRATION_DIR / "fused_similarity_matrix.npy"
    
    if fused_matrix_path.exists():
        try:
            fused_matrix = np.load(fused_matrix_path)
            print(f"已加载融合相似性矩阵: {fused_matrix.shape}")
            return fused_matrix
        except Exception as e:
            print(f"加载融合相似性矩阵出错: {e}")
            return None
    else:
        print("未找到融合相似性矩阵，尝试重建...")
        return None

def load_sample_ids():
    """加载样本ID列表"""
    try:
        # 尝试从整合聚类文件中获取样本ID
        integrated_clusters_path = INTEGRATION_DIR / "integrated_clusters.csv"
        if integrated_clusters_path.exists():
            clusters = pd.read_csv(integrated_clusters_path, index_col=0)
            sample_ids = list(clusters.index)
            print(f"从整合聚类结果中获取样本ID: {len(sample_ids)}个样本")
            return sample_ids
        else:
            print("错误: 未找到整合聚类结果文件")
            return None
    except Exception as e:
        print(f"加载样本ID时出错: {e}")
        return None

def rebuild_similarity_matrix(integrated_clusters):
    """从整合聚类结果重建相似性矩阵"""
    try:
        # 确保integrated_clusters是DataFrame，并且有至少1列
        if isinstance(integrated_clusters, pd.Series):
            integrated_clusters = pd.DataFrame(integrated_clusters)
        
        # 创建初始矩阵
        n_samples = len(integrated_clusters)
        similarity_matrix = np.zeros((n_samples, n_samples))
        
        # 填充相似度：同一聚类内样本相似度为1，不同聚类间为0
        cluster_labels = integrated_clusters.iloc[:, 0].values
        for i in range(n_samples):
            for j in range(n_samples):
                if cluster_labels[i] == cluster_labels[j]:
                    similarity_matrix[i, j] = 1.0
        
        print(f"已重建相似性矩阵，形状: {similarity_matrix.shape}")
        return similarity_matrix
    except Exception as e:
        print(f"重建相似性矩阵时出错: {e}")
        return None

def try_multiple_clustering_methods(fused_matrix, sample_ids, n_clusters=4):
    """尝试多种聚类方法"""
    # 确保融合矩阵有效
    fused_matrix = np.clip(fused_matrix, 0, 1)
    
    # 计算距离矩阵
    distance_matrix = 1 - fused_matrix
    np.fill_diagonal(distance_matrix, 0)
    distance_matrix = np.nan_to_num(distance_matrix, nan=0)
    
    # 初始化结果字典
    results = {}
    
    # 1. 谱聚类
    print("\n执行谱聚类...")
    try:
        spectral = SpectralClustering(
            n_clusters=n_clusters, 
            affinity='precomputed',
            assign_labels='discretize',
            random_state=42,
            n_init=10
        )
        spectral_labels = spectral.fit_predict(fused_matrix)
        results['spectral'] = spectral_labels
        print(f"  谱聚类完成，识别出{len(np.unique(spectral_labels))}个聚类")
    except Exception as e:
        print(f"  谱聚类失败: {e}")
        # 创建一个默认的分类结果作为替代
        results['spectral'] = np.zeros(len(sample_ids), dtype=int)
    
    # 2. 层次聚类
    print("执行层次聚类...")
    try:
        # 使用scipy的层次聚类，更稳定
        Z = linkage(squareform(distance_matrix), method='average')
        hierarchical_labels = fcluster(Z, n_clusters, criterion='maxclust') - 1  # 减1使标签从0开始
        results['hierarchical'] = hierarchical_labels
        print(f"  层次聚类完成，识别出{len(np.unique(hierarchical_labels))}个聚类")
    except Exception as e:
        print(f"  层次聚类失败: {e}")
        # 使用谱聚类结果作为替代
        if 'spectral' in results:
            results['hierarchical'] = results['spectral'].copy()
        else:
            results['hierarchical'] = np.zeros(len(sample_ids), dtype=int)
    
    # 3. MDS + K-means
    print("执行MDS降维 + K-means聚类...")
    try:
        # MDS降维
        mds = MDS(n_components=min(10, len(sample_ids)-1), 
                 dissimilarity='precomputed', 
                 random_state=42,
                 n_init=1,
                 normalized_stress='auto')
        mds_result = mds.fit_transform(distance_matrix)
        
        # K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(mds_result)
        results['kmeans'] = kmeans_labels
        print(f"  K-means聚类完成，识别出{len(np.unique(kmeans_labels))}个聚类")
    except Exception as e:
        print(f"  MDS+K-means聚类失败: {e}")
        # 使用层次聚类结果作为替代
        if 'hierarchical' in results:
            results['kmeans'] = results['hierarchical'].copy()
        else:
            results['kmeans'] = np.zeros(len(sample_ids), dtype=int)
    
    # 4. 高斯混合模型
    print("执行高斯混合模型聚类...")
    try:
        if mds_result is not None:
            gmm = GaussianMixture(n_components=n_clusters, random_state=42, n_init=10)
            gmm_labels = gmm.fit_predict(mds_result)
            results['gmm'] = gmm_labels
            print(f"  GMM聚类完成，识别出{len(np.unique(gmm_labels))}个聚类")
        else:
            raise ValueError("MDS结果不可用")
    except Exception as e:
        print(f"  GMM聚类失败: {e}")
        # 使用K-means结果作为替代
        if 'kmeans' in results:
            results['gmm'] = results['kmeans'].copy()
        else:
            results['gmm'] = np.zeros(len(sample_ids), dtype=int)
    
    # 评估各方法稳定性
    evaluate_clustering_stability(results)
    
    # 整合所有结果
    combined_labels = combine_clustering_results(results, sample_ids, n_clusters)
    
    return combined_labels, results

def evaluate_clustering_stability(clustering_results):
    """评估不同聚类方法的一致性"""
    methods = list(clustering_results.keys())
    
    print("\n评估聚类方法的一致性:")
    for i in range(len(methods)):
        for j in range(i+1, len(methods)):
            method1 = methods[i]
            method2 = methods[j]
            
            # 计算调整兰德指数
            try:
                ari = adjusted_rand_score(
                    clustering_results[method1],
                    clustering_results[method2]
                )
                print(f"  {method1} 与 {method2}: ARI = {ari:.4f}")
            except Exception as e:
                print(f"  计算{method1}与{method2}的ARI时出错: {e}")

def combine_clustering_results(clustering_results, sample_ids, n_clusters):
    """整合多种聚类结果为最终亚型"""
    # 创建包含所有方法结果的数据框
    results_df = pd.DataFrame(index=sample_ids)
    
    for method, labels in clustering_results.items():
        results_df[method] = labels
    
    # 保存所有聚类结果
    results_df.to_csv(ADV_CLUSTER_DIR / "all_clustering_results.csv")
    
    # 创建共识聚类
    # 方法：将每个样本与其他样本的共聚类频率作为相似度，然后再次聚类
    n_samples = len(sample_ids)
    co_occurrence = np.zeros((n_samples, n_samples))
    
    # 计算样本对之间的共聚类次数
    for method, labels in clustering_results.items():
        for i in range(n_samples):
            for j in range(i, n_samples):
                if labels[i] == labels[j]:
                    co_occurrence[i, j] += 1
                    co_occurrence[j, i] += 1
    
    # 归一化
    co_occurrence /= max(1, len(clustering_results))
    
    # 基于共聚类矩阵进行最终聚类
    try:
        final_spectral = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=42
        )
        final_labels = final_spectral.fit_predict(co_occurrence)
    except Exception as e:
        print(f"使用谱聚类整合结果失败: {e}")
        
        # 备选方法：使用层次聚类
        try:
            Z = linkage(squareform(1 - co_occurrence), method='average')
            final_labels = fcluster(Z, n_clusters, criterion='maxclust') - 1
        except Exception as e2:
            print(f"使用层次聚类整合结果也失败: {e2}")
            
            # 最后的备选：使用投票
            print("使用多数投票方式整合结果")
            from sklearn.cluster import AgglomerativeClustering
            
            # 转换每种方法的标签以确保一致的标签解释
            aligned_labels = {}
            for method in clustering_results:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                # 使用每种方法的标签作为一维特征进行聚类
                aligned_labels[method] = kmeans.fit_predict(clustering_results[method].reshape(-1, 1))
            
            # 使用多数投票
            votes = np.zeros((n_samples, n_clusters))
            for labels in aligned_labels.values():
                for i in range(n_samples):
                    votes[i, labels[i]] += 1
                    
            final_labels = np.argmax(votes, axis=1)
    
    # 确保标签从0开始连续编号
    unique_labels = np.unique(final_labels)
    label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
    final_labels_mapped = np.array([label_map[lbl] for lbl in final_labels])
    
    # 创建最终的亚型结果
    final_subtypes = pd.Series(final_labels_mapped, index=sample_ids, name='subtype')
    final_subtypes.to_csv(ADV_CLUSTER_DIR / "final_subtypes.csv")
    
    # 可视化亚型分布
    print("\n最终亚型分布:")
    subtype_counts = pd.Series(final_labels_mapped).value_counts().sort_index()
    for subtype, count in subtype_counts.items():
        print(f"  亚型 {subtype}: {count}个样本 ({count/len(final_labels_mapped)*100:.1f}%)")
    
    # 可视化
    visualize_combined_results(co_occurrence, final_labels_mapped, sample_ids)
    
    return final_subtypes

def visualize_combined_results(co_occurrence, final_labels, sample_ids):
    """可视化聚类结果"""
    # 1. 共聚类热图
    try:
        plt.figure(figsize=(12, 10))
        
        # 按亚型排序
        idx = np.argsort(final_labels)
        sorted_matrix = co_occurrence[idx, :][:, idx]
        
        # 计算每个亚型的分界线位置
        label_positions = [0]
        current_label = final_labels[idx[0]]
        for i, label in enumerate(final_labels[idx]):
            if label != current_label:
                label_positions.append(i)
                current_label = label
        label_positions.append(len(idx))
        
        # 绘制热图
        plt.imshow(sorted_matrix, cmap='viridis', aspect='auto')
        plt.colorbar(label='共聚类频率')
        
        # 添加亚型分界线
        for pos in label_positions:
            plt.axhline(y=pos-0.5, color='white', linestyle='-')
            plt.axvline(x=pos-0.5, color='white', linestyle='-')
        
        plt.title('样本共聚类频率矩阵')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "co_occurrence_matrix.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"绘制共聚类热图时出错: {e}")
    
    # 2. MDS可视化
    try:
        plt.figure(figsize=(10, 8))
        
        # 将共聚类矩阵转换为距离
        distance = 1 - co_occurrence
        np.fill_diagonal(distance, 0)
        
        # MDS降维
        mds = MDS(n_components=2, 
                 dissimilarity='precomputed', 
                 random_state=42, 
                 n_init=1, 
                 normalized_stress='auto')
        mds_result = mds.fit_transform(distance)
        
        # 创建可视化数据框
        viz_df = pd.DataFrame({
            'MDS1': mds_result[:, 0],
            'MDS2': mds_result[:, 1],
            'Subtype': final_labels
        })
        
        # 绘制散点图
        sns.scatterplot(
            data=viz_df, 
            x='MDS1', 
            y='MDS2', 
            hue='Subtype', 
            palette='viridis', 
            s=100, 
            alpha=0.7
        )
        plt.title('整合后的胃癌分子亚型')
        plt.legend(title='亚型', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "integrated_subtypes_mds.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"绘制MDS可视化时出错: {e}")

def main():
    print("开始执行高级聚类分析...")
    
    # 1. 加载样本ID和整合聚类结果
    sample_ids = load_sample_ids()
    if sample_ids is None:
        print("错误: 无法继续分析，未找到样本ID")
        return
    
    # 2. 尝试加载融合矩阵
    fused_matrix = load_fused_matrix()
    
    # 3. 如果融合矩阵加载失败，尝试从整合聚类结果重建
    if fused_matrix is None:
        print("尝试从整合聚类结果重建融合矩阵...")
        # 加载整合聚类结果
        integrated_clusters_path = INTEGRATION_DIR / "integrated_clusters.csv"
        if integrated_clusters_path.exists():
            integrated_clusters = pd.read_csv(integrated_clusters_path, index_col=0)
            print(f"已加载整合聚类结果: {len(integrated_clusters)}个样本")
            
            # 重建相似性矩阵
            fused_matrix = rebuild_similarity_matrix(integrated_clusters)
            if fused_matrix is None:
                print("错误: 重建融合矩阵失败，无法继续分析")
                return
                
            # 保存重建的融合矩阵，以便后续使用
            np.save(INTEGRATION_DIR / "fused_similarity_matrix.npy", fused_matrix)
        else:
            print("错误: 未找到整合聚类结果，无法继续分析")
            return
    
    # 4. 执行多种聚类方法
    print("\n使用多种聚类方法获取4个胃癌分子亚型...")
    final_subtypes, all_results = try_multiple_clustering_methods(fused_matrix, sample_ids, n_clusters=4)
    
    print("\n高级聚类分析完成！")
    print(f"最终亚型结果已保存至: {ADV_CLUSTER_DIR / 'final_subtypes.csv'}")
    print("请使用最终亚型结果继续分子亚型特征分析")

if __name__ == "__main__":
    main()