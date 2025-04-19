#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TCGA-STAD胃癌突变数据预处理脚本
用途：处理体细胞突变数据，计算突变负荷，识别高频突变基因，分析突变特征
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# 设置项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures" / "mutation"

# 确保输出目录存在
PROCESSED_DIR.mkdir(exist_ok=True, parents=True)
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

def load_data():
    """加载原始突变数据和处理后的临床数据"""
    # 加载突变数据
    mutation_path = RAW_DIR / "genomic" / "STAD_mc3_gene_level.txt"
    mutation_data = pd.read_csv(mutation_path, sep='\t', index_col=0)
    
    # 加载处理后的临床数据(用于后续整合分析)
    clinical_path = PROCESSED_DIR / "clinical_processed.csv"
    clinical_data = pd.read_csv(clinical_path, index_col=0)
    
    print(f"成功加载突变数据: {mutation_data.shape[0]}个基因 x {mutation_data.shape[1]}个样本")
    print(f"成功加载临床数据: {clinical_data.shape[0]}行 x {clinical_data.shape[1]}列")
    
    return mutation_data, clinical_data

def preprocess_mutation_data(mutation_data):
    """预处理突变数据：过滤、标准化列名等"""
    # 转置数据，使样本为行，基因为列
    mutation_transposed = mutation_data.T
    
    # 确保所有值为二进制(0/1)
    mutation_transposed = mutation_transposed.applymap(lambda x: 1 if x > 0 else 0)
    
    # 统一样本ID格式(如果需要)
    # mutation_transposed.index = mutation_transposed.index.str.upper()
    
    return mutation_transposed

def calculate_mutation_burden(mutation_data):
    """计算每个样本的突变负荷(总突变数量)"""
    mutation_burden = mutation_data.sum(axis=1)
    return mutation_burden

def identify_frequent_mutations(mutation_data, min_freq=0.05):
    """识别高频突变基因(在至少min_freq比例样本中出现突变的基因)"""
    # 计算每个基因的突变频率
    mutation_freq = mutation_data.mean(axis=0)
    
    # 筛选高频突变基因
    high_freq_genes = mutation_freq[mutation_freq >= min_freq]
    high_freq_genes_sorted = high_freq_genes.sort_values(ascending=False)
    
    print(f"识别出{len(high_freq_genes_sorted)}个高频突变基因(突变频率 >= {min_freq*100}%)")
    
    return high_freq_genes_sorted

def analyze_mutation_by_msi(mutation_data, clinical_data):
    """分析不同MSI状态的突变模式差异"""
    # 合并突变负荷与临床数据
    mutation_burden = calculate_mutation_burden(mutation_data)
    burden_df = pd.DataFrame({'mutation_burden': mutation_burden})
    
    # 只保留共有的样本
    common_samples = list(set(burden_df.index) & set(clinical_data.index))
    burden_df = burden_df.loc[common_samples]
    clinical_subset = clinical_data.loc[common_samples]
    
    # 合并数据
    merged_data = pd.concat([burden_df, clinical_subset[['MSI_status']]], axis=1)
    
    # 统计不同MSI状态的突变负荷
    msi_burden = merged_data.groupby('MSI_status')['mutation_burden'].agg(['mean', 'std', 'count'])
    
    # 可视化
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='MSI_status', y='mutation_burden', data=merged_data)
    plt.title('MSI状态与突变负荷关系')
    plt.xlabel('MSI状态 (0=MSS, 1=MSI-L, 2=MSI-H)')
    plt.ylabel('突变负荷(突变基因数量)')
    plt.savefig(FIGURES_DIR / 'msi_vs_mutation_burden.png', dpi=300)
    
    return msi_burden

def create_mutation_matrix(mutation_data, top_genes=100):
    """创建用于后续分析的突变矩阵(仅包含高频突变基因)"""
    # 计算每个基因的突变频率
    mutation_freq = mutation_data.mean(axis=0)
    
    # 获取前top_genes个高频突变基因
    top_genes_list = mutation_freq.sort_values(ascending=False).head(top_genes).index.tolist()
    
    # 创建包含高频突变基因的矩阵
    mutation_matrix = mutation_data[top_genes_list]
    
    return mutation_matrix

def visualize_mutation_data(mutation_data, clinical_data, top_genes=20):
    """可视化突变数据，包括热图和突变频率分布"""
    # 获取高频突变基因
    mutation_freq = mutation_data.mean(axis=0).sort_values(ascending=False)
    top_genes = mutation_freq.head(top_genes).index.tolist()
    
    # 绘制热图
    plt.figure(figsize=(12, 8))
    mutation_subset = mutation_data[top_genes]
    
    # 如果样本数量太多，可以随机选择一部分样本
    if mutation_subset.shape[0] > 100:
        mutation_subset = mutation_subset.sample(100)
    
    # 获取这些样本的MSI状态
    common_samples = list(set(mutation_subset.index) & set(clinical_data.index))
    if common_samples:
        mutation_subset = mutation_subset.loc[common_samples]
        msi_status = clinical_data.loc[common_samples, 'MSI_status']
        
        # 对样本进行排序，按MSI状态分组
        sample_order = msi_status.sort_values().index
        mutation_subset = mutation_subset.loc[sample_order]
        
        # 绘制热图，行标签为MSI状态
        g = sns.clustermap(
            mutation_subset, 
            cmap='viridis',
            row_colors=msi_status.map({0: 'blue', 1: 'green', 2: 'red'}).loc[sample_order],
            figsize=(15, 10),
            col_cluster=True,
            row_cluster=False,
            yticklabels=False
        )
        plt.title('Top突变基因热图(按MSI状态分组)', fontsize=14)
        plt.savefig(FIGURES_DIR / 'mutation_heatmap.png', dpi=300)
    else:
        # 如果没有共有样本，则直接绘制热图
        sns.heatmap(mutation_subset, cmap='viridis')
        plt.title('Top突变基因热图', fontsize=14)
        plt.savefig(FIGURES_DIR / 'mutation_heatmap_no_msi.png', dpi=300)
    
    # 绘制突变频率条形图
    plt.figure(figsize=(12, 8))
    sns.barplot(x=mutation_freq.head(30).index, y=mutation_freq.head(30).values)
    plt.title('Top 30高频突变基因')
    plt.xlabel('基因')
    plt.ylabel('突变频率')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'top_mutated_genes.png', dpi=300)

def main():
    """主函数"""
    print(f"开始处理突变数据: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 加载数据
    mutation_data, clinical_data = load_data()
    
    # 2. 预处理突变数据
    mutation_processed = preprocess_mutation_data(mutation_data)
    print(f"预处理后的突变数据: {mutation_processed.shape[0]}个样本 x {mutation_processed.shape[1]}个基因")
    
    # 3. 计算突变负荷
    mutation_burden = calculate_mutation_burden(mutation_processed)
    print(f"平均每个样本突变基因数: {mutation_burden.mean():.2f}")
    print(f"突变负荷范围: {mutation_burden.min()} - {mutation_burden.max()}")
    
    # 4. 识别高频突变基因
    high_freq_genes = identify_frequent_mutations(mutation_processed, min_freq=0.05)
    print("Top 10高频突变基因:")
    for gene, freq in high_freq_genes.head(10).items():
        print(f"{gene}: {freq*100:.2f}%")
    
    # 5. 分析MSI状态与突变的关系
    msi_burden = analyze_mutation_by_msi(mutation_processed, clinical_data)
    print("\nMSI状态与突变负荷关系:")
    print(msi_burden)
    
    # 6. 创建突变矩阵(仅包含高频突变基因)
    mutation_matrix = create_mutation_matrix(mutation_processed, top_genes=100)
    
    # 7. 可视化突变数据
    visualize_mutation_data(mutation_processed, clinical_data)
    
    # 8. 保存处理后的数据
    # 保存突变矩阵
    mutation_matrix.to_csv(PROCESSED_DIR / "mutation_matrix.csv")
    
    # 保存突变负荷数据
    mutation_burden_df = pd.DataFrame({'mutation_burden': mutation_burden})
    mutation_burden_df.to_csv(PROCESSED_DIR / "mutation_burden.csv")
    
    print(f"处理后的突变数据已保存至: {PROCESSED_DIR}")
    print(f"突变数据处理完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()