#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TCGA-STAD胃癌miRNA数据预处理脚本
用途：处理miRNA表达数据，进行标准化、缺失值处理和特征选择
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures" / "mirna"

# 确保输出目录存在
PROCESSED_DIR.mkdir(exist_ok=True, parents=True)
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

def load_data():
    """加载miRNA数据和处理后的临床数据"""
    # 加载miRNA表达数据
    mirna_path = RAW_DIR / "mirna" / "miRNA_HiSeq_gene"
    mirna_data = pd.read_csv(mirna_path, sep='\t', index_col=0)
    
    # 加载处理后的临床数据
    clinical_path = PROCESSED_DIR / "clinical_processed.csv"
    clinical_data = pd.read_csv(clinical_path, index_col=0)
    
    print(f"成功加载miRNA数据: {mirna_data.shape[0]}个miRNA x {mirna_data.shape[1]}个样本")
    print(f"成功加载临床数据: {clinical_data.shape[0]}行 x {clinical_data.shape[1]}列")
    
    return mirna_data, clinical_data

def preprocess_mirna_data(mirna_data):
    """预处理miRNA数据：标准化、转置等"""
    # 转置数据，使样本为行，miRNA为列
    mirna_transposed = mirna_data.T
    
    # 标准化样本ID格式
    mirna_transposed.index = mirna_transposed.index.str.upper()
    
    # 处理缺失值(如果有)
    na_count = mirna_transposed.isna().sum().sum()
    if na_count > 0:
        print(f"检测到{na_count}个缺失值，使用0填充")
        mirna_transposed = mirna_transposed.fillna(0)
    
    # 检查表达值的分布
    min_expr = mirna_transposed.values.min()
    max_expr = mirna_transposed.values.max()
    print(f"miRNA表达值范围: {min_expr:.4f} - {max_expr:.4f}")
    
    # 确保所有值为正数，避免log变换时出现问题
    if min_expr <= 0:
        print("检测到零或负值，在log变换前将所有表达值加上0.01")
        mirna_transposed = mirna_transposed + 0.01
    
    # 查看数据分布，如果极度偏离正态，可以进行log变换
    if max_expr / mirna_transposed.median().median() > 1000:
        print("数据分布极度偏态，进行log2变换")
        mirna_transposed = np.log2(mirna_transposed)
        
        # 检查log变换后是否有无穷值
        inf_count = np.isinf(mirna_transposed.values).sum()
        if inf_count > 0:
            print(f"变换后检测到{inf_count}个无穷值，将其替换为0")
            mirna_transposed = mirna_transposed.replace([np.inf, -np.inf], 0)
            
    # 最后检查是否有NaN值
    na_count = mirna_transposed.isna().sum().sum()
    if na_count > 0:
        print(f"处理后检测到{na_count}个NaN值，使用0填充")
        mirna_transposed = mirna_transposed.fillna(0)
    
    return mirna_transposed

def select_variable_mirnas(mirna_data, top_n=1000):
    """选择变异性最大的miRNA"""
    # 计算每个miRNA的标准差，比变异系数更稳定
    mirna_std = mirna_data.std(axis=0)
    
    # 选择标准差最高的top_n个miRNA
    top_mirnas = mirna_std.sort_values(ascending=False).index[:top_n]
    selected_data = mirna_data[top_mirnas]
    
    print(f"选择了{len(top_mirnas)}个高变异miRNA")
    
    return selected_data, mirna_std

def analyze_mirna_by_msi(mirna_data, clinical_data):
    """分析不同MSI状态的miRNA表达模式"""
    # 只保留共有的样本
    common_samples = list(set(mirna_data.index) & set(clinical_data.index))
    
    if not common_samples:
        print("警告: miRNA数据和临床数据没有共同样本，跳过MSI分析")
        return pd.DataFrame()
    
    print(f"miRNA和临床数据共有{len(common_samples)}个样本")
    
    try:
        mirna_subset = mirna_data.loc[common_samples]
        clinical_subset = clinical_data.loc[common_samples]
        
        # 计算每个样本的平均miRNA表达
        mean_expression = mirna_subset.mean(axis=1)
        
        # 创建用于可视化的数据框
        msi_data = pd.DataFrame({
            'MSI_status': clinical_subset['MSI_status'],
            'mean_expression': mean_expression
        })
        
        # 可视化不同MSI状态的全局miRNA表达
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='MSI_status', y='mean_expression', data=msi_data)
        plt.title('MSI状态与全局miRNA表达关系')
        plt.xlabel('MSI状态 (0=MSS, 1=MSI-L, 2=MSI-H)')
        plt.ylabel('平均miRNA表达')
        plt.savefig(FIGURES_DIR / 'msi_vs_global_mirna.png', dpi=300)
        
        # 计算不同MSI状态的平均miRNA表达
        msi_mirna_avg = pd.DataFrame()
        for status in msi_data['MSI_status'].unique():
            status_samples = clinical_subset[clinical_subset['MSI_status'] == status].index
            if len(status_samples) > 0:
                msi_mirna_avg[f'MSI_{status}'] = mirna_subset.loc[status_samples].mean()
        
        return msi_mirna_avg
    
    except Exception as e:
        print(f"MSI分析出错: {e}")
        return pd.DataFrame()

def visualize_mirna_data(mirna_data, clinical_data, top_n=50):
    """可视化miRNA数据"""
    print("开始生成miRNA可视化...")
    
    try:
        # 1. 全局表达分布
        plt.figure(figsize=(10, 6))
        sample_data = mirna_data.values.flatten()
        # 避免使用太多数据点，可能导致内存问题
        if len(sample_data) > 100000:
            sample_data = np.random.choice(sample_data, 100000, replace=False)
        sns.histplot(sample_data, bins=50, kde=True)
        plt.title('miRNA表达分布')
        plt.xlabel('表达值')
        plt.ylabel('频率')
        plt.savefig(FIGURES_DIR / 'mirna_distribution.png', dpi=300)
        print("生成了表达分布图")
        
        # 2. 热图 (选择top miRNAs)
        # 计算变异度
        mirna_std = mirna_data.std(axis=0).sort_values(ascending=False)
        top_mirnas = mirna_std.index[:min(top_n, len(mirna_std))]
        
        # 如果样本太多，随机选择一部分
        if mirna_data.shape[0] > 30:
            mirna_subset = mirna_data.sample(30)
        else:
            mirna_subset = mirna_data
            
        # 提取top miRNAs子集并确保数据干净
        subset_data = mirna_subset[top_mirnas].copy()
        subset_data = subset_data.replace([np.inf, -np.inf], 0)
        subset_data = subset_data.fillna(0)
        
        # 简单热图，避免聚类问题
        plt.figure(figsize=(12, 8))
        sns.heatmap(subset_data, cmap='viridis', yticklabels=False)
        plt.title('Top miRNA表达热图')
        plt.savefig(FIGURES_DIR / 'mirna_heatmap.png', dpi=300)
        print("生成了表达热图")
        
        # 3. PCA分析
        from sklearn.decomposition import PCA
        
        # 确保数据中没有无穷值或NaN
        pca_data = mirna_data.fillna(0).replace([np.inf, -np.inf], 0)
        
        # 只有数据足够时才运行PCA
        if pca_data.shape[0] > 3 and pca_data.shape[1] > 3:
            # 标准化数据以提高PCA质量
            pca_scaled = (pca_data - pca_data.mean()) / (pca_data.std() + 1e-10)
            
            pca = PCA(n_components=2)
            try:
                pca_result = pca.fit_transform(pca_scaled)
                
                plt.figure(figsize=(10, 8))
                plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
                plt.title('miRNA表达PCA分析')
                plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
                plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
                
                # 尝试添加MSI状态颜色标记
                common_samples = list(set(mirna_data.index) & set(clinical_data.index))
                if common_samples:
                    # 创建映射字典，从样本ID到PCA结果索引
                    sample_to_idx = {sample: idx for idx, sample in enumerate(pca_data.index)}
                    
                    # 获取common_samples的PCA结果和MSI状态
                    common_indices = [sample_to_idx[s] for s in common_samples if s in sample_to_idx]
                    pca_subset = pca_result[common_indices]
                    msi_status = clinical_data.loc[common_samples, 'MSI_status'].values
                    
                    # 绘制带MSI信息的PCA图
                    plt.figure(figsize=(10, 8))
                    scatter = plt.scatter(pca_subset[:, 0], pca_subset[:, 1], c=msi_status, 
                                         cmap='viridis', alpha=0.7)
                    plt.colorbar(scatter, label='MSI状态 (0=MSS, 1=MSI-L, 2=MSI-H)')
                    plt.title('miRNA表达PCA分析 (按MSI状态着色)')
                    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
                    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
                
                plt.savefig(FIGURES_DIR / 'mirna_pca.png', dpi=300)
                print("生成了PCA分析图")
            
            except Exception as e:
                print(f"PCA分析失败: {e}")
    
    except Exception as e:
        print(f"可视化过程中出错: {e}")

def main():
    """主函数"""
    start_time = datetime.now()
    print(f"开始处理miRNA数据: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 1. 加载数据
        mirna_data, clinical_data = load_data()
        
        # 2. 预处理miRNA数据
        processed_mirna = preprocess_mirna_data(mirna_data)
        print(f"处理后的miRNA数据: {processed_mirna.shape[0]}个样本 x {processed_mirna.shape[1]}个miRNA")
        
        # 3. 选择高变异miRNA
        variable_mirnas, mirna_std = select_variable_mirnas(processed_mirna, top_n=1000)
        
        # 4. MSI状态与miRNA表达分析
        msi_mirna_avg = analyze_mirna_by_msi(variable_mirnas, clinical_data)
        
        # 5. 可视化miRNA数据
        visualize_mirna_data(variable_mirnas, clinical_data)
        
        # 6. 保存处理后的数据
        variable_mirnas.to_csv(PROCESSED_DIR / "mirna_filtered.csv")
        
        # 保存miRNA变异值
        mirna_std_df = pd.DataFrame({'STD': mirna_std})
        mirna_std_df.to_csv(PROCESSED_DIR / "mirna_std_values.csv")
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds() / 60
        
        print(f"处理后的miRNA数据已保存至: {PROCESSED_DIR / 'mirna_filtered.csv'}")
        print(f"miRNA数据处理完成: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"总处理时间: {processing_time:.2f}分钟")
        
    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()