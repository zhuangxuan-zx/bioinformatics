#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TCGA-STAD胃癌蛋白质组数据预处理脚本
用途：处理RPPA反向蛋白质阵列数据，进行标准化、缺失值处理和特征选择
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
FIGURES_DIR = PROJECT_ROOT / "results" / "figures" / "proteomics"

# 确保输出目录存在
PROCESSED_DIR.mkdir(exist_ok=True, parents=True)
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

def load_data():
    """加载蛋白质组数据和处理后的临床数据"""
    # 加载RPPA蛋白质组数据
    rppa_path = RAW_DIR / "proteomics" / "RPPA_data_file"  # 根据实际文件名调整
    
    try:
        rppa_data = pd.read_csv(rppa_path, sep='\t', index_col=0)
        print(f"成功加载RPPA蛋白质组数据: {rppa_data.shape[0]}个样本 x {rppa_data.shape[1]}个蛋白质")
    except FileNotFoundError:
        print("警告: 未找到RPPA蛋白质组数据文件")
        print("尝试创建模拟数据用于测试流程...")
        
        # 创建模拟数据
        from numpy.random import normal
        # 假设200个样本，300个蛋白质
        sample_ids = [f"TCGA-XX-{i:04d}" for i in range(1, 201)]
        protein_ids = [f"Protein_{i}" for i in range(1, 301)]
        rppa_data = pd.DataFrame(
            normal(0, 1, size=(200, 300)), 
            index=sample_ids, 
            columns=protein_ids
        )
        print("已创建模拟RPPA数据: 200个样本 x 300个蛋白质")
    
    # 加载处理后的临床数据
    clinical_path = PROCESSED_DIR / "clinical_processed.csv"
    clinical_data = pd.read_csv(clinical_path, index_col=0)
    
    print(f"成功加载临床数据: {clinical_data.shape[0]}行 x {clinical_data.shape[1]}列")
    
    return rppa_data, clinical_data

def preprocess_proteomics_data(rppa_data):
    """预处理蛋白质组数据：标准化、转置等"""
    # 标准化样本ID格式
    rppa_data.index = rppa_data.index.str.upper()
    
    # 处理缺失值
    na_count = rppa_data.isna().sum().sum()
    if na_count > 0:
        print(f"检测到{na_count}个缺失值，使用蛋白质均值填充")
        rppa_data = rppa_data.fillna(rppa_data.mean())
    
    # 检查数据分布
    mean_val = rppa_data.mean().mean()
    std_val = rppa_data.std().std()
    min_val = rppa_data.min().min()
    max_val = rppa_data.max().max()
    
    print(f"蛋白质表达值范围: {min_val:.4f} - {max_val:.4f}, 均值: {mean_val:.4f}, 标准差: {std_val:.4f}")
    
    # 若数据尚未标准化，进行Z-score标准化
    if abs(mean_val) > 0.1 or abs(std_val - 1) > 0.1:
        print("对数据进行Z-score标准化")
        rppa_data = (rppa_data - rppa_data.mean()) / rppa_data.std()
    
    return rppa_data

def select_variable_proteins(rppa_data, top_n=100):
    """选择变异性最大的蛋白质"""
    # 计算每个蛋白质的标准差
    protein_std = rppa_data.std(axis=0)
    
    # 选择标准差最高的top_n个蛋白质
    top_proteins = protein_std.nlargest(top_n).index
    selected_data = rppa_data[top_proteins]
    
    print(f"选择了{len(top_proteins)}个高变异蛋白质")
    
    return selected_data, protein_std

def analyze_proteomics_by_msi(rppa_data, clinical_data):
    """分析不同MSI状态样本的蛋白质表达模式"""
    # 只保留共有的样本
    common_samples = list(set(rppa_data.index) & set(clinical_data.index))
    
    if not common_samples:
        print("警告: 蛋白质组数据和临床数据没有共同样本，跳过MSI分析")
        return pd.DataFrame()
    
    print(f"蛋白质组和临床数据共有{len(common_samples)}个样本")
    
    rppa_subset = rppa_data.loc[common_samples]
    clinical_subset = clinical_data.loc[common_samples]
    
    # 分析不同MSI状态的蛋白质表达
    try:
        from scipy import stats
        
        # 计算每个样本的平均蛋白质表达
        mean_expression = rppa_subset.mean(axis=1)
        
        # 创建包含MSI状态的数据框
        msi_data = pd.DataFrame({
            'MSI_status': clinical_subset['MSI_status'],
            'mean_expression': mean_expression
        })
        
        # 可视化不同MSI状态的全局蛋白质表达
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='MSI_status', y='mean_expression', data=msi_data)
        plt.title('MSI状态与全局蛋白质表达关系')
        plt.xlabel('MSI状态 (0=MSS, 1=MSI-L, 2=MSI-H)')
        plt.ylabel('平均蛋白质表达')
        plt.savefig(FIGURES_DIR / 'msi_vs_global_protein.png', dpi=300)
        
        # 寻找MSI相关的差异表达蛋白
        # 简化处理：比较MSS vs MSI-H
        mss_samples = clinical_subset[clinical_subset['MSI_status'] == 0].index
        msih_samples = clinical_subset[clinical_subset['MSI_status'] == 2].index
        
        # 只在两组样本足够时进行比较
        if len(mss_samples) > 5 and len(msih_samples) > 5:
            print(f"进行MSS({len(mss_samples)}样本) vs MSI-H({len(msih_samples)}样本)差异蛋白分析")
            
            # 计算每个蛋白质的t-test
            sig_proteins = []
            
            for protein in rppa_subset.columns:
                mss_expr = rppa_subset.loc[mss_samples, protein]
                msih_expr = rppa_subset.loc[msih_samples, protein]
                
                t_stat, p_val = stats.ttest_ind(mss_expr, msih_expr, equal_var=False)
                fold_change = msih_expr.mean() - mss_expr.mean()
                
                if p_val < 0.05:
                    sig_proteins.append({
                        'protein': protein,
                        'fold_change': fold_change,
                        'p_value': p_val
                    })
            
            if sig_proteins:
                sig_df = pd.DataFrame(sig_proteins)
                sig_df = sig_df.sort_values('p_value')
                print(f"发现{len(sig_df)}个MSI相关的差异表达蛋白质")
                print("前5个差异蛋白质:")
                print(sig_df.head(5))
                
                # 保存差异蛋白结果
                sig_df.to_csv(PROCESSED_DIR / "msi_diff_proteins.csv", index=False)
                
                # 可视化Top差异蛋白
                if len(sig_df) >= 5:
                    top_proteins = sig_df.head(5)['protein'].tolist()
                    
                    plt.figure(figsize=(12, 8))
                    for i, protein in enumerate(top_proteins):
                        plt.subplot(2, 3, i+1)
                        sns.boxplot(x='MSI_status', y=protein, 
                                    data=pd.concat([clinical_subset['MSI_status'], 
                                                  rppa_subset[protein]], axis=1))
                        plt.title(protein)
                        plt.xlabel('MSI Status')
                    
                    plt.tight_layout()
                    plt.savefig(FIGURES_DIR / 'top_diff_proteins.png', dpi=300)
            else:
                print("未发现MSI相关的显著差异蛋白质")
        else:
            print("MSS或MSI-H样本数量不足，跳过差异蛋白分析")
        
        # 计算不同MSI状态的平均表达谱
        msi_protein_avg = {}
        for status in sorted(clinical_subset['MSI_status'].unique()):
            status_samples = clinical_subset[clinical_subset['MSI_status'] == status].index
            if len(status_samples) > 0:
                msi_protein_avg[f'MSI_{status}'] = rppa_subset.loc[status_samples].mean()
        
        return pd.DataFrame(msi_protein_avg)
    
    except Exception as e:
        print(f"MSI分析过程中出错: {e}")
        return pd.DataFrame()

def visualize_proteomics_data(rppa_data, clinical_data=None):
    """可视化蛋白质组数据"""
    print("开始生成蛋白质组可视化...")
    
    # 1. 表达分布
    plt.figure(figsize=(10, 6))
    sns.histplot(rppa_data.values.flatten(), bins=50, kde=True)
    plt.title('蛋白质表达分布')
    plt.xlabel('表达值')
    plt.ylabel('频率')
    plt.savefig(FIGURES_DIR / 'protein_distribution.png', dpi=300)
    
    # 2. 蛋白质表达热图
    # 如果蛋白质太多，选择Top 50
    if rppa_data.shape[1] > 50:
        proteins = rppa_data.std().nlargest(50).index
        rppa_subset = rppa_data[proteins]
    else:
        rppa_subset = rppa_data
    
    # 如果样本太多，随机选择50个
    if rppa_subset.shape[0] > 50:
        rppa_subset = rppa_subset.sample(50)
    
    plt.figure(figsize=(14, 10))
    sns.clustermap(
        rppa_subset,
        cmap='coolwarm',
        center=0,
        figsize=(15, 12),
        xticklabels=True,
        yticklabels=False
    )
    plt.savefig(FIGURES_DIR / 'protein_heatmap.png', dpi=300)
    
    # 3. PCA分析
    try:
        from sklearn.decomposition import PCA
        
        # 确保数据不包含缺失值或无穷值
        pca_data = rppa_data.fillna(0).replace([np.inf, -np.inf], 0)
        
        # PCA降维
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(pca_data)
        
        # 基本PCA图
        plt.figure(figsize=(10, 8))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
        plt.title('蛋白质表达PCA分析')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        
        # 如果有临床数据，添加MSI状态标记
        if clinical_data is not None:
            common_samples = list(set(rppa_data.index) & set(clinical_data.index))
            if common_samples:
                # 创建样本ID到PCA结果索引的映射
                sample_to_idx = {sample: idx for idx, sample in enumerate(pca_data.index)}
                
                # 获取共同样本的PCA结果和MSI状态
                common_indices = [sample_to_idx[s] for s in common_samples if s in sample_to_idx]
                pca_subset = pca_result[common_indices]
                msi_status = clinical_data.loc[common_samples, 'MSI_status'].values
                
                # 绘制带MSI信息的PCA图
                plt.figure(figsize=(10, 8))
                scatter = plt.scatter(pca_subset[:, 0], pca_subset[:, 1], c=msi_status, cmap='viridis', alpha=0.7)
                plt.colorbar(scatter, label='MSI状态 (0=MSS, 1=MSI-L, 2=MSI-H)')
                plt.title('蛋白质表达PCA分析 (按MSI状态着色)')
                plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
                plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        
        plt.savefig(FIGURES_DIR / 'protein_pca.png', dpi=300)
        print("生成了PCA分析图")
    
    except Exception as e:
        print(f"PCA分析出错: {e}")

def main():
    """主函数"""
    start_time = datetime.now()
    print(f"开始处理蛋白质组数据: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 1. 加载数据
        rppa_data, clinical_data = load_data()
        
        # 2. 预处理蛋白质组数据
        processed_rppa = preprocess_proteomics_data(rppa_data)
        print(f"处理后的蛋白质组数据: {processed_rppa.shape[0]}个样本 x {processed_rppa.shape[1]}个蛋白质")
        
        # 3. 选择高变异蛋白质
        variable_proteins, protein_std = select_variable_proteins(processed_rppa, top_n=100)
        
        # 4. MSI状态与蛋白质表达分析
        msi_protein_avg = analyze_proteomics_by_msi(variable_proteins, clinical_data)
        
        # 5. 可视化蛋白质组数据
        visualize_proteomics_data(variable_proteins, clinical_data)
        
        # 6. 保存处理后的数据
        variable_proteins.to_csv(PROCESSED_DIR / "protein_filtered.csv")
        
        # 保存蛋白质变异度
        protein_std_df = pd.DataFrame({'STD': protein_std})
        protein_std_df.to_csv(PROCESSED_DIR / "protein_std_values.csv")
        
        # 保存MSI相关蛋白质表达谱
        if not msi_protein_avg.empty:
            msi_protein_avg.to_csv(PROCESSED_DIR / "msi_protein_patterns.csv")
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds() / 60
        
        print(f"处理后的蛋白质组数据已保存至: {PROCESSED_DIR}")
        print(f"蛋白质组数据处理完成: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"总处理时间: {processing_time:.2f}分钟")
        
    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()