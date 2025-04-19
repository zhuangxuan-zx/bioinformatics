#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TCGA-STAD胃癌基因表达数据预处理脚本
用途：处理RNA-seq表达数据，进行标准化、缺失值处理、批次效应校正和特征选择
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
FIGURES_DIR = PROJECT_ROOT / "results" / "figures" / "expression"

# 确保输出目录存在
PROCESSED_DIR.mkdir(exist_ok=True, parents=True)
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

def load_data():
    """加载基因表达数据和处理后的临床数据"""
    # 加载基因表达数据
    expr_path = RAW_DIR / "expression" / "HiSeqV2"
    expr_data = pd.read_csv(expr_path, sep='\t', index_col=0)
    
    # 加载处理后的临床数据
    clinical_path = PROCESSED_DIR / "clinical_processed.csv"
    clinical_data = pd.read_csv(clinical_path, index_col=0)
    
    print(f"成功加载基因表达数据: {expr_data.shape[0]}个基因 x {expr_data.shape[1]}个样本")
    print(f"成功加载临床数据: {clinical_data.shape[0]}行 x {clinical_data.shape[1]}列")
    
    return expr_data, clinical_data

def preprocess_expression_data(expr_data):
    """预处理基因表达数据：标准化、转置等"""
    # 转置数据，使样本为行，基因为列
    expr_transposed = expr_data.T
    
    # 标准化样本ID格式
    expr_transposed.index = expr_transposed.index.str.upper()
    
    # 处理缺失值
    na_count = expr_transposed.isna().sum().sum()
    if na_count > 0:
        print(f"检测到{na_count}个缺失值，使用最小值填充")
        min_val = expr_transposed.min().min() / 2  # 用最小值的一半填充缺失值
        expr_transposed = expr_transposed.fillna(min_val)
    
    # 检查表达值的分布
    min_expr = expr_transposed.values.min()
    max_expr = expr_transposed.values.max()
    print(f"基因表达值范围: {min_expr:.4f} - {max_expr:.4f}")
    
    # 确定是否需要log变换
    # 如果数据已经是log转换过的，可以跳过这一步
    if max_expr > 100 and max_expr / expr_transposed.median().median() > 100:
        print("数据可能未log转换，执行log2(x+1)变换")
        expr_transposed = np.log2(expr_transposed + 1)
    
    return expr_transposed

def filter_low_expression_genes(expr_data, min_samples=0.2, min_expression=0):
    """过滤低表达基因：去除在大多数样本中表达量较低的基因"""
    # 计算每个基因在多少样本中的表达量超过阈值
    samples_count = expr_data.shape[0]
    min_samples_count = int(samples_count * min_samples)
    
    # 计算每个基因表达超过阈值的样本数
    genes_pass = (expr_data > min_expression).sum(axis=0)
    
    # 过滤基因
    filtered_genes = genes_pass[genes_pass >= min_samples_count].index
    filtered_data = expr_data[filtered_genes]
    
    print(f"初始基因数: {expr_data.shape[1]}")
    print(f"过滤后的基因数: {filtered_data.shape[1]} (去除在少于{min_samples_count}个样本中表达的基因)")
    
    return filtered_data

def select_variable_genes(expr_data, top_n=2000):
    """选择表达变异性最大的基因"""
    # 计算每个基因的标准差
    gene_std = expr_data.std(axis=0)
    
    # 选择标准差最高的top_n个基因
    top_genes = gene_std.nlargest(top_n).index
    selected_data = expr_data[top_genes]
    
    print(f"选择了{top_n}个高变异基因用于后续分析")
    
    return selected_data, gene_std

def analyze_expression_by_msi(expr_data, clinical_data):
    """分析不同MSI状态样本的基因表达模式"""
    # 只保留共有的样本
    common_samples = list(set(expr_data.index) & set(clinical_data.index))
    
    if not common_samples:
        print("警告: 表达数据和临床数据没有共同样本，跳过MSI分析")
        return pd.DataFrame()
    
    print(f"表达数据和临床数据共有{len(common_samples)}个样本")
    
    expr_subset = expr_data.loc[common_samples]
    clinical_subset = clinical_data.loc[common_samples]
    
    # 计算不同MSI状态的平均表达
    msi_expr_avg = {}
    for status in sorted(clinical_subset['MSI_status'].unique()):
        status_samples = clinical_subset[clinical_subset['MSI_status'] == status].index
        if len(status_samples) > 0:
            msi_expr_avg[f'MSI_{status}'] = expr_subset.loc[status_samples].mean()
    
    msi_avg_df = pd.DataFrame(msi_expr_avg)
    
    # 可视化不同MSI状态的全局表达
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='MSI_status', y='mean_expression', 
                data=pd.DataFrame({
                    'MSI_status': clinical_subset['MSI_status'],
                    'mean_expression': expr_subset.mean(axis=1)
                }))
    plt.title('MSI状态与全局基因表达关系')
    plt.xlabel('MSI状态 (0=MSS, 1=MSI-L, 2=MSI-H)')
    plt.ylabel('平均基因表达')
    plt.savefig(FIGURES_DIR / 'msi_vs_global_expression.png', dpi=300)
    
    # 寻找MSI状态差异表达基因
    try:
        from scipy import stats
        
        # 简单处理：MSS vs MSI-H比较
        mss_samples = clinical_subset[clinical_subset['MSI_status'] == 0].index
        msih_samples = clinical_subset[clinical_subset['MSI_status'] == 2].index
        
        if len(mss_samples) > 0 and len(msih_samples) > 0:
            print(f"进行MSS({len(mss_samples)}样本) vs MSI-H({len(msih_samples)}样本)差异表达分析")
            
            # 计算每个基因的t-test
            pvals = []
            fold_changes = []
            gene_names = []
            
            # 只分析前2000个高变异基因，提高效率
            top_var_genes = expr_subset.var(axis=0).nlargest(2000).index
            
            for gene in top_var_genes:
                mss_expr = expr_subset.loc[mss_samples, gene]
                msih_expr = expr_subset.loc[msih_samples, gene]
                
                t_stat, p_val = stats.ttest_ind(mss_expr, msih_expr, equal_var=False)
                fold_change = msih_expr.mean() - mss_expr.mean()  # log值的差相当于log fold change
                
                pvals.append(p_val)
                fold_changes.append(fold_change)
                gene_names.append(gene)
            
            # 创建差异表达结果数据框
            de_results = pd.DataFrame({
                'gene': gene_names,
                'log_fold_change': fold_changes,
                'p_value': pvals
            })
            
            # 应用多重检验校正
            de_results['adjusted_p'] = stats.false_discovery_rate_correction(de_results['p_value'])[1]
            
            # 过滤显著差异表达基因
            sig_de = de_results[(de_results['adjusted_p'] < 0.05) & (abs(de_results['log_fold_change']) > 1)]
            print(f"识别出{len(sig_de)}个MSI-H与MSS之间的差异表达基因")
            
            # 保存差异表达结果
            de_results.sort_values('adjusted_p').to_csv(
                PROCESSED_DIR / "mss_vs_msih_de_genes.csv", index=False)
            
            # 可视化：火山图
            plt.figure(figsize=(10, 8))
            plt.scatter(
                de_results['log_fold_change'], 
                -np.log10(de_results['p_value']),
                alpha=0.5
            )
            
            # 标记显著上调和下调基因
            up_genes = de_results[(de_results['adjusted_p'] < 0.05) & (de_results['log_fold_change'] > 1)]
            down_genes = de_results[(de_results['adjusted_p'] < 0.05) & (de_results['log_fold_change'] < -1)]
            
            plt.scatter(
                up_genes['log_fold_change'], 
                -np.log10(up_genes['p_value']),
                color='red',
                alpha=0.7
            )
            plt.scatter(
                down_genes['log_fold_change'], 
                -np.log10(down_genes['p_value']),
                color='blue',
                alpha=0.7
            )
            
            plt.xlabel('Log2 Fold Change (MSI-H vs MSS)')
            plt.ylabel('-Log10 P-value')
            plt.title('MSI-H vs MSS差异表达火山图')
            plt.axhline(-np.log10(0.05), color='gray', linestyle='--')
            plt.axvline(1, color='gray', linestyle='--')
            plt.axvline(-1, color='gray', linestyle='--')
            plt.savefig(FIGURES_DIR / 'msi_vs_mss_volcano.png', dpi=300)
    
    except Exception as e:
        print(f"差异表达分析出错: {e}")
    
    return msi_avg_df

def visualize_expression_data(expr_data, clinical_data, top_n=50):
    """可视化基因表达数据"""
    print("开始生成基因表达可视化...")
    
    # 1. 全局表达分布
    plt.figure(figsize=(10, 6))
    # 采样数据点以避免过大的图表
    sample_values = expr_data.values.flatten()
    if len(sample_values) > 100000:
        sample_values = np.random.choice(sample_values, 100000, replace=False)
    sns.histplot(sample_values, bins=50, kde=True)
    plt.title('基因表达分布')
    plt.xlabel('表达值')
    plt.ylabel('频率')
    plt.savefig(FIGURES_DIR / 'expression_distribution.png', dpi=300)
    
    # 2. 热图 - 显示高变异基因
    gene_var = expr_data.var(axis=0).sort_values(ascending=False)
    top_genes = gene_var.index[:min(top_n, len(gene_var))]
    
    # 随机选择一部分样本，避免图表过大
    if expr_data.shape[0] > 50:
        sample_subset = expr_data.sample(50)
    else:
        sample_subset = expr_data
    
    plt.figure(figsize=(15, 10))
    sns.heatmap(
        sample_subset[top_genes], 
        cmap='viridis', 
        yticklabels=False, 
        xticklabels=False
    )
    plt.title(f'Top {len(top_genes)} 高变异基因表达热图')
    plt.savefig(FIGURES_DIR / 'expression_heatmap.png', dpi=300)
    
    # 3. PCA分析
    from sklearn.decomposition import PCA
    
    try:
        # 确保数据不包含无穷值或NaN
        pca_data = expr_data.fillna(0).replace([np.inf, -np.inf], 0)
        
        # 降维前标准化数据
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(pca_data)
        
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
        plt.title('基因表达PCA分析')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        
        # 结合MSI状态
        common_samples = list(set(expr_data.index) & set(clinical_data.index))
        if common_samples:
            # 创建样本ID到PCA结果索引的映射
            sample_to_idx = {sample: idx for idx, sample in enumerate(pca_data.index)}
            
            # 获取共同样本的PCA结果和MSI状态
            common_indices = [sample_to_idx[s] for s in common_samples if s in sample_to_idx]
            pca_subset = pca_result[common_indices]
            msi_status = clinical_data.loc[common_samples, 'MSI_status'].values
            
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(pca_subset[:, 0], pca_subset[:, 1], c=msi_status, cmap='viridis', alpha=0.7)
            plt.title('基因表达PCA分析 (按MSI状态着色)')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
            plt.colorbar(scatter, label='MSI状态 (0=MSS, 1=MSI-L, 2=MSI-H)')
        
        plt.savefig(FIGURES_DIR / 'expression_pca.png', dpi=300)
        print("生成了PCA分析图")
        
    except Exception as e:
        print(f"PCA分析出错: {e}")

def main():
    """主函数"""
    start_time = datetime.now()
    print(f"开始处理基因表达数据: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 1. 加载数据
        expr_data, clinical_data = load_data()
        
        # 2. 预处理表达数据
        processed_expr = preprocess_expression_data(expr_data)
        print(f"处理后的表达数据: {processed_expr.shape[0]}个样本 x {processed_expr.shape[1]}个基因")
        
        # 3. 过滤低表达基因
        filtered_expr = filter_low_expression_genes(processed_expr)
        
        # 4. 选择高变异基因
        variable_genes, gene_std = select_variable_genes(filtered_expr, top_n=2000)
        
        # 5. MSI状态与表达分析
        msi_expr_avg = analyze_expression_by_msi(variable_genes, clinical_data)
        
        # 6. 可视化表达数据
        visualize_expression_data(variable_genes, clinical_data)
        
        # 7. 保存处理后的数据
        variable_genes.to_csv(PROCESSED_DIR / "expression_filtered.csv")
        
        # 保存基因变异度
        gene_std_df = pd.DataFrame({'STD': gene_std})
        gene_std_df.to_csv(PROCESSED_DIR / "gene_std_values.csv")
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds() / 60
        
        print(f"处理后的基因表达数据已保存至: {PROCESSED_DIR / 'expression_filtered.csv'}")
        print(f"基因表达数据处理完成: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"总处理时间: {processing_time:.2f}分钟")
        
    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()