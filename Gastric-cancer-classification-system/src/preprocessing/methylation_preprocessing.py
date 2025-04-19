#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TCGA-STAD胃癌DNA甲基化数据预处理脚本 - 快速版本
使用采样和简化处理步骤大幅提高处理速度
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
FIGURES_DIR = PROJECT_ROOT / "results" / "figures" / "methylation"

# 确保输出目录存在
PROCESSED_DIR.mkdir(exist_ok=True, parents=True)
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

# 控制采样比例的参数 - 可以调整以平衡速度和结果质量
PROBE_SAMPLE_RATIO = 0.05  # 只处理5%的探针
MAX_PROBES = 20000         # 最多处理的探针数量
TOP_VARIABLE_PROBES = 5000  # 选择的高变异探针数量

def load_data():
    """加载甲基化数据和探针注释，使用采样优化内存使用"""
    # 加载甲基化探针注释
    probe_map_path = RAW_DIR / "methylation" / "probeMap_illuminaMethyl450_hg19_GPL16304_TCGAlegacy"
    
    # 只读取必要的列，节省内存
    try:
        probe_map = pd.read_csv(probe_map_path, sep='\t', usecols=[0, 1, 2])
        probe_cols = probe_map.columns.tolist()
        id_col = probe_cols[0]
        chr_col = next((col for col in probe_cols if 'chr' in col.lower()), probe_cols[1])
        gene_col = next((col for col in probe_cols if 'gene' in col.lower()), probe_cols[-1])
    except Exception as e:
        print(f"读取探针注释文件出错，尝试不指定列名: {e}")
        probe_map = pd.read_csv(probe_map_path, sep='\t')
        probe_cols = probe_map.columns.tolist()
        id_col = probe_cols[0]
        chr_col = probe_cols[1] if len(probe_cols) > 1 else None
        gene_col = probe_cols[-1] if len(probe_cols) > 2 else None
    
    print(f"探针注释文件列名: {probe_map.columns.tolist()}")
    print(f"使用 '{id_col}' 作为ID列，'{chr_col}' 作为染色体列，'{gene_col}' 作为基因列")
    
    # 加载甲基化β值数据 - 使用采样加速
    methyl_path = RAW_DIR / "methylation" / "HumanMethylation450"
    
    # 首先只读取前几行获取列名
    methyl_sample = pd.read_csv(methyl_path, sep='\t', nrows=5)
    all_samples = methyl_sample.columns[1:].tolist()
    print(f"甲基化数据包含 {len(all_samples)} 个样本")
    
    # 读取行名(探针ID)
    probe_ids = pd.read_csv(methyl_path, sep='\t', usecols=[0])
    
    # 随机采样探针
    probe_count = len(probe_ids)
    sample_size = min(int(probe_count * PROBE_SAMPLE_RATIO), MAX_PROBES)
    sampled_indices = np.sort(np.random.choice(probe_count, size=sample_size, replace=False))
    
    print(f"从 {probe_count} 个探针中随机采样 {sample_size} 个进行处理")
    
    # 使用chunk处理，只选取采样的行
    chunks = []
    chunk_size = 10000
    for chunk in pd.read_csv(methyl_path, sep='\t', chunksize=chunk_size):
        chunk_indices = chunk.index.intersection(sampled_indices)
        if len(chunk_indices) > 0:
            chunks.append(chunk.loc[chunk_indices])
    
    methyl_data = pd.concat(chunks) if chunks else pd.DataFrame()
    
    # 将探针ID作为索引
    if not methyl_data.empty:
        methyl_data.set_index(methyl_data.columns[0], inplace=True)
    
    # 加载处理后的临床数据
    clinical_path = PROCESSED_DIR / "clinical_processed.csv"
    clinical_data = pd.read_csv(clinical_path, index_col=0)
    
    print(f"成功加载甲基化数据: {methyl_data.shape[0]}个探针 x {methyl_data.shape[1]}个样本")
    print(f"成功加载临床数据: {clinical_data.shape[0]}行 x {clinical_data.shape[1]}列")
    
    return methyl_data, probe_map, clinical_data, id_col, chr_col, gene_col

def process_methylation_data(methyl_data):
    """快速处理甲基化数据"""
    # 转置数据，使样本为行，探针为列
    print("转置甲基化数据矩阵...")
    methyl_transposed = methyl_data.T
    
    # 处理缺失值 - 使用中位数填充，比均值计算快
    missing_values = methyl_transposed.isnull().sum().sum()
    if missing_values > 0:
        print(f"检测到{missing_values}个缺失值，使用中位数填充")
        # 列方向的中位数填充
        methyl_transposed = methyl_transposed.fillna(methyl_transposed.median())
    
    print(f"处理后的矩阵: {methyl_transposed.shape[0]}个样本 × {methyl_transposed.shape[1]}个探针")
    return methyl_transposed

def select_variable_probes(methyl_data, n_probes=TOP_VARIABLE_PROBES):
    """快速选择高变异的甲基化探针"""
    print(f"选择{n_probes}个高变异探针...")
    
    # 使用方差而不是MAD或标准差，计算更快
    variance = methyl_data.var(axis=0)
    top_probes = variance.nlargest(n_probes).index.tolist()
    
    return methyl_data[top_probes]

def analyze_methylation_by_msi(methyl_data, clinical_data):
    """根据MSI状态分析全局甲基化水平"""
    print("分析MSI状态与甲基化关系...")
    
    # 只保留共有的样本
    common_samples = list(set(methyl_data.index) & set(clinical_data.index))
    if not common_samples:
        print("警告: 没有共同的样本，跳过MSI分析")
        return
        
    methyl_subset = methyl_data.loc[common_samples]
    clinical_subset = clinical_data.loc[common_samples]
    
    # 计算每个样本的平均甲基化水平
    mean_methylation = methyl_subset.mean(axis=1)
    
    # 分析不同MSI状态的甲基化
    try:
        msi_data = pd.DataFrame({
            'MSI_status': clinical_subset['MSI_status'],
            'mean_methylation': mean_methylation
        })
        
        # 去除缺失值
        msi_data = msi_data.dropna()
        
        if len(msi_data) > 0:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='MSI_status', y='mean_methylation', data=msi_data)
            plt.title('MSI状态与全局甲基化水平关系')
            plt.xlabel('MSI状态 (0=MSS, 1=MSI-L, 2=MSI-H)')
            plt.ylabel('平均甲基化水平(β值)')
            plt.savefig(FIGURES_DIR / 'msi_vs_global_methylation.png', dpi=300)
            print("生成MSI vs 甲基化分析图")
    except Exception as e:
        print(f"MSI分析出错: {e}")

def visualize_methylation_data(methyl_data):
    """简化的甲基化数据可视化"""
    print("生成可视化图表...")
    try:
        # 1. 全局甲基化分布直方图 - 只使用部分样本加速
        sample_values = methyl_data.iloc[:5, :100].values.flatten()
        plt.figure(figsize=(10, 6))
        sns.histplot(sample_values, bins=50, kde=True)
        plt.title('甲基化水平分布样本')
        plt.xlabel('甲基化水平(β值)')
        plt.ylabel('频率')
        plt.savefig(FIGURES_DIR / 'methylation_distribution.png', dpi=300)
        
        # 2. 简单热图 - 取较小子集避免内存问题
        if methyl_data.shape[0] > 20:
            methyl_subset = methyl_data.sample(min(20, methyl_data.shape[0]))
        else:
            methyl_subset = methyl_data
            
        if methyl_subset.shape[1] > 50:
            methyl_subset = methyl_subset.iloc[:, :50]
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(methyl_subset, cmap='coolwarm', yticklabels=False)
        plt.title('甲基化水平热图(样本)')
        plt.savefig(FIGURES_DIR / 'methylation_heatmap_sample.png', dpi=300)
    except Exception as e:
        print(f"可视化过程出错: {e}")

def main():
    """主函数 - 优化流程"""
    start_time = datetime.now()
    print(f"开始处理甲基化数据(快速版): {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 1. 加载数据(使用采样)
        methyl_data, probe_map, clinical_data, id_col, chr_col, gene_col = load_data()
        
        # 2. 处理甲基化数据
        processed_methyl = process_methylation_data(methyl_data)
        
        # 3. 选择高变异探针
        variable_methyl = select_variable_probes(processed_methyl)
        
        # 4. MSI状态分析
        analyze_methylation_by_msi(variable_methyl, clinical_data)
        
        # 5. 简化可视化
        visualize_methylation_data(variable_methyl)
        
        # 6. 保存处理后的数据
        print("保存处理后的数据...")
        variable_methyl.to_csv(PROCESSED_DIR / "methylation_filtered.csv")
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds() / 60
        
        print(f"处理后的甲基化数据已保存至: {PROCESSED_DIR / 'methylation_filtered.csv'}")
        print(f"甲基化数据处理完成: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"总处理时间: {processing_time:.2f}分钟")
        
    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()