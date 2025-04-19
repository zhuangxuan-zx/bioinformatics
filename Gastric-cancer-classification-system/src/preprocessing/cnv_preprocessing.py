#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TCGA-STAD胃癌拷贝数变异(CNV)数据预处理脚本
用途：处理GISTIC2拷贝数变异数据，分析基因水平的拷贝数改变模式
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
FIGURES_DIR = PROJECT_ROOT / "results" / "figures" / "cnv"

# 确保输出目录存在
PROCESSED_DIR.mkdir(exist_ok=True, parents=True)
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

def load_data():
    """加载拷贝数变异数据和处理后的临床数据"""
    # 加载GISTIC2拷贝数数据
    cnv_path = RAW_DIR / "cnv" / "GISTIC2_COPYNUMBER_GISTIC2_ALL_THRESHOLDED.BY_GENES"
    cnv_data = pd.read_csv(cnv_path, sep='\t', index_col=0)
    
    # 加载处理后的临床数据
    clinical_path = PROCESSED_DIR / "clinical_processed.csv"
    clinical_data = pd.read_csv(clinical_path, index_col=0)
    
    print(f"成功加载CNV数据: {cnv_data.shape[0]}个基因 x {cnv_data.shape[1]}个样本")
    print(f"成功加载临床数据: {clinical_data.shape[0]}行 x {clinical_data.shape[1]}列")
    
    return cnv_data, clinical_data

def preprocess_cnv_data(cnv_data):
    """预处理CNV数据：转置、检查数据格式等"""
    # 转置数据，使样本为行，基因为列
    cnv_transposed = cnv_data.T
    
    # 标准化样本ID格式
    cnv_transposed.index = cnv_transposed.index.str.upper()
    
    # 检查CNV值的分布
    value_counts = cnv_transposed.values.flatten()
    unique_values = np.unique(value_counts)
    print(f"CNV数据中的唯一值: {unique_values}")
    
    # 检查CNV类型：是离散值(-2,-1,0,1,2)还是连续值
    is_discrete = all(val in [-2, -1, 0, 1, 2] for val in unique_values)
    print(f"CNV数据类型: {'离散' if is_discrete else '连续'}")
    
    # 处理缺失值(如果有)
    na_count = cnv_transposed.isna().sum().sum()
    if na_count > 0:
        print(f"检测到{na_count}个缺失值，使用0填充")
        cnv_transposed = cnv_transposed.fillna(0)
    
    return cnv_transposed

def select_frequently_altered_genes(cnv_data, amp_threshold=0.1, del_threshold=0.1):
    """选择频繁发生拷贝数变异的基因"""
    # 计算每个基因的扩增和缺失频率
    amplification_freq = (cnv_data > 0).mean(axis=0)
    deletion_freq = (cnv_data < 0).mean(axis=0)
    
    # 选择扩增或缺失频率高于阈值的基因
    amp_genes = amplification_freq[amplification_freq >= amp_threshold].index.tolist()
    del_genes = deletion_freq[deletion_freq >= del_threshold].index.tolist()
    
    # 合并频繁改变的基因列表
    altered_genes = list(set(amp_genes + del_genes))
    selected_data = cnv_data[altered_genes]
    
    print(f"扩增频率≥{amp_threshold*100}%的基因: {len(amp_genes)}个")
    print(f"缺失频率≥{del_threshold*100}%的基因: {len(del_genes)}个")
    print(f"共选择了{len(altered_genes)}个频繁改变的基因")
    
    # 创建频率数据框，用于后续分析
    freq_df = pd.DataFrame({
        'amplification_freq': amplification_freq,
        'deletion_freq': deletion_freq
    })
    
    return selected_data, freq_df

def analyze_cnv_by_msi(cnv_data, clinical_data):
    """分析不同MSI状态样本的CNV模式"""
    # 只保留共有的样本
    common_samples = list(set(cnv_data.index) & set(clinical_data.index))
    
    if not common_samples:
        print("警告: CNV数据和临床数据没有共同样本，跳过MSI分析")
        return
    
    print(f"CNV数据和临床数据共有{len(common_samples)}个样本")
    
    cnv_subset = cnv_data.loc[common_samples]
    clinical_subset = clinical_data.loc[common_samples]
    
    # 计算每个样本的CNV负荷(被改变基因的百分比)
    cnv_burden = (cnv_subset != 0).sum(axis=1) / cnv_subset.shape[1]
    
    # 结合MSI状态
    cnv_msi_df = pd.DataFrame({
        'CNV_burden': cnv_burden,
        'MSI_status': clinical_subset['MSI_status']
    })
    
    # 可视化不同MSI状态的CNV负荷
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='MSI_status', y='CNV_burden', data=cnv_msi_df)
    plt.title('MSI状态与CNV负荷关系')
    plt.xlabel('MSI状态 (0=MSS, 1=MSI-L, 2=MSI-H)')
    plt.ylabel('CNV负荷(被改变基因的比例)')
    plt.savefig(FIGURES_DIR / 'msi_vs_cnv_burden.png', dpi=300)
    
    # 计算不同MSI状态的平均CNV模式
    msi_cnv_avg = {}
    for status in sorted(clinical_subset['MSI_status'].unique()):
        status_samples = clinical_subset[clinical_subset['MSI_status'] == status].index
        if len(status_samples) > 0:
            msi_cnv_avg[f'MSI_{status}'] = cnv_subset.loc[status_samples].mean()
    
    return pd.DataFrame(msi_cnv_avg)

def identify_focal_events(cnv_data, window_size=50):
    """识别局部扩增和缺失事件"""
    # 只有在有染色体位置信息的情况下才能执行此功能
    # 这里假设基因名是按染色体位置排序的
    # 实际应用中可能需要使用额外的注释文件获取基因的染色体位置信息
    
    print("注意: 局部事件检测需要基因的染色体位置信息。这里使用滑动窗口近似法。")
    
    # 计算每个窗口的平均CNV值
    focal_events = []
    
    for i in range(0, cnv_data.shape[1] - window_size, window_size // 2):
        window_genes = cnv_data.columns[i:i+window_size]
        window_mean = cnv_data[window_genes].mean(axis=1)
        
        # 识别显著扩增和缺失的窗口
        amp_samples = window_mean[window_mean > 0.5].index.tolist()
        del_samples = window_mean[window_mean < -0.5].index.tolist()
        
        if amp_samples or del_samples:
            event = {
                'genes': window_genes.tolist(),
                'center_gene': window_genes[len(window_genes)//2],
                'amp_samples': len(amp_samples),
                'del_samples': len(del_samples),
                'amp_ratio': len(amp_samples) / cnv_data.shape[0],
                'del_ratio': len(del_samples) / cnv_data.shape[0]
            }
            focal_events.append(event)
    
    # 转换为DataFrame
    if focal_events:
        events_df = pd.DataFrame(focal_events)
        top_events = events_df.sort_values(by=['amp_ratio', 'del_ratio'], ascending=False).head(20)
        print("\n显著的局部CNV事件:")
        print(top_events[['center_gene', 'amp_ratio', 'del_ratio']].head(10))
        return events_df
    else:
        print("未检测到显著的局部CNV事件")
        return pd.DataFrame()

def visualize_cnv_data(cnv_data, clinical_data=None, top_n=100):
    """可视化CNV数据"""
    print("开始生成CNV可视化...")
    
    # 1. CNV值分布
    plt.figure(figsize=(10, 6))
    sns.histplot(cnv_data.values.flatten(), bins=50, kde=True)
    plt.title('CNV值分布')
    plt.xlabel('拷贝数变异值')
    plt.ylabel('频率')
    plt.savefig(FIGURES_DIR / 'cnv_distribution.png', dpi=300)
    
    # 2. 热图 - 显示频繁变异的基因
    # 计算每个基因的变异频率(不管是扩增还是缺失)
    alteration_freq = (cnv_data != 0).mean(axis=0)
    top_genes = alteration_freq.nlargest(top_n).index
    
    # 如果样本太多，随机选择一部分
    if cnv_data.shape[0] > 50:
        cnv_subset = cnv_data.sample(50)
    else:
        cnv_subset = cnv_data
    
    plt.figure(figsize=(15, 10))
    sns.heatmap(
        cnv_subset[top_genes],
        cmap='RdBu_r',
        center=0,
        yticklabels=False,
        xticklabels=False
    )
    plt.title(f'Top {len(top_genes)} 频繁变异基因CNV热图')
    plt.savefig(FIGURES_DIR / 'cnv_heatmap.png', dpi=300)
    
    # 3. 扩增和缺失频率图
    amp_freq = (cnv_data > 0).mean(axis=0)
    del_freq = (cnv_data < 0).mean(axis=0)
    
    # 取最高频率的基因
    top_amp_genes = amp_freq.nlargest(20).index
    top_del_genes = del_freq.nlargest(20).index
    
    # 创建扩增频率图
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_amp_genes, y=amp_freq[top_amp_genes])
    plt.title('Top 20 频繁扩增基因')
    plt.xticks(rotation=90)
    plt.xlabel('基因')
    plt.ylabel('扩增频率')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'top_amplified_genes.png', dpi=300)
    
    # 创建缺失频率图
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_del_genes, y=del_freq[top_del_genes])
    plt.title('Top 20 频繁缺失基因')
    plt.xticks(rotation=90)
    plt.xlabel('基因')
    plt.ylabel('缺失频率')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'top_deleted_genes.png', dpi=300)
    
    # 4. MSI状态与CNV关联(如果有临床数据)
    if clinical_data is not None:
        # 在analyze_cnv_by_msi函数中已经实现
        pass

def main():
    """主函数"""
    start_time = datetime.now()
    print(f"开始处理CNV数据: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 1. 加载数据
        cnv_data, clinical_data = load_data()
        
        # 2. 预处理CNV数据
        processed_cnv = preprocess_cnv_data(cnv_data)
        print(f"处理后的CNV数据: {processed_cnv.shape[0]}个样本 x {processed_cnv.shape[1]}个基因")
        
        # 3. 选择频繁变异的基因
        selected_cnv, freq_df = select_frequently_altered_genes(
            processed_cnv, amp_threshold=0.1, del_threshold=0.1)
        
        # 4. MSI状态与CNV分析
        msi_cnv_avg = analyze_cnv_by_msi(selected_cnv, clinical_data)
        
        # 5. 识别局部事件
        focal_events = identify_focal_events(selected_cnv)
        
        # 6. 可视化CNV数据
        visualize_cnv_data(selected_cnv, clinical_data)
        
        # 7. 保存处理后的数据
        selected_cnv.to_csv(PROCESSED_DIR / "cnv_processed.csv")
        
        # 保存频率数据
        freq_df.to_csv(PROCESSED_DIR / "cnv_frequency.csv")
        
        # 保存MSI相关CNV模式
        if msi_cnv_avg is not None:
            msi_cnv_avg.to_csv(PROCESSED_DIR / "msi_cnv_patterns.csv")
        
        # 保存局部事件
        if not focal_events.empty:
            focal_events.to_csv(PROCESSED_DIR / "cnv_focal_events.csv", index=False)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds() / 60
        
        print(f"处理后的CNV数据已保存至: {PROCESSED_DIR}")
        print(f"CNV数据处理完成: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"总处理时间: {processing_time:.2f}分钟")
        
    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()