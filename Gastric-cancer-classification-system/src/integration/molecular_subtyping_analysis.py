#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
胃癌分子亚型分析脚本
用途：分析整合聚类得到的分子亚型特征、与临床相关性及生物学意义
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import matplotlib.gridspec as gridspec
from scipy import stats
import argparse
import warnings
warnings.filterwarnings('ignore')

# 设置项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
INTEGRATION_DIR = RESULTS_DIR / "integration"
SUBTYPE_DIR = RESULTS_DIR / "molecular_subtypes"
FIGURES_DIR = SUBTYPE_DIR / "figures"

# 确保输出目录存在
SUBTYPE_DIR.mkdir(exist_ok=True, parents=True)
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

def load_integrated_clusters(custom_file=None):
    """加载整合聚类结果，支持自定义文件路径"""
    # 优先使用自定义文件路径
    if custom_file:
        clusters_path = Path(custom_file)
    else:
        clusters_path = INTEGRATION_DIR / "integrated_clusters.csv"
        
    if clusters_path.exists():
        try:
            clusters = pd.read_csv(clusters_path, index_col=0)
            
            # 确保clusters是DataFrame格式
            if isinstance(clusters, pd.Series):
                clusters = pd.DataFrame(clusters)
                
            print(f"加载亚型分类结果: {len(clusters)}个样本, {clusters.iloc[:, 0].nunique()}个亚型")
            return clusters
        except Exception as e:
            print(f"加载亚型分类文件时出错: {e}")
            return None
    else:
        print(f"错误: 未找到亚型分类文件 {clusters_path}")
        return None

def load_omics_and_clinical_data():
    """加载各组学数据和临床数据"""
    omics_data = {}
    
    # 加载各组学数据，优先加载经过特征选择的文件
    omics_types = ["expression", "methylation", "mirna", "cnv", "mutation"]
    for omics_type in omics_types:
        # 按优先级尝试加载不同处理阶段的文件
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
                    print(f"加载{omics_type}数据: {data.shape[0]}个样本 x {data.shape[1]}个特征")
                    break
                except Exception as e:
                    print(f"加载{omics_type}数据时出错: {e}")
    
    # 加载临床数据
    clinical_path = PROCESSED_DIR / "clinical_processed.csv"
    clinical_data = None
    if clinical_path.exists():
        try:
            clinical_data = pd.read_csv(clinical_path, index_col=0)
            print(f"加载临床数据: {clinical_data.shape[0]}个样本 x {clinical_data.shape[1]}列")
        except Exception as e:
            print(f"加载临床数据时出错: {e}")
    
    return omics_data, clinical_data

def analyze_clinical_characteristics(integrated_clusters, clinical_data):
    """分析分子亚型与临床特征的关联"""
    if clinical_data is None:
        print("未找到临床数据，跳过临床特征分析")
        return None
    
    # 合并聚类结果与临床数据
    common_samples = list(set(integrated_clusters.index) & set(clinical_data.index))
    
    if not common_samples:
        print("整合聚类结果与临床数据没有共同样本")
        return None
    
    # 确保integrated_clusters是包含一列的DataFrame
    if isinstance(integrated_clusters, pd.Series):
        integrated_clusters = pd.DataFrame(integrated_clusters)
    
    merged_data = pd.DataFrame({'Subtype': integrated_clusters.loc[common_samples].iloc[:, 0]})
    merged_data = pd.concat([merged_data, clinical_data.loc[common_samples]], axis=1)
    
    # 打印子类型分布情况
    print("\n亚型样本分布:")
    subtype_counts = merged_data['Subtype'].value_counts().sort_index()
    for subtype, count in subtype_counts.items():
        print(f"  亚型 {subtype}: {count}个样本")
    
    # 保存临床特征分析数据
    merged_data.to_csv(SUBTYPE_DIR / "subtypes_with_clinical.csv")
    
    # 尝试转换临床特征为数值型
    clinical_features = ['MSI_status', 'stage_code', 'vital_status_code', 'gender_code', 'age_at_initial_pathologic_diagnosis']
    for feature in clinical_features:
        if feature in merged_data.columns:
            try:
                merged_data[feature] = pd.to_numeric(merged_data[feature], errors='coerce')
            except:
                print(f"无法将{feature}转换为数值型")
    
    try:
        # 分析关键临床特征
        key_features = ['MSI_status', 'stage_code', 'vital_status_code', 'gender_code', 'age_at_initial_pathologic_diagnosis']
        
        # 创建多图布局
        fig, axes = plt.subplots(3, 2, figsize=(18, 15))
        axes = axes.flatten()
        
        plot_idx = 0
        for feature in key_features:
            if feature in merged_data.columns and plot_idx < len(axes):
                ax = axes[plot_idx]
                
                # 确保有足够的非缺失数据
                valid_data = merged_data[['Subtype', feature]].dropna()
                if len(valid_data) < 10:
                    print(f"警告: {feature}有效数据太少({len(valid_data)}行), 跳过可视化")
                    continue
                    
                try:
                    if feature == 'age_at_initial_pathologic_diagnosis':
                        # 对于连续变量，使用箱线图，但先检查每个亚型是否有数据
                        subtypes = sorted(valid_data['Subtype'].unique())
                        
                        # 检查每个亚型是否有数据点
                        subtype_data = {}
                        for subtype in subtypes:
                            subset = valid_data[valid_data['Subtype'] == subtype]
                            if len(subset) > 0:
                                subtype_data[subtype] = subset[feature]
                        
                        if subtype_data:
                            sns.boxplot(x='Subtype', y=feature, data=valid_data, ax=ax)
                            ax.set_title(f'亚型与{feature}关系')
                            
                            # 计算统计显著性（只有在有多个亚型的情况下）
                            if len(subtype_data) > 1:
                                f_stat, p_val = stats.f_oneway(*[values for values in subtype_data.values()])
                                ax.text(0.05, 0.95, f'ANOVA: p={p_val:.4f}', transform=ax.transAxes, va='top')
                    else:
                        # 对分类变量，转换为分类型并创建计数图
                        valid_data['Subtype'] = valid_data['Subtype'].astype('category')
                        valid_data[feature] = valid_data[feature].astype('category')
                        
                        # 创建计数图
                        sns.countplot(x='Subtype', hue=feature, data=valid_data, ax=ax)
                        ax.set_title(f'亚型与{feature}关系')
                        ax.legend(title=feature)
                        
                        # 创建频率表并检查统计显著性
                        try:
                            crosstab = pd.crosstab(valid_data['Subtype'], valid_data[feature])
                            if crosstab.size > 0:  # 确保表格不为空
                                chi2, p, _, _ = stats.chi2_contingency(crosstab)
                                ax.text(0.05, 0.95, f'Chi2: p={p:.4f}', transform=ax.transAxes, va='top')
                        except Exception as e:
                            print(f"计算{feature}的统计显著性时出错: {e}")
                    
                    plot_idx += 1
                except Exception as e:
                    print(f"绘制{feature}关联图时出错: {str(e)}")
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "clinical_associations.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"临床特征分析时出错: {e}")
    
    # 生存分析
    try:
        perform_survival_analysis(merged_data)
    except Exception as e:
        print(f"执行生存分析时出错: {str(e)}")
    
    return merged_data

def perform_survival_analysis(merged_data):
    """进行生存分析"""
    if 'OS' in merged_data.columns and 'OS.time' in merged_data.columns:
        try:
            from lifelines import KaplanMeierFitter
            from lifelines.statistics import logrank_test
            
            # 绘制Kaplan-Meier曲线
            plt.figure(figsize=(10, 8))
            kmf = KaplanMeierFitter()
            
            subtypes = sorted(merged_data['Subtype'].unique())
            colors = sns.color_palette("viridis", len(subtypes))
            
            for i, subtype in enumerate(subtypes):
                subtype_data = merged_data[merged_data['Subtype'] == subtype]
                
                # 确保OS.time为数值型
                times = pd.to_numeric(subtype_data['OS.time'], errors='coerce')
                events = pd.to_numeric(subtype_data['OS'], errors='coerce')
                valid_idx = ~(times.isna() | events.isna())
                
                if valid_idx.sum() > 0:
                    kmf.fit(
                        times[valid_idx], 
                        events[valid_idx], 
                        label=f'亚型 {subtype} (n={valid_idx.sum()})'
                    )
                    kmf.plot(ci_show=False, color=colors[i])
            
            plt.title('胃癌分子亚型生存分析')
            plt.xlabel('时间(天)')
            plt.ylabel('生存概率')
            plt.grid(alpha=0.3)
            
            # 计算logrank检验p值
            if len(subtypes) > 1:
                # 所有亚型的多组比较
                all_p_values = []
                
                for i in range(len(subtypes)):
                    for j in range(i+1, len(subtypes)):
                        s1 = merged_data[merged_data['Subtype'] == subtypes[i]]
                        s2 = merged_data[merged_data['Subtype'] == subtypes[j]]
                        
                        t1 = pd.to_numeric(s1['OS.time'], errors='coerce')
                        e1 = pd.to_numeric(s1['OS'], errors='coerce')
                        t2 = pd.to_numeric(s2['OS.time'], errors='coerce')
                        e2 = pd.to_numeric(s2['OS'], errors='coerce')
                        
                        # 去除缺失值
                        idx1 = ~(t1.isna() | e1.isna())
                        idx2 = ~(t2.isna() | e2.isna())
                        
                        if idx1.sum() > 0 and idx2.sum() > 0:
                            results = logrank_test(
                                t1[idx1], t2[idx2],
                                e1[idx1], e2[idx2]
                            )
                            all_p_values.append(results.p_value)
                
                if all_p_values:
                    min_p = min(all_p_values)
                    plt.text(0.05, 0.05, f'Log-rank test: p={min_p:.4f}', 
                             transform=plt.gca().transAxes)
            
            plt.savefig(FIGURES_DIR / "survival_analysis.png", dpi=300)
            plt.close()
            
        except ImportError:
            print("未安装lifelines库，跳过生存分析")
        except Exception as e:
            print(f"生存分析出错: {e}")

def identify_subtype_features(integrated_clusters, omics_data):
    """识别各分子亚型的特征模式"""
    results = {}
    
    for omics_type, data in omics_data.items():
        # 找出共同样本
        common_samples = list(set(data.index) & set(integrated_clusters.index))
        
        if not common_samples:
            print(f"{omics_type}数据与整合聚类结果没有共同样本，跳过")
            continue
            
        print(f"\n分析{omics_type}数据的亚型特征...")
        
        # 获取该组学数据的亚型样本
        subtype_data = {}
        all_data = data.loc[common_samples]
        
        # 确保integrated_clusters是DataFrame格式
        if isinstance(integrated_clusters, pd.Series):
            clusters = pd.DataFrame(integrated_clusters)
        else:
            clusters = integrated_clusters
            
        clusters = clusters.loc[common_samples]
        
        # 对每个亚型分别获取数据
        subtypes = sorted(clusters.iloc[:, 0].unique())
        for subtype in subtypes:
            subtype_samples = clusters[clusters.iloc[:, 0] == subtype].index
            subtype_data[subtype] = all_data.loc[subtype_samples]
            print(f"  亚型 {subtype}: {len(subtype_samples)}个样本")
        
        # 计算每个亚型的特征模式
        subtype_means = {}
        for subtype, s_data in subtype_data.items():
            subtype_means[subtype] = s_data.mean()
        
        subtype_means_df = pd.DataFrame(subtype_means)
        
        # 标准化每个特征，突出亚型间差异
        scaler = StandardScaler()
        subtype_means_scaled = pd.DataFrame(
            scaler.fit_transform(subtype_means_df),
            index=subtype_means_df.index,
            columns=subtype_means_df.columns
        )
        
        # 寻找每个亚型的特征性基因/探针
        try:
            # 计算各亚型特异性分数
            specificity_scores = pd.DataFrame(index=subtype_means_df.index)
            
            for subtype in subtypes:
                # 计算特征在该亚型中的特异性分数
                # (该亚型平均值 - 所有亚型中最大的其他亚型平均值) / 标准差
                other_subtypes = [s for s in subtypes if s != subtype]
                if other_subtypes:
                    other_max = subtype_means_df[other_subtypes].max(axis=1)
                    specificity = (subtype_means_df[subtype] - other_max) / subtype_means_df.std(axis=1).replace(0, 1)
                    specificity_scores[f'Subtype_{subtype}'] = specificity
            
            # 选择每个亚型top特征（特异性最高的特征）
            n_top_features = min(50, len(specificity_scores))
            top_features = {}
            
            for subtype in subtypes:
                top_features[subtype] = specificity_scores[f'Subtype_{subtype}'].nlargest(n_top_features).index.tolist()
            
            # 绘制亚型特征热图
            plt.figure(figsize=(14, 10))
            
            # 选取每个亚型的top特征
            all_top_features = []
            for subtype_features in top_features.values():
                all_top_features.extend(subtype_features[:10])  # 每个亚型取前10个特征
            
            all_top_features = list(dict.fromkeys(all_top_features))  # 去重
            all_top_features = all_top_features[:min(100, len(all_top_features))]
            
            if omics_type == 'expression':
                title = '基因表达亚型特征'
            elif omics_type == 'methylation':
                title = 'DNA甲基化亚型特征'
            elif omics_type == 'mirna':
                title = 'miRNA表达亚型特征'
            elif omics_type == 'cnv':
                title = '拷贝数变异亚型特征'
            elif omics_type == 'mutation':
                title = '突变亚型特征'
            else:
                title = f'{omics_type}亚型特征'
            
            sns.clustermap(
                subtype_means_scaled.loc[all_top_features],
                cmap='coolwarm',
                center=0,
                figsize=(12, 14),
                row_cluster=True,
                col_cluster=False,
                xticklabels=True,
                yticklabels=False
            )
            plt.suptitle(title, y=1.02, fontsize=16)
            plt.savefig(FIGURES_DIR / f"{omics_type}_subtype_features.png", dpi=300)
            plt.close()
            
            # 保存结果
            subtype_means_df.to_csv(SUBTYPE_DIR / f"{omics_type}_subtype_means.csv")
            specificity_scores.to_csv(SUBTYPE_DIR / f"{omics_type}_feature_specificity.csv")
            
            # 记录top特征
            results[omics_type] = {
                'means': subtype_means_df,
                'top_features': top_features,
                'specificity': specificity_scores
            }
            
            # 打印每个亚型的top特征
            for subtype, features in top_features.items():
                print(f"亚型 {subtype} 的前5个{omics_type}特征:")
                for i, feature in enumerate(features[:5]):
                    if omics_type == 'expression' or omics_type == 'cnv' or omics_type == 'mutation':
                        print(f"  {i+1}. {feature} (特征值: {subtype_means_df.loc[feature, subtype]:.2f})")
                    else:
                        print(f"  {i+1}. {feature}")
            
        except Exception as e:
            print(f"分析{omics_type}亚型特征时出错: {e}")
    
    return results

def characterize_molecular_subtypes(subtype_features, merged_clinical):
    """综合特征为每个亚型创建分子特征描述"""
    if not subtype_features:
        print("无法创建分子亚型特征描述：缺少特征数据")
        return
        
    # 获取亚型列表
    all_subtypes = set()
    for omics_type in subtype_features:
        if 'means' in subtype_features[omics_type]:
            all_subtypes.update(subtype_features[omics_type]['means'].columns)
    
    subtypes = sorted(all_subtypes)
    print("\n胃癌分子亚型特征概述:")
    
    subtype_descriptions = {}
    
    for subtype in subtypes:
        description = f"分子亚型 {subtype} 特征概述:\n"
        
        # 添加临床特征
        if merged_clinical is not None:
            subtype_clinical = merged_clinical[merged_clinical['Subtype'] == subtype]
            
            # 生存情况
            if 'OS' in subtype_clinical.columns and 'OS.time' in subtype_clinical.columns:
                os_events = pd.to_numeric(subtype_clinical['OS'], errors='coerce').sum()
                total_samples = len(subtype_clinical)
                if total_samples > 0:
                    os_rate = os_events / total_samples * 100
                    description += f"- 生存事件: {os_events}/{total_samples} ({os_rate:.1f}%)\n"
            
            # MSI状态
            if 'MSI_status' in subtype_clinical.columns:
                msi_counts = subtype_clinical['MSI_status'].value_counts()
                if 2 in msi_counts:
                    msih_rate = msi_counts[2] / len(subtype_clinical) * 100
                    description += f"- MSI-H比例: {msih_rate:.1f}%\n"
            
            # 临床阶段
            if 'stage_code' in subtype_clinical.columns:
                try:
                    subtype_clinical['stage_code'] = pd.to_numeric(subtype_clinical['stage_code'], errors='coerce')
                    late_stage = subtype_clinical['stage_code'].isin([3, 4]).sum()
                    if len(subtype_clinical) > 0:
                        late_rate = late_stage / len(subtype_clinical) * 100
                        description += f"- 晚期(III+IV)比例: {late_rate:.1f}%\n"
                except:
                    pass
        
        # 添加各组学特征
        for omics_type, features in subtype_features.items():
            if 'top_features' in features and subtype in features['top_features']:
                top5 = features['top_features'][subtype][:5]
                
                if omics_type == 'expression':
                    description += f"- 高表达基因: {', '.join(top5)}\n"
                elif omics_type == 'methylation':
                    description += f"- 显著甲基化位点: {', '.join(top5)}\n"
                elif omics_type == 'mirna':
                    description += f"- 特征miRNA: {', '.join(top5)}\n"
                elif omics_type == 'cnv':
                    means = features['means']
                    amp_genes = [g for g in top5 if g in means.index and means.loc[g, subtype] > 0]
                    del_genes = [g for g in top5 if g in means.index and means.loc[g, subtype] < 0]
                    if amp_genes:
                        description += f"- 显著扩增基因: {', '.join(amp_genes)}\n"
                    if del_genes:
                        description += f"- 显著缺失基因: {', '.join(del_genes)}\n"
                elif omics_type == 'mutation':
                    description += f"- 高频突变基因: {', '.join(top5)}\n"
        
        subtype_descriptions[subtype] = description
        print(f"\n{description}")
    
    # 保存分子亚型描述
    with open(SUBTYPE_DIR / "molecular_subtype_descriptions.txt", "w", encoding='utf-8') as f:
        for subtype, desc in subtype_descriptions.items():
            f.write(f"\n{desc}\n")
    
    return subtype_descriptions

def main():
    # 添加命令行参数
    parser = argparse.ArgumentParser(description="胃癌分子亚型分析")
    parser.add_argument('--subtypes_file', type=str, help='亚型分类文件路径', default=None)
    args = parser.parse_args()
    
    print("开始胃癌分子亚型分析...")
    
    # 1. 加载整合聚类结果
    integrated_clusters = load_integrated_clusters(args.subtypes_file)
    if integrated_clusters is None:
        print("错误: 无法继续分析，未找到亚型分类结果")
        return
    
    # 2. 加载各组学数据和临床数据
    omics_data, clinical_data = load_omics_and_clinical_data()
    
    # 3. 分析临床特征
    print("\n分析分子亚型的临床特征...")
    merged_clinical = analyze_clinical_characteristics(integrated_clusters, clinical_data)
    
    # 4. 识别亚型特征
    print("\n识别各分子亚型的特征模式...")
    subtype_features = identify_subtype_features(integrated_clusters, omics_data)
    
    # 5. 综合特征描述
    print("\n创建分子亚型特征概述...")
    subtype_descriptions = characterize_molecular_subtypes(subtype_features, merged_clinical)
    
    print("\n胃癌分子亚型分析完成！")
    print(f"结果已保存至: {SUBTYPE_DIR}")

if __name__ == "__main__":
    main()