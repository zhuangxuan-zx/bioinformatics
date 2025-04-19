#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TCGA-STAD甲基化数据修复脚本
用途：过滤已处理甲基化数据，只保留真正的甲基化探针(cg开头)
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

# 设置项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def main():
    """过滤已处理的甲基化数据，只保留cg开头的探针"""
    start_time = datetime.now()
    print(f"开始修复甲基化数据: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 加载已处理的甲基化数据
        methyl_path = PROCESSED_DIR / "methylation_filtered.csv"
        methyl_data = pd.read_csv(methyl_path, index_col=0)
        
        print(f"原始处理后甲基化数据: {methyl_data.shape[0]}个样本 x {methyl_data.shape[1]}个探针")
        
        # 统计不同类型的探针
        cg_probes = [col for col in methyl_data.columns if col.startswith('cg')]
        rs_probes = [col for col in methyl_data.columns if col.startswith('rs')]
        other_probes = [col for col in methyl_data.columns if not col.startswith('cg') and not col.startswith('rs')]
        
        print(f"甲基化探针(cg): {len(cg_probes)}")
        print(f"SNP标记(rs): {len(rs_probes)}")
        print(f"其他标记: {len(other_probes)}")
        
        # 只保留cg开头的甲基化探针
        methyl_filtered = methyl_data.loc[:, cg_probes]
        
        print(f"过滤后的甲基化数据: {methyl_filtered.shape[0]}个样本 x {methyl_filtered.shape[1]}个探针")
        
        # 保存过滤后的数据
        methyl_filtered.to_csv(PROCESSED_DIR / "methylation_filtered_cg_only.csv2")
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds() / 60
        
        print(f"修复后的甲基化数据已保存至: {PROCESSED_DIR / 'methylation_filtered_cg_only.csv'}")
        print(f"处理完成: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"处理时间: {processing_time:.2f}分钟")
        
    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()