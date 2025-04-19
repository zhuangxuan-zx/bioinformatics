#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
胃癌分子亚型预测模型
用途：构建机器学习模型预测样本的分子亚型归属
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# 设置项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SUBTYPE_DIR = RESULTS_DIR / "molecular_subtypes"
PREDICTION_DIR = RESULTS_DIR / "prediction_models"
FIGURES_DIR = PREDICTION_DIR / "figures"
INTEGRATION_DIR = RESULTS_DIR / "integration"  # 添加这一行

# 创建输出目录
PREDICTION_DIR.mkdir(exist_ok=True, parents=True)
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

def load_subtypes_and_omics_data():
    """加载亚型分类结果和各组学数据"""
    # 优先加载高级聚类结果
    subtype_files = [
        RESULTS_DIR / "advanced_clustering" / "final_subtypes.csv",
        SUBTYPE_DIR / "subtypes_with_clinical.csv",
        INTEGRATION_DIR / "integrated_clusters.csv"
    ]
    
    subtypes = None
    for file in subtype_files:
        if file.exists():
            try:
                data = pd.read_csv(file, index_col=0)
                if 'subtype' in data.columns:
                    subtypes = data['subtype']
                elif 'Subtype' in data.columns:
                    subtypes = data['Subtype']
                elif 'integrated_cluster' in data.columns:
                    subtypes = data['integrated_cluster']
                else:
                    # 假设第一列是亚型
                    subtypes = data.iloc[:, 0]
                print(f"加载亚型分类结果: {len(subtypes)}个样本")
                break
            except Exception as e:
                print(f"加载{file}时出错: {e}")
    
    if subtypes is None:
        print("错误: 未找到亚型分类结果")
        return None, None
    
    # 加载组学数据
    omics_data = {}
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
    
    return subtypes, omics_data

def prepare_features_for_model(subtypes, omics_data, top_features_per_omics=100):
    """准备用于模型训练的特征数据"""
    # 找出所有数据集共同的样本
    common_samples = set(subtypes.index)
    for data in omics_data.values():
        common_samples &= set(data.index)
    
    common_samples = sorted(common_samples)
    print(f"所有数据集共有{len(common_samples)}个样本")
    
    if len(common_samples) < 30:
        print("警告: 共同样本数过少，可能影响模型质量")
        # 尝试找出具有最多共同样本的组学组合
        best_omics = []
        max_samples = 0
        
        for omics_type in omics_data:
            test_set = set(subtypes.index) & set(omics_data[omics_type].index)
            if len(test_set) > max_samples:
                max_samples = len(test_set)
                best_omics = [omics_type]
        
        print(f"推荐使用{', '.join(best_omics)}数据构建模型，共有{max_samples}个样本")
        
        # 更新共同样本
        common_samples = set(subtypes.index)
        for omics_type in best_omics:
            common_samples &= set(omics_data[omics_type].index)
        common_samples = sorted(common_samples)
    
    # 准备目标变量（亚型标签）
    y = subtypes.loc[common_samples].values
    
    # 为每种组学数据选择最具区分性的特征
    selected_features = {}
    X_combined = pd.DataFrame(index=common_samples)
    
    for omics_type, data in omics_data.items():
        if set(common_samples).issubset(set(data.index)):
            # 提取共同样本的数据
            X_omics = data.loc[common_samples]
            
            # 使用ANOVA进行特征选择
            selector = SelectKBest(f_classif, k=min(top_features_per_omics, X_omics.shape[1]))
            selector.fit(X_omics, y)
            
            # 获取选定的特征
            top_features_idx = selector.get_support(indices=True)
            top_features = X_omics.columns[top_features_idx].tolist()
            
            # 保存选定的特征
            selected_features[omics_type] = top_features
            
            # 将特征添加到组合数据集
            for feature in top_features:
                X_combined[f"{omics_type}_{feature}"] = X_omics[feature]
            
            print(f"从{omics_type}中选择了{len(top_features)}个特征")
    
    print(f"组合特征集: {X_combined.shape[1]}个特征")
    
    # 保存选定的特征
    features_df = pd.DataFrame({
        'omics_type': [k for k, v in selected_features.items() for _ in v],
        'feature': [f for v in selected_features.values() for f in v]
    })
    features_df.to_csv(PREDICTION_DIR / "selected_model_features.csv", index=False)
    
    return X_combined, y, common_samples

def train_and_evaluate_models(X, y):
    """训练并评估多个分类模型"""
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 定义要尝试的模型
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    # 定义参数网格
    param_grids = {
        'RandomForest': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf', 'linear']
        }
    }
    
    best_models = {}
    best_scores = {}
    
    for model_name, model in models.items():
        print(f"\n训练{model_name}模型...")
        
        # 进行网格搜索
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            model, param_grids[model_name],
            scoring='accuracy', cv=cv, n_jobs=-1
        )
        grid_search.fit(X_train_scaled, y_train)
        
        # 获取最佳模型
        best_model = grid_search.best_estimator_
        best_score = grid_search.best_score_
        best_params = grid_search.best_params_
        
        print(f"  最佳参数: {best_params}")
        print(f"  交叉验证准确率: {best_score:.4f}")
        
        # 在测试集上评估
        y_pred = best_model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_pred)
        print(f"  测试集准确率: {test_accuracy:.4f}")
        
        # 保存结果
        best_models[model_name] = best_model
        best_scores[model_name] = test_accuracy
        
        # 创建混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_name} 混淆矩阵')
        plt.ylabel('真实亚型')
        plt.xlabel('预测亚型')
        plt.savefig(FIGURES_DIR / f"{model_name}_confusion_matrix.png", dpi=300)
        plt.close()
        
        # 创建分类报告
        report = classification_report(y_test, y_pred)
        print("  分类报告:")
        print(report)
        
        with open(PREDICTION_DIR / f"{model_name}_classification_report.txt", "w") as f:
            f.write(report)
    
    # 选择最佳模型
    best_model_name = max(best_scores, key=best_scores.get)
    best_model = best_models[best_model_name]
    
    print(f"\n最佳模型: {best_model_name}, 准确率: {best_scores[best_model_name]:.4f}")
    
    # 分析特征重要性（仅对随机森林）
    if 'RandomForest' in best_models:
        rf_model = best_models['RandomForest']
        feature_importances = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # 保存特征重要性
        feature_importances.to_csv(PREDICTION_DIR / "feature_importances.csv", index=False)
        
        # 可视化前20个重要特征
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importances.head(20))
        plt.title('特征重要性 (随机森林)')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "feature_importances.png", dpi=300)
        plt.close()
    
    # 保存模型
    for model_name, model in best_models.items():
        model_path = PREDICTION_DIR / f"{model_name}_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"已保存{model_name}模型: {model_path}")
    
    # 保存数据预处理器
    with open(PREDICTION_DIR / "scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    
    return best_models, best_scores, X_train, X_test, y_train, y_test

def create_prediction_function():
    """创建用于预测新样本的函数并保存到文件"""
    predict_code = '''
def predict_subtype(new_data):
    """
    预测新样本的胃癌分子亚型
    参数:
        new_data: 包含特征的DataFrame
    返回:
        预测的亚型标签
    """
    import pickle
    import pandas as pd
    import numpy as np
    from pathlib import Path
    
    # 设置模型路径
    model_dir = Path(__file__).parent
    
    # 加载模型和预处理器
    with open(model_dir / "RandomForest_model.pkl", "rb") as f:
        model = pickle.load(f)
        
    with open(model_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    # 加载特征列表
    features = pd.read_csv(model_dir / "selected_model_features.csv")
    
    # 检查输入数据中是否包含所需特征
    required_features = []
    for _, row in features.iterrows():
        feature_name = f"{row['omics_type']}_{row['feature']}"
        required_features.append(feature_name)
    
    missing_features = [f for f in required_features if f not in new_data.columns]
    if missing_features:
        print(f"警告: 输入数据缺少{len(missing_features)}个特征")
        # 为缺失特征填充0
        for feature in missing_features:
            new_data[feature] = 0
    
    # 提取和排序特征
    X = new_data[required_features]
    
    # 标准化
    X_scaled = scaler.transform(X)
    
    # 预测
    prediction = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    
    return prediction, probabilities
'''
    
    with open(PREDICTION_DIR / "predict_subtype.py", "w") as f:
        f.write(predict_code)
    
    print(f"已生成预测函数: {PREDICTION_DIR / 'predict_subtype.py'}")

def main():
    print("开始构建胃癌分子亚型预测模型...")
    
    # 1. 加载亚型和组学数据
    subtypes, omics_data = load_subtypes_and_omics_data()
    if subtypes is None or not omics_data:
        print("错误: 数据加载失败，无法继续构建模型")
        return
    
    # 2. 准备模型特征
    X, y, common_samples = prepare_features_for_model(subtypes, omics_data)
    
    # 保存用于模型训练的数据
    model_data = pd.DataFrame(X)
    model_data['subtype'] = y
    model_data.to_csv(PREDICTION_DIR / "model_training_data.csv")
    
    # 3. 训练和评估模型
    models, scores, X_train, X_test, y_train, y_test = train_and_evaluate_models(X, y)
    
    # 4. 创建预测函数
    create_prediction_function()
    
    print("\n胃癌分子亚型预测模型构建完成！")
    print(f"所有模型和结果已保存至: {PREDICTION_DIR}")

if __name__ == "__main__":
    main()