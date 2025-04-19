
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
