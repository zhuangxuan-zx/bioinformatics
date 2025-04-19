
def predict_subtype(new_data):
    """
    Ԥ����������θ����������
    ����:
        new_data: ����������DataFrame
    ����:
        Ԥ������ͱ�ǩ
    """
    import pickle
    import pandas as pd
    import numpy as np
    from pathlib import Path
    
    # ����ģ��·��
    model_dir = Path(__file__).parent
    
    # ����ģ�ͺ�Ԥ������
    with open(model_dir / "RandomForest_model.pkl", "rb") as f:
        model = pickle.load(f)
        
    with open(model_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    # ���������б�
    features = pd.read_csv(model_dir / "selected_model_features.csv")
    
    # ��������������Ƿ������������
    required_features = []
    for _, row in features.iterrows():
        feature_name = f"{row['omics_type']}_{row['feature']}"
        required_features.append(feature_name)
    
    missing_features = [f for f in required_features if f not in new_data.columns]
    if missing_features:
        print(f"����: ��������ȱ��{len(missing_features)}������")
        # Ϊȱʧ�������0
        for feature in missing_features:
            new_data[feature] = 0
    
    # ��ȡ����������
    X = new_data[required_features]
    
    # ��׼��
    X_scaled = scaler.transform(X)
    
    # Ԥ��
    prediction = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    
    return prediction, probabilities
