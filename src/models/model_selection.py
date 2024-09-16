import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def train_best_model(X_train, y_train):
    models = {
        'RandomForest': RandomForestClassifier(),
        'SVM': SVC()
    }
    
    param_grids = {
        'RandomForest': {
            'n_estimators': [50, 100],
            'max_depth': [10, 20]
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }
    }
    
    best_model = None
    best_score = 0
    
    for model_name, model in models.items():
        grid_search = GridSearchCV(model, param_grids[model_name], cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_model = grid_search.best_estimator_
    
    return best_model

def train_anomaly_detection(X_train):
    model = IsolationForest(contamination=0.1)
    model.fit(X_train)
    return model

if __name__ == "__main__":
    # Đường dẫn thư mục
    model_dir = '../model'

    # Tạo thư mục nếu nó chưa tồn tại
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    X_train = pd.read_csv('/home/meli0das/IDS_ML /data/processed/X_train.csv')
    y_train = pd.read_csv('/home/meli0das/IDS_ML /data/processed/X_train.csv').values.ravel()
    
    model = train_best_model(X_train, y_train)
    
    # Lưu mô hình
    joblib.dump(model, os.path.join(model_dir, 'best_model.pkl'))
