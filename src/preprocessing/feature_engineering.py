import pandas as pd
from sklearn.preprocessing import StandardScaler

def feature_engineering(df):
    # Kiểm tra các cột dạng chuỗi để mã hóa
    categorical_columns = df.select_dtypes(include=['object']).columns

    # Mã hóa các cột dạng chuỗi
    df_encoded = pd.get_dummies(df, columns=categorical_columns)

    # Chuẩn hóa các cột số (sau khi mã hóa)
    scaler = StandardScaler()
    
    # Giả sử 'label' là cột chứa nhãn cần dự đoán, nếu không có thì bỏ qua
    if 'label' in df_encoded.columns:
        features = df_encoded.drop(columns=['label'])
        labels = df_encoded['label']
    else:
        features = df_encoded
        labels = pd.Series()

    # Chuẩn hóa các đặc trưng
    scaled_features = scaler.fit_transform(features)

    return scaled_features, labels

if __name__ == "__main__":
    # Đọc dữ liệu đã tiền xử lý
    train_df = pd.read_csv('/home/meli0das/IDS_ML /data/processed/train_processed.csv')

    # Thực hiện chuẩn hóa và trích xuất đặc trưng
    X_train, y_train = feature_engineering(train_df)

    # Lưu kết quả ra file CSV
    pd.DataFrame(X_train).to_csv('/home/meli0das/IDS_ML /data/processed.', index=False)
    pd.DataFrame(y_train).to_csv('/home/meli0das/IDS_ML /data/processed', index=False)
