import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Load data from file KDDTrain.txt
def load_data(filepath):
    columns = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", 
               "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", 
               "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", 
               "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", 
               "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", 
               "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", 
               "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", 
               "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", 
               "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]

    # Read data
    df = pd.read_csv(filepath, header=None, names=columns)
    return df

# Preprocess data
def preprocess_data(df):
    # Encode string columns into numeric
    df = pd.get_dummies(df, columns=["protocol_type", "service", "flag"])

    # Remove useless columns 
    df.drop(columns=["num_outbound_cmds"], inplace=True)

    # Change label in to binary (0 - normal, 1 - attack)
    df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)

    return df

# Save result after preprocessing data
def save_processed_data(df, output_path):
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    # Load and preprocess data
    train_df = load_data('/home/meli0das/IDS_ML /data/raw/KDDTrain+.txt')
    train_processed = preprocess_data(train_df)

    # Save file train_processed.csv
    save_processed_data(train_processed, '/home/meli0das/IDS_ML /data/processed/train_processed.csv')
