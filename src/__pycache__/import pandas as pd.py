import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml

class FederatedDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        if isinstance(y, pd.Series):
            y = y.values
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_domain_data(domain="Healthcare"):
    """
    Dynamically loads and preprocesses data based on the chosen domain.
    Healthcare: Heart Disease
    Finance: Credit-g (fraud/default equivalent)
    Benchmark: Adult Census Income
    """
    if domain == "Healthcare":
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
        df = pd.read_csv(url, names=columns, na_values="?")
        df = df.dropna()
        
        df['target'] = (df['target'] > 0).astype(int)
        df['protected_group'] = (df['age'] > 55).astype(int)
        
        target_col = 'target'
        protected_col = 'protected_group'

    elif domain == "Finance":
        data = fetch_openml(data_id=31, as_frame=True, parser='auto')
        df = data.frame
        df = df.dropna()
        
        df['target'] = (df['class'] == 'bad').astype(int)
        df['protected_group'] = (df['age'] > 30).astype(int)
        
        df = df.drop(columns=['class'])
        target_col = 'target'
        protected_col = 'protected_group'
        
    elif domain == "Benchmark":
        data = fetch_openml(data_id=1590, as_frame=True, parser='auto')
        df = data.frame
        df = df.dropna()
        
        df['target'] = (df['class'] == '>50K').astype(int)
        df['protected_group'] = (df['sex'] == 'Male').astype(int)
        
        df = df.drop(columns=['class'])
        target_col = 'target'
        protected_col = 'protected_group'
        df = df.sample(n=5000, random_state=42)

    else:
        raise ValueError("Invalid domain selected")

    y = df[target_col].values
    protected = df[protected_col].values
    X_raw = df.drop(columns=[target_col, protected_col])

    categorical_cols = X_raw.select_dtypes(include=['object', 'category']).columns.tolist()
    X_encoded = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X_encoded)

    return X, y, protected, df

def partition_data_for_clients(X, y, protected, num_clients=3):
    """ Splits the data into partitions for Federated Learning clients """
    partitions = []
    indices = np.random.permutation(len(X))
    split_size = len(X) // num_clients
    
    for i in range(num_clients):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i < num_clients - 1 else len(X)
        
        client_indices = indices[start_idx:end_idx]
        
        X_client = X[client_indices]
        y_client = y[client_indices]
        protected_client = protected[client_indices]
        
        X_train, X_test, y_train, y_test, p_train, p_test = train_test_split(
            X_client, y_client, protected_client, test_size=0.2, random_state=42
        )
        
        partitions.append({
            'train': (X_train, y_train, p_train),
            'test': (X_test, y_test, p_test)
        })
        
    return partitions

def get_dataloaders(X, y, batch_size=32):
    dataset = FederatedDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
