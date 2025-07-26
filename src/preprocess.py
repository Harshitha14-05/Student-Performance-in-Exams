# preprocess.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)

    # Check for missing values
    df.dropna(inplace=True)

    # Encode target column (result: Pass/Fail to 1/0)
    le = LabelEncoder()
    df['Result'] = le.fit_transform(df['Result'])

    X = df.drop('Result', axis=1)
    y = df['Result']

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
