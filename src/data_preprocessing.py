import pandas as pd


def load_data(file_path):
    return pd.read_csv(file_path)


def preprocess_data(data):
    data = data.dropna()
    data = pd.get_dummies(data, drop_first=True)
    X = data.drop('Churn_Yes', axis=1)
    y = data['Churn_Yes']
    return X, y
