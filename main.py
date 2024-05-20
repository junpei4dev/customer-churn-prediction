import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime
import os

# データの読み込み
data = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# 前処理
data = data.dropna()
data = pd.get_dummies(data, drop_first=True)

# 特徴量とラベルの分割
X = data.drop('Churn_Yes', axis=1)
y = data['Churn_Yes']

# データの分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# モデルの訓練
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 予測と評価
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# 結果の保存
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_dir = 'results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# モデルの保存
model_filename = f'{results_dir}/model_{timestamp}.pkl'
pd.to_pickle(model, model_filename)

# 評価結果の保存
results_filename = f'{results_dir}/results_{timestamp}.txt'
with open(results_filename, 'w') as f:
    f.write(f'Accuracy: {accuracy:.2f}\n')
    f.write(f'Model saved as: {model_filename}\n')
