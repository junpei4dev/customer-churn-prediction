import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime
import os
import joblib

# データの読み込み
file_path = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
data = pd.read_csv(file_path)

# データの基本情報を表示（デバッグ用）
print(data.info())
print(data.describe())
print(data.head())

# データの前処理
data = data.dropna()

# カテゴリカルデータのエンコーディング
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    if column != 'customerID':
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

# 特徴量とラベルの分割
X = data.drop(['Churn', 'customerID'], axis=1)
y = data['Churn']

# データのスケーリング
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# データの分割
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# モデルとハイパーパラメータの設定
models = {
    'RandomForest': RandomForestClassifier(),
    'SVM': SVC()
}

params = {
    'RandomForest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30]
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    }
}

# 最適なモデルとハイパーパラメータの検索
best_models = {}
for model_name in models:
    grid_search = GridSearchCV(
        models[model_name], params[model_name], cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_models[model_name] = grid_search.best_estimator_
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")

# 最適なモデルの評価と結果の保存
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_dir = 'results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

results = {}
for model_name, model in best_models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of {model_name}: {accuracy:.2f}")

    # クロスバリデーションの実行
    cv_scores = cross_val_score(model, X_scaled, y, cv=5)
    print(f"Cross-validation scores for {model_name}: {cv_scores}")
    print(f"Mean CV score for {model_name}: {cv_scores.mean():.2f}")

    # テストデータでの詳細な評価
    clf_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Classification report for {model_name}:\n{clf_report}")
    print(f"Confusion matrix for {model_name}:\n{conf_matrix}")

    # モデルの保存
    model_filename = f'{results_dir}/model_{model_name}_{timestamp}.pkl'
    joblib.dump(model, model_filename)

    # 評価結果の保存
    results_filename = f'{results_dir}/results_{model_name}_{timestamp}.txt'
    with open(results_filename, 'w') as f:
        f.write(f'Accuracy: {accuracy:.2f}\n')
        f.write(f'Cross-validation scores: {cv_scores}\n')
        f.write(f'Mean CV score: {cv_scores.mean():.2f}\n')
        f.write(f'Classification report:\n{clf_report}\n')
        f.write(f'Confusion matrix:\n{conf_matrix}\n')
        f.write(f'Model saved as: {model_filename}\n')

    # 結果を辞書に保存
    results[model_name] = {
        'accuracy': accuracy,
        'cv_scores': cv_scores,
        'mean_cv_score': cv_scores.mean(),
        'classification_report': clf_report,
        'confusion_matrix': conf_matrix,
        'model_filename': model_filename,
        'results_filename': results_filename
    }

print(f"Results saved in {results_dir} directory.")
