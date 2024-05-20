from src.data_preprocessing import load_data, preprocess_data
from src.model_training import train_model
from src.model_evaluation import evaluate_model
from src.utils import save_model, save_results

# データの読み込み
file_path = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
data = load_data(file_path)

# データの前処理
X, y = preprocess_data(data)

# モデルの訓練
model, X_test, y_test = train_model(X, y)

# モデルの評価
accuracy = evaluate_model(model, X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

# 結果の保存
model_filename = save_model(model, 'random_forest')
results_filename = save_results('random_forest', accuracy, model_filename)
print(f"Results saved in {results_filename}")
