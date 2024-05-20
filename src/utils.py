from datetime import datetime
import os
import joblib


def save_model(model, model_name, results_dir='results'):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    model_filename = f'{results_dir}/model_{model_name}_{timestamp}.pkl'
    joblib.dump(model, model_filename)
    return model_filename


def save_results(model_name, accuracy, model_filename, results_dir='results'):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_filename = f'{results_dir}/results_{model_name}_{timestamp}.txt'
    with open(results_filename, 'w') as f:
        f.write(f'Accuracy: {accuracy:.2f}\n')
        f.write(f'Model saved as: {model_filename}\n')
    return results_filename
