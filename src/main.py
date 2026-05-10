import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import build_pipeline, save_pipeline
from monitoring import setup_mlflow, log_optuna_run, log_final_metrics, check_model_drift
from eda import run_eda
from visualization import generate_plots

# Настройки путей
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "job_salary_prediction_dataset.csv")
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "final_pipeline.pkl")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# КОнфигурации
TARGET = "salary"
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_TRIALS = 20

def objective(trial):
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET] > 0]
    df = df[df[TARGET] < df[TARGET].quantile(0.99)]
    
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]
    
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    model_type = trial.suggest_categorical('model', ['gb', 'rf', 'ridge'])
    
    if model_type == 'gb':
        params = {
            'model': 'gb',
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10)
        }
    elif model_type == 'rf':
        params = {
            'model': 'rf',
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20)
        }
    else:
        params = {
            'model': 'ridge',
            'alpha': trial.suggest_float('alpha', 0.1, 100.0, log=True)
        }
        
    pipeline = build_pipeline(params)
    score = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
    return -score.mean()

def main():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET] > 0]
    df = df[df[TARGET] < df[TARGET].quantile(0.99)]
    print(f"{len(df)} записей")
    
    # EDA
    run_eda(df, target=TARGET, output_path=os.path.join(RESULTS_DIR, 'eda_report.txt'))
    
    # Разделение
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    # Optuna
    setup_mlflow()
    study = optuna.create_study(direction='minimize', study_name='salary_optuna')
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    
    log_optuna_run(study)
    print(f"Лучшая конфигурация: {study.best_params}")
    
    # Финальное обучение
    final_pipeline = build_pipeline(study.best_params)
    final_pipeline.fit(X_train, y_train)
    
    # Оценка
    y_pred = final_pipeline.predict(X_test)
    metrics = {
        'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
        'mae': float(mean_absolute_error(y_test, y_pred)),
        'r2': float(r2_score(y_test, y_pred)),
        'mape_percent': float(mean_absolute_percentage_error(y_test, y_pred) * 100),
        'model': study.best_params['model']
    }
    
    log_final_metrics(metrics)
    
    # Мониторинг
    check_model_drift(y_test, y_pred, baseline_mape=metrics['mape_percent'])
    
    # Сохранение
    save_pipeline(final_pipeline, path=MODEL_SAVE_PATH)

    # Визуализация
    generate_plots(df, final_pipeline, output_dir=os.path.join(RESULTS_DIR, 'plots'))

    # Тесты    
    test_result = pytest.main([
        os.path.join(PROJECT_ROOT, 'tests'), # папка с тестами
        '-v', # подробный вывод
        '--tb=short', # краткий traceback при ошибках
        '-x' # остановка на первой ошибке
    ])
    
    if test_result == 0:
        print("Тесты пройдены")
    else:
        print("Некоторые тесты не пройдены")

    # Примеры
    os.makedirs('results', exist_ok=True)
    sample = X_test.head(10).copy()
    sample['actual_salary'] = y_test.head(10).values
    sample['predicted_salary'] = final_pipeline.predict(sample)
    sample.to_csv('results/sample_predictions.csv', index=False)
    
    print(f"Метрики: RMSE=${metrics['rmse']:,.0f} | R²={metrics['r2']:.3f} | MAPE={metrics['mape_percent']:.1f}%")

if __name__ == "__main__":
    main()