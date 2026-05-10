import mlflow
import numpy as np
import pandas as pd

# Инициализация MLflow
def setup_mlflow(experiment_name: str = "Job_Salary_Prediction"):
    mlflow.set_experiment(experiment_name)
    return mlflow

# Логирование результатов поиска Optuna
def log_optuna_run(study, run_name: str = "optuna_search"):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_cv_rmse", study.best_value)
        mlflow.log_param("best_model", study.best_params['model'])

# Логирование финальных метрик на тесте
def log_final_metrics(metrics: dict, run_name: str = "final_evaluation"):
    with mlflow.start_run(run_name=run_name):
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)
            else:
                mlflow.log_param(k, v)

# Мониторинг качества: сравнивает текущую ошибку с базовой линией
# Если MAPE вырос > на 20% — сигнализирует о возможном дрейфе данных
def check_model_drift(y_true: pd.Series, y_pred: np.ndarray, baseline_mape: float = 15.0) -> dict:
    current_mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    drift_ratio = current_mape / baseline_mape
    is_drift = drift_ratio > 1.2
    
    result = {
        "current_mape": current_mape,
        "baseline_mape": baseline_mape,
        "drift_ratio": drift_ratio,
        "drift_detected": is_drift
    }
    
    # Логируем в MLflow для отслеживания
    with mlflow.start_run(run_name="monitoring_check", nested=True):
        mlflow.log_metrics(result)
        
    status = "Обнаружен дрейф данных!" if is_drift else "Модель стабильна"
    print(f"\nМониторинг: {status} (MAPE: {current_mape:.1f}%, Дрейф: {drift_ratio:.2f}x)")
    return result