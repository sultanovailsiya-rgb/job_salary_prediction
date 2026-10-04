"""Логирование экспериментов в MLflow и мониторинг качества модели."""

import logging
from types import ModuleType

import mlflow
import numpy as np
import optuna
import pandas as pd

logger = logging.getLogger(__name__)


def setup_mlflow(experiment_name: str = "Job_Salary_Prediction") -> ModuleType:
    """Инициализирует эксперимент MLflow."""
    mlflow.set_experiment(experiment_name)
    return mlflow


def log_optuna_run(study: optuna.Study, run_name: str = "optuna_search") -> None:
    """Логирует результаты поиска Optuna в MLflow."""
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_cv_rmse", study.best_value)
        mlflow.log_param("best_model", study.best_params["model"])


def log_final_metrics(metrics: dict, run_name: str = "final_evaluation") -> None:
    """Логирует финальные метрики: числа как метрики, остальное как параметры."""
    with mlflow.start_run(run_name=run_name):
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)
            else:
                mlflow.log_param(k, v)


def check_model_drift(
    y_true: pd.Series, y_pred: np.ndarray, baseline_mape: float = 15.0
) -> dict:
    """Сравнивает текущую ошибку (MAPE) с базовой линией.

    Если MAPE вырос более чем на 20% - сигнализирует о возможном дрейфе данных.

    Args:
        y_true: Фактические значения.
        y_pred: Предсказания модели.
        baseline_mape: Базовое значение MAPE в процентах.

    Returns:
        Словарь с текущим MAPE, базой, отношением и флагом дрейфа.
    """
    current_mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    drift_ratio = current_mape / baseline_mape
    is_drift = drift_ratio > 1.2

    result = {
        "current_mape": current_mape,
        "baseline_mape": baseline_mape,
        "drift_ratio": drift_ratio,
        "drift_detected": is_drift,
    }

    # Логируем в MLflow для отслеживания
    with mlflow.start_run(run_name="monitoring_check", nested=True):
        mlflow.log_metrics(result)

    status = "Обнаружен дрейф данных!" if is_drift else "Модель стабильна"
    logger.info(
        "Мониторинг: %s (MAPE: %.1f%%, Дрейф: %.2fx)",
        status,
        current_mape,
        drift_ratio,
    )
    return result
