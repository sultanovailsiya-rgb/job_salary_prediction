"""Точка входа: EDA, подбор гиперпараметров Optuna, обучение и оценка модели."""

import logging
from functools import partial
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, train_test_split

from src.config import DATA_PATH, MODEL_PATH, RANDOM_STATE, RESULTS_DIR, TARGET
from src.eda import run_eda
from src.model import build_pipeline, save_pipeline
from src.monitoring import (
    check_model_drift,
    log_final_metrics,
    log_optuna_run,
    setup_mlflow,
)
from src.visualization import generate_plots

logger = logging.getLogger(__name__)

# Конфигурация обучения
TEST_SIZE = 0.2
N_TRIALS = 20
CV_FOLDS = 3
OUTLIER_QUANTILE = 0.99


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Загружает датасет и фильтрует целевую переменную.

    Удаляет пропуски, неположительные значения и верхние выбросы
    (выше квантиля ``OUTLIER_QUANTILE``).

    Args:
        path: Путь к CSV-файлу с данными.

    Returns:
        Очищенный датафрейм.
    """
    df = pd.read_csv(path)
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET] > 0]
    df = df[df[TARGET] < df[TARGET].quantile(OUTLIER_QUANTILE)]
    return df


def objective(trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series) -> float:
    """Целевая функция Optuna: средний CV RMSE для предложенной конфигурации.

    Args:
        trial: Текущий trial Optuna.
        X_train: Обучающие признаки.
        y_train: Обучающая целевая переменная.

    Returns:
        Средний RMSE по кросс-валидации (минимизируется).
    """
    model_type = trial.suggest_categorical("model", ["gb", "rf", "ridge"])

    if model_type == "gb":
        params = {
            "model": "gb",
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
        }
    elif model_type == "rf":
        params = {
            "model": "rf",
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        }
    else:
        params = {
            "model": "ridge",
            "alpha": trial.suggest_float("alpha", 0.1, 100.0, log=True),
        }

    pipeline = build_pipeline(params)
    score = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=CV_FOLDS,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    return -score.mean()


def main() -> None:
    """Запускает полный цикл: данные, EDA, Optuna, обучение, оценка, графики."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    df = load_data()
    logger.info("%d записей", len(df))

    run_eda(df, target=TARGET, output_path=RESULTS_DIR / "eda_report.txt")

    X = df.drop(TARGET, axis=1)
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    setup_mlflow()
    study = optuna.create_study(direction="minimize", study_name="salary_optuna")
    study.optimize(
        partial(objective, X_train=X_train, y_train=y_train),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    log_optuna_run(study)
    logger.info("Лучшая конфигурация: %s", study.best_params)

    final_pipeline = build_pipeline(study.best_params)
    final_pipeline.fit(X_train, y_train)

    y_pred = final_pipeline.predict(X_test)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "mape_percent": float(mean_absolute_percentage_error(y_test, y_pred) * 100),
        "model": study.best_params["model"],
    }

    log_final_metrics(metrics)
    check_model_drift(y_test, y_pred, baseline_mape=metrics["mape_percent"])

    save_pipeline(final_pipeline, path=MODEL_PATH)
    generate_plots(df, final_pipeline, output_dir=RESULTS_DIR / "plots")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sample = X_test.head(10).copy()
    sample["actual_salary"] = y_test.head(10).values
    sample["predicted_salary"] = final_pipeline.predict(sample)
    sample.to_csv(RESULTS_DIR / "sample_predictions.csv", index=False)

    logger.info(
        "Метрики: RMSE=$%s | R²=%.3f | MAPE=%.1f%%",
        f"{metrics['rmse']:,.0f}",
        metrics["r2"],
        metrics["mape_percent"],
    )


if __name__ == "__main__":
    main()
