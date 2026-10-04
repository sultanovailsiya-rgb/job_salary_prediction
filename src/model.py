"""Сборка ML-пайплайна: предобработка, модели и сохранение."""

from pathlib import Path

import joblib
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import MODEL_PATH, RANDOM_STATE

NUMERICAL_COLS = ["experience_years", "skills_count", "certifications"]
CATEGORICAL_COLS = [
    "job_title",
    "education_level",
    "industry",
    "company_size",
    "location",
    "remote_work",
]


def create_preprocessor() -> ColumnTransformer:
    """Создаёт предобработку: масштабирование чисел и one-hot для категорий."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_COLS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLS,
            ),
        ],
        remainder="drop",
    )


def create_model(best_params: dict) -> BaseEstimator:
    """Создаёт модель по параметрам, подобранным Optuna.

    Args:
        best_params: Параметры модели; ключ ``model`` — gb, rf или ridge.

    Returns:
        Нефитованная модель scikit-learn.
    """
    model_type = best_params["model"]
    if model_type == "gb":
        return GradientBoostingRegressor(
            n_estimators=best_params["n_estimators"],
            learning_rate=best_params["learning_rate"],
            max_depth=best_params["max_depth"],
            random_state=RANDOM_STATE,
        )
    elif model_type == "rf":
        return RandomForestRegressor(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            min_samples_split=best_params["min_samples_split"],
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:  # ridge
        return Ridge(alpha=best_params["alpha"])


def build_pipeline(best_params: dict) -> Pipeline:
    """Собирает полный пайплайн: предобработка и модель."""
    return Pipeline(
        [
            ("preprocessor", create_preprocessor()),
            ("model", create_model(best_params)),
        ]
    )


def save_pipeline(pipeline: Pipeline, path: Path = MODEL_PATH) -> None:
    """Сохраняет пайплайн на диск, создавая каталог при необходимости."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_pipeline(path: Path = MODEL_PATH) -> Pipeline:
    """Загружает сохранённый пайплайн.

    Raises:
        FileNotFoundError: Если файл модели не найден.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Модель не найдена: {path}")
    return joblib.load(path)
