"""Тесты данных, подготовки данных и сборки пайплайна."""

import joblib
import pandas as pd
import pytest

from src.config import DATA_PATH, MODEL_PATH, TARGET
from src.main import load_data
from src.model import (
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
    build_pipeline,
    create_model,
    load_pipeline,
)

REQUIRED_COLS = NUMERICAL_COLS + CATEGORICAL_COLS + [TARGET]


@pytest.fixture
def tiny_df() -> pd.DataFrame:
    """Небольшой синтетический датасет для быстрых тестов."""
    rows = 120
    return pd.DataFrame(
        {
            "job_title": ["Engineer", "Analyst", "Manager", "Designer"] * 30,
            "experience_years": list(range(rows)),
            "education_level": ["BSc", "MSc", "PhD"] * 40,
            "skills_count": [i % 10 for i in range(rows)],
            "industry": ["IT", "Finance"] * 60,
            "company_size": ["Small", "Large"] * 60,
            "location": ["US", "UK", "DE"] * 40,
            "remote_work": ["Yes", "No"] * 60,
            "certifications": [i % 4 for i in range(rows)],
            TARGET: [50_000 + 1_000 * i for i in range(rows)],
        }
    )


# Тесты для проверки целостности данных
class TestData:
    def test_file_exists(self):
        assert DATA_PATH.exists(), f"Файл {DATA_PATH} не найден"

    def test_required_columns(self):
        df = pd.read_csv(DATA_PATH, nrows=1000)
        for col in REQUIRED_COLS:
            assert col in df.columns, f"Отсутствует обязательная колонка: {col}"

    def test_salary_valid(self):
        df = pd.read_csv(DATA_PATH)
        salaries = df[TARGET].dropna()
        assert (salaries > 0).all(), "Есть неположительные зарплаты"
        assert salaries.median() < 500_000, "Медиана зарплаты нереалистична"


class TestLoadData:
    def test_filters_target(self, tmp_path, tiny_df):
        dirty = tiny_df.copy()
        dirty.loc[0, TARGET] = None
        dirty.loc[1, TARGET] = -5
        dirty.loc[2, TARGET] = 10_000_000
        path = tmp_path / "data.csv"
        dirty.to_csv(path, index=False)

        df = load_data(path)

        assert df[TARGET].notna().all()
        assert (df[TARGET] > 0).all()
        assert df[TARGET].max() < 10_000_000


class TestPipeline:
    def test_create_model_ridge(self):
        model = create_model({"model": "ridge", "alpha": 1.0})
        assert hasattr(model, "fit")

    def test_build_pipeline_fit_predict(self, tiny_df):
        X = tiny_df.drop(TARGET, axis=1)
        y = tiny_df[TARGET]
        pipeline = build_pipeline({"model": "ridge", "alpha": 1.0})

        pipeline.fit(X, y)
        pred = pipeline.predict(X)

        assert len(pred) == len(X)

    def test_load_pipeline_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_pipeline(tmp_path / "no_model.pkl")


# Тесты сохранённой модели
class TestSavedModel:
    @pytest.mark.skipif(not MODEL_PATH.exists(), reason="Модель ещё не сохранена")
    def test_model_loads(self):
        model = joblib.load(MODEL_PATH)
        assert hasattr(model, "predict"), "Модель не имеет метода predict"
