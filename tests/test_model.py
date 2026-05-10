import pytest
import pandas as pd
import json
import os
import joblib

# Пути (автоматически находим корень проекта)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'job_salary_prediction_dataset.csv')
METRICS_PATH = os.path.join(PROJECT_ROOT, 'results', 'metrics.json')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'final_pipeline.pkl')

# Тесты для проверки целостности данных
class TestData:    
    def test_file_exists(self):
        assert os.path.exists(DATA_PATH), f"Файл {DATA_PATH} не найден"
    
    def test_required_columns(self):
        df = pd.read_csv(DATA_PATH)
        required = ['job_title', 'experience_years', 'education_level', 
                   'skills_count', 'industry', 'company_size', 
                   'location', 'remote_work', 'certifications', 'salary']
        for col in required:
            assert col in df.columns, f"Отсутствует обязательная колонка: {col}"
    
    def test_salary_valid(self):
        df = pd.read_csv(DATA_PATH)
        salaries = df['salary'].dropna()
        assert (salaries > 0).all(), "Есть неположительные зарплаты"
        assert salaries.median() < 500_000, "Медиана зарплаты нереалистична"


# Тесты для проверки работы модели и метрик
class TestModel:    
    @pytest.mark.skipif(not os.path.exists(METRICS_PATH), reason="Метрики ещё не сгенерированы")
    def test_metrics_valid(self):
        with open(METRICS_PATH, 'r', encoding='utf-8') as f:
            m = json.load(f)
        assert m['r2'] > 0, "R² <= 0 — модель бесполезна"
        assert m['mape_percent'] < 30, f"MAPE слишком высок: {m['mape_percent']}%"
        assert m['rmse'] > 0, "RMSE должен быть положительным"
    
    @pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Модель ещё не сохранена")
    def test_model_loads(self):
        model = joblib.load(MODEL_PATH)
        assert hasattr(model, 'predict'), "Модель не имеет метода predict"