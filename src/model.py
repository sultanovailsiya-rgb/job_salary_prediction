import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

NUMERICAL_COLS = ['experience_years', 'skills_count', 'certifications']
CATEGORICAL_COLS = ['job_title', 'education_level', 'industry', 'company_size', 'location', 'remote_work']

# Автоматическая предобработка признаков
def create_preprocessor():
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_COLS),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_COLS)
        ],
        remainder='drop'
    )

# Создание модели на основе лучших параметров Optuna
def create_model(best_params: dict):
    model_type = best_params['model']
    if model_type == 'gb':
        return GradientBoostingRegressor(
            n_estimators=best_params['n_estimators'],
            learning_rate=best_params['learning_rate'],
            max_depth=best_params['max_depth'],
            random_state=42
        )
    elif model_type == 'rf':
        return RandomForestRegressor(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            random_state=42,
            n_jobs=-1
        )
    else:  # ridge
        return Ridge(alpha=best_params['alpha'])

# Сборка полного ML-пайплайна
def build_pipeline(best_params: dict) -> Pipeline:
    return Pipeline([
        ('preprocessor', create_preprocessor()),
        ('model', create_model(best_params))
    ])

def save_pipeline(pipeline: Pipeline, path: str = 'models/final_pipeline.pkl'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)

def load_pipeline(path: str = 'models/final_pipeline.pkl') -> Pipeline:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Модель не найдена: {path}")
    return joblib.load(path)