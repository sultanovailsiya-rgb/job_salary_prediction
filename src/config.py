"""Общие пути и константы проекта."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "job_salary_prediction_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "final_pipeline.pkl"
RESULTS_DIR = PROJECT_ROOT / "results"

TARGET = "salary"
RANDOM_STATE = 42
