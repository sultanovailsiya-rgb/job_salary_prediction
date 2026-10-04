from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE, RESULTS_DIR, TARGET
from src.model import CATEGORICAL_COLS, NUMERICAL_COLS


def generate_plots(
    df: pd.DataFrame, pipeline: Pipeline, output_dir: Path = RESULTS_DIR / "plots"
) -> None:
    """Генерирует и сохраняет графики: важность признаков, прогноз, распределения.

    Args:
        df: Датафрейм с признаками и целевой переменной.
        pipeline: Обученный пайплайн с шагами preprocessor и model.
        output_dir: Каталог для сохранения PNG-файлов.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Быстрая выборка для scatter plot
    sample = df.sample(5000, random_state=RANDOM_STATE)
    X_sample = sample.drop(TARGET, axis=1)
    y_true = sample[TARGET]
    y_pred = pipeline.predict(X_sample)

    # 1. Feature Importance
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    ohe = preprocessor.named_transformers_["cat"]

    cat_features = ohe.get_feature_names_out(CATEGORICAL_COLS)
    all_features = list(NUMERICAL_COLS) + list(cat_features)

    importances = pd.Series(model.feature_importances_, index=all_features)
    top_10 = importances.sort_values(ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_10.values, y=top_10.index, color="#2ca02c")
    plt.title("Top 10 Feature Importance (Gradient Boosting)")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=300)
    plt.close()

    # 2. Actual vs Predicted
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, s=20, color="#1f77b4")
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], "r--", lw=2, label="Ideal Prediction")
    plt.xlabel("Actual Salary ($)")
    plt.ylabel("Predicted Salary ($)")
    plt.title("Actual vs Predicted Salary")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "actual_vs_predicted.png", dpi=300)
    plt.close()

    # 3. Salary Distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(df[TARGET], kde=True, bins=60, color="#ff7f0e")
    plt.title("Annual Salary Distribution")
    plt.xlabel("Salary ($)")
    plt.tight_layout()
    plt.savefig(output_dir / "salary_distribution.png", dpi=300)
    plt.close()

    # 4. Salary vs Experience
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df, x="experience_years", y=TARGET, color="#9467bd", showfliers=False
    )
    plt.title("Salary vs Experience (Years)")
    plt.xlabel("Experience (Years)")
    plt.ylabel("Salary ($)")
    plt.tight_layout()
    plt.savefig(output_dir / "salary_vs_experience.png", dpi=300)
    plt.close()
