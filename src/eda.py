import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RESULTS_DIR

logger = logging.getLogger(__name__)


def run_eda(
    df: pd.DataFrame, target: str, output_path: Path = RESULTS_DIR / "eda_report.txt"
) -> dict:
    """Строит текстовый EDA-отчёт, пишет его в файл и логирует ключевые факты.

    Args:
        df: Датафрейм с признаками и целевой переменной.
        target: Название целевой колонки.
        output_path: Путь для сохранения отчёта.

    Returns:
        Словарь со сводкой: пропуски, дубликаты, доля выбросов, корреляции.
    """

    report = []
    report.append(f"=== EDA Report ===\nDataset shape: {df.shape}\n")

    # 1. Пропуски
    logger.info("\n1. Пропущенные значения:")
    report.append("1. MISSING VALUES:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"count": missing, "percent": missing_pct})
    missing_df = missing_df[missing_df["count"] > 0]

    if len(missing_df) == 0:
        logger.info("Пропусков нет")
        report.append("No missing values\n")
    else:
        logger.info(missing_df.to_string())
        report.append(missing_df.to_string() + "\n")

    # 2. Дубликаты
    logger.info("\n2. Дубликаты:")
    report.append("2. DUPLICATES:")
    full_dupes = df.duplicated().sum()
    logger.info(f"Полных дубликатов строк: {full_dupes}")
    report.append(f"Full row duplicates: {full_dupes}\n")

    # 3. Уникальные значения категориальных признаков
    logger.info("\n3. Категориальные признаки (топ-5 значений):")
    report.append("3. CATEGORICAL FEATURES (top-5):")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        if col != target:
            top_vals = df[col].value_counts().head(5)
            logger.info(f"   {col}: {dict(top_vals)}")
            report.append(f"   {col}: {dict(top_vals)}")
    report.append("")

    # 4. Статистика по числовым признакам
    logger.info("\n4. Статистика по числовым признакам:")
    report.append("4. NUMERICAL FEATURES STATISTICS:")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target in num_cols:
        num_cols.remove(target)

    for col in num_cols:
        stats = df[col].describe()
        logger.info(f"\n   {col}:")
        logger.info(f"      mean={stats['mean']:.2f}, std={stats['std']:.2f}")
        logger.info(f"      min={stats['min']:.2f}, max={stats['max']:.2f}")
        report.append(
            f"\n   {col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, "
            f"min={stats['min']:.2f}, max={stats['max']:.2f}"
        )
    report.append("")

    # 5. Выбросы в целевой переменной (метод IQR)
    logger.info(f"\n5. Выбросы в целевой переменной '{target}' (метод IQR):")
    report.append(f"5. OUTLIERS IN TARGET '{target}' (IQR method):")

    Q1 = df[target].quantile(0.25)
    Q3 = df[target].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[target] < lower_bound) | (df[target] > upper_bound)]
    outlier_pct = len(outliers) / len(df) * 100

    logger.info(f"   IQR: {IQR:,.0f}")
    logger.info(f"   Границы: [{lower_bound:,.0f}, {upper_bound:,.0f}]")
    logger.info(f"   Выбросов: {len(outliers)} ({outlier_pct:.2f}%)")

    report.append(f"   IQR: {IQR:,.0f}")
    report.append(f"   Bounds: [{lower_bound:,.0f}, {upper_bound:,.0f}]")
    report.append(f"   Outliers: {len(outliers)} ({outlier_pct:.2f}%)\n")

    # 6. Выбросы в числовых признаках (опционально)
    logger.info("\n6. Выбросы в числовых признаках (IQR):")
    report.append("6. OUTLIERS IN NUMERICAL FEATURES:")
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        n_outliers = len(df[(df[col] < lower) | (df[col] > upper)])
        pct = n_outliers / len(df) * 100
        if pct > 0:
            logger.info(f"   {col}: {n_outliers} выбросов ({pct:.2f}%)")
            report.append(f"   {col}: {n_outliers} outliers ({pct:.2f}%)")
    report.append("")

    # 7. Корреляции (только числовые)
    logger.info("\n7. Корреляция признаков с целевой переменной:")
    report.append("7. CORRELATION WITH TARGET:")
    num_df = df.select_dtypes(include=[np.number])
    if target in num_df.columns:
        corr = num_df.corr()[target].drop(target).sort_values(ascending=False)
        for feat, val in corr.items():
            logger.info(f"   {feat}: {val:+.3f}")
            report.append(f"   {feat}: {val:+.3f}")

    # Сохранение отчёта
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    return {
        "missing_count": missing.sum(),
        "duplicates": full_dupes,
        "target_outliers_pct": outlier_pct,
        "correlations": corr.to_dict() if "corr" in locals() else {},
    }
