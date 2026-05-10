import pandas as pd
import numpy as np
import os

def run_eda(df: pd.DataFrame, target: str, output_path: str = 'results/eda_report.txt'):

    import numpy as np
        
    report = []
    report.append(f"=== EDA Report ===\nDataset shape: {df.shape}\n")
    
    # 1. Пропуски
    print("\n1. Пропущенные значения:")
    report.append("1. MISSING VALUES:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({'count': missing, 'percent': missing_pct})
    missing_df = missing_df[missing_df['count'] > 0]
    
    if len(missing_df) == 0:
        print("Пропусков нет")
        report.append("No missing values\n")
    else:
        print(missing_df.to_string())
        report.append(missing_df.to_string() + "\n")
    
    # 2. Дубликаты
    print("\n2. Дубликаты:")
    report.append("2. DUPLICATES:")
    full_dupes = df.duplicated().sum()
    print(f"Полных дубликатов строк: {full_dupes}")
    report.append(f"Full row duplicates: {full_dupes}\n")
    
    # 3. Уникальные значения категориальных признаков
    print("\n3. Категориальные признаки (топ-5 значений):")
    report.append("3. CATEGORICAL FEATURES (top-5):")
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        if col != target:
            top_vals = df[col].value_counts().head(5)
            print(f"   {col}: {dict(top_vals)}")
            report.append(f"   {col}: {dict(top_vals)}")
    report.append("")
    
    # 4. Статистика по числовым признакам
    print("\n4. Статистика по числовым признакам:")
    report.append("4. NUMERICAL FEATURES STATISTICS:")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target in num_cols:
        num_cols.remove(target)
    
    for col in num_cols:
        stats = df[col].describe()
        print(f"\n   {col}:")
        print(f"      mean={stats['mean']:.2f}, std={stats['std']:.2f}")
        print(f"      min={stats['min']:.2f}, max={stats['max']:.2f}")
        report.append(f"\n   {col}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}")
    report.append("")
    
    # 5. Выбросы в целевой переменной (метод IQR)
    print(f"\n5. Выбросы в целевой переменной '{target}' (метод IQR):")
    report.append(f"5. OUTLIERS IN TARGET '{target}' (IQR method):")
    
    Q1 = df[target].quantile(0.25)
    Q3 = df[target].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[target] < lower_bound) | (df[target] > upper_bound)]
    outlier_pct = len(outliers) / len(df) * 100
    
    print(f"   IQR: {IQR:,.0f}")
    print(f"   Границы: [{lower_bound:,.0f}, {upper_bound:,.0f}]")
    print(f"   Выбросов: {len(outliers)} ({outlier_pct:.2f}%)")
    
    report.append(f"   IQR: {IQR:,.0f}")
    report.append(f"   Bounds: [{lower_bound:,.0f}, {upper_bound:,.0f}]")
    report.append(f"   Outliers: {len(outliers)} ({outlier_pct:.2f}%)\n")
    
    # 6. Выбросы в числовых признаках (опционально)
    print("\n6. Выбросы в числовых признаках (IQR):")
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
            print(f"   {col}: {n_outliers} выбросов ({pct:.2f}%)")
            report.append(f"   {col}: {n_outliers} outliers ({pct:.2f}%)")
    report.append("")
    
    # 7. Корреляции (только числовые)
    print("\n7. Корреляция признаков с целевой переменной:")
    report.append("7. CORRELATION WITH TARGET:")
    num_df = df.select_dtypes(include=[np.number])
    if target in num_df.columns:
        corr = num_df.corr()[target].drop(target).sort_values(ascending=False)
        for feat, val in corr.items():
            print(f"   {feat}: {val:+.3f}")
            report.append(f"   {feat}: {val:+.3f}")
    
    # Сохранение отчёта
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
        
    return {
        'missing_count': missing.sum(),
        'duplicates': full_dupes,
        'target_outliers_pct': outlier_pct,
        'correlations': corr.to_dict() if 'corr' in locals() else {}
    }