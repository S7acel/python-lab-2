"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput

FEATURES = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]
TARGET = "MRI_Count"


def _compute_group_correlations(group_df: pd.DataFrame) -> dict[str, float]:
    """Вычисляет корреляции Пирсона заданных признаков с MRI_Count."""
    corr_series = group_df[FEATURES + [TARGET]].corr(method="pearson")[TARGET]
    return {feat: float(corr_series[feat]) for feat in FEATURES}


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt.

    Разделите наблюдения по полу и для каждой группы вычислите корреляции
    признаков FSIQ, VIQ, PIQ, Weight, Height с MRI_Count методом Пирсона.
    В strongest_mri_feature верните название признака с наибольшим модулем
    корреляции с MRI_Count среди объединённых результатов двух групп.
    """
    df = pd.read_csv(data.csv_path, sep="\t", na_values=["NA"])

    women_df = df[df["Gender"] == "Female"]
    men_df = df[df["Gender"] == "Male"]

    women_corr = _compute_group_correlations(women_df)
    men_corr = _compute_group_correlations(men_df)

    strongest_feature = max(
        FEATURES,
        key=lambda feat: max(abs(women_corr[feat]), abs(men_corr[feat])),
    )

    return BrainCorrelationSummary(
        men_count=int(len(men_df)),
        women_count=int(len(women_df)),
        women_mri_correlation=women_corr,
        men_mri_correlation=men_corr,
        strongest_mri_feature=strongest_feature,
    )