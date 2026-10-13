import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from grader_contracts.pandas_tasks import TitanicInput
from grader_contracts.correlation_tasks import BrainDataInput
from pandas_tasks import analyze_titanic
from correlation_tasks import analyze_brain_correlations

if __name__ == "__main__":
    print("=== ЧАСТЬ 1: TITANIC ===")
    titanic_res = analyze_titanic(TitanicInput(csv_path="titanic.csv"))
    print(titanic_res)

    print("\n=== ЧАСТЬ 2: BRAINSIZE ===")
    brain_res = analyze_brain_correlations(BrainDataInput(csv_path="brainsize.txt"))
    print(brain_res)

    # Построение и показ графиков
    df_brain = pd.read_csv("brainsize.txt", sep="\t", na_values="NA")
    sns.set_theme(style="whitegrid")
    cols = ["FSIQ", "VIQ", "PIQ", "Weight", "Height", "MRI_Count"]

    # 1. Окно с тепловыми картами (Heatmap)
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(
        df_brain[df_brain["Gender"] == "Female"][cols].corr(),
        annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=axes1[0]
    )
    axes1[0].set_title("Матрица корреляций (Female)")

    sns.heatmap(
        df_brain[df_brain["Gender"] == "Male"][cols].corr(),
        annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=axes1[1]
    )
    axes1[1].set_title("Матрица корреляций (Male)")
    plt.tight_layout()
    fig1.savefig("brain_heatmaps.png", dpi=150)

    # 2. Окно с точечными графиками (Scatter)
    features = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]
    fig2, axes2 = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
    for i, feat in enumerate(features):
        sns.scatterplot(
            data=df_brain, x=feat, y="MRI_Count",
            hue="Gender", style="Gender", s=70, ax=axes2[i]
        )
        axes2[i].set_title(f"{feat} vs MRI_Count")
    plt.tight_layout()
    fig2.savefig("brain_scatter.png", dpi=150)

    print("\nГрафики сохранены в файлы brain_heatmaps.png и brain_scatter.png.")
    print("Открываю окна с графиками...")
    plt.show()