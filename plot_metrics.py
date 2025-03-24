# plot_metrics.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ✅ 指定中文字型（避免亂碼或方框）
plt.rcParams['font.family'] = 'Microsoft JhengHei'  # 適用於 Windows，如用 macOS 可改為 'Arial Unicode MS'

def plot_classification_report(report_dict, save_path='metrics_plot.png'):
    """
    report_dict: sklearn classification_report(output_dict=True)
    save_path: where to save the plot
    """
    # 選出每個類別（過濾掉 macro avg / weighted avg / accuracy）
    filtered = {
        label: values
        for label, values in report_dict.items()
        if label not in ("accuracy", "macro avg", "weighted avg")
    }

    df = pd.DataFrame.from_dict(filtered, orient="index")
    df = df[["precision", "recall", "f1-score"]]

    # 繪圖
    plt.figure(figsize=(10, 6))
    df.plot(kind="bar", ylim=(0, 1.05), legend=True)
    plt.title("📊 分類指標報告（各類別）", fontsize=14)
    plt.ylabel("分數", fontsize=12)
    plt.xlabel("類別", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
