
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import json
from plot_metrics import plot_classification_report
import matplotlib.pyplot as plt

# 模型路徑
model_path = "finetuned_laborlaw_model"

# 載入模型與 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
plt.rcParams['font.family'] = 'Microsoft JhengHei'

# 建立 pipeline
clf = pipeline("text-classification", model=model, tokenizer=tokenizer)

# label2id 映射表（與訓練時保持一致）
label2id = {
  "假別": 0,
  "其他": 1,
  "契約與聘僱關係": 2,
  "工時": 3,
  "終止與解僱": 4,
  "職場安全與性別平等": 5,
  "薪資": 6
}
id2label = {v: k for k, v in label2id.items()}

# 讀取已標註資料
df = pd.read_csv("laborlaw_sentences_labeled.csv")
df_eval = df[df["label"].notna()].copy()

# 執行預測
preds = clf(list(df_eval["text"]), truncation=True)

# 將中文標籤名稱轉換為 id
df_eval["label_id"] = df_eval["label"].map(label2id)
df_eval["pred_label_name"] = [p["label"] for p in preds]
df_eval["pred_label_id"] = df_eval["pred_label_name"].map(label2id)

# 儲存結果
df_eval.to_csv("laborlaw_predicted.csv", index=False, encoding='utf-8-sig')

# 去除有缺值的樣本
df_eval = df_eval.dropna(subset=["label_id", "pred_label_id"])

# 計算指標
y_true = df_eval["label_id"]
y_pred = df_eval["pred_label_id"]

report = classification_report(
    y_true, y_pred,
    target_names=[id2label[i] for i in sorted(id2label)],
    output_dict=True
)

print("\n🔍 Classification Report:")
print(classification_report(y_true, y_pred, target_names=[id2label[i] for i in sorted(id2label)]))

print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# 儲存為 JSON
with open("classification_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

# 繪圖
plot_classification_report(report, save_path="metrics_plot.png")

print("✅ 完成評估報告與圖表生成")