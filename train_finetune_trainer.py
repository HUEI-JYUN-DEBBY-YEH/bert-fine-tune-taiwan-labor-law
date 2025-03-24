from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer
from datasets import load_from_disk
import json

# 載入 tokenizer 資訊
model_name = "bert-base-chinese"  # 可替換為其他中文模型
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 載入 tokenized 資料集與 label2id 對應表
dataset = load_from_disk("tokenized_dataset")
with open("label2id.json", "r", encoding="utf-8") as f:
    label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}

# 載入模型，設定類別數與對應標籤
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# 設定訓練參數
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

# 定義評估指標（這裡用準確率）
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="macro")
    }

# 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics
)

# 開始訓練
trainer.train()

# 儲存模型與 tokenizer
model.save_pretrained("finetuned_laborlaw_model")
tokenizer.save_pretrained("finetuned_laborlaw_model")
