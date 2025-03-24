import pandas as pd
import json
from datasets import Dataset
from transformers import AutoTokenizer

def build_label2id(labels):
    unique_labels = sorted(set(labels))
    return {label: idx for idx, label in enumerate(unique_labels)}

def main():
    # 輸入檔案路徑
    csv_path = "laborlaw_sentences_labeled.csv"  # 標註完成的 CSV
    model_name = "bert-base-chinese"             # 可替換為其他模型名稱

    # 讀取 CSV
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["text", "label"])     # 移除缺 label 資料

    # 建立 label2id 對應表
    label2id = build_label2id(df["label"])
    print("✅ label2id 映射：", label2id)

    # 將文字 label 轉為數值
    df["label_id"] = df["label"].map(label2id)

    # 建立 Hugging Face Dataset
    dataset = Dataset.from_pandas(df[["text", "label_id"]])

    # 載入 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 定義 tokenizer 函式
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    def preprocess(example):
        encoding = tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )
        encoding["labels"] = example["label"]  # ✅ 加這一行
        return encoding

    # 對 dataset 做 tokenizer 處理
    def tokenize_and_add_labels(example):
        encoding = tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )
        encoding["labels"] = example["label_id"]  # 🔥 加入 labels 欄位
        return encoding

    tokenized_dataset = dataset.map(tokenize_and_add_labels)

    print("✅ Tokenizer 處理完成")
    print(tokenized_dataset.column_names)
    # 應該包含：['input_ids', 'token_type_ids', 'attention_mask', 'labels']
    print(tokenized_dataset[0])

    # 分割 train/test（80/20）
    split_dataset = tokenized_dataset.train_test_split(test_size=0.2)
    split_dataset.save_to_disk("tokenized_dataset")

    # 儲存 label2id 對應表
    with open("label2id.json", "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)
    print("✅ Dataset 與 label2id 已儲存完畢")

if __name__ == "__main__":
    main()
