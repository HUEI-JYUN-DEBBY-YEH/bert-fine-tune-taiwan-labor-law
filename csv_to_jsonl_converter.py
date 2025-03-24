import pandas as pd
import json

def csv_to_jsonl(csv_path, output_path):
    df = pd.read_csv(csv_path)

    # 確保有 text 與 label 欄位
    assert 'text' in df.columns and 'label' in df.columns, "CSV 必須包含 'text' 和 'label' 欄位"

    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            item = {
                "text": row["text"],
                "label": row["label"]
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ 轉換完成，已輸出至: {output_path}")

if __name__ == "__main__":
    csv_file = "laborlaw_sentences_labeled.csv"       # 請替換為你標註完成的 CSV 檔
    jsonl_output = "laborlaw_dataset.jsonl"            # 將產出這個 JSONL 檔
    csv_to_jsonl(csv_file, jsonl_output)
