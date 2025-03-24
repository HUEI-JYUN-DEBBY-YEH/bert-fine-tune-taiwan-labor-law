import json
import pandas as pd

def json_to_csv_with_label(json_path, csv_output):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 加入空白 label 欄位
    for item in data:
        item["label"] = ""

    df = pd.DataFrame(data)
    df.to_csv(csv_output, index=False, encoding="utf-8-sig")
    print(f"✅ 轉換完成，已輸出至: {csv_output}")

if __name__ == "__main__":
    json_file = "laborlaw_sentences_unlabeled.json"     # 原始句子資料
    csv_output = "laborlaw_sentences_labeled.csv"       # 輸出的標註用 CSV
    json_to_csv_with_label(json_file, csv_output)
