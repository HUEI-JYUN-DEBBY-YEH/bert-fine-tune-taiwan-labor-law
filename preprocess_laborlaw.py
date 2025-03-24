import os
import json
import re

def preprocess_laborlaw_text(raw_text):
    # 切句：依據每一條條文為單位
    raw_sentences = re.split(r'\n|第\d+條', raw_text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    dataset = [{"id": i + 1, "text": sentence, "label": ""} for i, sentence in enumerate(sentences)]
    return dataset

def batch_process_txt_files(input_dir, output_path):
    all_sentences = []
    file_count = 0

    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
            processed = preprocess_laborlaw_text(raw_text)
            all_sentences.extend(processed)
            file_count += 1

    print(f"✅ 處理完成，共讀取 {file_count} 個檔案，輸出 {len(all_sentences)} 條句子。")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_sentences, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # 🗂️ 設定輸入資料夾與輸出路徑
    input_folder = "raw_laborlaw_txt"
    output_file = "laborlaw_sentences_unlabeled.json"

    # 🚀 執行批次處理
    batch_process_txt_files(input_folder, output_file)
