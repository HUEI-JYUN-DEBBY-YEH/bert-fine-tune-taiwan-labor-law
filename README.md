This repository focuses on fine-tuning a BERT model for classifying clauses within Taiwan's Labor Standards Act. The model is trained to categorize legal texts into predefined classes, facilitating efficient legal document analysis and information retrieval.

# 🇹🇼 BERT Fine-tune for Taiwan Labor Law Classification  
微調 BERT 中文模型應用於台灣勞基法分類任務  
[→ 🔗 Hugging Face 模型卡連結](https://huggingface.co/DEBBY-YEH/finetuned-laborlaw-bert)

---

## 📘 專案介紹 Project Overview

本專案以 `bert-base-chinese` 為基礎，針對台灣勞基法問題句進行 fine-tuning 微調訓練，達成分類任務，預測問題所屬法條主題分類，支援應用於 AI Chatbot 智能問答場景。

> This project fine-tunes a Chinese BERT model for multi-class classification on Taiwan Labor Law QA data. The goal is to classify user questions into predefined legal topics and support intelligent response systems such as chatbots.

---

## 🛠 訓練與模型說明 Model & Training

- **Base model**：`bert-base-chinese` (via Hugging Face Transformers)
- **Fine-tune dataset**：台灣勞基法 QA 語料，涵蓋 8 大類別
- **Loss Function**：CrossEntropyLoss
- **Optimizer**：AdamW
- **Framework**：Transformers + PyTorch + Trainer API
- **前處理**：(1)條文句子斷句，去除特殊符號與空白、(2)以人工方式標註為 7～8 類主題標籤、(3)轉為 Hugging Face 格式進行訓練
- **Label 分類**：
  1. 工時
  2. 薪資
  3. 假別
  4. 契約與聘僱關係
  5. 終止與解僱
  6. 職場安全與性別平等
  7. 其他綜合規範
  
- 訓練腳本 `train_finetune_trainer.py` 可快速再訓練本模型。

## 🧾 資料結構 Data Files
  ├── raw_laborlaw_txt/              # 勞基法原始資料（未上傳 GitHub）
  ├── laborlaw_sentences_labeled.csv # 已標註的句子資料
  ├── laborlaw_dataset.json          # 訓練用 JSON 格式
  ├── label2id.json                  # 類別對應表

## 📂 評估檔案：
- classification_report.json：Precision / Recall / F1
- laborlaw_predicted.csv：預測結果與 Ground Truth 比較
---

## 🔍 使用方式 How to Use

### 🖥 CLI 測試：  
```bash
python inference.py --text "我想請育嬰留停"
```

### 📤 輸出結果：
```
預測結果：職場安全與性別平等
信心分數：0.313
```
![](./demo_output.png)

### 📦 檔案說明：
- `inference.py`：推論腳本，支援 CLI 測試與 API 整合
- `label2id.json`：標籤與索引對應表
- `model/`：包含微調後的 `.safetensors` 模型與 config

---

## 🤖 應用延伸：整合 AI Chatbot 法規問答

本模型已成功整合至下列專案：

👉 [🔗 AI Chatbot 法規問答系統 GitHub Repo](https://github.com/HUEI-JYUN-DEBBY-YEH/AI_Chatbot)

在該專案中，本模型擔任語意理解模組（NLU），用於判別使用者問題主題並導引向量檢索階段，進一步提供最相關法條條文／解釋函。

---

## 🧠 模型託管｜Hugging Face Model Hosting

🔗 [模型卡 Model Card 連結](https://huggingface.co/DEBBY-YEH/finetuned-laborlaw-bert)  
包含以下資訊：
- 模型摘要與訓練背景
- 使用語言與任務
- 標籤分類詳解
- 推論方式與限制說明

---

## ✅ TODO｜後續待辦與優化方向

- [ ] 整合 Flask Web API 版本，方便應用部署
- [ ] 於 Hugging Face Space 建立 Web Demo 展示頁
- [ ] 提供前端頁面或 Chat UI 與模型對接
- [ ] 擴增 QA 訓練資料與語義標註粒度
- [ ] 評估多標籤分類 / 長文本處理支援

---
