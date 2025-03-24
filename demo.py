import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
from plot_metrics import plot_classification_report
from PIL import Image

# === 模型與 tokenizer 路徑 ===
MODEL_PATH = "finetuned_laborlaw_model"
LABEL_MAPPING_PATH = './label2id.json'
REPORT_PATH = './classification_report.json'  # 預先儲存的 dict

# === 載入 tokenizer & 模型 ===
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

@st.cache_data
def load_labels():
    with open(LABEL_MAPPING_PATH, 'r', encoding='utf-8') as f:
        label2id = json.load(f)
    id2label = {v: k for k, v in label2id.items()}
    return id2label

# === 預測邏輯 ===
def predict(text, tokenizer, model, id2label):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1).squeeze()
    top_id = torch.argmax(probs).item()
    return id2label[top_id], probs[top_id].item()

# === 介面開始 ===
st.title("🧠 勞基法文本分類 Demo")
st.markdown("本模型為 fine-tuned BERT，支援多類別文本分類")

tokenizer, model = load_model()
id2label = load_labels()

text_input = st.text_area("請輸入勞基法相關文本：", height=150)

if st.button("預測類別"):
    if text_input.strip():
        label, score = predict(text_input, tokenizer, model, id2label)
        st.success(f"**預測類別：{label}**（信心度：{score:.2f}）")
    else:
        st.warning("請先輸入文本")

# === 顯示評估圖表 ===
with st.expander("📊 查看模型分類表現"):
    with open(REPORT_PATH, 'r', encoding='utf-8') as f:
        report = json.load(f)
    plot_classification_report(report)
    st.image(Image.open("metrics_plot.png"), caption="Classification Report")

st.markdown("---")
st.caption("© 2025 勞基法 NLP 分類專案 | Powered by BERT")