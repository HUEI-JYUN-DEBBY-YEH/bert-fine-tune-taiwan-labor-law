import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 模型與標籤設定
MODEL_NAME = "DEBBY-YEH/finetuned-laborlaw-bert"
LABELS = ["假別", "其他", "契約與聘僱關係", "工時", "終止與解僱", "職場安全與性別平等", "薪資"]


# 載入模型與 tokenizer
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

# 推論函數
def predict(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    predicted_class_id = logits.argmax().item()
    confidence = probs[0][predicted_class_id].item()
    return LABELS[predicted_class_id], round(confidence, 3)


# 主程式入口：可接受 CLI 輸入
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True, help="欲分類之勞基法問句")
    args = parser.parse_args()

    tokenizer, model = load_model()
    label, confidence = predict(args.text, tokenizer, model)
    print(f"預測結果：{label}")
    print(f"信心分數：{confidence}")


if __name__ == "__main__":
    main()
