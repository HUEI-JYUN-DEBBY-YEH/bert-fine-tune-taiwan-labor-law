### 🧑‍⚖️ BERT Fine-Tuning for Taiwan Labor Law Classification
This project fine-tunes a pre-trained BERT model to classify legal text and inquiries under the Taiwan Labor Standards Act (LSA). It enables category prediction for labor-related questions, supporting the development of AI-powered legal assistants or HR tools for compliance and consultation.

## 📌 Why This Project Matters
Navigating labor law is often difficult for both employers and employees. Traditional legal chatbots struggle with low-resource domains and context-specific laws.
This project demonstrates how fine-tuning BERT on a targeted legal domain—Taiwan’s LSA—can enable more accurate classification, interpretability, and human-aligned decision support in the HR and legal tech landscape.

## 🧾 Dataset Overview
# Source:
- Manually annotated clauses from the Taiwan Labor Standards Act
- 50 test queries generated via GPT-4 to simulate common labor-related HR questions
# Format:
- CSV with the following fields: text, category, label, context
- Used for multi-class classification
# Classification Categories (7 classes):
- Working Hour
- Wage
- Leaves
- Employment (Contract & Relations)
- Termination
- Workplace Safety and Gender Equality
- Others (General Provisions)

## ⚙️ Model Training
- Base model: bert-base-chinese
- Training method: Simple fine-tuning using HuggingFace's Trainer API
- Training samples: 500+ annotated legal texts
- Evaluation set: 50 GPT-generated questions labeled by hand

## 📊 Results
# Model	Accuracy
- Fine-tuned BERT	90%
- High precision in classifying legal topics from real-world questions
- Particularly strong in handling legal intent and sentence structure in Mandarin
- Weaknesses observed in ambiguous or multi-label phrasing, indicating potential for future multi-label extension

## 🧪 Usage
Inference Example
``bash
python inference.py --text "每日工作不得超過幾小時？"

``Output:
Predicted Category: Working Hour
Confidence Score: 0.88

## 🧭 System Architecture
[Input Text] → [Tokenizer] → [Fine-tuned BERT] → [Softmax Classifier] → [Category Label]

## 🛠 Tools & Libraries
- Hugging Face Transformers
- PyTorch
- Pandas, Scikit-learn
- Jupyter Notebook for evaluation

## 🔍 Key Insights
- Fine-tuning BERT in Mandarin for legal domains is feasible and effective with limited data.
- Even a small labeled dataset can yield high-accuracy performance when domain-aligned.
- Combining classification with retrieval (e.g., FAISS) may further enhance answer generation in a chatbot setting.

## 🧱 Related Projects
- AI Chatbot: Taiwan Labor Law QA System – integrates this classifier into a working chatbot (https://github.com/HUEI-JYUN-DEBBY-YEH/AI_Chatbot)
- Medium Article: Building a Legal Chatbot with BERT (https://medium.com/@debby.yeh1994/bert-%E4%B8%AD%E6%96%87%E5%88%86%E9%A1%9E%E5%AF%A6%E4%BD%9C-%E6%89%93%E9%80%A0%E5%8F%B0%E7%81%A3%E5%8B%9E%E5%9F%BA%E6%B3%95-chatbot-%E6%99%BA%E8%83%BD%E6%A0%B8%E5%BF%83-e6c7c72f82de)

## 👩‍💻 Author
Debby Yeh
NLP Application Engineer
Specialized in: LLM, legal NLP, HR automation, chatbot systems
🔗 Portfolio(https://www.notion.so/Debby-Yeh-Portfolio-1ca5118474d2801caa58de564fb53e38)
