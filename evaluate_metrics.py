# evaluate_model.py
import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1️⃣ Load model, tokenizer, encoder
model_path = "airline_distilbert_model"
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
encoder = joblib.load("label_encoder.pkl")

# 2️⃣ Load test dataset
df = pd.read_csv("airline_requests.csv")  # or a separate test CSV
texts = df["utterance"].tolist()
labels = df["label"].tolist()

# 3️⃣ Encode labels
y_true = encoder.transform(labels)

# 4️⃣ Predict
y_pred = []
model.eval()
with torch.no_grad():
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        pred_index = torch.argmax(outputs.logits, dim=1).item()
        y_pred.append(pred_index)

# 5️⃣ Evaluation metrics
print("Accuracy:", accuracy_score(y_true, y_pred))
print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=encoder.classes_))
print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))
