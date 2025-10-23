# train_model.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
import joblib
import os

def retrain_model():
    print("🔄 Starting model (re)training...")

    # Step 1: Load the main dataset
    base_file = "airline_requests.csv"
    if not os.path.exists(base_file):
        raise FileNotFoundError("❌ airline_requests.csv not found. Please provide the dataset first.")

    df = pd.read_csv(base_file)
    if "utterance" not in df.columns or "label" not in df.columns:
        raise ValueError("❌ CSV must contain 'utterance' and 'label' columns.")

    # Step 2: Add feedback data if exists
    feedback_file = "feedback_log.csv"
    if os.path.exists(feedback_file):
        feedback_df = pd.read_csv(feedback_file)
        # Rename to match base dataset
        feedback_df = feedback_df.rename(columns={"correct_intent": "label"})
        if "utterance" in feedback_df.columns and "label" in feedback_df.columns:
            df = pd.concat([df, feedback_df[["utterance", "label"]]], ignore_index=True)
            print(f"✅ Feedback data merged: {len(feedback_df)} new samples added.")
        else:
            print("⚠️ Feedback file exists but missing required columns — skipping merge.")

    print(f"📊 Total training samples: {len(df)}")

    # Step 3: Prepare data
    texts = df["utterance"].astype(str).tolist()
    labels = df["label"].astype(str).tolist()

    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)

    # Step 4: Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, encoded_labels, test_size=0.2, random_state=42
    )

    # Step 5: Tokenization
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)

    class AirlineDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = AirlineDataset(train_encodings, train_labels)
    val_dataset = AirlineDataset(val_encodings, val_labels)

    # Step 6: Load model (new or existing)
    model_path = "airline_distilbert_model"
    if os.path.exists(model_path):
        print("🔁 Loading existing model for retraining...")
        model = DistilBertForSequenceClassification.from_pretrained(
            model_path, num_labels=len(encoder.classes_)
        )
    else:
        print("🆕 Training new model from scratch...")
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=len(encoder.classes_)
        )

    # Step 7: Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_dir="./logs",
)

    # Step 8: Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Step 9: Train model
    trainer.train()

    # Step 10: Save everything
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    joblib.dump(encoder, "label_encoder.pkl")

    print("✅ Training complete. Model, tokenizer, and encoder saved successfully.")


if __name__ == "__main__":
    retrain_model()
