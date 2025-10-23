from flask import Flask, request, jsonify, render_template
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import pandas as pd
import joblib
import os
import threading
import time
import sqlite3
from train_model import retrain_model

app = Flask(__name__)

# -----------------------------
# Model, tokenizer, encoder
# -----------------------------
model_path = "airline_distilbert_model"
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
encoder = joblib.load("label_encoder.pkl")

# -----------------------------
# SQLite DB setup for chat history
# -----------------------------
DB_FILE = "chat_history.db"
ARCHIVE_DIR = "chat_archive"
MAX_LENGTH = 500
os.makedirs(ARCHIVE_DIR, exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender TEXT,
            message TEXT,
            intent TEXT,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_chat(sender, message, intent=None, confidence=None):
    truncated_message = message[:MAX_LENGTH]
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chats (sender, message, intent, confidence) VALUES (?, ?, ?, ?)",
        (sender, truncated_message, intent, confidence)
    )
    chat_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    # Save full message in archive
    archive_file = os.path.join(ARCHIVE_DIR, f"{chat_id}_{sender}.txt")
    with open(archive_file, "w", encoding="utf-8") as f:
        f.write(message)

init_db()

# -----------------------------
# Feedback storage
# -----------------------------
FEEDBACK_FILE = "feedback_log.csv"
ARCHIVE_FILE = "archived_feedback.csv"
if os.path.exists(FEEDBACK_FILE):
    feedback_df = pd.read_csv(FEEDBACK_FILE)
else:
    feedback_df = pd.DataFrame(columns=["utterance", "predicted_intent", "correct_intent"])
    feedback_df.to_csv(FEEDBACK_FILE, index=False)

# -----------------------------
# Responses dictionary
# -----------------------------
responses = {
    "Cancel Trip": "Sure! Please share your booking ID so we can cancel your trip.",
    "Change Flight": "No problem! Tell me your booking number and preferred date/time.",
    "Flight Status": "Let me check... please provide your flight number.",
    "Missing Bag": "I’m sorry to hear that! Could you share your baggage tag number?",
    "Complaints": "We’re sorry for the inconvenience. Please describe your issue briefly.",
    "Discounts": "Current offers are listed on our promotions page. Would you like the link?",
    "Insurance": "You can add insurance during booking or view available plans here.",
    "Carry On Luggage Faq": "Hand luggage must be under 7kg and fit overhead storage.",
    "Check In Luggage Faq": "Each passenger can check in up to 15kg for free.",
    "Pet Travel": "Yes, we allow pets! Please provide breed and size details.",
    "Damaged Bag": "We’ll help you with that. Please visit the baggage claim desk immediately.",
    "Medical Policy": "We offer special assistance for medical needs. Need a wheelchair?",
    "Fare Check": "Sure! Please share your source and destination.",
    "Seat Availability": "Checking seat availability... please provide flight number.",
    "Flights Info": "Please provide your route and travel date to view available flights.",
    "Prohibited Items Faq": "Sharp objects, explosives, and liquids over 100ml are not allowed.",
    "Cancellation Policy": "Tickets can be cancelled 24 hrs before departure for partial refund.",
    "Sports Music Gear": "You can check in sports or musical gear with additional fees."
}

# -----------------------------
# Small-talk / greetings
# -----------------------------
small_talk = {
    "hi": "Hello! How can I assist you with your flight today?",
    "hello": "Hi there! What can I help you with regarding your trip?",
    "hey": "Hey! Looking for flight info or something else?",
    "good morning": "Good morning! How can I help you with your travel plans?",
    "good afternoon": "Good afternoon! Need assistance with your flights?",
    "good evening": "Good evening! How can I assist you today?",
    "bye": "Goodbye! Have a safe flight!",
    "goodbye": "Bye! Hope to assist you again soon!",
    "see you": "See you! Safe travels!",
    "thanks": "You're welcome! Anything else I can do?",
    "thank you": "Glad I could help! Need anything else?",
    "thank you so much": "My pleasure! Is there anything else?",
    "how are you": "I'm just a bot, but I'm here to help you with your flights!",
    "what's up": "I'm here to help with your flight inquiries. How can I assist?",
    "how is it going": "All good! Ready to assist with your travel needs.",
    "hey there": "Hey! How can I assist you today?",
    "hi there": "Hello! What can I help you with regarding your trip?"
}

# -----------------------------
# Helper: reload model
# -----------------------------
def reload_model():
    global model, tokenizer, encoder
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    encoder = joblib.load("label_encoder.pkl")
    print("✅ Model reloaded after retraining.")

# -----------------------------
# Background retraining
# -----------------------------
def background_retraining():
    while True:
        try:
            if os.path.exists(FEEDBACK_FILE):
                df = pd.read_csv(FEEDBACK_FILE)
                if len(df) >= 10:
                    print("🔄 Retraining triggered automatically with 10 feedbacks...")
                    retrain_model()
                    reload_model()
                    # Archive old feedback
                    if os.path.exists(ARCHIVE_FILE):
                        archived_df = pd.read_csv(ARCHIVE_FILE)
                        archived_df = pd.concat([archived_df, df], ignore_index=True)
                    else:
                        archived_df = df.copy()
                    archived_df.to_csv(ARCHIVE_FILE, index=False)
                    # Clear feedback log
                    df = pd.DataFrame(columns=["utterance", "predicted_intent", "correct_intent"])
                    df.to_csv(FEEDBACK_FILE, index=False)
                    print("✅ Feedback archived and feedback_log cleared.")
        except Exception as e:
            print("⚠️ Retraining error:", e)
        time.sleep(30)

threading.Thread(target=background_retraining, daemon=True).start()

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/agent_view")
def agent_view():
    return render_template("agent.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data["message"].strip().lower()
    save_chat("user", text)

    # Small talk first
    for key, reply in small_talk.items():
        if key in text:
            save_chat("bot", reply, intent="small_talk", confidence=1.0)
            return jsonify({
                "prediction": reply,
                "confidence": 1.0,
                "show_feedback": False,
                "predicted_intent": "small_talk"
            })

    # Intent prediction
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred_index = torch.argmax(probs, dim=1).item()
    intent = encoder.inverse_transform([pred_index])[0]
    confidence = probs[0][pred_index].item()

    # Response
    response_text = responses.get(intent, f"I detected your intent as '{intent}'. Could you please provide more details?")
    save_chat("bot", response_text, intent=intent, confidence=confidence)
    show_feedback = confidence < 0.6 or intent not in responses

    return jsonify({
        "prediction": response_text,
        "confidence": float(confidence),
        "show_feedback": show_feedback,
        "predicted_intent": intent
    })

@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.json
    new_row = {
        "utterance": data["message"],
        "predicted_intent": data.get("predicted", "Unknown"),
        "correct_intent": data["correct_type"]
    }
    global feedback_df
    feedback_df = pd.concat([feedback_df, pd.DataFrame([new_row])], ignore_index=True)
    feedback_df.to_csv(FEEDBACK_FILE, index=False)
    return jsonify({"status": "ok"})

@app.route("/chat_history", methods=["GET"])
def chat_history():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, sender, message, intent, confidence, timestamp FROM chats ORDER BY id ASC")
    rows = cursor.fetchall()
    conn.close()

    messages = []
    for chat_id, s, m, i, c, t in rows:
        full_message_file = os.path.join(ARCHIVE_DIR, f"{chat_id}_{s}.txt")
        full_message = m
        if os.path.exists(full_message_file):
            with open(full_message_file, "r", encoding="utf-8") as f:
                full_message = f.read()
        messages.append({
            "id": chat_id,
            "sender": s,
            "message": m,
            "full_message": full_message,
            "intent": i,
            "confidence": c,
            "timestamp": t
        })
    return jsonify(messages)

# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
