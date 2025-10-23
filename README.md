
 Airline Customer Chatbot

This project is a Flask-based chatbot for airline customer support. It uses DistilBERT for intent classification and stores chat history in SQLite. The project includes:

* User interface to chat with the bot.
* Agent dashboard to monitor chat history, predicted intents, and confidence scores.
* Feedback mechanism to improve the model over time.

---

 Features

* Intent recognition with confidence scores using DistilBERT.
* Predefined responses for common airline queries (flight status, cancellations, baggage, etc.).
* Small-talk handling for greetings and casual conversation.
* Feedback collection to improve prediction accuracy.
* Agent dashboard for monitoring chats and reviewing model predictions.
* Automatic model retraining when feedback reaches a threshold (10+ entries).

---

 Project Structure


airline-chatbot/
│
├─ app.py                # Flask app with routes
├─ train_model.py        # Script for retraining model
├─ airline_distilbert_model/  # Pretrained DistilBERT model
├─ label_encoder.pkl     # Label encoder for intents
├─ chat_history.db       # SQLite database for chat logs
├─ feedback_log.csv      # CSV file storing feedback
├─ archived_feedback.csv # Archived feedback for retraining
├─ templates/
│   ├─ index.html        # User chat interface
│   └─ agent.html        # Agent dashboard


## Setup Instructions

1. Clone the repository

git clone <repo-url>
cd airline-chatbot


2. **Create a virtual environment**

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows


3. Install dependencies
pip install -r requirements.txt


4. Run the Flask app

python app.py


5. Access interfaces

* User chat interface: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)
* Agent dashboard: [http://127.0.0.1:5000/agent_view](http://127.0.0.1:5000/agent_view)

---

 Usage

1. **User interface**: Type your message in the input box and click *Send*. The bot responds based on intent predictions.
2. **Feedback**: If the bot prediction is low confidence or unknown, provide the correct intent via the dropdown.
3. **Agent dashboard**: View all chat history, including sender, truncated message (expandable), predicted intent, confidence, and timestamp.

---

 Notes

* Debug Mode is enabled for development. Do not use in production.
* SQLite is lightweight and used here for simplicity. For production, consider PostgreSQL or MySQL.
* The automatic retraining runs in the background and archives feedback after retraining.

---

Future Improvements

* Deploy the app with Gunicorn or uWSGI and Nginx for production.
* Add real-time notifications for agents.
* Enhance user interface with chat bubbles and typing indicators.
* Integrate external APIs for flight status or booking info.


