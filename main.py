import os
import re
import json
import logging
import random
import pickle
import numpy as np
import nltk
import mysql.connector
import tensorflow as tf
from dotenv import load_dotenv
from telegram import (
    Update,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
)
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import load_model

# ===================[ Setup ]===================
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("punkt")
lemmatizer = nltk.WordNetLemmatizer()
load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

application = Application.builder().token(TOKEN).build()
logging.basicConfig(level=logging.INFO)


# ===================[ Koneksi Database MySQL ]===================
def get_db_connection():
    return mysql.connector.connect(
        host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=DB_NAME
    )


# ===================[ Load Model Chatbot ]===================
with open("translated_intents.json", "r", encoding="utf-8") as file:
    intents = json.load(file)
with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)
with open("label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)
model = load_model("chatbot_model.keras")


def predict_class(text):
    words = nltk.word_tokenize(text)
    processed_text = " ".join([lemmatizer.lemmatize(w.lower()) for w in words])
    X_input = vectorizer.transform([processed_text]).toarray()
    prediction = model.predict(X_input)
    predicted_class = np.argmax(prediction)
    tag = label_encoder.inverse_transform([predicted_class])[0]
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "Maaf, saya tidak mengerti. Bisa Anda jelaskan lebih lanjut?"


# ===================[ Skrining DASS-21 ]===================
user_sessions = {}
dass_questions = [
    {
        "question": "Apakah akhir-akhir ini Anda merasa sulit untuk tenang?",
        "scale": "stress",
    },
    {
        "question": "Apakah Anda sering merasa mulut kering saat cemas?",
        "scale": "anxiety",
    },
    {
        "question": "Apakah Anda merasa kesulitan menemukan hal-hal yang positif dalam hidup?",
        "scale": "depression",
    },
]

dass_interpretation = {
    "depression": [
        (0, 9, "Normal"),
        (10, 13, "Ringan"),
        (14, 20, "Sedang"),
        (21, 27, "Berat"),
        (28, 42, "Sangat Berat"),
    ],
    "anxiety": [
        (0, 7, "Normal"),
        (8, 9, "Ringan"),
        (10, 14, "Sedang"),
        (15, 19, "Berat"),
        (20, 42, "Sangat Berat"),
    ],
    "stress": [
        (0, 14, "Normal"),
        (15, 18, "Ringan"),
        (19, 25, "Sedang"),
        (26, 33, "Berat"),
        (34, 42, "Sangat Berat"),
    ],
}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    reply_keyboard = [["CareBot", "Skrining DASS-21"]]
    markup = ReplyKeyboardMarkup(
        reply_keyboard, one_time_keyboard=True, resize_keyboard=True
    )
    await update.message.reply_text(
        "👋 Selamat datang di CareBot!", reply_markup=markup
    )


async def handle_menu_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_choice = update.message.text
    if user_choice == "CareBot":
        context.user_data["in_dass"] = False
        await update.message.reply_text(
            "Silakan kirim pertanyaan Anda.", reply_markup=ReplyKeyboardRemove()
        )
    elif user_choice == "Skrining DASS-21":
        context.user_data["in_dass"] = True
        await start_dass(update, context)


async def start_dass(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_sessions[user_id] = {
        "current_question": 0,
        "scores": {"depression": 0, "anxiety": 0, "stress": 0},
    }
    await ask_dass_question(update, context)


async def ask_dass_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    session = user_sessions.get(user_id)
    if not session or session["current_question"] >= len(dass_questions):
        await conclude_dass(update, context)
        return

    question_data = dass_questions[session["current_question"]]
    keyboard = [
        [InlineKeyboardButton("0 - Tidak Sama Sekali", callback_data="0")],
        [InlineKeyboardButton("1 - Kadang-kadang", callback_data="1")],
        [InlineKeyboardButton("2 - Sering", callback_data="2")],
        [InlineKeyboardButton("3 - Hampir Selalu", callback_data="3")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        question_data["question"], reply_markup=reply_markup
    )


async def handle_dass_response(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    session = user_sessions.get(user_id)
    if session:
        question = dass_questions[session["current_question"]]
        session["scores"][question["scale"]] += int(query.data)
        session["current_question"] += 1
        await ask_dass_question(update, context)


async def conclude_dass(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    session = user_sessions.pop(user_id, None)
    if not session:
        await update.message.reply_text("Sesi tidak ditemukan.")
        return

    results = []
    for scale, score in session["scores"].items():
        category = next(
            cat
            for min_v, max_v, cat in dass_interpretation[scale]
            if min_v <= score <= max_v
        )
        results.append(f"*{scale.capitalize()}*: {score} ({category})")

    result_text = "\n".join(results)
    await update.message.reply_text(
        f"📊 *Hasil Skrining DASS-21 Anda:*\n\n{result_text}", parse_mode="Markdown"
    )

    # Simpan ke database
    db = get_db_connection()
    cursor = db.cursor()
    cursor.execute(
        "INSERT INTO dass_results (user_id, depression, anxiety, stress) VALUES (%s, %s, %s, %s)",
        (
            user_id,
            session["scores"]["depression"],
            session["scores"]["anxiety"],
            session["scores"]["stress"],
        ),
    )
    db.commit()
    db.close()


async def handle_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.user_data.get("in_dass", False):
        await handle_dass_response(update, context)
    else:
        response = predict_class(update.message.text)
        await update.message.reply_text(response)


application.add_handler(CommandHandler("start", start))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_chat))
application.add_handler(CallbackQueryHandler(handle_dass_response))

if __name__ == "__main__":
    application.run_polling()
 