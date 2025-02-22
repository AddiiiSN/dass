import re
import json
import logging
import random
import pickle
import numpy as np
import nltk
import tensorflow as tf
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import load_model
import threading
import asyncio
import mysql.connector


nltk.download('wordnet')
nltk.download('omw-1.4')

# ===================[ Konfigurasi Database ]===================
DB_CONFIG = {
    "host": "localhost",
    "user": "root",  # Ganti dengan username database Anda
    "password": "",  # Ganti dengan password database Anda
    "database": "carebot_db",  # Ganti dengan nama database Anda
}


def connect_db():
    """Membuka koneksi ke database."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None


def save_dass_result(user_id, depression, anxiety, stress):
    """Menyimpan hasil skrining DASS-21 ke database."""
    conn = connect_db()
    if conn:
        cursor = conn.cursor()
        query = """
        INSERT INTO dass_results (user_id, depression, anxiety, stress, created_at)
        VALUES (%s, %s, %s, %s, NOW())
        """
        cursor.execute(query, (user_id, depression, anxiety, stress))
        conn.commit()
        cursor.close()
        conn.close()


def log_chat(user_id, user_message, bot_response):
    """Menyimpan log percakapan chatbot ke database."""
    conn = connect_db()
    if conn:
        cursor = conn.cursor()
        query = """
        INSERT INTO chat_logs (user_id, user_message, bot_response, timestamp)
        VALUES (%s, %s, %s, NOW())
        """
        cursor.execute(query, (user_id, user_message, bot_response))
        conn.commit()
        cursor.close()
        conn.close()

# ===================[ Konfigurasi Logging ]===================
logging.basicConfig(level=logging.INFO)

# ===================[ Load Model Chatbot (Neural Network) ]===================
nltk.download('punkt')
lemmatizer = nltk.WordNetLemmatizer()

# Load intents JSON
with open("translated_intents.json", "r", encoding="utf-8") as file:
    intents = json.load(file)

# Load model AI & vectorizer
with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

with open("label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)

model = load_model("chatbot_model.keras")

def predict_class(text):
    """Memprediksi kelas teks pengguna dengan Neural Network"""
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

# ===================[ Mapping Keyword ke Skor Skrining ]===================
keywords = {
    0: ["tidak", "biasa saja", "normal", "santai", "tidak pernah", "baik-baik saja", "tidak ada masalah", "saya sangat", "saya merasa lebih baik"],
    1: ["kadang-kadang", "terkadang", "sesekali", "agak terganggu", "lumayan", "sekali-sekali", "terasa sedikit"],
    2: ["sering", "beberapa kali", "cukup berat", "mengganggu", "berulang kali", "sering merasa", "lumayan sering"],
    3: ["sangat sulit", "tidak bisa", "parah", "berat sekali", "sangat mengganggu", "tak tertahankan", "ekstrem"],
}


# Kata negasi yang perlu diperhatikan
negation_words = ["tidak", "bukan", "tidak pernah", "belum", "tidak merasa"]

# ===================[ Fungsi Analisis Jawaban ]===================
def detect_negation(response):
    """Mendeteksi apakah ada kata negasi dalam jawaban pengguna."""
    return any(neg_word in response for neg_word in negation_words)

def analyze_response(response, question):
    """Menganalisis jawaban pengguna dan memberikan skor yang sesuai."""
    response = response.lower().strip()
    detected_scores = set()

    has_negation = detect_negation(response)

    for score in sorted(keywords.keys(), reverse=True):
        if any(re.search(rf"\b{re.escape(word)}\b", response) for word in keywords[score]):
            detected_scores.add(score)

    if has_negation and ("kurang" in question.lower() or "tidak" in question.lower()):
        return 0

    return max(detected_scores, default=0)

# ===================[ Skrining DASS-21 ]===================
user_sessions = {}

dass_questions = [
    {"question": "Apakah akhir-akhir ini Anda merasa sulit untuk tenang?", "scale": "stress"},
    {"question": "Apakah Anda sering merasa mulut kering saat cemas?", "scale": "anxiety"},
    {"question": "Apakah Anda merasa kesulitan menemukan hal-hal yang positif dalam hidup?", "scale": "depression"},
    {"question": "Apakah Anda merasa kesulitan untuk bekerja sebaik biasanya?", "scale": "stress"},
    {"question": "Apakah Anda merasa bereaksi secara berlebihan terhadap situasi tertentu?", "scale": "stress"},
    {"question": "Apakah Anda sering mengalami kesulitan bernapas, seperti sesak napas tanpa aktivitas fisik?", "scale": "anxiety"},
    {"question": "Apakah Anda merasa kurang memiliki inisiatif untuk melakukan sesuatu?", "scale": "depression"},
    {"question": "Apakah Anda cenderung bereaksi secara emosional terhadap situasi sehari-hari?", "scale": "stress"},
    {"question": "Apakah Anda sering merasa tangan gemetar tanpa sebab yang jelas?", "scale": "anxiety"},
    {"question": "Apakah Anda merasa kehilangan semangat terhadap hal-hal yang biasanya menarik bagi Anda?", "scale": "depression"},
    {"question": "Apakah Anda sering merasa cemas tanpa alasan yang jelas?", "scale": "anxiety"},
    {"question": "Apakah Anda merasa tidak ada harapan untuk masa depan?", "scale": "depression"},
    {"question": "Apakah Anda merasa mudah terganggu oleh hal-hal kecil?", "scale": "stress"},
    {"question": "Apakah Anda merasa sulit untuk benar-benar rileks?", "scale": "stress"},
    {"question": "Apakah Anda merasa cemas hampir sepanjang waktu?", "scale": "anxiety"},
    {"question": "Apakah Anda sering merasa sedih atau tertekan?", "scale": "depression"},
    {"question": "Apakah Anda merasa kurang sabar terhadap hal-hal yang mengganggu Anda?", "scale": "stress"},
    {"question": "Apakah Anda merasa hampir panik tanpa alasan yang jelas?", "scale": "anxiety"},
    {"question": "Apakah Anda merasa kehilangan antusiasme terhadap hal-hal yang biasanya menyenangkan bagi Anda?", "scale": "depression"},
    {"question": "Apakah Anda merasa sulit mentoleransi gangguan saat sedang fokus melakukan sesuatu?", "scale": "stress"},
    {"question": "Apakah Anda pernah merasa bahwa hidup ini tidak berharga?", "scale": "depression"},
]

dass_interpretation = {
    "depression": [(0, 9, "Normal"), (10, 13, "Ringan"), (14, 20, "Sedang"), (21, 27, "Berat"), (28, 42, "Sangat Berat")],
    "anxiety": [(0, 7, "Normal"), (8, 9, "Ringan"), (10, 14, "Sedang"), (15, 19, "Berat"), (20, 42, "Sangat Berat")],
    "stress": [(0, 14, "Normal"), (15, 18, "Ringan"), (19, 25, "Sedang"), (26, 33, "Berat"), (34, 42, "Sangat Berat")],
}

# ===================[ Fungsi Skrining DASS-21 ]===================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Menampilkan menu utama"""
    reply_keyboard = [["CareBot", "Skrining DASS-21"]]
    markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True, resize_keyboard=True)
    await update.message.reply_text(
        "👋 Selamat datang di *CareBot*!\n\n"
        "Saya adalah asisten virtual yang dapat membantu Anda dalam:\n"
        "1️⃣ *Skrining DASS-21* 🧠 - Tes untuk mengukur tingkat Depresi, Kecemasan, dan Stres.\n"
        "2️⃣ *CareBot* 🤖 - Chatbot yang bisa menjawab pertanyaan Anda tentang kesehatan mental.\n\n"
        "🔹 Ketik `/start` kapan saja untuk kembali ke menu utama.\n"
        "🔹 Pilih salah satu fitur di bawah ini untuk memulai.\n"
    ,
        reply_markup=markup
    )

async def handle_menu_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Menangani pilihan fitur dari menu utama"""
    user_choice = update.message.text

    if user_choice == "CareBot":
        context.user_data["in_dass"] = False
        await update.message.reply_text("Anda memilih CareBot. Silakan kirim pertanyaan Anda.", reply_markup=ReplyKeyboardRemove())

    elif user_choice == "Skrining DASS-21":
        context.user_data["in_dass"] = True
        await start_dass(update, context)

async def start_dass(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Memulai sesi skrining DASS-21"""
    user_id = update.effective_user.id
    user_sessions[user_id] = {"current_question": 0, "scores": {"depression": 0, "anxiety": 0, "stress": 0}}
    await ask_dass_question(update, context)

async def ask_dass_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Menampilkan pertanyaan DASS-21"""
    user_id = update.effective_user.id
    session = user_sessions.get(user_id)

    if not session or session["current_question"] >= len(dass_questions):
        await conclude_dass(update, context)
        return

    question_data = dass_questions[session["current_question"]]
    await update.message.reply_text(question_data["question"])

async def handle_dass_response(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Menangani jawaban pengguna dalam skrining"""
    user_id = update.effective_user.id
    session = user_sessions.get(user_id)

    if session:
        question = dass_questions[session["current_question"]]
        score = analyze_response(update.message.text, question["question"]) * 2
        session["scores"][question["scale"]] += score
        session["current_question"] += 1
        await ask_dass_question(update, context)

async def conclude_dass(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Menampilkan hasil akhir skrining dengan penjelasan yang sesuai, lalu kembali ke menu utama."""
    user_id = update.effective_user.id
    session = user_sessions.pop(user_id, None)

    if not session:
        await update.message.reply_text("Sesi tidak ditemukan.")
        return

    explanations = {
        "Normal": "Hasil Anda menunjukkan bahwa Anda berada dalam batas normal. Tetap jaga kesehatan mental Anda! 😊",
        "Ringan": "Anda mengalami sedikit gejala, tetapi masih dalam tingkat ringan. Coba luangkan waktu untuk relaksasi dan menjaga keseimbangan hidup. 🧘",
        "Sedang": "Gejala yang Anda alami berada di tingkat sedang. Pertimbangkan untuk berbicara dengan seseorang yang bisa membantu, seperti teman dekat atau keluarga. 💬",
        "Berat": "Hasil menunjukkan tingkat yang cukup tinggi. Mungkin ini saatnya untuk mencari dukungan lebih lanjut, seperti berkonsultasi dengan profesional. 🩺",
        "Sangat Berat": "Tingkat yang Anda alami cukup serius. Sangat disarankan untuk berbicara dengan ahli kesehatan mental atau psikolog. Jangan ragu mencari bantuan. 🤝",
    }

    results = []
    for scale, score in session["scores"].items():
        category = next(cat for min_v, max_v, cat in dass_interpretation[scale] if min_v <= score <= max_v)
        explanation = explanations[category]  # Menambahkan penjelasan dari kategori hasil
        results.append(f"**{scale.capitalize()}**: {score} ({category})\n➡️ {explanation}")

    result_text = "\n\n".join(results)
    
    await update.message.reply_text(
        f"📊 **Hasil Skrining DASS-21 Anda:**\n\n{result_text}", 
        parse_mode="Markdown"
    )

    await start(update, context)  # Kembali ke menu utama

async def handle_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Menangani chat dengan CareBot jika tidak dalam sesi skrining"""
    if context.user_data.get("in_dass", False):
        await handle_dass_response(update, context)
    else:
        response = predict_class(update.message.text)
        await update.message.reply_text(response)


# ===================[ Konfigurasi Bot Telegram ]===================
application = (
    Application.builder()
    .token("7111956700:AAH3MTcF2UxZuU6JhhaFYYP7UARBlTveWvY")
    .read_timeout(30)  
    .connect_timeout(30)
    .build()
)

# Handler
application.add_handler(CommandHandler("start", start))
application.add_handler(MessageHandler(filters.Regex("^(CareBot|Skrining DASS-21)$"), handle_menu_choice))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_chat))

# Jalankan bot
def run_bot():
    application.run_polling()

if __name__ == "__main__":
    application.run_polling()
