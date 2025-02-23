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
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# ===================[ KONFIGURASI LOGGING ]===================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ===================[ SETUP MODEL BERT ]===================
MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4)

def predict_dass21_score(answer):
    inputs = tokenizer(answer, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.argmax(outputs.logits, dim=1).item()# Pilih skor tertinggi (0,1,2,3)
    return predicted_score

# ===================[ LOAD MODEL CHATBOT ]===================
nltk.download("punkt")
nltk.download("wordnet")



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
    """Memprediksi intent dari input pengguna tanpa mencampur terlalu banyak riwayat"""

    words = nltk.word_tokenize(text)  # Gunakan hanya input terbaru
    processed_text = " ".join([lemmatizer.lemmatize(w.lower()) for w in words])

    X_input = vectorizer.transform([processed_text]).toarray()
    prediction = model.predict(X_input)
    predicted_class = np.argmax(prediction)

    tag = label_encoder.inverse_transform([predicted_class])[0]

    # Cari respons yang sesuai dari intents JSON
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
        await update.message.reply_text("Anda memilih CareBot. ada yang bisa saya bantu ,Silakan kirim pertanyaan Anda.", reply_markup=ReplyKeyboardRemove())

    elif user_choice == "Skrining DASS-21":
        context.user_data["in_dass"] = True
        await start_dass(update, context)

async def start_dass(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Memulai sesi skrining DASS-21"""
    user_id = update.effective_user.id
    user_sessions[user_id] = {"current_question": 0, "scores": {"depression": 0, "anxiety": 0, "stress": 0}}
    await ask_dass_question(update, context)

async def ask_dass_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Menampilkan pertanyaan DASS-21 satu per satu"""

    user_id = update.effective_user.id
    session = user_sessions.get(user_id)

    if not session or session["current_question"] >= len(dass_questions):
        # **Pastikan skrining selesai dengan memanggil conclude_dass()**
        await conclude_dass(update, context)
        return  

    # **Tampilkan pertanyaan berikutnya**
    question_data = dass_questions[session["current_question"]]
    await update.message.reply_text(question_data["question"])

# ========================
# 5. Prediksi Skor dengan BERT (Dikali 2)
# ========================
if 'user_answers' not in globals() or 'questions' not in globals():
    user_answers = []  # Jika tidak ada jawaban, inisialisasi list kosong
    questions = []

dass_scores = [predict_dass21_score(answer) * 2 for answer in user_answers]

# ========================
# 6. Menampilkan Hasil Skrining
# ========================
print("\n=== Hasil Skrining DASS-21 ===")
for i, (question, answer, score) in enumerate(zip(questions, user_answers, dass_scores)):
    print(f"\nQ{i+1}: {question}")
    print(f"User: {answer}")
    print(f"Predicted Score (x2): {score}")  # Pastikan skor sudah dikali 2

async def handle_dass_response(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Menangani jawaban pengguna dalam skrining"""
    user_id = update.effective_user.id
    session = user_sessions.get(user_id)

    if session:
        # Cek apakah masih ada pertanyaan tersisa
        if session["current_question"] >= len(dass_questions):
            await conclude_dass(update, context)
            return

        question = dass_questions[session["current_question"]]
        score = analyze_response(update.message.text, question["question"]) * 2  # Pastikan skor dikali 2
        session["scores"][question["scale"]] += score
        session["current_question"] += 1

        # Tampilkan pertanyaan berikutnya atau akhiri skrining
        if session["current_question"] < len(dass_questions):
            await ask_dass_question(update, context)
        else:
            await conclude_dass(update, context)  # Skrining selesai

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

     # **Pastikan status skrining dimatikan**
    context.user_data["in_dass"] = False


    await start(update, context)  # Kembali ke menu utama

async def handle_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Menangani chat dengan CareBot atau skrining DASS-21"""

    user_id = update.effective_user.id
    user_message = update.message.text.strip()

    # **Jika pengguna sedang dalam sesi skrining, paksa ke handler skrining**
    if context.user_data.get("in_dass", False):
        await handle_dass_response(update, context)
        return  # Pastikan CareBot tidak menangani pesan ini!

    # **Jika tidak dalam skrining, lanjutkan ke CareBot seperti biasa**
    history = context.user_data.get("chat_history", [])
    history.append({"role": "user", "text": user_message})

    response = predict_class(user_message)  # Prediksi jawaban berdasarkan input pengguna saja

    history.append({"role": "bot", "text": response})

    if len(history) > 10:
        history = history[-10:]

    context.user_data["chat_history"] = history  

    await update.message.reply_text(response)

async def reset_chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Menghapus riwayat percakapan dan memulai dari awal"""
    context.user_data["chat_history"] = []
    await update.message.reply_text("Riwayat percakapan telah dihapus. Silakan mulai percakapan baru.")


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

application.add_handler(CommandHandler("reset", reset_chat))

# Jalankan bot
def run_bot():
    application.run_polling()

if __name__ == "__main__":
    application.run_polling()


