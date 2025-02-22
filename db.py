from pymongo import MongoClient

try:
    client = MongoClient("mongodb://localhost:27017/")  # Pastikan MongoDB berjalan
    db = client["nama_database"]  # Ganti dengan nama database Anda
    print("Koneksi ke MongoDB berhasil!")
except Exception as e:
    print(f"Error: {e}")
