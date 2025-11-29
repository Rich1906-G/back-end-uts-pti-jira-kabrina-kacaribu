from flask import Flask, request, jsonify, send_from_directory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import re

app = Flask(__name__)

# ============================================
# 1. BACA DATASET TOKOPEDIA & BUAT LABEL
# ============================================

DATA_PATH = "./tokopedia-product-reviews-2019.csv"  # ganti jika nama file beda

df = pd.read_csv(DATA_PATH)

# Gunakan hanya kolom teks dan rating
df = df[["text", "rating"]].dropna()

# Mapping rating → sentimen
def rating_to_sentiment(r):
    if r >= 4:
        return "positif"
    elif r == 3:
        return "netral"
    else:
        return "negatif"

df["sentiment"] = df["rating"].apply(rating_to_sentiment)

# Pisahkan berdasarkan kelas
neg = df[df["sentiment"] == "negatif"]
neu = df[df["sentiment"] == "netral"]
pos = df[df["sentiment"] == "positif"]

# ==========================
# BALANCING DATASET
# ==========================
# Target: ambil jumlah positif = 2x (negatif + netral)
target_pos = 2 * (len(neg) + len(neu))

# Cegah error kalau dataset terlalu kecil
target_pos = min(target_pos, len(pos))

pos_sampled = pos.sample(target_pos, random_state=42)

# Gabungkan kembali dataset balanced
train_df = pd.concat([neg, neu, pos_sampled]).reset_index(drop=True)

# Untuk memastikan balance
print("Distribusi data training:")
print(train_df["sentiment"].value_counts())

# ============================================
# 2. PREPROCESSING TEKS
# ============================================

STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "untuk", "dengan",
    "atau", "itu", "ini", "saya", "aku", "kami", "kita",
    "anda", "kamu", "dia", "mereka", "lagi", "sudah",
    "belum", "kok", "loh", "deh"
}

# PERHATIKAN: KITA **TIDAK** menghapus kata "tidak" dan "kurang"
# karena itu penanda NEGASI yang sangat penting dalam sentimen.


def preprocess_text(text: str):
    original = str(text).strip()
    lower = original.lower()

    # tokenisasi sederhana
    tokens = [t for t in re.split(r"[^a-z0-9]+", lower) if t]

    # hapus stopword, tetapi tidak hapus "tidak" & "kurang"
    no_stop = [t for t in tokens if t not in STOPWORDS]

    return original, lower, tokens, no_stop

# Siapkan teks training (gunakan kata tanpa stopword)
train_corpus = [" ".join(preprocess_text(t)[3]) for t in train_df["text"]]
y_train = train_df["sentiment"].values

# ============================================
# 3. TF-IDF DENGAN N-GRAM (1 DAN 2)
# ============================================

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),     # ← inilah yang membuat model mengerti frasa "tidak bagus"
    max_features=20000,
    min_df=2
)

X_train = vectorizer.fit_transform(train_corpus)

# Train model Multinomial Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

print("Model ML dengan TF-IDF NGRAM (1,2) siap digunakan.")

# ============================================
# 4. ROUTE FRONTEND / INDEX.HTML
# ============================================

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

# ============================================
# 5. API PREDIKSI
# ============================================

@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "Teks review kosong"}), 400

    original, lower, tokens, no_stop = preprocess_text(text)

    # ==============================
    # 1. CEK POLA NEGATIF KUAT
    # ==============================
    strong_neg_patterns = [
        "tidak bagus",
        "sangat tidak bagus",
        "sangat jelek",
        "tidak memuaskan",
        "tidak puas",
        "sangat mengecewakan",
        "kurang bagus",
        "kualitasnya sangat tidak bagus",
    ]

    # Kalau mengandung frasa negatif kuat → langsung NEGATIF
    if any(pat in lower for pat in strong_neg_patterns):
        label = "negatif"
        confidence = 0.99
    else:
        # ==============================
        # 2. JALANKAN MODEL ML SEPERTI BIASA
        # ==============================
        X = vectorizer.transform([" ".join(no_stop)])
        proba = model.predict_proba(X)[0]
        label = model.classes_[proba.argmax()]
        confidence = float(proba.max())

    return jsonify({
        "sentiment": label,
        "confidence": confidence,
        "preprocess": {
            "original": original,
            "lower": lower,
            "tokens": tokens,
            "no_stopwords": no_stop
        }
    })

if __name__ == "__main__":
    app.run(debug=True)
