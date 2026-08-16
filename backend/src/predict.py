import os
import pickle
import re

try:
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    _HAS_NLTK = True
except Exception:
    _HAS_NLTK = False

# ---------------------------
# Paths
# ---------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CATEGORY_MODEL_PATH = os.path.join(BASE_DIR, "models", "email_classifier.pkl")
CATEGORY_VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")
URGENCY_MODEL_PATH = os.path.join(BASE_DIR, "models", "urgency_model.pkl")
URGENCY_VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "urgency_vectorizer.pkl")

# ---------------------------
# Load models
# ---------------------------
category_model = pickle.load(open(CATEGORY_MODEL_PATH, "rb"))
category_vectorizer = pickle.load(open(CATEGORY_VECTORIZER_PATH, "rb"))

urgency_model = pickle.load(open(URGENCY_MODEL_PATH, "rb"))
urgency_vectorizer = pickle.load(open(URGENCY_VECTORIZER_PATH, "rb"))

# ---------------------------
# Category mapping
# ---------------------------
CATEGORY_MAPPING = {
    0: "Forum",
    1: "Promotions",
    2: "Social Media",
    3: "Spam",
    4: "Updates",
    5: "Verify Code",
}

# ---------------------------
# Text cleaning
# ---------------------------
if _HAS_NLTK:
    stop_words = set(stopwords.words("english"))
else:
    stop_words = set()


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    if _HAS_NLTK:
        tokens = word_tokenize(text)
    else:
        tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)


# ---------------------------
# Prediction function
# ---------------------------
def predict_email(text):
    text = clean_text(text)

    # Category prediction
    text_vector = category_vectorizer.transform([text])
    category_id = category_model.predict(text_vector)[0]
    category = CATEGORY_MAPPING.get(category_id, "Unknown")
    category_conf = max(category_model.predict_proba(text_vector)[0]) * 100

    # Urgency prediction
    urgency_vector = urgency_vectorizer.transform([text])
    urgency = urgency_model.predict(urgency_vector)[0]
    urgency_conf = max(urgency_model.predict_proba(urgency_vector)[0]) * 100

    return category, category_conf, urgency, urgency_conf


# ---------------------------
# CLI testing
# ---------------------------
if __name__ == "__main__":
    user_input = input("Enter email text: ")

    cat, cat_conf, urg, urg_conf = predict_email(user_input)

    print("\nPrediction Result")
    print("------------------")
    print(f"Category: {cat} ({cat_conf:.2f}%)")
    print(f"Urgency: {urg} ({urg_conf:.2f}%)")
