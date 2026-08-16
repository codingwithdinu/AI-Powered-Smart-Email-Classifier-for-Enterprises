import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle

# -----------------------------
# Paths (robust to CWD)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "module3_final_train_output.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "urgency_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "urgency_vectorizer.pkl")

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv(DATA_PATH)

X = df["text"]
y = df["urgency"]

# -----------------------------
# TF-IDF Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_tfidf = vectorizer.fit_transform(X)

# -----------------------------
# Logistic Regression Model
# -----------------------------
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_tfidf, y)

# -----------------------------
# Prediction
# -----------------------------
y_pred = model.predict(X_tfidf)

# -----------------------------
# Evaluation
# -----------------------------
print("\nAccuracy:", accuracy_score(y, y_pred))

print("\nClassification Report:\n")
print(classification_report(y, y_pred))

# -----------------------------
# Save Model
# -----------------------------
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
pickle.dump(model, open(MODEL_PATH, "wb"))
pickle.dump(vectorizer, open(VECTORIZER_PATH, "wb"))

print("\nUrgency model saved successfully.")
