import os
import sys
import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# Make sibling modules importable regardless of CWD.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate_model import evaluate_model, plot_confusion_matrix


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

train_path = os.path.join(BASE_DIR, "data", "processed", "train_clean.csv")
test_path = os.path.join(BASE_DIR, "data", "processed", "test_clean.csv")


# -----------------------------
# Load processed data
# -----------------------------
print("Loading processed dataset...")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)


# -----------------------------
# Select features
# -----------------------------
X_train = train_df["clean_text"]
y_train = train_df["category_id"]

X_test = test_df["clean_text"]
y_test = test_df["category_id"]


# -----------------------------
# TF-IDF Vectorization
# -----------------------------
print("Vectorizing text...")

vectorizer = TfidfVectorizer(max_features=5000)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# -----------------------------
# Train Logistic Regression
# -----------------------------
print("Training model...")

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_tfidf, y_train)


# -----------------------------
# Training Accuracy
# -----------------------------
train_pred = log_model.predict(X_train_tfidf)
print("\nTraining Accuracy:", accuracy_score(y_train, train_pred))


# -----------------------------
# Evaluation
# -----------------------------
y_pred = evaluate_model(log_model, X_test_tfidf, y_test)
plot_confusion_matrix(y_test, y_pred)


# -----------------------------
# Cross Validation
# -----------------------------
cv_scores = cross_val_score(log_model, X_train_tfidf, y_train, cv=5)

print("\nCross Validation Scores:", cv_scores)
print("Mean CV Score:", cv_scores.mean())


# -----------------------------
# Save model
# -----------------------------
model_path = os.path.join(BASE_DIR, "models", "email_classifier.pkl")
vectorizer_path = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

os.makedirs(os.path.dirname(model_path), exist_ok=True)
pickle.dump(log_model, open(model_path, "wb"))
pickle.dump(vectorizer, open(vectorizer_path, "wb"))

print("\nModel saved successfully.")
