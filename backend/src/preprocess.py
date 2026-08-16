import os
import re
import nltk
import pandas as pd
from datasets import load_dataset
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# -----------------------------
# Download NLTK resources
# -----------------------------
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# -----------------------------
# Clean Text Function
# -----------------------------
def clean_text(text):

    if not isinstance(text, str):
        return ""

    text = text.lower()

    # remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # remove emails
    text = re.sub(r"\S+@\S+", " ", text)

    # remove special characters & numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # tokenize
    tokens = word_tokenize(text)

    # remove stopwords
    tokens = [
        word for word in tokens
        if word not in stop_words and len(word) > 2
    ]

    # lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)


# -----------------------------
# Load Dataset
# -----------------------------
print("Loading dataset...")

dataset = load_dataset("jason23322/high-accuracy-email-classifier")

train_df = pd.DataFrame(dataset["train"])
test_df = pd.DataFrame(dataset["test"])

# -----------------------------
# Apply Cleaning
# -----------------------------
print("Cleaning training data...")
train_df["clean_text"] = train_df["text"].apply(clean_text)

print("Cleaning test data...")
test_df["clean_text"] = test_df["text"].apply(clean_text)

# -----------------------------
# Create processed folder
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

processed_path = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(processed_path, exist_ok=True)

# -----------------------------
# Save Clean Data
# -----------------------------
train_df.to_csv(os.path.join(processed_path, "train_clean.csv"), index=False)
test_df.to_csv(os.path.join(processed_path, "test_clean.csv"), index=False)

print("\n Clean data saved in data/processed/")