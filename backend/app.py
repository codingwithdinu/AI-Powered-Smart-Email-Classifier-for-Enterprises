"""
AI-Powered Smart Email Classifier for Enterprises
-------------------------------------------------
Flask backend that serves the frontend UI and exposes:
  - GET  /                    -> classify page (index.html)
  - GET  /dashboard           -> analytics dashboard (dashboard.html)
  - POST /predict             -> form-based prediction (renders dashboard)
  - POST /api/predict         -> JSON prediction API
  - GET  /api/history         -> JSON prediction history + summary
  - GET  /api/training-summary-> JSON urgency model training analytics

Run:  python app.py   (or)  gunicorn app:app
"""

import os
import re
import pickle
from datetime import datetime

import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------------------------
# Optional NLTK (used to mirror the training-time text cleaning pipeline).
# If NLTK is unavailable we fall back to a built-in stopword list so the app
# still runs without crashing.
# ---------------------------------------------------------------------------
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    _HAS_NLTK = True
except Exception:  # pragma: no cover
    _HAS_NLTK = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))       # .../backend
ROOT_DIR = os.path.dirname(BASE_DIR)                        # project root
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

app = Flask(
    __name__,
    template_folder=FRONTEND_DIR,
    static_folder=os.path.join(FRONTEND_DIR, "static"),
)

# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------
category_model = pickle.load(
    open(os.path.join(MODELS_DIR, "email_classifier.pkl"), "rb")
)
category_vectorizer = pickle.load(
    open(os.path.join(MODELS_DIR, "vectorizer.pkl"), "rb")
)
urgency_model = pickle.load(
    open(os.path.join(MODELS_DIR, "urgency_model.pkl"), "rb")
)
urgency_vectorizer = pickle.load(
    open(os.path.join(MODELS_DIR, "urgency_vectorizer.pkl"), "rb")
)

# ---------------------------------------------------------------------------
# Category mapping
# The category model was trained on "category_id" (0..5). We derive the
# human-readable names dynamically from the training CSV so the mapping can
# never drift out of sync with the model.
# ---------------------------------------------------------------------------
_DEFAULT_CATEGORY_MAPPING = {
    0: "Forum",
    1: "Promotions",
    2: "Social Media",
    3: "Spam",
    4: "Updates",
    5: "Verify Code",
}


def _build_category_mapping():
    mapping = dict(_DEFAULT_CATEGORY_MAPPING)
    path = os.path.join(DATA_DIR, "train_clean.csv")
    try:
        df = pd.read_csv(path, usecols=["category", "category_id"])
        for _, row in df.drop_duplicates("category_id").iterrows():
            mapping[int(row["category_id"])] = (
                str(row["category"]).replace("_", " ").title()
            )
    except Exception:
        pass
    return mapping


CATEGORY_MAPPING = _build_category_mapping()

# ---------------------------------------------------------------------------
# Text cleaning (mirrors src/preprocess.py so predictions match training)
# ---------------------------------------------------------------------------
_DEFAULT_STOPWORDS = set(
    """a about above after again against all am an and any are aren't as at be
    because been before being below between both but by can can't cannot could
    couldn't did didn't do does doesn't doing don't down during each few for
    from further had hadn't has hasn't have haven't having he he'd he'll he's
    her here here's hers herself him himself his how how's i i'd i'll i'm i've
    if in into is isn't it it's its itself let's me more most mustn't my myself
    no nor not of off on once only or other ought our ours ourselves out over
    own same shan't she she'd she'll she's should shouldn't so some such than
    that that's the their theirs them themselves then there there's these they
    they'd they'll they're they've this those through to too under until up very
    was wasn't we we'd we'll we're we've were weren't what what's when when's
    where where's which while who who's whom why why's with won't would
    wouldn't you you'd you'll you're you've your yours yourself yourselves""".split()
)


def _get_stopwords():
    if _HAS_NLTK:
        try:
            return set(stopwords.words("english"))
        except Exception:
            pass
    return _DEFAULT_STOPWORDS


def clean_text(text):
    """Lowercase, strip URLs/emails/punctuation, remove stopwords, lemmatize."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = text.split()
    stop = _get_stopwords()
    tokens = [w for w in tokens if w not in stop and len(w) > 2]
    if _HAS_NLTK:
        try:
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(w) for w in tokens]
        except Exception:
            pass
    return " ".join(tokens)


def normalize_urgency(value):
    """The urgency model emits lowercase labels; normalize for display."""
    v = str(value).strip().lower()
    return {"low": "Low", "medium": "Medium", "high": "High"}.get(v, str(value).title())


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------
def predict_email(email_text):
    # Category model was trained on *cleaned* text.
    cleaned = clean_text(email_text)
    cat_vec = category_vectorizer.transform([cleaned])
    cat_id = int(category_model.predict(cat_vec)[0])
    category = CATEGORY_MAPPING.get(cat_id, "Unknown")
    cat_conf = round(float(max(category_model.predict_proba(cat_vec)[0])) * 100, 2)

    # Urgency model was trained on *raw* text.
    urg_vec = urgency_vectorizer.transform([email_text])
    urgency = normalize_urgency(urgency_model.predict(urg_vec)[0])
    urg_conf = round(float(max(urgency_model.predict_proba(urg_vec)[0])) * 100, 2)

    return {
        "category": category,
        "category_confidence": cat_conf,
        "urgency": urgency,
        "urgency_confidence": urg_conf,
    }


# In-memory prediction history (resets on restart).
prediction_history = []


def build_history_summary(history):
    total = len(history)
    if total == 0:
        return {
            "total": 0,
            "high_count": 0,
            "high_rate": 0.0,
            "avg_cat_conf": 0.0,
            "avg_urg_conf": 0.0,
        }

    high_count = sum(1 for r in history if str(r.get("urgency", "")).lower() == "high")
    avg_cat_conf = sum(float(r.get("category_confidence", 0.0)) for r in history) / total
    avg_urg_conf = sum(float(r.get("urgency_confidence", 0.0)) for r in history) / total

    return {
        "total": total,
        "high_count": high_count,
        "high_rate": round((high_count / total) * 100, 2),
        "avg_cat_conf": round(avg_cat_conf, 2),
        "avg_urg_conf": round(avg_urg_conf, 2),
    }


def build_urgency_training_summary():
    summary = {
        "training_rows": 0,
        "accuracy": 0.0,
        "distribution": {},
        "class_metrics": [],
        "vectorizer_max_features": getattr(urgency_vectorizer, "max_features", None),
        "model_max_iter": getattr(urgency_model, "max_iter", None),
        "model_class_weight": getattr(urgency_model, "class_weight", None),
    }

    dataset_path = os.path.join(DATA_DIR, "module3_final_train_output.csv")
    if not os.path.exists(dataset_path):
        return summary

    df = pd.read_csv(dataset_path)
    if "text" not in df.columns or "urgency" not in df.columns or df.empty:
        return summary

    y_true = df["urgency"].astype(str)
    X_tfidf = urgency_vectorizer.transform(df["text"].fillna(""))
    y_pred = urgency_model.predict(X_tfidf)

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    summary["training_rows"] = int(len(df))
    summary["accuracy"] = round(accuracy_score(y_true, y_pred) * 100, 2)
    summary["distribution"] = {
        str(label): int(count)
        for label, count in df["urgency"].value_counts().to_dict().items()
    }

    for label in summary["distribution"].keys():
        if label not in report:
            continue
        row = report[label]
        summary["class_metrics"].append(
            {
                "label": normalize_urgency(label),
                "precision": round(row["precision"] * 100, 2),
                "recall": round(row["recall"] * 100, 2),
                "f1": round(row["f1-score"] * 100, 2),
                "support": int(row["support"]),
            }
        )

    return summary


TRAINING_SUMMARY = build_urgency_training_summary()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    return render_template(
        "dashboard.html",
        history=prediction_history,
        history_summary=build_history_summary(prediction_history),
        training_summary=TRAINING_SUMMARY,
    )


@app.route("/predict", methods=["POST"])
def predict():
    email_text = request.form.get("email", "").strip()
    if not email_text:
        return render_template(
            "dashboard.html",
            error="Please enter some email text to classify.",
            history=prediction_history,
            history_summary=build_history_summary(prediction_history),
            training_summary=TRAINING_SUMMARY,
        )

    result = predict_email(email_text)
    result.update(
        {
            "email": email_text[:80] + "..." if len(email_text) > 80 else email_text,
            "full_email": email_text,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    prediction_history.append(result)

    return render_template(
        "dashboard.html",
        result=result,
        history=prediction_history,
        history_summary=build_history_summary(prediction_history),
        training_summary=TRAINING_SUMMARY,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    email_text = (data.get("email") or "").strip()
    if not email_text:
        return jsonify({"error": "Missing or empty 'email' field."}), 400

    result = predict_email(email_text)
    result["email"] = email_text[:80] + "..." if len(email_text) > 80 else email_text
    result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prediction_history.append(result)

    return jsonify(result)


@app.route("/api/history", methods=["GET"])
def api_history():
    return jsonify(
        {
            "history": list(reversed(prediction_history)),
            "summary": build_history_summary(prediction_history),
        }
    )


@app.route("/api/training-summary", methods=["GET"])
def api_training_summary():
    return jsonify(TRAINING_SUMMARY)


@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({"status": "ok", "categories": CATEGORY_MAPPING})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
