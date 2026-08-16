# 📧 AI-Powered Smart Email Classifier for Enterprises

A complete **machine-learning web application** that automatically **classifies emails into categories** and **detects their urgency level** — built with Python, scikit-learn, and Flask.

---

## 📌 What is this project?

This project reads any email text and instantly answers **two questions**:

1. **What is this email about?** → It assigns one of **6 categories** (Spam, Promotions, Updates, etc.)
2. **How urgent is it?** → It labels it as **Low**, **Medium**, or **High** urgency.

It also shows a **confidence score (0–100%)** for each prediction, so you know how sure the AI is.

> 💡 **Example:** You paste *"URGENT: Production server is down, please fix immediately!"* → the app replies **Category: Updates**, **Urgency: High** (with confidence percentages).

---

## 🎯 Why does this matter?

Enterprises receive **thousands of emails every day**. Manually reading and sorting them wastes a lot of time. This tool helps by:

- ✅ **Sorting emails automatically** — no human effort needed
- ✅ **Flagging urgent emails first** — critical issues are never missed
- ✅ **Filtering spam & promotions** — the inbox stays clean
- ✅ **Working instantly** — through a web UI or a REST API

---

## ✨ Features

- 🗂️ **6-way email classification** (Logistic Regression + TF-IDF)
- ⚡ **3-level urgency detection** (Low / Medium / High)
- 📊 **Confidence scores** for every prediction
- 🖥️ **Modern web UI** — paste an email, get the result instantly
- 📈 **Live dashboard** with prediction history & model analytics
- 🔌 **JSON REST API** for integration with other systems
- 🔄 **Retraining scripts** to rebuild the models on new data

---

## 🧠 How it works (the ML pipeline)

The project uses **two machine-learning models** working together:

### 1️⃣ Category Model

| Item | Value |
|---|---|
| Algorithm | Logistic Regression |
| Text representation | TF-IDF (Term Frequency–Inverse Document Frequency) |
| Trained on | ~10,000 emails (Hugging Face dataset) |
| Output | One of 6 categories |

### 2️⃣ Urgency Model

| Item | Value |
|---|---|
| Algorithm | Logistic Regression (with class balancing) |
| Text representation | TF-IDF |
| Output | Low / Medium / High |

### 🧹 Text preprocessing (before prediction)
Before an email is classified, it goes through a cleaning pipeline:

1. **Lowercase** → "URGENT" becomes "urgent"
2. **Remove URLs & emails** → `http://...`, `name@mail.com` removed
3. **Remove punctuation & numbers** → only letters remain
4. **Remove stopwords** → common words like "the", "is", "and" removed
5. **Lemmatization** → words reduced to base form ("running" → "run")

> ⚠️ **Important detail:** the *category* model uses **cleaned** text, while the *urgency* model uses **raw** text (this matches how each was trained).

---

## 🗂️ The 6 Categories

| # | Category | What it means | Example email |
|---|---|---|---|
| 0 | **Forum** | Discussion / community posts | "How do I reset my password?" |
| 1 | **Promotions** | Marketing, offers, discounts | "50% off this weekend only!" |
| 2 | **Social Media** | Notifications from social platforms | "John liked your post" |
| 3 | **Spam** | Scams, phishing, fake offers | "You won $5000! Claim now" |
| 4 | **Updates** | Account / system notifications | "Your order has shipped" |
| 5 | **Verify Code** | Login / verification codes | "Your code is 482913" |

---

## ⚡ Urgency Levels

| Level | Meaning | Typical keywords |
|---|---|---|
| 🔴 **High** | Needs **immediate** action | "urgent", "ASAP", "emergency", "down", "deadline" |
| 🟡 **Medium** | Needs attention **soon** | "review", "request", "meeting", "approval" |
| 🟢 **Low** | No rush | general info, newsletters |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3, Flask |
| **Machine Learning** | scikit-learn, pandas, NumPy |
| **Frontend** | HTML5, CSS3, JavaScript (vanilla, no framework) |
| **Deployment** | Docker, Gunicorn, Heroku (Procfile) |

---

## 📁 Project Structure

```
.
├── backend/                       # Flask API + ML pipeline
│   ├── app.py                     # Flask application (entry point)
│   ├── requirements.txt           # Python dependencies
│   ├── Dockerfile                 # Container build
│   ├── Procfile                   # Heroku / gunicorn process
│   ├── wsgi.py                    # WSGI entry point
│   ├── models/                    # Trained .pkl models + vectorizers
│   │   ├── email_classifier.pkl   # Category model
│   │   ├── vectorizer.pkl         # Category TF-IDF vectorizer
│   │   ├── urgency_model.pkl      # Urgency model
│   │   └── urgency_vectorizer.pkl # Urgency TF-IDF vectorizer
│   ├── data/processed/            # Processed CSV datasets
│   │   ├── train_clean.csv        # Cleaned training data
│   │   ├── test_clean.csv         # Cleaned test data
│   │   └── module3_final_train_output.csv  # Urgency training data
│   └── src/                       # Training / preprocessing scripts
│       ├── preprocess.py          # Download + clean dataset
│       ├── train_model.py         # Train category model
│       ├── train_urgency_model.py # Train urgency model
│       ├── evaluate_model.py      # Evaluation helpers
│       └── predict.py             # CLI prediction script
│
└── frontend/                      # UI / UX
    ├── index.html                 # Classify page (single-page app)
    ├── dashboard.html             # Analytics dashboard
    └── static/
        ├── css/style.css          # Design system
        └── js/script.js           # Frontend logic (fetch API)
```

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.10+** (3.13 recommended)
- **Git** (to clone the repo)

### Step 1 — Clone the repository
```bash
git clone https://github.com/codingwithdinu/AI-Powered-Smart-Email-Classifier-for-Enterprises.git
cd AI-Powered-Smart-Email-Classifier-for-Enterprises
```

### Step 2 — Create a virtual environment
```bash
python -m venv .venv
```

Activate it:
```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r backend/requirements.txt
```

### Step 4 — (Optional) Download NLTK data
The text-cleaning pipeline uses NLTK. Download the required data once:
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```
> ℹ️ If NLTK data is missing, the app still works — it falls back to a built-in stopword list.

---

## ▶️ How to Run

### Development server
```bash
cd backend
python app.py
```
Now open your browser at **http://localhost:5000** 🎉

### Production (Gunicorn)
```bash
cd backend
gunicorn --bind 0.0.0.0:5000 app:app
```

---

## 💻 How to Use the App (step by step)

1. Open **http://localhost:5000**
2. **Paste** any email text into the textarea — or click a **sample chip** (e.g. "🚨 Urgent outage")
3. Click the **"Classify Email"** button
4. See the result instantly:
   - 🏷️ **Category badge** (e.g. Spam, Updates)
   - ⚡ **Urgency badge** (Low / Medium / High)
   - 📊 **Confidence bars** (animated)
5. Scroll down to see:
   - **Session Overview** (total predictions, high-urgency count, avg confidence)
   - **Prediction History** (a table of all your predictions)
   - **Urgency Model Analytics** (training accuracy, class distribution, precision/recall/F1)

> 💡 The **Dashboard** page (`/dashboard`) shows the same analytics in a dedicated view.

---

## 🔌 API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Classify page (UI) |
| `GET` | `/dashboard` | Analytics dashboard (UI) |
| `POST` | `/predict` | Form-based prediction (renders dashboard) |
| `POST` | `/api/predict` | JSON prediction |
| `GET` | `/api/history` | Prediction history + summary |
| `GET` | `/api/training-summary` | Urgency model training analytics |
| `GET` | `/api/health` | Health check + category mapping |

### Example — classify an email (JSON)
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"email": "URGENT: production server is down, please fix immediately!"}'
```

**Response:**
```json
{
  "category": "Updates",
  "category_confidence": 87.42,
  "urgency": "High",
  "urgency_confidence": 91.13,
  "email": "URGENT: production server is down, please fix immediately!",
  "timestamp": "2026-08-16 10:30:00"
}
```

### Example — health check
```bash
curl http://localhost:5000/api/health
```
```json
{
  "status": "ok",
  "categories": {
    "0": "Forum", "1": "Promotions", "2": "Social Media",
    "3": "Spam", "4": "Updates", "5": "Verify Code"
  }
}
```

---

## 🔄 Retraining the Models

If you have new data and want to rebuild the models from scratch:

```bash
cd backend/src

# 1. Download & clean the dataset
python preprocess.py

# 2. Train the category model
python train_model.py

# 3. Train the urgency model (uses module3_final_train_output.csv)
python train_urgency_model.py
```

> 📝 **Note:** `preprocess.py` pulls the dataset from Hugging Face
> (`jason23322/high-accuracy-email-classifier`). If the dataset is gated,
> set your token first: `export HF_TOKEN=your_token` (Linux/macOS) or
> `set HF_TOKEN=your_token` (Windows).

---

## 🐳 Docker

```bash
cd backend
docker build -t email-classifier .
docker run -p 5000:5000 email-classifier
```

---

## ❓ Troubleshooting

| Problem | Solution |
|---|---|
| **Port 5000 already in use** | Stop the other process, or run on another port: `python app.py` → change `port=5000` in `app.py` |
| **`InconsistentVersionWarning` (scikit-learn)** | Harmless. The `.pkl` models were saved with scikit-learn 1.5.x. To silence it, install `pip install "scikit-learn==1.5.1"` |
| **`ModuleNotFoundError: No module named 'flask'`** | Run `pip install -r backend/requirements.txt` |
| **NLTK data missing** | Run `python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"` (or ignore — the app has a fallback) |
| **Hugging Face dataset download fails** | Set your `HF_TOKEN` environment variable (see Retraining section) |

---

## 📄 License

See [LICENSE](LICENSE).

---

## 👥 Contributors

- **Dinesh Patel** — [GitHub](https://github.com/codingwithdinu)



