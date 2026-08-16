# AI-Powered Smart Email Classifier for Enterprises

An AI-powered email classification system that categorizes incoming emails and
detects their urgency level using scikit-learn machine learning models, served
through a Flask backend with a modern, responsive web UI.

## Project Structure

```
.
├── backend/                    # Flask API + ML pipeline
│   ├── app.py                  # Flask application (entry point)
│   ├── requirements.txt        # Python dependencies
│   ├── Dockerfile              # Container build
│   ├── Procfile                # Heroku / gunicorn process
│   ├── wsgi.py                 # WSGI entry point
│   ├── models/                 # Trained .pkl models + vectorizers
│   ├── data/processed/         # Processed CSV datasets
│   └── src/                    # Training / preprocessing scripts
│       ├── preprocess.py
│       ├── train_model.py
│       ├── train_urgency_model.py
│       ├── evaluate_model.py
│       └── predict.py
└── frontend/                   # UI / UX
    ├── index.html              # Classify page (single-page app)
    ├── dashboard.html          # Analytics dashboard
    └── static/
        ├── css/style.css       # Design system
        └── js/script.js        # Frontend logic
```

## Features

- **Email categorization** into 6 classes: Forum, Promotions, Social Media,
  Spam, Updates, Verify Code (Logistic Regression + TF-IDF).
- **Urgency detection** (Low / Medium / High) via a second Logistic Regression
  model.
- **Confidence scores** for both predictions.
- **Live dashboard** with prediction history, session metrics, and urgency
  model training analytics.

## Setup

```bash
# 1. Create & activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r backend/requirements.txt

# 3. (Optional) Download NLTK data used by the text-cleaning pipeline
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

> **Note on scikit-learn:** the pre-trained `.pkl` models were pickled with
> scikit-learn 1.5.x. Loading them with a newer scikit-learn version works but
> prints an `InconsistentVersionWarning`. This is harmless; to silence it,
> install a matching version, e.g. `pip install "scikit-learn==1.5.1"` (subject
> to Python-version wheel availability).


## Running

```bash
# Development server
cd backend
python app.py
# -> http://localhost:5000

# Production (gunicorn)
cd backend
gunicorn --bind 0.0.0.0:5000 app:app
```

## API Endpoints

| Method | Path                    | Description                              |
|--------|-------------------------|------------------------------------------|
| GET    | `/`                     | Classify page (UI)                       |
| GET    | `/dashboard`            | Analytics dashboard (UI)                 |
| POST   | `/predict`              | Form-based prediction (renders dashboard)|
| POST   | `/api/predict`          | JSON prediction                          |
| GET    | `/api/history`          | Prediction history + summary             |
| GET    | `/api/training-summary` | Urgency model training analytics         |
| GET    | `/api/health`           | Health check + category mapping          |

### Example: `/api/predict`

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"email": "URGENT: production server is down, please fix immediately!"}'
```

Response:

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

## Retraining the Models

```bash
cd backend/src

# 1. Download & clean the dataset
python preprocess.py

# 2. Train the category model
python train_model.py

# 3. Train the urgency model (uses module3_final_train_output.csv)
python train_urgency_model.py
```

> **Note:** `preprocess.py` pulls the dataset from Hugging Face
> (`jason23322/high-accuracy-email-classifier`) and requires a valid
> `HF_TOKEN` environment variable if the dataset is gated.

## Docker

```bash
cd backend
docker build -t email-classifier .
docker run -p 5000:5000 email-classifier
```

## License

See [LICENSE](LICENSE).

