# BanglaGuard: Bengali SMS Spam Detection System

BanglaGuard is an academic Natural Language Processing (NLP) system designed to detect and classify spam in Bengali SMS messages. The application provides both an interactive web dashboard and a RESTful API powered by FastAPI, featuring custom Bengali text preprocessing and a TF-IDF + Logistic Regression classification model.

---

## Features

- **FastAPI Web Interface & API**: Unified ASGI application serving both the Jinja2 HTML/CSS frontend and JSON prediction endpoints.
- **Bengali NLP Preprocessor**: Unicode normalization (NFKC) and entity recognition/masking (`<URL>`, `<PHONE>`, `<EMAIL>`, `<MONEY>`, `<NUMBER>`).
- **Trained Baseline Model**: TF-IDF vectorizer (8,746 features) paired with balanced Logistic Regression classifier (~97% test accuracy / F1-score).
- **Lightweight & Fast**: Managed with `uv` for fast dependency resolution and reproducible environments.

---

## Project Structure

```
BanglaGuard/
├── app/
│   ├── __init__.py
│   └── main.py              # FastAPI application & route handlers
├── models/
│   ├── tfidf_vectorizer.joblib     # Serialized TF-IDF vectorizer
│   ├── logistic_regression.joblib  # Trained Logistic Regression model
│   └── baseline_metadata.json      # Model hyperparameters & metrics
├── templates/
│   └── index.html           # Jinja2 web interface template
├── static/
│   └── style.css            # Stylesheet for web dashboard
├── training/
│   ├── __init__.py
│   ├── preprocessor.py      # BengaliTextPreprocessor class
│   ├── data_prep.py         # Dataset cleaning & preparation
│   ├── baseline_model.py    # Training & evaluation pipeline
│   └── train_baseline.py    # Model training CLI script
├── tests/
│   ├── test_api.py          # API & route test suite
│   ├── test_baseline_model.py # Model & inference tests
│   ├── test_preprocessor.py # Preprocessing & normalization tests
│   └── test_data_prep.py    # Data preparation tests
├── data/
│   └── processed/           # Processed datasets
├── Procfile                 # Cloud deployment process file
├── pyproject.toml           # Project dependencies & configuration
├── uv.lock                  # Pinned dependency lockfile
└── README.md
```

---

## Getting Started

### 1. Installation & Dependency Sync

Clone the repository and install dependencies using [uv](https://github.com/astral-sh/uv):

```bash
# Sync all dependencies from uv.lock
uv sync
```

### 2. Running Locally

Start the local development server:

```bash
uv run uvicorn app.main:app --reload
```

- **Web Dashboard**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **API Health**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- **Interactive OpenAPI Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Cloud Deployment

BanglaGuard is ready for deployment on cloud hosting platforms (e.g., Render, Railway, Fly.io, Heroku, Hugging Face Spaces, GCP, AWS).

### Production Start Command

The platform will dynamically assign the `PORT` environment variable:

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Alternatively, standard ASGI execution:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

### Runtime Model Artifacts

> **Important**: The trained model artifacts inside the `models/` directory (`tfidf_vectorizer.joblib`, `logistic_regression.joblib`, and `baseline_metadata.json`) are required at runtime. Ensure these files are committed and present in the deployment repository.

---

## API Reference

### `GET /health`
Returns service status and confirms model loading state:
```json
{
  "status": "healthy",
  "service": "BanglaGuard Web & API",
  "model_loaded": true,
  "model_type": "TF-IDF + Logistic Regression"
}
```

### `POST /predict`
Classifies a Bengali SMS message:

**Request:**
```json
{
  "text": "অভিনন্দন! আপনি জিতেছেন নগদ ৳৫০,০০০ টাকা! পুরস্কার পেতে কল করুন ০১৭১২৩৪৫৬৭৮"
}
```

**Response:**
```json
{
  "status": "success",
  "prediction": "spam",
  "label": "spam",
  "label_id": 1,
  "confidence": 0.8477,
  "spam_probability": 0.8477,
  "ham_probability": 0.1523,
  "original_text": "অভিনন্দন! আপনি জিতেছেন নগদ ৳৫০,০০০ টাকা! পুরস্কার পেতে কল করুন ০১৭১২৩৪৫৬৭৮",
  "processed_text": "অভিনন্দন! আপনি জিতেছেন নগদ <MONEY> টাকা! পুরস্কার পেতে কল করুন <PHONE>",
  "model_name": "TF-IDF + Logistic Regression"
}
```

---

## Running Tests

Execute the full test suite (31 unit tests):

```bash
uv run python -m unittest discover -s tests -p "test_*.py"
```
