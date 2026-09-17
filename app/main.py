"""
BanglaGuard FastAPI Application.

Serves:
1. Jinja2 web interface (GET /).
2. Static assets (/static).
3. Health probe endpoint (GET /health).
4. NLP Spam Detection prediction endpoint (POST /predict).
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure standard output/error supports UTF-8 on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from training.baseline_model import BaselinePredictor


# Resolve project directories
BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
MODELS_DIR = BASE_DIR / "models"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    Loads the trained TF-IDF + Logistic Regression model once on startup.
    """
    try:
        app.state.predictor = BaselinePredictor.from_saved(model_dir=MODELS_DIR)
        app.state.model_loaded = True
        print(f"[INFO] BanglaGuard Baseline NLP model successfully loaded from {MODELS_DIR.resolve()}")
    except Exception as e:
        print(f"[WARNING] Could not preload baseline model from {MODELS_DIR}: {e}")
        app.state.predictor = None
        app.state.model_loaded = False
    yield
    # Shutdown / cleanup
    app.state.predictor = None
    app.state.model_loaded = False


app = FastAPI(
    title="BanglaGuard",
    description="Academic Bengali SMS Spam Detection System",
    version="0.1.0",
    lifespan=lifespan,
)

# Mount static files and initialize Jinja2 templates
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


class SMSRequest(BaseModel):
    text: str = Field(..., description="Bengali SMS text to classify")


class SMSResponse(BaseModel):
    status: str
    prediction: str
    label: str
    label_id: int
    confidence: float
    spam_probability: float
    ham_probability: float
    original_text: str
    processed_text: str
    model_name: str


def get_predictor(app_instance: FastAPI) -> BaselinePredictor:
    """Helper to retrieve or lazily initialize the predictor."""
    predictor = getattr(app_instance.state, "predictor", None)
    if predictor is None:
        try:
            predictor = BaselinePredictor.from_saved(model_dir=MODELS_DIR)
            app_instance.state.predictor = predictor
            app_instance.state.model_loaded = True
        except Exception as err:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"NLP Model is not loaded or artifacts are missing: {err}",
            )
    return predictor


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Render the BanglaGuard web interface."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request, "app_title": "BanglaGuard"},
    )


@app.get("/health")
def health():
    """Health check endpoint reporting API and model readiness."""
    try:
        predictor = get_predictor(app)
        is_loaded = predictor is not None
    except Exception:
        is_loaded = False

    return {
        "status": "healthy",
        "service": "BanglaGuard Web & API",
        "model_loaded": is_loaded,
        "model_type": "TF-IDF + Logistic Regression",
    }


@app.post("/predict", response_model=SMSResponse)
def predict_sms_endpoint(payload: SMSRequest):
    """
    Classify a Bengali SMS message as Spam (1) or Ham (0).

    Applies:
    1. Input validation.
    2. Bengali NLP preprocessing (Unicode normalization, entity tokenization).
    3. TF-IDF vectorization and Logistic Regression inference.
    """
    raw_text = payload.text.strip() if payload.text else ""
    if not raw_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="অনুগ্রহ করে একটি বৈধ বাংলা এসএমএস টেক্সট প্রদান করুন (SMS text cannot be empty).",
        )

    predictor = get_predictor(app)

    try:
        res = predictor.predict(raw_text)
        return SMSResponse(
            status="success",
            prediction=res["label"],
            label=res["label"],
            label_id=res["label_id"],
            confidence=res["confidence"],
            spam_probability=res["spam_probability"],
            ham_probability=res["ham_probability"],
            original_text=res["original_text"],
            processed_text=res["processed_text"],
            model_name="TF-IDF + Logistic Regression",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction processing error: {str(e)}",
        )


@app.post("/analyze", response_model=SMSResponse)
def analyze_sms_alias(payload: SMSRequest):
    """Alias for /predict endpoint for backward compatibility."""
    return predict_sms_endpoint(payload)


@app.post("/test-sms")
def test_sms(payload: SMSRequest):
    """Legacy echo endpoint maintained for backward compatibility."""
    return {
        "status": "success",
        "message": "SMS received successfully by BanglaGuard API",
        "received_text": payload.text,
    }
