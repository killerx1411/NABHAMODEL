"""
app/main.py — Production FastAPI disease prediction API
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from typing import List, Optional
from contextlib import asynccontextmanager
import numpy as np
import pandas as pd
import joblib
import json
import os
import time
import logging
from app.symptom_mapper import mapper

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# ─────────────────────────────────────────────────────────────
# LOAD ARTIFACTS AT STARTUP
# ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model artifacts...")
    app.state.model        = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    app.state.le           = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    app.state.symptom_list = joblib.load(os.path.join(MODEL_DIR, "symptom_list.pkl"))

    with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
        app.state.metadata = json.load(f)

    logger.info(f"✅ Model loaded: {app.state.metadata['best_model']}")
    logger.info(f"✅ Symptoms: {app.state.metadata['n_symptoms']}")
    logger.info(f"✅ Diseases: {app.state.metadata['n_diseases']}")
    yield
    logger.info("Shutting down...")

# ─────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Disease Predictor API",
    description="""
## Disease Prediction API

Provide a list of symptoms and receive the top predicted diseases with confidence scores.

### How to use
- `GET /symptoms` — Get full list of valid symptom names
- `POST /predict` — Submit symptoms, get predictions
- `GET /diseases` — Get full list of diseases the model can predict
- `GET /health` — Check API status

### Notes
- Symptoms must match the exact names from `/symptoms`
- Minimum 2 symptoms required
- Returns top 3 predictions with confidence percentages
    """,
    version="1.0.0",
    lifespan=lifespan
)

# ─────────────────────────────────────────────────────────────
# CORS — allow any frontend to call this API
# ─────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# REQUEST LOGGING MIDDLEWARE
# ─────────────────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = round((time.time() - start) * 1000, 2)
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({duration}ms)")
    return response

# ─────────────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    symptoms: List[str]

    @field_validator("symptoms")
    @classmethod
    def validate_symptoms(cls, v):
        if not v:
            raise ValueError("Symptoms list cannot be empty.")
        if len(v) < 2:
            raise ValueError("Please provide at least 2 symptoms.")
        return [s.strip() for s in v]   # mapper handles normalization

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "symptoms": ["fever", "chills", "throwing up", "headache"]
                }
            ]
        }
    }


class PredictionResult(BaseModel):
    rank: int
    disease: str
    confidence: float
    confidence_label: str   # "High", "Moderate", "Low"


class PredictResponse(BaseModel):
    predictions: List[PredictionResult]
    recognized_symptoms: List[str]
    unrecognized_symptoms: List[str]
    warning: Optional[str] = None
    model_used: str


class HealthResponse(BaseModel):
    status: str
    model: str
    n_symptoms: int
    n_diseases: int

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def symptoms_to_vector(symptoms: List[str], symptom_list: List[str]):
    vector = np.zeros(len(symptom_list))
    recognized = []
    unrecognized = []

    for s in symptoms:
        if s in symptom_list:
            vector[symptom_list.index(s)] = 1
            recognized.append(s)
        else:
            unrecognized.append(s)

    return vector, recognized, unrecognized


def confidence_label(score: float) -> str:
    if score >= 70:
        return "High"
    elif score >= 40:
        return "Moderate"
    return "Low"

# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────
@app.get("/", tags=["General"])
def root():
    return {
        "name": "Disease Predictor API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health(request: Request):
    meta = request.app.state.metadata
    return HealthResponse(
        status="healthy",
        model=meta["best_model"],
        n_symptoms=meta["n_symptoms"],
        n_diseases=meta["n_diseases"],
    )


@app.get("/symptoms", tags=["Data"])
def get_symptoms(request: Request):
    """Returns all valid symptom names. Use these exact strings in /predict."""
    symptom_list = request.app.state.symptom_list
    return {
        "total": len(symptom_list),
        "symptoms": sorted(symptom_list)
    }


@app.get("/diseases", tags=["Data"])
def get_diseases(request: Request):
    """Returns all diseases the model can predict."""
    le = request.app.state.le
    return {
        "total": len(le.classes_),
        "diseases": sorted(le.classes_.tolist())
    }


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: Request, body: PredictRequest):
    """
    Submit a list of symptoms and receive top-3 disease predictions.

    Each prediction includes the disease name, confidence percentage,
    and a confidence label (High / Moderate / Low).
    """
    model        = request.app.state.model
    le           = request.app.state.le
    symptom_list = request.app.state.symptom_list
    meta         = request.app.state.metadata

    # ── Map natural language symptoms to model vocabulary ──
    mapping = mapper.map_symptoms(body.symptoms)
    recognized = mapping["mapped"]
    unrecognized = mapping["unmatched"]
    mapping_warnings = mapping["warnings"]

    # Require at least 2 recognized symptoms
    if len(recognized) < 2:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Not enough recognized symptoms.",
                "recognized": recognized,
                "unrecognized": unrecognized,
                "suggestions": {
                    r["input"]: r["suggestions"]
                    for r in mapping["results"] if not r["mapped"]
                },
                "hint": "Use GET /symptoms for valid names, or try rephrasing."
            }
        )

    # Build binary vector from mapped symptoms
    vector = np.zeros(len(symptom_list))
    for s in recognized:
        if s in symptom_list:
            vector[symptom_list.index(s)] = 1

    # Predict probabilities
    vector_df = pd.DataFrame([vector], columns=symptom_list)
    probs = model.predict_proba(vector_df)[0]

    # Top 3
    top3_idx = np.argsort(probs)[::-1][:3]
    predictions = [
        PredictionResult(
            rank=rank + 1,
            disease=le.inverse_transform([idx])[0],
            confidence=round(float(probs[idx]) * 100, 2),
            confidence_label=confidence_label(float(probs[idx]) * 100)
        )
        for rank, idx in enumerate(top3_idx)
    ]

    warning = None
    all_warnings = mapping_warnings
    if unrecognized:
        all_warnings.append(
            f"Symptoms not recognized: {unrecognized}. Use GET /symptoms for valid names."
        )

    return PredictResponse(
        predictions=predictions,
        recognized_symptoms=recognized,
        unrecognized_symptoms=unrecognized,
        warning="; ".join(all_warnings) if all_warnings else None,
        model_used=meta["best_model"]
    )


# ─────────────────────────────────────────────────────────────
# GLOBAL ERROR HANDLER
# ─────────────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error. Please try again."}
    )


# ─────────────────────────────────────────────────────────────
# NATURAL LANGUAGE PREDICTION
# Accepts full sentences/paragraphs, extracts symptoms, predicts
# ─────────────────────────────────────────────────────────────
from app.sentence_parser import parser

class NaturalLanguageRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError("Text cannot be empty.")
        if len(v) > 2000:
            raise ValueError("Text too long. Max 2000 characters.")
        return v.strip()

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "I have a fever and I've been throwing up all night. My head hurts and I feel dizzy."
                }
            ]
        }
    }


@app.post("/predict_natural", response_model=PredictResponse, tags=["Prediction"])
def predict_from_natural_language(request: Request, body: NaturalLanguageRequest):
    """
    Submit natural language description and get disease predictions.
    
    This endpoint extracts symptoms from full sentences/paragraphs,
    maps them to medical terms, and returns predictions.
    
    Example:
    "I have a fever and I've been throwing up all night. My head hurts."
    → Extracts: fever, throwing up, head hurts
    → Maps to: high_fever, vomiting, headache
    → Predicts diseases
    """
    model        = request.app.state.model
    le           = request.app.state.le
    symptom_list = request.app.state.symptom_list
    meta         = request.app.state.metadata

    # ── Step 1: Extract symptom phrases from natural language ──
    parse_result = parser.parse(body.text)
    extracted_phrases = parse_result["extracted"]

    if not extracted_phrases:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Could not identify any symptoms in your text.",
                "hint": "Try describing your symptoms more explicitly, e.g., 'I have a fever and headache.'"
            }
        )

    # ── Step 2: Map extracted phrases to model vocabulary ──
    mapping = mapper.map_symptoms(extracted_phrases)
    recognized = mapping["mapped"]
    unrecognized = mapping["unmatched"]
    mapping_warnings = mapping["warnings"]

    # Add extraction info to warnings
    extraction_info = f"Extracted {len(extracted_phrases)} symptom phrases from your text: {extracted_phrases}"
    all_warnings = [extraction_info] + mapping_warnings

    if len(recognized) < 2:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Not enough recognized symptoms.",
                "extracted": extracted_phrases,
                "recognized": recognized,
                "unrecognized": unrecognized,
                "hint": "Please describe your symptoms more explicitly."
            }
        )

    # ── Step 3: Build binary vector and predict ──
    vector = np.zeros(len(symptom_list))
    for s in recognized:
        if s in symptom_list:
            vector[symptom_list.index(s)] = 1

    vector_df = pd.DataFrame([vector], columns=symptom_list)
    probs = model.predict_proba(vector_df)[0]

    # Top 3
    top3_idx = np.argsort(probs)[::-1][:3]
    predictions = [
        PredictionResult(
            rank=rank + 1,
            disease=le.inverse_transform([idx])[0],
            confidence=round(float(probs[idx]) * 100, 2),
            confidence_label=confidence_label(float(probs[idx]) * 100)
        )
        for rank, idx in enumerate(top3_idx)
    ]

    return PredictResponse(
        predictions=predictions,
        recognized_symptoms=recognized,
        unrecognized_symptoms=unrecognized,
        warning="; ".join(all_warnings) if all_warnings else None,
        model_used=meta["best_model"]
    )