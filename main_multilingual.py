"""
app/main_multilingual.py — Multilingual Disease Prediction API
Supports English, Hindi, and Punjabi with native language models
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import List, Optional
from contextlib import asynccontextmanager
from langdetect import detect
import numpy as np
import pandas as pd
import joblib
import json
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# ─────────────────────────────────────────────────────────────
# LOAD ALL LANGUAGE MODELS AT STARTUP
# ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading multilingual models...")
    
    # English model
    app.state.models = {}
    app.state.models["en"] = {
        "model": joblib.load(os.path.join(MODEL_DIR, "best_model.pkl")),
        "le": joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl")),
        "symptom_list": joblib.load(os.path.join(MODEL_DIR, "symptom_list.pkl")),
        "metadata": json.load(open(os.path.join(MODEL_DIR, "metadata.json")))
    }
    logger.info(f"✅ English model loaded: {app.state.models['en']['metadata']['n_diseases']} diseases")
    
    # Hindi model
    try:
        app.state.models["hi"] = {
            "model": joblib.load(os.path.join(MODEL_DIR, "hindi", "best_model.pkl")),
            "le": joblib.load(os.path.join(MODEL_DIR, "hindi", "label_encoder.pkl")),
            "symptom_list": joblib.load(os.path.join(MODEL_DIR, "hindi", "symptom_list.pkl")),
            "metadata": json.load(open(os.path.join(MODEL_DIR, "hindi", "metadata.json"), encoding='utf-8'))
        }
        logger.info(f"✅ Hindi model loaded: {app.state.models['hi']['metadata']['n_diseases']} diseases")
    except FileNotFoundError:
        logger.warning("⚠️  Hindi model not found. Run train_hindi.py first.")
        app.state.models["hi"] = None
    
    # Punjabi model
    try:
        app.state.models["pa"] = {
            "model": joblib.load(os.path.join(MODEL_DIR, "punjabi", "best_model.pkl")),
            "le": joblib.load(os.path.join(MODEL_DIR, "punjabi", "label_encoder.pkl")),
            "symptom_list": joblib.load(os.path.join(MODEL_DIR, "punjabi", "symptom_list.pkl")),
            "metadata": json.load(open(os.path.join(MODEL_DIR, "punjabi", "metadata.json"), encoding='utf-8'))
        }
        logger.info(f"✅ Punjabi model loaded: {app.state.models['pa']['metadata']['n_diseases']} diseases")
    except FileNotFoundError:
        logger.warning("⚠️  Punjabi model not found. Run train_punjabi.py first.")
        app.state.models["pa"] = None
    
    yield
    logger.info("Shutting down...")

# ─────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Multilingual Disease Predictor API",
    description="Disease prediction in English, Hindi (हिन्दी), and Punjabi (ਪੰਜਾਬੀ)",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str
    language: Optional[str] = None  # "en", "hi", "pa", or None for auto-detect

    @field_validator("text")
    @classmethod
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError("Text cannot be empty.")
        return v.strip()


class PredictionResult(BaseModel):
    rank: int
    disease: str
    confidence: float
    confidence_label: str


class PredictResponse(BaseModel):
    predictions: List[PredictionResult]
    detected_language: str
    language_name: str
    model_used: str
    warning: Optional[str] = None


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def detect_language(text: str) -> str:
    """Detect language from text. Returns 'en', 'hi', or 'pa'."""
    try:
        lang = detect(text)
        # Map detected language codes
        if lang in ['hi', 'mr', 'ne']:  # Hindi, Marathi, Nepali (Devanagari script)
            return 'hi'
        elif lang in ['pa']:  # Punjabi
            return 'pa'
        else:
            return 'en'
    except:
        return 'en'  # Default to English


def confidence_label(score: float) -> str:
    if score >= 70:
        return "High"
    elif score >= 40:
        return "Moderate"
    return "Low"


def extract_symptoms_from_text(text: str, symptom_list: List[str]) -> List[str]:
    """
    Extract symptoms from natural language text.
    For Hindi/Punjabi, looks for symptom mentions directly.
    """
    text_lower = text.lower()
    found_symptoms = []
    
    for symptom in symptom_list:
        if symptom.lower() in text_lower:
            found_symptoms.append(symptom)
    
    return found_symptoms


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name": "Multilingual Disease Predictor API",
        "version": "2.0.0",
        "languages": ["English", "हिन्दी (Hindi)", "ਪੰਜਾਬੀ (Punjabi)"],
        "docs": "/docs"
    }


@app.get("/models")
def get_models(request: Request):
    """Get information about loaded models."""
    models_info = {}
    for lang, model_data in request.app.state.models.items():
        if model_data:
            models_info[lang] = {
                "language": model_data["metadata"].get("language", "English"),
                "n_diseases": model_data["metadata"]["n_diseases"],
                "n_symptoms": model_data["metadata"]["n_symptoms"],
                "best_model": model_data["metadata"]["best_model"]
            }
    return models_info


@app.post("/predict", response_model=PredictResponse)
def predict(request: Request, body: PredictRequest):
    """
    Multilingual disease prediction.
    
    Automatically detects language (English/Hindi/Punjabi) or accepts explicit language parameter.
    Uses native language model for Hindi and Punjabi for better accuracy.
    """
    
    # Detect or use provided language
    if body.language:
        lang = body.language.lower()
        if lang not in ['en', 'hi', 'pa']:
            raise HTTPException(status_code=400, detail=f"Unsupported language: {lang}")
    else:
        lang = detect_language(body.text)
    
    # Get model for detected language
    model_data = request.app.state.models.get(lang)
    
    if not model_data:
        # Fallback to English if requested language model not available
        logger.warning(f"Model for '{lang}' not available, falling back to English")
        lang = 'en'
        model_data = request.app.state.models['en']
    
    model = model_data["model"]
    le = model_data["le"]
    symptom_list = model_data["symptom_list"]
    metadata = model_data["metadata"]
    
    # Extract symptoms from text
    found_symptoms = extract_symptoms_from_text(body.text, symptom_list)
    
    if len(found_symptoms) < 2:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Could not identify enough symptoms in your text.",
                "found_symptoms": found_symptoms,
                "hint": f"Please describe your symptoms more explicitly in {'English' if lang == 'en' else 'Hindi' if lang == 'hi' else 'Punjabi'}."
            }
        )
    
    # Build binary vector
    vector = np.zeros(len(symptom_list))
    for s in found_symptoms:
        if s in symptom_list:
            vector[symptom_list.index(s)] = 1
    
    # Predict
    vector_df = pd.DataFrame([vector], columns=symptom_list)
    probs = model.predict_proba(vector_df)[0]
    
    # Top 3
    top3_idx = np.argsort(probs)[::-1][:3]
    predictions = [
        PredictionResult(
            rank=i + 1,
            disease=le.inverse_transform([idx])[0],
            confidence=round(float(probs[idx]) * 100, 2),
            confidence_label=confidence_label(float(probs[idx]) * 100)
        )
        for i, idx in enumerate(top3_idx)
    ]
    
    # Language names
    lang_names = {"en": "English", "hi": "हिन्दी (Hindi)", "pa": "ਪੰਜਾਬੀ (Punjabi)"}
    
    return PredictResponse(
        predictions=predictions,
        detected_language=lang,
        language_name=lang_names.get(lang, "English"),
        model_used=metadata["best_model"],
        warning=f"Found {len(found_symptoms)} symptoms: {', '.join(found_symptoms[:5])}"
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return {"error": "Internal server error. Please try again."}
