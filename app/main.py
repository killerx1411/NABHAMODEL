"""
app/main.py — Multilingual Disease Prediction API with Interactive Diagnosis
Supports English, Hindi, and Punjabi with native language models
Includes sequential symptom elicitation for interactive diagnosis
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, field_validator
from typing import List, Optional
from contextlib import asynccontextmanager
from langdetect import detect
from fastapi.templating import Jinja2Templates
from app.interactive_diagnosis import create_session, answer_question, add_text_to_session, SESSIONS
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
    
    # For backward compatibility with interactive diagnosis, set these at app.state level
    app.state.model = app.state.models["en"]["model"]
    app.state.le = app.state.models["en"]["le"]
    app.state.symptom_list = app.state.models["en"]["symptom_list"]
    app.state.metadata = app.state.models["en"]["metadata"]
    
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
    title="Multilingual Disease Predictor API with Interactive Diagnosis",
    description="Disease prediction in English, Hindi (हिन्दी), and Punjabi (ਪੰਜਾਬੀ) with sequential symptom elicitation",
    version="2.0.0",
    lifespan=lifespan
)
templates = Jinja2Templates(directory="app/templates")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def detect_language(text: str) -> str:
    """Detect language from text. Returns 'en', 'hi', or 'pa'."""
    try:
        lang = detect(text)
        if lang in ['hi', 'mr', 'ne']:  # Devanagari-script languages → treat as Hindi
            return 'hi'
        elif lang in ['pa']:
            return 'pa'
        else:
            return 'en'
    except Exception:
        return 'en'  # Default to English


def confidence_label(score: float) -> str:
    if score >= 70:
        return "High"
    elif score >= 40:
        return "Moderate"
    return "Low"


def extract_symptoms_from_text(text: str, symptom_list: List[str]) -> List[str]:
    """
    Extract symptoms from natural language text by case-insensitive substring match.
    """
    text_lower = text.lower()
    return [symptom for symptom in symptom_list if symptom.lower() in text_lower]


def _resolve_model(request: Request, lang: str):
    """
    Return (model, le, symptom_list, metadata, resolved_lang) for the given language.
    Falls back to English if the requested language model is not loaded.
    """
    model_data = request.app.state.models.get(lang)
    if not model_data:
        logger.warning(f"Model for '{lang}' not available, falling back to English")
        lang = "en"
        model_data = request.app.state.models["en"]
    return (
        model_data["model"],
        model_data["le"],
        model_data["symptom_list"],
        model_data["metadata"],
        lang,
    )


# ─────────────────────────────────────────────────────────────
# SCHEMAS - REGULAR PREDICTION
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
# SCHEMAS - INTERACTIVE DIAGNOSIS
# ─────────────────────────────────────────────────────────────

class InteractiveStartRequest(BaseModel):
    text: str
    language: Optional[str] = None  # ← NEW: "en", "hi", "pa", or None for auto-detect

    @field_validator("text")
    @classmethod
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError("Text cannot be empty.")
        return v.strip()

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"text": "I have fever, headache and vomiting"},
                {"text": "मुझे बुखार और सिरदर्द है", "language": "hi"},
                {"text": "ਮੈਨੂੰ ਬੁਖਾਰ ਅਤੇ ਸਿਰ ਦਰਦ ਹੈ", "language": "pa"},
            ]
        }
    }


class InteractiveAnswerRequest(BaseModel):
    session_id: str
    symptom: str
    answer: bool  # True = yes, False = no
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "session_id": "abc-123-def",
                "symptom": "chills",
                "answer": True
            }]
        }
    }


class InteractiveAddTextRequest(BaseModel):
    session_id: str
    text: str

    @field_validator("text")
    @classmethod
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError("Text cannot be empty.")
        return v.strip()

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "session_id": "abc-123-def",
                "text": "I also have chills and sweating"
            }]
        }
    }


# ─────────────────────────────────────────────────────────────
# ROUTES - GENERAL
# ─────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "name": "Multilingual Disease Predictor API with Interactive Diagnosis",
        "version": "2.0.0",
        "languages": ["English", "हिन्दी (Hindi)", "ਪੰਜਾਬੀ (Punjabi)"],
        "features": ["Batch prediction", "Interactive diagnosis", "Multilingual support"],
        "docs": "/docs",
        "interactive_demo": "/predict_interactive/demo"
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


# ─────────────────────────────────────────────────────────────
# ROUTES - REGULAR PREDICTION
# ─────────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: Request, body: PredictRequest):
    """
    Multilingual disease prediction (batch mode).
    
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
    
    model, le, symptom_list, metadata, lang = _resolve_model(request, lang)
    
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
    
    probs = model.predict_proba(vector.reshape(1, -1))[0]
    
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


# ─────────────────────────────────────────────────────────────
# ROUTES - INTERACTIVE DIAGNOSIS
# ─────────────────────────────────────────────────────────────
@app.post("/predict_interactive/start", tags=["Interactive Diagnosis"])
def start_interactive_diagnosis(request: Request, body: InteractiveStartRequest):
    """
    Start an interactive diagnosis session.

    Accepts free-text describing symptoms in English, Hindi, or Punjabi.
    Pass `language` explicitly ("en"/"hi"/"pa") or let the API auto-detect it.

    The system extracts recognized symptoms, then asks follow-up yes/no questions
    IN THE SAME LANGUAGE to narrow down the diagnosis.

    Returns:
        - session_id: Use this in subsequent /answer and /add_text calls
        - current_predictions: Top 3 diseases with probabilities
        - next_question: First question (in the detected/specified language)
        - status: "questioning" or "complete"
        - language: Resolved language code used for this session
    """
    # ── resolve language ──────────────────────────────────────
    lang = body.language.lower() if body.language else detect_language(body.text)
    if lang not in ['en', 'hi', 'pa']:
        lang = 'en'

    # ── pick the right model ──────────────────────────────────
    model, le, symptom_list, _, lang = _resolve_model(request, lang)

    # ── extract symptoms ──────────────────────────────────────
    recognized = extract_symptoms_from_text(body.text, symptom_list)

    if len(recognized) < 1:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "No recognized symptoms found in your text.",
                "provided_text": body.text,
                "hint": "Try describing your symptoms more explicitly, e.g. 'I have fever and headache'.",
                "language_detected": lang
            }
        )

    # ── create session with language ──────────────────────────
    result = create_session(model, le, symptom_list, recognized, lang=lang)
    return result


@app.post("/predict_interactive/answer", tags=["Interactive Diagnosis"])
def answer_interactive_question(request: Request, body: InteractiveAnswerRequest):
    """
    Answer a yes/no question in an interactive diagnosis session.

    Language is remembered from the session automatically — no need to pass it again.
    The next question will be returned in the same language as the session.

    After answering, the system will either:
    - Ask another question (status="questioning")
    - Provide final diagnosis (status="complete")
    """
    # ── look up session to get language, then resolve matching model ──
    session = SESSIONS.get(body.session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {body.session_id} not found")

    lang = session.get("lang", "en")
    model, le, symptom_list, _, _ = _resolve_model(request, lang)

    try:
        result = answer_question(
            body.session_id,
            body.symptom,
            body.answer,
            model,
            le,
            symptom_list
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/predict_interactive/add_text", tags=["Interactive Diagnosis"])
def add_text_to_active_session(request: Request, body: InteractiveAddTextRequest):
    """
    Add more free-text symptoms to an active diagnosis session.

    Extracts any new symptoms from the provided text and merges them into
    the existing session without resetting it. Uses the session's original
    language model. Probabilities are recalculated.

    Returns:
        - session_id
        - updated_predictions: Recalculated top 3 diseases
        - new_symptoms_added: Symptoms extracted and added from the new text
    """
    # ── look up session to get language, then resolve matching model ──
    session = SESSIONS.get(body.session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {body.session_id} not found")

    lang = session.get("lang", "en")
    model, le, symptom_list, _, _ = _resolve_model(request, lang)

    try:
        result = add_text_to_session(
            body.session_id,
            body.text,
            model,
            le,
            symptom_list
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/predict_interactive/demo", tags=["Interactive Diagnosis"])
def interactive_demo_page():
    """
    Serve a simple HTML demo page for interactive diagnosis.
    Supports English, Hindi, and Punjabi with a language selector.
    """
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Interactive Disease Diagnosis</title>
    <meta charset="utf-8">
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 { color: #2c3e50; }
        .step { margin: 20px 0; }
        .predictions {
            background: #e8f4f8;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
        }
        .disease {
            padding: 10px;
            margin: 5px 0;
            background: white;
            border-left: 4px solid #3498db;
        }
        .question {
            background: #fff3cd;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
            font-size: 18px;
        }
        button {
            padding: 12px 30px;
            margin: 10px 5px;
            font-size: 16px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .btn-yes { background: #28a745; color: white; }
        .btn-no { background: #dc3545; color: white; }
        .btn-start { background: #007bff; color: white; }
        .btn-add { background: #6f42c1; color: white; }
        input, select {
            width: 100%;
            padding: 10px;
            font-size: 16px;
            border: 2px solid #ddd;
            border-radius: 5px;
            box-sizing: border-box;
            margin-bottom: 10px;
        }
        .complete { background: #d4edda; padding: 20px; border-radius: 5px; margin: 20px 0; }
        .add-text-box { background: #f0e6ff; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .lang-badge {
            display: inline-block;
            background: #17a2b8;
            color: white;
            border-radius: 4px;
            padding: 2px 10px;
            font-size: 13px;
            margin-left: 8px;
            vertical-align: middle;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏥 Interactive Disease Diagnosis</h1>
        <p>Describe your symptoms in English, Hindi, or Punjabi. Follow-up questions will appear in the same language.</p>

        <div id="step1" class="step">
            <h3>Step 1: Choose language &amp; describe symptoms</h3>
            <select id="langSelect">
                <option value="">Auto-detect language</option>
                <option value="en">English</option>
                <option value="hi">हिन्दी (Hindi)</option>
                <option value="pa">ਪੰਜਾਬੀ (Punjabi)</option>
            </select>
            <input type="text" id="symptoms"
                placeholder="e.g. I have fever, headache and vomiting  /  मुझे बुखार और सिरदर्द है  /  ਮੈਨੂੰ ਬੁਖਾਰ ਅਤੇ ਸਿਰ ਦਰਦ ਹੈ">
            <button class="btn-start" onclick="startDiagnosis()">Start Diagnosis</button>
        </div>

        <div id="step2" class="step" style="display:none;">
            <h3>Current Predictions <span id="langBadge" class="lang-badge"></span></h3>
            <div id="predictions" class="predictions"></div>

            <div class="add-text-box">
                <strong>Add More Symptoms:</strong>
                <input type="text" id="addTextInput"
                    placeholder="Describe additional symptoms in the same language...">
                <button class="btn-add" onclick="addMoreSymptoms()">➕ Add Symptoms</button>
            </div>

            <div id="questionBox" class="question" style="display:none;">
                <strong>Question:</strong>
                <p id="questionText"></p>
                <button class="btn-yes" onclick="answerQuestion(true)">✓ Yes</button>
                <button class="btn-no" onclick="answerQuestion(false)">✗ No</button>
            </div>

            <div id="complete" class="complete" style="display:none;">
                <h3>✅ Diagnosis Complete</h3>
                <p id="stopReason"></p>
                <p id="questionsAsked"></p>
                <button class="btn-start" onclick="location.reload()">Start New Diagnosis</button>
            </div>
        </div>
    </div>

    <script>
        let sessionId = null;
        let currentSymptom = null;

        const LANG_LABELS = { en: 'English', hi: 'हिन्दी', pa: 'ਪੰਜਾਬੀ' };

        async function startDiagnosis() {
            const text = document.getElementById('symptoms').value.trim();
            const langVal = document.getElementById('langSelect').value;
            if (!text) return;

            const body = { text };
            if (langVal) body.language = langVal;

            const response = await fetch('/predict_interactive/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });

            const data = await response.json();
            if (!response.ok) { alert(JSON.stringify(data.detail)); return; }

            sessionId = data.session_id;
            const lang = data.language || 'en';
            document.getElementById('langBadge').textContent = LANG_LABELS[lang] || lang;
            document.getElementById('step1').style.display = 'none';
            document.getElementById('step2').style.display = 'block';
            displayResults(data);
        }

        async function answerQuestion(answer) {
            const response = await fetch('/predict_interactive/answer', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId, symptom: currentSymptom, answer })
            });
            const data = await response.json();
            displayResults(data);
        }

        async function addMoreSymptoms() {
            const text = document.getElementById('addTextInput').value.trim();
            if (!text) return;

            const response = await fetch('/predict_interactive/add_text', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId, text })
            });
            const data = await response.json();
            if (!response.ok) { alert(JSON.stringify(data.detail)); return; }

            document.getElementById('addTextInput').value = '';

            const predictions = data.updated_predictions;
            let html = '';
            predictions.forEach(p => {
                html += `<div class="disease"><strong>${p.rank}. ${p.disease}</strong> — ${p.confidence}% confidence</div>`;
            });
            document.getElementById('predictions').innerHTML = html;
        }

        function displayResults(data) {
            const predictions = data.current_predictions || data.final_predictions;
            let html = '';
            predictions.forEach(p => {
                html += `<div class="disease"><strong>${p.rank}. ${p.disease}</strong> — ${p.confidence}% confidence</div>`;
            });
            document.getElementById('predictions').innerHTML = html;

            if (data.status === 'questioning' && data.next_question) {
                document.getElementById('questionText').textContent = data.next_question.question;
                currentSymptom = data.next_question.symptom;
                document.getElementById('questionBox').style.display = 'block';
                document.getElementById('complete').style.display = 'none';
            } else {
                document.getElementById('questionBox').style.display = 'none';
                document.getElementById('complete').style.display = 'block';
                document.getElementById('stopReason').textContent = data.stop_reason || 'Diagnosis complete';
                document.getElementById('questionsAsked').textContent = `Questions asked: ${data.questions_asked || 0}`;
            }
        }
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


@app.get("/interactive", response_class=HTMLResponse)
def interactive_page(request: Request):
    return templates.TemplateResponse("interactive.html", {"request": request})

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error. Please try again."}
    )