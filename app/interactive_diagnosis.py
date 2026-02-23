"""
app/interactive_diagnosis.py — Sequential symptom elicitation with information gain

This module implements interactive diagnosis where the system asks discriminating
questions to narrow down the disease, mimicking real physician diagnostic reasoning.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy.stats import entropy
import uuid

# ─────────────────────────────────────────────────────────────
# SESSION STORE (in-memory, use Redis for production)
# ─────────────────────────────────────────────────────────────
SESSIONS = {}

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.70  # Stop when top disease > 70%
MAX_QUESTIONS = 5            # Max questions to ask
MIN_INFORMATION_GAIN = 0.00  # Stop if no question provides meaningful info


# ─────────────────────────────────────────────────────────────
# MULTILINGUAL SYMPTOM TRANSLATIONS
# ─────────────────────────────────────────────────────────────

SYMPTOM_TRANSLATIONS_HI = {
    "fever": "बुखार", "headache": "सिरदर्द", "vomiting": "उल्टी",
    "nausea": "जी मिचलाना", "cough": "खांसी", "fatigue": "थकान",
    "chills": "ठंड लगना", "sweating": "पसीना आना", "joint_pain": "जोड़ों में दर्द",
    "body_pain": "शरीर में दर्द", "stomach_pain": "पेट दर्द",
    "abdominal_pain": "पेट में दर्द", "chest_pain": "सीने में दर्द",
    "back_pain": "पीठ दर्द", "neck_pain": "गर्दन दर्द",
    "muscle_pain": "मांसपेशियों में दर्द", "skin_rash": "त्वचा पर चकत्ते",
    "rash": "चकत्ते", "itching": "खुजली",
    "yellowing_of_eyes": "आंखों का पीला होना", "yellow_skin": "त्वचा पीली होना",
    "weight_loss": "वजन कम होना", "weight_gain": "वजन बढ़ना",
    "loss_of_appetite": "भूख न लगना", "diarrhea": "दस्त",
    "constipation": "कब्ज", "breathlessness": "सांस की तकलीफ",
    "shortness_of_breath": "सांस फूलना", "swelling": "सूजन",
    "swelled_lymph_nodes": "लसिका ग्रंथियों में सूजन",
    "runny_nose": "नाक बहना", "sore_throat": "गला दर्द",
    "throat_pain": "गले में दर्द", "dizziness": "चक्कर आना",
    "blurred_vision": "धुंधली दृष्टि", "weakness": "कमजोरी",
    "high_fever": "तेज बुखार", "irritability": "चिड़चिड़ापन",
    "mood_swings": "मूड में बदलाव", "depression": "अवसाद",
    "anxiety": "चिंता", "insomnia": "नींद न आना",
    "excessive_hunger": "बहुत अधिक भूख", "frequent_urination": "बार-बार पेशाब",
    "burning_micturition": "पेशाब में जलन", "dark_urine": "गहरे रंग का पेशाब",
    "pale_stool": "हल्के रंग का मल", "blood_in_urine": "पेशाब में खून",
    "blood_in_stool": "मल में खून", "increased_thirst": "अधिक प्यास",
    "increased_hunger": "अधिक भूख", "muscle_weakness": "मांसपेशियों में कमजोरी",
    "numbness": "सुन्नपन", "tingling": "झनझनाहट",
    "palpitations": "दिल की धड़कन तेज होना", "acidity": "एसिडिटी",
    "indigestion": "अपच", "heartburn": "सीने में जलन",
    "ulcers_on_tongue": "जीभ पर छाले", "patches_in_throat": "गले में धब्बे",
    "swollen_legs": "पैरों में सूजन", "movement_stiffness": "हलचल में अकड़न",
    "spinning_movements": "चक्कर", "loss_of_smell": "सूंघने की शक्ति खत्म होना",
    "loss_of_taste": "स्वाद खत्म होना", "muscle_cramps": "मांसपेशियों में ऐंठन",
}

SYMPTOM_TRANSLATIONS_PA = {
    "fever": "ਬੁਖਾਰ", "headache": "ਸਿਰ ਦਰਦ", "vomiting": "ਉਲਟੀ",
    "nausea": "ਮਤਲੀ", "cough": "ਖੰਘ", "fatigue": "ਥਕਾਵਟ",
    "chills": "ਕੰਬਣੀ", "sweating": "ਪਸੀਨਾ ਆਉਣਾ",
    "joint_pain": "ਜੋੜਾਂ ਵਿੱਚ ਦਰਦ", "body_pain": "ਸਰੀਰ ਵਿੱਚ ਦਰਦ",
    "stomach_pain": "ਪੇਟ ਦਰਦ", "abdominal_pain": "ਢਿੱਡ ਵਿੱਚ ਦਰਦ",
    "chest_pain": "ਛਾਤੀ ਵਿੱਚ ਦਰਦ", "back_pain": "ਪਿੱਠ ਦਰਦ",
    "neck_pain": "ਗਰਦਨ ਦਰਦ", "muscle_pain": "ਮਾਸਪੇਸ਼ੀਆਂ ਵਿੱਚ ਦਰਦ",
    "skin_rash": "ਚਮੜੀ 'ਤੇ ਧੱਫੜ", "rash": "ਧੱਫੜ", "itching": "ਖਾਰਸ਼",
    "yellowing_of_eyes": "ਅੱਖਾਂ ਦਾ ਪੀਲਾ ਹੋਣਾ", "yellow_skin": "ਚਮੜੀ ਪੀਲੀ ਹੋਣਾ",
    "weight_loss": "ਭਾਰ ਘਟਣਾ", "weight_gain": "ਭਾਰ ਵਧਣਾ",
    "loss_of_appetite": "ਭੁੱਖ ਨਾ ਲੱਗਣਾ", "diarrhea": "ਦਸਤ",
    "constipation": "ਕਬਜ਼", "breathlessness": "ਸਾਹ ਲੈਣ ਵਿੱਚ ਤਕਲੀਫ਼",
    "shortness_of_breath": "ਸਾਹ ਦੀ ਤਕਲੀਫ਼", "swelling": "ਸੋਜ",
    "swelled_lymph_nodes": "ਲਿੰਫ ਗ੍ਰੰਥੀਆਂ ਵਿੱਚ ਸੋਜ",
    "runny_nose": "ਨੱਕ ਵਗਣਾ", "sore_throat": "ਗਲੇ ਵਿੱਚ ਦਰਦ",
    "throat_pain": "ਗਲੇ ਵਿੱਚ ਦਰਦ", "dizziness": "ਚੱਕਰ ਆਉਣਾ",
    "blurred_vision": "ਧੁੰਦਲੀ ਨਜ਼ਰ", "weakness": "ਕਮਜ਼ੋਰੀ",
    "high_fever": "ਤੇਜ਼ ਬੁਖਾਰ", "irritability": "ਚਿੜਚਿੜਾਪਣ",
    "mood_swings": "ਮੂਡ ਵਿੱਚ ਬਦਲਾਅ", "depression": "ਉਦਾਸੀ",
    "anxiety": "ਚਿੰਤਾ", "insomnia": "ਨੀਂਦ ਨਾ ਆਉਣਾ",
    "excessive_hunger": "ਬਹੁਤ ਜ਼ਿਆਦਾ ਭੁੱਖ", "frequent_urination": "ਵਾਰ-ਵਾਰ ਪਿਸ਼ਾਬ",
    "burning_micturition": "ਪਿਸ਼ਾਬ ਵਿੱਚ ਜਲਣ", "dark_urine": "ਗਾੜ੍ਹੇ ਰੰਗ ਦਾ ਪਿਸ਼ਾਬ",
    "pale_stool": "ਹਲਕੇ ਰੰਗ ਦਾ ਮਲ", "blood_in_urine": "ਪਿਸ਼ਾਬ ਵਿੱਚ ਖੂਨ",
    "blood_in_stool": "ਮਲ ਵਿੱਚ ਖੂਨ", "increased_thirst": "ਜ਼ਿਆਦਾ ਪਿਆਸ",
    "increased_hunger": "ਜ਼ਿਆਦਾ ਭੁੱਖ",
    "muscle_weakness": "ਮਾਸਪੇਸ਼ੀਆਂ ਵਿੱਚ ਕਮਜ਼ੋਰੀ", "numbness": "ਸੁੰਨ ਹੋਣਾ",
    "tingling": "ਝਣਝਣਾਹਟ", "palpitations": "ਦਿਲ ਦੀ ਧੜਕਣ ਤੇਜ਼",
    "acidity": "ਐਸਿਡਿਟੀ", "indigestion": "ਬਦਹਜ਼ਮੀ",
    "heartburn": "ਛਾਤੀ ਵਿੱਚ ਜਲਣ", "ulcers_on_tongue": "ਜੀਭ 'ਤੇ ਛਾਲੇ",
    "patches_in_throat": "ਗਲੇ ਵਿੱਚ ਧੱਬੇ", "swollen_legs": "ਲੱਤਾਂ ਵਿੱਚ ਸੋਜ",
    "movement_stiffness": "ਹਿਲਜੁਲ ਵਿੱਚ ਅਕੜਾਅ", "spinning_movements": "ਚੱਕਰ ਆਉਣਾ",
    "loss_of_smell": "ਸੁੰਘਣ ਦੀ ਸ਼ਕਤੀ ਖਤਮ", "loss_of_taste": "ਸੁਆਦ ਦੀ ਸ਼ਕਤੀ ਖਤਮ",
    "muscle_cramps": "ਮਾਸਪੇਸ਼ੀਆਂ ਵਿੱਚ ਮਰੋੜ",
}


def translate_symptom(symptom: str, lang: str) -> str:
    """Translate an English symptom key to the target language readable string.
    Falls back to humanised English if no translation exists."""
    if lang == "hi":
        return SYMPTOM_TRANSLATIONS_HI.get(symptom, symptom.replace("_", " "))
    elif lang == "pa":
        return SYMPTOM_TRANSLATIONS_PA.get(symptom, symptom.replace("_", " "))
    return symptom.replace("_", " ")


# ─────────────────────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────

def calculate_entropy(probabilities: np.ndarray) -> float:
    """Calculate Shannon entropy of probability distribution."""
    # Filter out zero probabilities to avoid log(0)
    probs = probabilities[probabilities > 0]
    return entropy(probs)


def predict_with_symptoms(model, symptom_list, present_symptoms):

    vector = np.zeros(len(symptom_list))

    for symptom in present_symptoms:
        if symptom in symptom_list:
            idx = symptom_list.index(symptom)
            vector[idx] = 1

    # Convert to DataFrame with feature names
    df_vector = pd.DataFrame([vector], columns=symptom_list)

    return model.predict_proba(df_vector)[0]

    # Pass numpy array directly instead of DataFrame — fixes pandas/xgboost mismatch
    return model.predict_proba(vector.reshape(1, -1))[0]


def calculate_information_gain(
    model,
    symptom_list: List[str],
    present_symptoms: List[str],
    candidate_symptom: str,
    current_probs: np.ndarray
) -> float:
    """
    Calculate expected information gain if we ask about candidate_symptom.

    Information gain = current_entropy - expected_entropy_after_answer
    """
    current_entropy = calculate_entropy(current_probs)

    # Scenario 1: User says YES
    probs_if_yes = predict_with_symptoms(
        model, symptom_list, present_symptoms + [candidate_symptom]
    )
    entropy_if_yes = calculate_entropy(probs_if_yes)

    # Scenario 2: User says NO — approximate with current entropy
    entropy_if_no = current_entropy

    # Assume 50/50 prior
    p_yes = 0.5
    p_no = 0.5

    expected_entropy = p_yes * entropy_if_yes + p_no * entropy_if_no
    information_gain = current_entropy - expected_entropy

    return information_gain


def select_next_question(
    model,
    le,
    symptom_list: List[str],
    present_symptoms: List[str],
    asked_symptoms: List[str],
    current_probs: np.ndarray,
    lang: str = "en",           # ← NEW: controls question language
    top_n_candidates: int = 5
) -> Optional[Dict]:
    """
    Select the most informative symptom to ask about next.

    Returns:
        Dictionary with 'symptom' and 'question' text, or None if no good question
    """
    # Get top disease candidates
    top_disease_indices = np.argsort(current_probs)[::-1][:top_n_candidates]

    # Find symptoms we haven't asked about yet
    unanswered_symptoms = [
        s for s in symptom_list
        if s not in present_symptoms and s not in asked_symptoms
    ]

    if not unanswered_symptoms:
        return None

    # Calculate information gain for each unanswered symptom
    gains = {}
    for symptom in unanswered_symptoms[:5]:  # Limit to top 50 for speed
        ig = calculate_information_gain(
            model, symptom_list, present_symptoms, symptom, current_probs
        )
        gains[symptom] = ig

    # Get symptom with highest IG
    best_symptom = max(gains, key=gains.get)
    best_ig = gains[best_symptom]

    if best_ig < MIN_INFORMATION_GAIN:
        return None  # No question provides meaningful information

    # Generate natural language question in the correct language
    question_text = generate_question_text(best_symptom, lang)  # ← pass lang

    return {
        "symptom": best_symptom,
        "question": question_text,
        "information_gain": round(float(best_ig), 4)
    }


def generate_question_text(symptom: str, lang: str = "en") -> str:
    """Convert symptom name to natural language question in the target language."""
    symptom_readable = translate_symptom(symptom, lang)

    if lang == "hi":
        pain_keywords = ["pain", "ache", "hurt"]
        if any(kw in symptom for kw in pain_keywords):
            return f"क्या आपको {symptom_readable} हो रहा है?"
        if symptom.startswith("increased_") or symptom.startswith("decreased_"):
            return f"क्या आपने {symptom_readable} महसूस किया है?"
        return f"क्या आपको {symptom_readable} है?"

    elif lang == "pa":
        pain_keywords = ["pain", "ache", "hurt"]
        if any(kw in symptom for kw in pain_keywords):
            return f"ਕੀ ਤੁਹਾਨੂੰ {symptom_readable} ਹੋ ਰਿਹਾ ਹੈ?"
        if symptom.startswith("increased_") or symptom.startswith("decreased_"):
            return f"ਕੀ ਤੁਸੀਂ {symptom_readable} ਮਹਿਸੂਸ ਕੀਤਾ ਹੈ?"
        return f"ਕੀ ਤੁਹਾਨੂੰ {symptom_readable} ਹੈ?"

    else:
        # Original English logic — unchanged
        pain_keywords = ["pain", "ache", "hurt"]
        if any(kw in symptom for kw in pain_keywords):
            return f"Do you have {symptom_readable}?"
        if symptom.startswith("increased_") or symptom.startswith("decreased_"):
            return f"Have you noticed {symptom_readable}?"
        if "fever" in symptom or "temperature" in symptom:
            return f"Do you have {symptom_readable}?"
        return f"Do you have {symptom_readable}?"


def should_stop_asking(
    current_probs: np.ndarray,
    questions_asked: int,
    has_next_question: bool
) -> Tuple[bool, str]:
    """
    Determine if we should stop asking questions.

    Returns:
        (should_stop, reason)
    """
    top_confidence = float(np.max(current_probs))

    if top_confidence >= CONFIDENCE_THRESHOLD:
        return True, f"High confidence reached ({top_confidence*100:.1f}%)"

    if questions_asked >= MAX_QUESTIONS:
        return True, f"Maximum questions ({MAX_QUESTIONS}) reached"

    if not has_next_question:
        return True, "No more discriminating questions available"

    return False, ""


# ─────────────────────────────────────────────────────────────
# SYMPTOM EXTRACTION (local copy — avoids circular import with main.py)
# ─────────────────────────────────────────────────────────────

def _extract_symptoms_from_text(text: str, symptom_list: List[str]) -> List[str]:
    """
    Extract recognized symptoms from free-text input.
    Case-insensitive substring match against the model vocabulary.
    Identical logic to extract_symptoms_from_text() in main.py.
    """
    text_lower = text.lower()
    return [symptom for symptom in symptom_list if symptom.lower() in text_lower]


# ─────────────────────────────────────────────────────────────
# SESSION MANAGEMENT
# ─────────────────────────────────────────────────────────────

def create_session(
    model,
    le,
    symptom_list: List[str],
    initial_symptoms: List[str],
    lang: str = "en"            # ← NEW: language stored in session
) -> Dict:
    """
    Start a new interactive diagnosis session.

    Returns:
        Session data with session_id, current_probs, next_question
    """
    session_id = str(uuid.uuid4())

    # Get initial predictions
    current_probs = predict_with_symptoms(model, symptom_list, initial_symptoms)

    # Get top diseases
    top3_idx = np.argsort(current_probs)[::-1][:3]
    predictions = [
        {
            "rank": i + 1,
            "disease": le.inverse_transform([idx])[0],
            "confidence": round(float(current_probs[idx]) * 100, 2)
        }
        for i, idx in enumerate(top3_idx)
    ]

    # Select next question (in the right language)
    next_q = select_next_question(
        model, le, symptom_list, initial_symptoms, [], current_probs, lang  # ← pass lang
    )

    # Check if we should stop immediately
    should_stop, stop_reason = should_stop_asking(
        current_probs, 0, next_q is not None
    )

    session = {
        "session_id": session_id,
        "present_symptoms": initial_symptoms,
        "asked_symptoms": [],
        "current_probs": current_probs.tolist(),
        "questions_asked": 0,
        "status": "complete" if should_stop else "questioning",
        "lang": lang            # ← NEW: store language so every answer call uses it
    }

    SESSIONS[session_id] = session

    return {
        "session_id": session_id,
        "current_predictions": predictions,
        "next_question": next_q,
        "status": session["status"],
        "stop_reason": stop_reason if should_stop else None,
        "language": lang        # ← NEW: expose in response
    }


def answer_question(
    session_id: str,
    symptom: str,
    answer: bool,
    model,
    le,
    symptom_list: List[str]
) -> Dict:
    """
    Process user's answer to a question and return next question or final result.

    Language is read automatically from the session — no API change needed.
    """
    if session_id not in SESSIONS:
        raise ValueError(f"Session {session_id} not found")

    session = SESSIONS[session_id]
    lang = session.get("lang", "en")    # ← NEW: read persisted language

    # Update session state
    if answer:
        session["present_symptoms"].append(symptom)
    session["asked_symptoms"].append(symptom)
    session["questions_asked"] += 1

    # Get updated predictions
    current_probs = predict_with_symptoms(
        model, symptom_list, session["present_symptoms"]
    )
    session["current_probs"] = current_probs.tolist()

    # Get top diseases
    top3_idx = np.argsort(current_probs)[::-1][:3]
    predictions = [
        {
            "rank": i + 1,
            "disease": le.inverse_transform([idx])[0],
            "confidence": round(float(current_probs[idx]) * 100, 2)
        }
        for i, idx in enumerate(top3_idx)
    ]

    # Select next question in session language
    next_q = select_next_question(
        model, le, symptom_list,
        session["present_symptoms"],
        session["asked_symptoms"],
        current_probs,
        lang                            # ← NEW: pass language
    )

    # Check stopping criteria
    should_stop, stop_reason = should_stop_asking(
        current_probs, session["questions_asked"], next_q is not None
    )

    if should_stop:
        session["status"] = "complete"
        return {
            "session_id": session_id,
            "final_predictions": predictions,
            "status": "complete",
            "stop_reason": stop_reason,
            "questions_asked": session["questions_asked"],
            "language": lang            # ← NEW
        }

    return {
        "session_id": session_id,
        "current_predictions": predictions,
        "next_question": next_q,
        "status": "questioning",
        "questions_asked": session["questions_asked"],
        "language": lang                # ← NEW
    }


def add_text_to_session(
    session_id: str,
    text: str,
    model,
    le,
    symptom_list: List[str]
) -> Dict:
    """
    Add more free-text symptoms to an active session without resetting it.

    Extracts newly recognized symptoms from text, merges them into the session's
    present_symptoms (skipping duplicates), and recalculates probabilities.
    Language is read automatically from the session.
    """
    if session_id not in SESSIONS:
        raise ValueError(f"Session {session_id} not found")

    session = SESSIONS[session_id]

    # Extract new symptoms from text
    extracted = _extract_symptoms_from_text(text, symptom_list)

    # Add only symptoms not already present
    new_symptoms = []
    for symptom in extracted:
        if symptom not in session["present_symptoms"]:
            session["present_symptoms"].append(symptom)
            new_symptoms.append(symptom)

    # Recompute probabilities
    current_probs = predict_with_symptoms(
        model,
        symptom_list,
        session["present_symptoms"]
    )
    session["current_probs"] = current_probs.tolist()

    # Get top diseases
    top3_idx = np.argsort(current_probs)[::-1][:3]
    predictions = [
        {
            "rank": i + 1,
            "disease": le.inverse_transform([idx])[0],
            "confidence": round(float(current_probs[idx]) * 100, 2)
        }
        for i, idx in enumerate(top3_idx)
    ]

    return {
        "session_id": session_id,
        "updated_predictions": predictions,
        "new_symptoms_added": new_symptoms,
        "language": session.get("lang", "en")   # ← NEW
    }