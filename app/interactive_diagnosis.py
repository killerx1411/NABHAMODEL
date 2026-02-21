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
MIN_INFORMATION_GAIN = 0.01  # Stop if no question provides meaningful info


# ─────────────────────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────

def calculate_entropy(probabilities: np.ndarray) -> float:
    """Calculate Shannon entropy of probability distribution."""
    # Filter out zero probabilities to avoid log(0)
    probs = probabilities[probabilities > 0]
    return entropy(probs)


def predict_with_symptoms(model, symptom_list: List[str], 
                         present_symptoms: List[str]) -> np.ndarray:
    """
    Get disease probabilities given a set of present symptoms.
    
    Args:
        model: Trained classifier
        symptom_list: Full vocabulary of symptoms
        present_symptoms: Symptoms the patient has reported
        
    Returns:
        Array of probabilities for each disease
    """
    # Build binary vector
    vector = np.zeros(len(symptom_list))
    for symptom in present_symptoms:
        if symptom in symptom_list:
            idx = symptom_list.index(symptom)
            vector[idx] = 1
    
    # Predict
    vector_df = pd.DataFrame([vector], columns=symptom_list)
    return model.predict_proba(vector_df)[0]


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
    
    # Scenario 1: User says YES (they have the symptom)
    probs_if_yes = predict_with_symptoms(
        model, symptom_list, present_symptoms + [candidate_symptom]
    )
    entropy_if_yes = calculate_entropy(probs_if_yes)
    
    # Scenario 2: User says NO (they don't have the symptom)
    # We can't directly model "absence" easily with the current setup,
    # so we approximate: if symptom is strong for top diseases but user says NO,
    # those diseases become less likely
    # For simplicity, use current_entropy as upper bound
    entropy_if_no = current_entropy
    
    # Estimate P(yes) from current top diseases
    # Simple heuristic: if top diseases have this symptom in training data,
    # P(yes) is higher
    # For now, assume 50/50 (can be refined with training data analysis)
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
    for symptom in unanswered_symptoms[:50]:  # Limit to top 50 for speed
        ig = calculate_information_gain(
            model, symptom_list, present_symptoms, symptom, current_probs
        )
        gains[symptom] = ig
    
    # Get symptom with highest IG
    best_symptom = max(gains, key=gains.get)
    best_ig = gains[best_symptom]
    
    if best_ig < MIN_INFORMATION_GAIN:
        return None  # No question provides meaningful information
    
    # Generate natural language question
    question_text = generate_question_text(best_symptom)
    
    return {
        "symptom": best_symptom,
        "question": question_text,
        "information_gain": round(float(best_ig), 4)
    }


def generate_question_text(symptom: str) -> str:
    """Convert symptom name to natural language question."""
    # Clean up symptom name
    symptom_readable = symptom.replace("_", " ").replace("  ", " ")
    
    # Generate question based on symptom type
    pain_keywords = ["pain", "ache", "hurt"]
    if any(kw in symptom for kw in pain_keywords):
        return f"Do you have {symptom_readable}?"
    
    if symptom.startswith("increased_") or symptom.startswith("decreased_"):
        return f"Have you noticed {symptom_readable}?"
    
    if "fever" in symptom or "temperature" in symptom:
        return f"Do you have {symptom_readable}?"
    
    # Default format
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
# SESSION MANAGEMENT
# ─────────────────────────────────────────────────────────────

def create_session(
    model,
    le,
    symptom_list: List[str],
    initial_symptoms: List[str]
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
    
    # Select next question
    next_q = select_next_question(
        model, le, symptom_list, initial_symptoms, [], current_probs
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
        "status": "complete" if should_stop else "questioning"
    }
    
    SESSIONS[session_id] = session
    
    return {
        "session_id": session_id,
        "current_predictions": predictions,
        "next_question": next_q,
        "status": session["status"],
        "stop_reason": stop_reason if should_stop else None
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
    
    Args:
        session_id: Session identifier
        symptom: The symptom that was asked about
        answer: True if user has it, False if not
        
    Returns:
        Updated session state with next question or final predictions
    """
    if session_id not in SESSIONS:
        raise ValueError(f"Session {session_id} not found")
    
    session = SESSIONS[session_id]
    
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
    
    # Select next question
    next_q = select_next_question(
        model, le, symptom_list,
        session["present_symptoms"],
        session["asked_symptoms"],
        current_probs
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
            "questions_asked": session["questions_asked"]
        }
    
    return {
        "session_id": session_id,
        "current_predictions": predictions,
        "next_question": next_q,
        "status": "questioning",
        "questions_asked": session["questions_asked"]
    }
