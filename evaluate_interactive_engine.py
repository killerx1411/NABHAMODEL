"""
evaluate_interactive_engine.py  — Fixed version

Key fixes vs original:
  1. Starts session with a SUBSET of true symptoms (simulates real user describing
     their main complaint), not empty — empty start means pure noise for 5 questions
  2. Caps question loop correctly so we never loop forever
  3. Reads questions_asked from session state regardless of final/questioning status
  4. Uses top_n_candidates=4 to match your 4-class model

Run:
  python evaluate_interactive_engine.py
"""

import joblib
import os
import numpy as np
from collections import defaultdict
from app.interactive_diagnosis import create_session, answer_question, SESSIONS

MODEL_DIR = "model"

print("📂 Loading model artifacts...")
model       = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
X_test      = joblib.load(os.path.join(MODEL_DIR, "X_test.pkl"))
y_test      = joblib.load(os.path.join(MODEL_DIR, "y_test.pkl"))
symptom_list = joblib.load(os.path.join(MODEL_DIR, "symptom_list.pkl"))
le          = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

print(f"   Test samples  : {len(X_test)}")
print(f"   Symptom count : {len(symptom_list)}")
print(f"   Classes       : {le.classes_.tolist()}")

# ── CONFIG ──────────────────────────────────────────────────
# How many symptoms to give at session start (simulate patient describing complaint)
# 0 = fully blind (hard),  2-3 = realistic,  all = trivial
SEED_SYMPTOMS = 2   # give the model the first 2 symptoms the patient has
MAX_LOOP      = 10  # safety cap on the while-loop regardless of engine config
# ────────────────────────────────────────────────────────────

correct        = 0
total_q        = 0
per_class_correct = defaultdict(int)
per_class_total   = defaultdict(int)

print(f"\n🧪 Evaluating {len(X_test)} test cases (seed_symptoms={SEED_SYMPTOMS})...\n")

for i in range(len(X_test)):
    if i % 50 == 0:
        print(f"   [{i:>4}/{len(X_test)}] running...")

    SESSIONS.clear()

    # Build ground-truth symptom list for this sample
    true_vector   = X_test.iloc[i].values
    true_label_idx = int(y_test[i])
    true_label     = le.inverse_transform([true_label_idx])[0]
    true_symptoms  = [symptom_list[j] for j in range(len(symptom_list)) if true_vector[j] == 1]

    # Seed: give the first SEED_SYMPTOMS symptoms upfront
    initial = true_symptoms[:SEED_SYMPTOMS]

    # ── Start session ────────────────────────────────────────
    result = create_session(model, le, symptom_list, initial_symptoms=initial, lang="en")
    session_id = result["session_id"]

    loop_count = 0

    # ── Question loop ────────────────────────────────────────
    while result["status"] == "questioning" and loop_count < MAX_LOOP:
        loop_count += 1

        next_q = result.get("next_question")
        if next_q is None:
            break

        symptom_asked = next_q["symptom"]
        user_answer   = symptom_asked in true_symptoms   # ground-truth oracle

        result = answer_question(session_id, symptom_asked, user_answer, model, le, symptom_list)

    # ── Extract final prediction ─────────────────────────────
    preds_key = "final_predictions" if "final_predictions" in result else "current_predictions"
    predicted = result[preds_key][0]["disease"]

    # questions_asked lives in result OR in session store
    q_asked = result.get("questions_asked", SESSIONS.get(session_id, {}).get("questions_asked", loop_count))

    # ── Score ────────────────────────────────────────────────
    if predicted == true_label:
        correct += 1
        per_class_correct[true_label] += 1
    per_class_total[true_label] += 1
    total_q += q_asked

n = len(X_test)
interactive_acc = correct / n
avg_q           = total_q / n

print("\n══════════════════════════════════════════════════")
print("📊  INTERACTIVE ENGINE EVALUATION RESULTS")
print("══════════════════════════════════════════════════")
print(f"  Interactive Accuracy : {interactive_acc*100:.2f}%  ({correct}/{n})")
print(f"  Avg Questions Asked  : {avg_q:.2f}")
print(f"  Seed Symptoms Given  : {SEED_SYMPTOMS}")
print()
print("  Per-class accuracy:")
for cls in sorted(per_class_total.keys()):
    c = per_class_correct[cls]
    t = per_class_total[cls]
    bar = "█" * int(20 * c / t)
    print(f"    {c:>3}/{t:<3}  {cls:<35} {bar}  {c/t*100:.1f}%")
print("══════════════════════════════════════════════════")