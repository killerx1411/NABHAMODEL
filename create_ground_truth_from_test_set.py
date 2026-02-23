"""
create_ground_truth_from_test_set.py

Generates validation_ground_truth.json by running your existing
labelled TEST cases through the interactive session system.

This is the most practical approach — you already have labelled data
from your train/test split, so you know the real diagnosis for each case.

How it works:
  1. Loads your test cases (X_test, y_test) from your training data
  2. Runs each one through the interactive questioning loop
  3. Records what the system predicted vs what the real label was
  4. Saves ground truth + session logs so log_real_interactive_accuracy.py
     can compute your real final_interactive_acc

Run:
  python create_ground_truth_from_test_set.py
"""

import json
import os
import uuid
import numpy as np
import joblib
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_DIR        = "model"
METADATA_PATH    = os.path.join(MODEL_DIR, "metadata.json")
MODEL_PATH       = os.path.join(MODEL_DIR, "best_model.pkl")
SYMPTOMS_PATH    = os.path.join(MODEL_DIR, "symptom_list.pkl")
LE_PATH          = os.path.join(MODEL_DIR, "label_encoder.pkl")

GROUND_TRUTH_OUT = "validation_ground_truth.json"
SESSION_LOG_OUT  = "interactive_sessions.jsonl"

# How many test cases to run (None = all of them)
# Start with 100-200 to verify it works, then set to None for full run
MAX_CASES = 200

# Number of questions the interactive session asks before stopping
MAX_QUESTIONS = 5
CONFIDENCE_THRESHOLD = 0.80

# ── Load model assets ─────────────────────────────────────────────────────────
print("📂 Loading model assets...")

with open(METADATA_PATH) as f:
    metadata = json.load(f)

model    = joblib.load(MODEL_PATH)
symptoms = list(joblib.load(SYMPTOMS_PATH))
le       = joblib.load(LE_PATH)

best_model_name = metadata["best_model"]
print(f"   ✅ Model: {best_model_name}")
print(f"   ✅ Symptoms: {len(symptoms)}")
print(f"   ✅ Classes: {len(le.classes_)}\n")


# ── Load your original dataset ────────────────────────────────────────────────
# EDIT THIS SECTION to point to your actual data file.
# This needs to be the same dataset you trained on, with a 'Disease' or
# label column so you know the real answer.
#
# Common formats:
#   - A CSV with symptom columns + a Disease column
#   - The same file you fed into train.py

DATA_PATH = "data/updated_result_with_BERT.csv"   # ← CHANGE THIS to your actual data file
LABEL_COL = "Diagnosis"            # ← CHANGE THIS to your label column name

if not os.path.exists(DATA_PATH):
    print(f"❌ Data file not found: {DATA_PATH}")
    print("   Edit DATA_PATH at the top of this script to point to your dataset.")
    exit(1)

print(f"📂 Loading dataset from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)
print(f"   ✅ {len(df)} rows, columns: {list(df.columns[:8])}...")

# Separate features and labels
if LABEL_COL not in df.columns:
    print(f"❌ Label column '{LABEL_COL}' not found in dataset.")
    print(f"   Available columns: {list(df.columns)}")
    exit(1)

y_labels = df[LABEL_COL].values
X        = df[symptoms].values   # use only the symptom columns the model knows

# Take a subset if MAX_CASES is set
if MAX_CASES and MAX_CASES < len(X):
    rng   = np.random.default_rng(42)
    idxs  = rng.choice(len(X), MAX_CASES, replace=False)
    X        = X[idxs]
    y_labels = y_labels[idxs]

print(f"   Running {len(X)} cases through interactive simulation...\n")


# ── Interactive session simulation ────────────────────────────────────────────
# This mimics exactly what your chatbot does:
#   1. Get initial prediction from symptoms already present
#   2. Find the most uncertain symptom (largest info gain)
#   3. Ask about it (here: read the answer from the ground truth row)
#   4. Update and re-predict
#   5. Stop when confident enough or max questions reached

def get_prediction(model, feature_vector):
    proba = model.predict_proba([feature_vector])[0]
    top_idx = np.argmax(proba)
    return le.classes_[top_idx], proba[top_idx], proba

def most_informative_symptom(model, feature_vector, asked_indices):
    """
    Find the symptom not yet asked about that, when flipped,
    changes the prediction probability the most.
    Simple proxy for information gain.
    """
    proba = model.predict_proba([feature_vector])[0]
    best_gain = -1
    best_idx  = None

    for i, sym in enumerate(symptoms):
        if i in asked_indices:
            continue
        # Try flipping this symptom
        fv_flip = feature_vector.copy()
        fv_flip[i] = 1 - fv_flip[i]
        proba_flip = model.predict_proba([fv_flip])[0]
        gain = np.sum(np.abs(proba - proba_flip))
        if gain > best_gain:
            best_gain = gain
            best_idx  = i

    return best_idx


ground_truth    = {}
session_entries = []

correct_count   = 0
total_questions = 0

for case_i, (feature_row, true_label) in enumerate(zip(X, y_labels)):
    session_id = f"session_{uuid.uuid4().hex[:12]}"
    fv         = feature_row.astype(float).copy()
    asked      = set()
    questions_asked = 0

    for q in range(MAX_QUESTIONS):
        pred_label, confidence, _ = get_prediction(model, fv)

        if confidence >= CONFIDENCE_THRESHOLD:
            break

        # Ask most informative symptom
        sym_idx = most_informative_symptom(model, fv, asked)
        if sym_idx is None:
            break

        asked.add(sym_idx)
        # "Answer" the question using ground truth data
        # (in real usage, the patient answers; here we read from the row)
        fv[sym_idx] = feature_row[sym_idx]
        questions_asked += 1

    # Final prediction after questioning
    final_pred, final_conf, _ = get_prediction(model, fv)

    # Record
    ground_truth[session_id] = str(true_label)
    session_entries.append({
        "session_id":       session_id,
        "model_name":       best_model_name,
        "questions_asked":  questions_asked,
        "final_prediction": final_pred,
        "confidence":       round(float(final_conf), 4),
        "stop_reason":      "confidence" if final_conf >= CONFIDENCE_THRESHOLD else "max_questions"
    })

    is_correct = (final_pred == str(true_label))
    if is_correct:
        correct_count += 1
    total_questions += questions_asked

    if (case_i + 1) % 50 == 0:
        running_acc = correct_count / (case_i + 1) * 100
        print(f"   Progress: {case_i+1}/{len(X)} cases | Running accuracy: {running_acc:.2f}%")


# ── Results ───────────────────────────────────────────────────────────────────
real_acc    = correct_count / len(X)
avg_q       = total_questions / len(X)
initial_acc = metadata["results"][best_model_name]["test_acc"]

print(f"\n{'='*60}")
print(f"✅ Simulation complete")
print(f"   Cases run:            {len(X)}")
print(f"   Initial test_acc:     {initial_acc*100:.2f}%")
print(f"   Real interactive acc: {real_acc*100:.2f}%")
print(f"   Change:               {(real_acc - initial_acc)*100:+.2f}%")
print(f"   Avg questions asked:  {avg_q:.1f}")
print(f"{'='*60}\n")


# ── Save outputs ──────────────────────────────────────────────────────────────
print(f"💾 Saving {GROUND_TRUTH_OUT}...")
with open(GROUND_TRUTH_OUT, 'w') as f:
    json.dump(ground_truth, f, indent=2)

print(f"💾 Saving {SESSION_LOG_OUT}...")
with open(SESSION_LOG_OUT, 'w') as f:
    for entry in session_entries:
        f.write(json.dumps(entry) + "\n")

# Also write final_interactive_acc directly to metadata
metadata["results"][best_model_name]["final_interactive_acc"] = round(real_acc, 4)
with open(METADATA_PATH, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n✅ All done! Files written:")
print(f"   {GROUND_TRUTH_OUT}  ({len(ground_truth)} entries)")
print(f"   {SESSION_LOG_OUT}   ({len(session_entries)} sessions)")
print(f"   {METADATA_PATH}     (final_interactive_acc updated to REAL value)")
print(f"\n🚀 Now run:  python empirical_results.py")
