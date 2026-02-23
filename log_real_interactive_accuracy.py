"""
log_real_interactive_accuracy.py

Reads REAL interactive session logs from your API and computes
the actual final_interactive_acc for each model.

HOW IT WORKS:
  - Reads interactive_sessions.jsonl (logged by your /predict_interactive API)
  - Matches sessions to ground truth labels
  - Computes real accuracy: was the top-1 final prediction correct?
  - Saves the real value to metadata.json

BEFORE RUNNING:
  1. Make sure your API logs sessions to interactive_sessions.jsonl
     (see SESSION LOGGING section below for what to add to your API)
  2. Make sure you have a validation ground truth file
     (see GROUND TRUTH section below)

Run:
  python log_real_interactive_accuracy.py
"""

import json
import os
import pandas as pd

MODEL_DIR            = "model"
METADATA_PATH        = os.path.join(MODEL_DIR, "metadata.json")
SESSION_LOG_PATH     = "interactive_sessions.jsonl"   # written by your API
GROUND_TRUTH_PATH    = "validation_ground_truth.json" # you provide this

# ══════════════════════════════════════════════════════════════
# STEP 1: Add this logging block to your FastAPI endpoint
# ══════════════════════════════════════════════════════════════
#
# In your /predict_interactive/answer endpoint, after the model
# returns a 'complete' result, append one JSON line per session:
#
#   from datetime import datetime
#   import json
#
#   if result['status'] == 'complete':
#       log_entry = {
#           "timestamp":         datetime.now().isoformat(),
#           "session_id":        body.session_id,
#           "model_name":        "XGBoost",          # whichever model handled it
#           "questions_asked":   result['questions_asked'],
#           "final_prediction":  result['final_predictions'][0]['disease'],
#           "confidence":        result['final_predictions'][0]['confidence'],
#           "stop_reason":       result['stop_reason']
#       }
#       with open("interactive_sessions.jsonl", "a") as f:
#           f.write(json.dumps(log_entry) + "\n")
#
# ══════════════════════════════════════════════════════════════
# STEP 2: Create validation_ground_truth.json
# ══════════════════════════════════════════════════════════════
#
# This maps session_id → true disease label for sessions where
# you know the real answer (e.g. doctor-confirmed cases).
#
# Format:
#   {
#     "session_abc123": "Lumbar Disc Herniation",
#     "session_def456": "Osteoarthritis",
#     ...
#   }
#
# ══════════════════════════════════════════════════════════════

def main():
    # ── Load metadata ──────────────────────────────────────────
    print(f"📂 Loading {METADATA_PATH}...")
    if not os.path.exists(METADATA_PATH):
        print(f"❌ Not found: {METADATA_PATH}. Run train.py first.")
        return

    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)

    # ── Check session log ──────────────────────────────────────
    if not os.path.exists(SESSION_LOG_PATH):
        print(f"❌ Session log not found: {SESSION_LOG_PATH}")
        print("   You need to log real interactive sessions from your API first.")
        print("   See the comments at the top of this file for what to add to your API.")
        return

    # ── Load session logs ──────────────────────────────────────
    sessions = []
    with open(SESSION_LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                sessions.append(json.loads(line))

    if not sessions:
        print(f"❌ No sessions found in {SESSION_LOG_PATH}")
        return

    print(f"✅ Loaded {len(sessions)} interactive sessions")

    # ── Load ground truth ──────────────────────────────────────
    if not os.path.exists(GROUND_TRUTH_PATH):
        print(f"❌ Ground truth not found: {GROUND_TRUTH_PATH}")
        print("   Create this file mapping session_id → true disease label.")
        print("   See the comments at the top of this file for the format.")
        return

    with open(GROUND_TRUTH_PATH) as f:
        ground_truth = json.load(f)

    print(f"✅ Loaded {len(ground_truth)} ground truth labels")

    # ── Build dataframe ────────────────────────────────────────
    df = pd.DataFrame(sessions)

    # Match ground truth
    df['true_disease'] = df['session_id'].map(ground_truth)

    # Drop sessions with no ground truth
    matched = df.dropna(subset=['true_disease'])
    unmatched = len(df) - len(matched)
    if unmatched > 0:
        print(f"⚠️  {unmatched} sessions had no ground truth label — skipped")

    if matched.empty:
        print("❌ No sessions could be matched to ground truth. Cannot compute accuracy.")
        return

    print(f"✅ {len(matched)} sessions matched to ground truth\n")

    # ── Calculate accuracy per model ───────────────────────────
    print("="*60)
    print("📊 Real Interactive Accuracy by Model")
    print("="*60)

    results = metadata.get("results", {})
    updated_models = []

    # Group by model if multiple models logged
    if 'model_name' in matched.columns:
        groups = matched.groupby('model_name')
    else:
        # Single model — apply to best model
        best = metadata.get("best_model")
        groups = [(best, matched)]

    for model_name, group in groups:
        if model_name not in results:
            print(f"⚠️  Model '{model_name}' not in metadata.json — skipping")
            continue

        group = group.copy()
        group['correct'] = group['final_prediction'] == group['true_disease']

        real_acc        = group['correct'].mean()
        avg_questions   = group['questions_asked'].mean() if 'questions_asked' in group.columns else None
        n_sessions      = len(group)
        initial_acc     = results[model_name].get("test_acc", 0)
        improvement     = (real_acc - initial_acc) * 100

        # Save to metadata
        results[model_name]['final_interactive_acc'] = round(real_acc, 4)
        updated_models.append(model_name)

        print(f"\n  Model:              {model_name}")
        print(f"  Sessions used:      {n_sessions}")
        print(f"  Initial test acc:   {initial_acc*100:.2f}%")
        print(f"  Final interactive:  {real_acc*100:.2f}%  ← REAL")
        if improvement >= 0:
            print(f"  Improvement:        +{improvement:.2f}%")
        else:
            print(f"  Change:             {improvement:.2f}%  (interactive was harder)")
        if avg_questions is not None:
            print(f"  Avg questions:      {avg_questions:.1f}")

    print("\n" + "="*60)

    if not updated_models:
        print("❌ No models were updated.")
        return

    # ── Save updated metadata ──────────────────────────────────
    print(f"\n💾 Saving real values to {METADATA_PATH}...")
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Done! Updated: {', '.join(updated_models)}")
    print("\n🚀 Now run:  python empirical_results.py")
    print("   Charts will show your REAL final accuracy (not simulated).")


if __name__ == "__main__":
    main()
