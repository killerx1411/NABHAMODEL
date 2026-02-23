"""
add_interactive_accuracy.py  — Fixed version

Runs the interactive evaluation for ALL 3 saved models (not just best_model.pkl)
so that D1 and D2 charts show separate curves per model.

Writes into metadata.json:
  final_interactive_acc          — best model's final accuracy (top-level, float)
  avg_questions_asked            — best model's avg questions (top-level, float)
  seed_symptoms                  — seed count used
  interactive_d2_thresholds      — best model D2 bar data (backward compat)

  interactive_models: {
    "Random Forest": {
      "d1_steps":       [0, 1, 2, ...],
      "d1_confidence":  [48.2, 63.1, ...],
      "d2_thresholds":  {"0.6": 1.2, "0.7": 2.1, ...},
      "final_acc":      0.9725,
      "avg_questions":  0.82
    },
    "XGBoost": { ... },
    "Gradient Boosting": { ... }
  }

IMPORTANT: For all 3 curves to appear, you need individual model .pkl files:
  model/random_forest.pkl
  model/xgboost.pkl
  model/gradient_boosting.pkl

Add these lines to train.py after fitting each model:
  joblib.dump(trained_models["Random Forest"],     "model/random_forest.pkl")
  joblib.dump(trained_models["XGBoost"],           "model/xgboost.pkl")
  joblib.dump(trained_models["Gradient Boosting"], "model/gradient_boosting.pkl")

Run:
  python add_interactive_accuracy.py
Then:
  python empirical_results.py
"""

import json
import os
import joblib
import numpy as np
from collections import defaultdict
from app.interactive_diagnosis import create_session, answer_question, SESSIONS

MODEL_DIR     = "model"
SEED_SYMPTOMS = 2
MAX_LOOP      = 8
THRESHOLDS    = [0.60, 0.70, 0.80, 0.90]

# ── Speed control ─────────────────────────────────────────────
# 100 samples ~30s per model, gives stable curves.
# Set to None to use full test set (accurate but slow ~5 min/model).
SAMPLE_SIZE = 60

# ── Load test data & artifacts ───────────────────────────────
print("📂 Loading artifacts...")
X_test       = joblib.load(os.path.join(MODEL_DIR, "X_test.pkl"))
y_test       = joblib.load(os.path.join(MODEL_DIR, "y_test.pkl"))
symptom_list = joblib.load(os.path.join(MODEL_DIR, "symptom_list.pkl"))
le           = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

meta_path = os.path.join(MODEL_DIR, "metadata.json")
with open(meta_path, "r", encoding="utf-8-sig", errors="replace") as f:
    metadata = json.load(f)

model_names = list(metadata["results"].keys())
best_name   = metadata["best_model"]

# Map model name → .pkl filename
MODEL_FILE_MAP = {
    "Random Forest":     "random_forest.pkl",
    "XGBoost":           "xgboost.pkl",
    "Gradient Boosting": "gradient_boosting.pkl",
}

def load_model(name):
    fname = MODEL_FILE_MAP.get(name)
    path  = os.path.join(MODEL_DIR, fname) if fname else None
    if path and os.path.exists(path):
        print(f"   ✅ {name} → {fname}")
        return joblib.load(path)
    if name == best_name:
        print(f"   ✅ {name} → best_model.pkl (fallback)")
        return joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    print(f"   ⚠  {name} → no .pkl found, skipping")
    return None

models_loaded = {}
for name in model_names:
    m = load_model(name)
    if m is not None:
        models_loaded[name] = m

if not models_loaded:
    raise RuntimeError("No models could be loaded.")

# Apply stratified sample if SAMPLE_SIZE is set
if SAMPLE_SIZE is not None and SAMPLE_SIZE < len(X_test):
    import pandas as pd
    from sklearn.utils import resample
    idx = list(range(len(X_test)))
    # Stratified: keep class proportions
    sampled_idx = []
    y_arr = y_test if hasattr(y_test, '__iter__') else list(y_test)
    classes = list(set(y_arr))
    per_class = max(1, SAMPLE_SIZE // len(classes))
    for c in classes:
        c_idx = [i for i in idx if y_arr[i] == c]
        sampled_idx += c_idx[:per_class]
    sampled_idx = sampled_idx[:SAMPLE_SIZE]
    import numpy as np
    if hasattr(X_test, 'iloc'):
        X_sample = X_test.iloc[sampled_idx]
    else:
        X_sample = X_test[sampled_idx]
    y_sample = [y_arr[i] for i in sampled_idx]
    print(f"   ⚡ Sampling {len(sampled_idx)}/{len(X_test)} test cases (set SAMPLE_SIZE=None for full run)")
else:
    X_sample = X_test
    y_sample = list(y_test)
    print(f"   📊 Using full test set ({len(X_test)} samples)")

print(f"\n📋 Evaluating: {list(models_loaded.keys())}")
print(f"   Test samples : {len(X_sample)}")
print(f"   Seed symptoms: {SEED_SYMPTOMS}\n")


# ── Evaluation function ──────────────────────────────────────
def evaluate_model(name, model):
    print(f"🧪 {name}...")

    correct      = 0
    total_q      = 0
    conf_by_step = defaultdict(list)
    threshold_qs = {t: [] for t in THRESHOLDS}

    for i in range(len(X_sample)):
        if i % 25 == 0:
            print(f"   [{i:>3}/{len(X_sample)}]")

        SESSIONS.clear()

        true_vector    = X_sample.iloc[i].values if hasattr(X_sample, 'iloc') else X_sample[i]
        true_label_idx = int(y_sample[i])
        true_label     = le.inverse_transform([true_label_idx])[0]
        true_symptoms  = [symptom_list[j] for j in range(len(symptom_list)) if true_vector[j] == 1]
        initial        = true_symptoms[:SEED_SYMPTOMS]

        result     = create_session(model, le, symptom_list, initial_symptoms=initial, lang="en")
        session_id = result["session_id"]

        init_conf = result["current_predictions"][0]["confidence"] / 100.0
        conf_by_step[0].append(init_conf)

        hit = {t: False for t in THRESHOLDS}
        for t in THRESHOLDS:
            if init_conf >= t:
                hit[t] = True
                threshold_qs[t].append(0)

        loop_count = 0
        while result["status"] == "questioning" and loop_count < MAX_LOOP:
            loop_count += 1
            next_q = result.get("next_question")
            if next_q is None:
                break
            symptom_asked = next_q["symptom"]
            user_answer   = symptom_asked in true_symptoms
            result        = answer_question(
                session_id, symptom_asked, user_answer, model, le, symptom_list
            )

            preds_key = "final_predictions" if "final_predictions" in result else "current_predictions"
            conf      = result[preds_key][0]["confidence"] / 100.0
            conf_by_step[loop_count].append(conf)

            for t in THRESHOLDS:
                if not hit[t] and conf >= t:
                    hit[t] = True
                    threshold_qs[t].append(loop_count)

        for t in THRESHOLDS:
            if not hit[t]:
                threshold_qs[t].append(MAX_LOOP)

        preds_key = "final_predictions" if "final_predictions" in result else "current_predictions"
        predicted = result[preds_key][0]["disease"]
        q_asked   = result.get("questions_asked", loop_count)

        if predicted == true_label:
            correct += 1
        total_q += q_asked

    n        = len(X_sample)
    final_acc = correct / n
    avg_q     = total_q / n
    max_step  = max(conf_by_step.keys())
    d1_steps  = list(range(max_step + 1))
    d1_confs  = [float(np.mean(conf_by_step[s])) * 100 for s in d1_steps]
    d2_data   = {str(t): float(np.mean(threshold_qs[t])) for t in THRESHOLDS}

    print(f"   ✅ acc={final_acc*100:.2f}%  avg_q={avg_q:.2f}\n")

    return {
        "d1_steps":      d1_steps,
        "d1_confidence": d1_confs,
        "d2_thresholds": d2_data,
        "final_acc":     round(final_acc, 4),
        "avg_questions": round(avg_q, 2),
    }


# ── Run all models ────────────────────────────────────────────
all_results = {}
for name, model in models_loaded.items():
    all_results[name] = evaluate_model(name, model)

# ── Write metadata.json ──────────────────────────────────────
best_result = all_results.get(best_name, list(all_results.values())[0])

# Top-level keys (backward compat)
metadata["final_interactive_acc"]     = best_result["final_acc"]
metadata["avg_questions_asked"]       = best_result["avg_questions"]
metadata["seed_symptoms"]             = SEED_SYMPTOMS
metadata["interactive_d1_steps"]      = best_result["d1_steps"]
metadata["interactive_d1_confidence"] = best_result["d1_confidence"]
metadata["interactive_d2_thresholds"] = best_result["d2_thresholds"]

# Per-model curves — what empirical_results.py now reads for D1/D2
metadata["interactive_models"] = all_results

with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

# ── Summary ───────────────────────────────────────────────────
print("══════════════════════════════════════════════════")
print("📊  WRITTEN TO metadata.json")
print("══════════════════════════════════════════════════")
print(f"  {'Model':<22} {'Final Acc':>10} {'Avg Qs':>8}")
print("  " + "-"*42)
for name, r in all_results.items():
    star = "★ " if name == best_name else "  "
    print(f"  {star+name:<22} {r['final_acc']*100:>9.2f}% {r['avg_questions']:>8.2f}")

missing = [n for n in model_names if n not in models_loaded]
if missing:
    print(f"\n  ⚠  Missing model files for: {missing}")
    print(f"     Add to train.py (after model.fit):")
    for n in missing:
        fname = MODEL_FILE_MAP.get(n, n.lower().replace(" ", "_") + ".pkl")
        print(f"       joblib.dump(trained_models['{n}'], 'model/{fname}')")

print("\n▶  Now run: python empirical_results.py")