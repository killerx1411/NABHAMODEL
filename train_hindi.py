"""
train_hindi.py — Hindi training pipeline with 4-class disease merge
Maps all 29 Hindi diseases → 4 canonical groups matching the English model.

FINAL 4 CLASSES (same as English train.py)
──────────────────────────────────────────
  1. Avascular Necrosis    — AVN variants, arthritic hip, BMES, DDH, infected hip, TB hip
  2. Osteoarthritis        — OA + RA (indistinguishable by symptoms)
  3. Hip & Bone Fracture   — all fracture variants
  4. Other Orthopaedic     — spinal tumors, discharge destination

RUN
───
  python train_hindi.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib
import os
import json
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH = "data/updated_result_with_AI_HINDI.csv"
MODEL_DIR = "model/hindi"
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# DISEASE MERGE MAP — All 29 Hindi labels → 4 canonical classes
# ─────────────────────────────────────────────────────────────────────────────

# ── GROUP 1: AVASCULAR NECROSIS ───────────────────────────────────────────────
# All AVN variants + conditions with same hip pain/mobility/stiffness profile
AVASCULAR_NECROSIS_HI = [
    "कूल्हे का एवास्कुलर नेक्रोसिस",
    "दोनों कूल्हे का अस्वास्कुलर नेक्रोसिस",
    "बाएं कूल्हे का अस्वास्कुलर नेक्रोसिस",
    "दाहिने कूल्हे का अस्वास्कुलर नेक्रोसिस",
    "रक्त संचार रुकने से हड्डी का गलना",        # standalone AVN
    "आर्थ्राइटिक कूल्हे",                        # arthritic hip
    "आर्थ्राइटिक कूल्हे (ऐसिटैबुलर फ्रैक्चर)",
    "दाहिनी कूल्हे का बोन मैरो एडिमा सिंड्रोम",  # BMES right
    "बायीं कूल्हे का बोन मैरो एडिमा सिंड्रोम",   # BMES left
    "दोनों कूल्हे्स का बोन मैरो एडिमा सिंड्रोम", # BMES bilateral
    "ट्रॉमा के बाद डायनैमिक कूल्हे स्क्रू",       # Dynamic Hip Screw
    "कूल्हे का तपेदिक",                            # TB hip
    "संक्रमित कूल्हे",                             # infected hip
    "पोस्ट-ट्रॉमैटिक कूल्हे इंजरी",               # post-traumatic hip
    "कूल्हे का विकासात्मक असामान्यता",             # DDH
]

# ── GROUP 2: OSTEOARTHRITIS ───────────────────────────────────────────────────
# OA + RA share indistinguishable symptom profiles (sim=0.996)
OSTEOARTHRITIS_HI = [
    " ओस्टियोआर्थराइटिस",           # note: leading space matches raw PKL output
    "ओस्टियोआर्थराइटिस",             # without leading space (safety net)
    "ऑस्टियोगठिया का निदान एक्स-रे से",
    "गठिया",                          # Arthritis
    "रुमेटॉयड गठिया",                 # Rheumatoid Arthritis
]

# ── GROUP 3: HIP & BONE FRACTURE ─────────────────────────────────────────────
HIP_BONE_FRACTURE_HI = [
    "ऐसिटैबुलर फ्रैक्चर",
    "ऐसिटैबुलर फ्रैक्चर / फीमर फ्रैक्चर",
    "नज़रअंदाज़ किया गया ऐसिटैबुलर फ्रैक्चर",
    "फीमर की गर्दन का फ्रैक्चर",
    "फीमर की गर्दन का फ्रैक्चर + शाफ्ट नॉन-यूनियन",
    "neglected फीमर की गर्दन का फ्रैक्चर",
    "ट्रोचैंटर फ्रैक्चर",
    "failed ट्रोचैंटर फ्रैक्चर",
]

# ── GROUP 4: OTHER ORTHOPAEDIC ────────────────────────────────────────────────
# Spinal + discharge outcomes — everything unmapped also falls here
OTHER_ORTHOPAEDIC_HI = [
    "रीढ़ की हड्डी के ट्यूमर",
    "सर्जरी के बाद डिस्चार्ज गंतव्य",
]

# Build merge dict
DISEASE_MERGE_HI = {}
for d in AVASCULAR_NECROSIS_HI:
    DISEASE_MERGE_HI[d] = "Avascular Necrosis"
for d in OSTEOARTHRITIS_HI:
    DISEASE_MERGE_HI[d] = "Osteoarthritis"
for d in HIP_BONE_FRACTURE_HI:
    DISEASE_MERGE_HI[d] = "Hip & Bone Fracture"
for d in OTHER_ORTHOPAEDIC_HI:
    DISEASE_MERGE_HI[d] = "Other Orthopaedic"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD DATASET
# ─────────────────────────────────────────────────────────────────────────────
print("📂 Loading Hindi medical dataset...")
df = pd.read_csv(DATA_PATH)

df = df[['Pseudonymized_Diagnosis', 'Pseudonymized_symptoms']].copy()
df.columns = ['Disease', 'symptoms']
df = df.dropna()
df['Disease']  = df['Disease'].str.strip()
df['symptoms'] = df['symptoms'].str.strip().str.lower()

print(f"   Raw rows: {len(df)} | Raw diseases: {df['Disease'].nunique()}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — APPLY MERGE MAP
# ─────────────────────────────────────────────────────────────────────────────
print("\n🔀 Applying disease merge map...")

# Any label not explicitly mapped → "Other Orthopaedic"
df['Disease'] = df['Disease'].map(DISEASE_MERGE_HI).fillna("Other Orthopaedic")

print(f"   After merge: {len(df)} rows | {df['Disease'].nunique()} canonical classes")
print(f"   Zero rows dropped — all {len(df)} rows used\n")
print("   Class distribution:")
for disease, count in df['Disease'].value_counts().items():
    bar = '█' * (count // 5)
    print(f"     {count:5d}  {disease:<25} {bar}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — BUILD SYMPTOM MATRIX
# ─────────────────────────────────────────────────────────────────────────────
print("\n🔧 Building symptom matrix...")

symptom_lists = df['symptoms'].str.split(',').apply(
    lambda x: [s.strip() for s in x if s.strip()]
)
all_symptoms = sorted(set(s for sublist in symptom_lists for s in sublist))
print(f"   Unique symptoms: {len(all_symptoms)}")

X = pd.DataFrame(0, index=df.index, columns=all_symptoms)
for i, symptoms in zip(df.index, symptom_lists):
    X.loc[i, symptoms] = 1

y = df['Disease']
print(f"   Matrix shape: {X.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — ENCODE LABELS
# ─────────────────────────────────────────────────────────────────────────────
le        = LabelEncoder()
y_encoded = le.fit_transform(y)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — SAVE ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────
symptom_list = X.columns.tolist()
disease_list = le.classes_.tolist()

joblib.dump(symptom_list, f"{MODEL_DIR}/symptom_list.pkl")
joblib.dump(le,           f"{MODEL_DIR}/label_encoder.pkl")

with open(f"{MODEL_DIR}/symptom_list.json", "w", encoding="utf-8") as f:
    json.dump(symptom_list, f, indent=2, ensure_ascii=False)
with open(f"{MODEL_DIR}/disease_list.json", "w", encoding="utf-8") as f:
    json.dump(disease_list, f, indent=2, ensure_ascii=False)

print(f"   ✅ Saved {len(symptom_list)} symptoms, {len(disease_list)} diseases")
print(f"   disease_list.json: {disease_list}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"\n📊 Train: {len(X_train)} | Test: {len(X_test)}")

# Convert to numpy for XGBoost compatibility
X_train_np = X_train.values
X_test_np  = X_test.values

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — TRAIN MODELS
# ─────────────────────────────────────────────────────────────────────────────
print("\n🏋️  Training models...\n")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=6,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0
    ),
    "Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    ),
}

results       = {}
trained_models = {}

for name, model in models.items():
    print(f"   Training {name}...")
    cv_scores = cross_val_score(
        model, X_train_np, y_train, cv=cv, scoring="accuracy", n_jobs=-1
    )
    model.fit(X_train_np, y_train)
    test_acc = accuracy_score(y_test, model.predict(X_test_np))

    results[name] = {
        "cv_mean":     round(float(cv_scores.mean()), 4),
        "cv_std":      round(float(cv_scores.std()), 4),
        "test_acc":    round(float(test_acc), 4),
        "fold_scores": cv_scores.tolist(),
    }
    trained_models[name] = model
    print(f"   {name}: CV={cv_scores.mean():.4f} ± {cv_scores.std():.4f} | Test={test_acc:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — SELECT & SAVE BEST MODEL
# ─────────────────────────────────────────────────────────────────────────────
best_name  = max(results, key=lambda k: results[k]["cv_mean"])
best_model = trained_models[best_name]
print(f"\n✅ Best model: {best_name}")

joblib.dump(best_model, f"{MODEL_DIR}/best_model.pkl")

y_pred = best_model.predict(X_test_np)
cm     = confusion_matrix(y_test, y_pred)
np.save(f"{MODEL_DIR}/confusion_matrix.npy", cm)

report = classification_report(
    y_test, y_pred,
    target_names=le.classes_,
    zero_division=0,
    output_dict=True
)

metadata = {
    "language":      "Hindi",
    "best_model":    best_name,
    "results":       results,
    "n_symptoms":    len(symptom_list),
    "n_diseases":    len(disease_list),
    "train_size":    len(X_train),
    "test_size":     len(X_test),
    "data_source":   "Hindi dataset — 4 merged classes, all rows used",
    "class_report":  report,
    "cv_mean":       results[best_name]["cv_mean"],
    "cv_std":        results[best_name]["cv_std"],
    "test_accuracy": results[best_name]["test_acc"],
}

with open(f"{MODEL_DIR}/metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — FINAL REPORT
# ─────────────────────────────────────────────────────────────────────────────
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))
print("✅ Done. Hindi artifacts saved in /model/hindi")
print(f"   disease_list.json: {disease_list}")