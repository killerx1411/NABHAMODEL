"""
train_punjabi.py — Punjabi training pipeline with 4-class disease merge
Maps all 27 Punjabi diseases → 4 canonical groups matching the English model.

FINAL 4 CLASSES (same as English train.py)
──────────────────────────────────────────
  1. Avascular Necrosis    — AVN variants, arthritic hip, BMES, DDH, infected hip, TB hip
  2. Osteoarthritis        — OA + RA (indistinguishable by symptoms)
  3. Hip & Bone Fracture   — all fracture variants
  4. Other Orthopaedic     — spinal tumors, discharge/surgery outcomes, standalone AVN label

RUN
───
  python train_punjabi.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib
import os
import json
import unicodedata
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH = "data/updated_result_with_AI_PUNJABI.csv"
MODEL_DIR = "model/punjabi"
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# DISEASE MERGE MAP — All 27 Punjabi labels → 4 canonical classes
# ─────────────────────────────────────────────────────────────────────────────

# ── GROUP 1: AVASCULAR NECROSIS ───────────────────────────────────────────────
AVASCULAR_NECROSIS_PA = [
    "ਕਮਰ ਦੇ avascular necrosis",
    "ਦੁਵੱਲੇ ਕੁੱਲ੍ਹੇ ਦਾ ਅਵੈਸਕੁਲਰ ਨੈਕਰੋਸਿਸ",
    "ਖੱਬੀ ਕਮਰ ਦਾ ਅਵੈਸਕੁਲਰ ਨੈਕਰੋਸਿਸ",
    "ਸੱਜੇ ਕਮਰ ਦਾ ਅਵੈਸਕੁਲਰ ਨੈਕਰੋਸਿਸ",
    "ਗਠੀਏ ਦੇ ਕਮਰ",                              # arthritic hip
    "ਗਠੀਏ ਦੇ ਕਮਰ (ਐਸੀਟੇਬੂਲਰ ਫ੍ਰੈਕਚਰ)",
    "ਖੱਬੇ ਕਮਰ ਦਾ ਬੋਨ ਮੈਰੋ ਐਡੀਮਾ ਸਿੰਡਰੋਮ",       # BMES left
    "ਸੱPER_REPLACEMENT ਦਾ ਬੋਨ ਮੈਰੋ ਐਡੀਮਾ ਸਿੰਡਰੋਮ", # BMES right (anonymised)
    "ਦੁਵੱਲੇ ਕੁੱਲ੍ਹੇ ਦਾ ਬੋਨ ਮੈਰੋ ਐਡੀਮਾ ਸਿੰਡਰੋਮ",   # BMES bilateral
    "ਡਾਇਨਾਮਿਕ ਹਿਪ ਸਕ੍ਰੂ ਪੋਸਟ-ਟਰਾਮਾ",             # Dynamic Hip Screw
    "ਕਮਰ ਦੀ ਟੀ",                                 # TB hip (ਟੀ = TB)
    "ਸੰਕਰਮਿਤ ਕਮਰ",                               # infected hip
    "ਪੋਸਟ-ਟਰਾਮੈਟਿਕ ਕਮਰ ਦੀ ਸੱਟ",                  # post-traumatic hip
    "ORG_REPLACEMENT ਸੰਬੰਧੀ ਡਿਸਪਲੇਸੀਆ (ਡੀਡੀਐਚ)",  # DDH (anonymised org name)
]

# ── GROUP 2: OSTEOARTHRITIS ───────────────────────────────────────────────────
# Note: Punjabi dataset has no OA X-ray diagnosis entry — only 2 diseases here
OSTEOARTHRITIS_PA = [
    "ਗਠੀਏ",                  # Arthritis / OA
    "ਰਾਇਮੇਟਾਇਡ ਗਠੀਏ",        # Rheumatoid Arthritis
]

# ── GROUP 3: HIP & BONE FRACTURE ─────────────────────────────────────────────
HIP_BONE_FRACTURE_PA = [
    "acetabular ਫ੍ਰੈਕਚਰ",
    "acetabular fracture / femur fracture",
    "ਨਜ਼ਰਅੰਦਾਜ਼ ਐਸੀਟਾਬੂਲਰ ਫ੍ਰੈਕਚਰ",
    "ਗਰਦਨ ਦੇ ਫਰੈਕਚਰ ਦੀ ਗਰਦਨ",                    # Neck of Femur Fracture
    "ਫੇਮਰ ਫ੍ਰੈਕਚਰ ਦੀ ਗਰਦਨ + ਸ਼ਾਫਟ ਗੈਰ-ਯੂਨੀਅਨ",    # NOF + Shaft Non-Union
    "PER_REPLACEMENTੈਕਚਰ ਦੀ ਅਣਦੇਖੀ ਗਰਦਨ",         # Neglected NOF (anonymised)
    "trochanter ਫ੍ਰੈਕਚਰ",
    "ਅਸਫਲ trochanter ਫ੍ਰੈਕਚਰ",
]

# ── GROUP 4: OTHER ORTHOPAEDIC ────────────────────────────────────────────────
# Spinal + discharge + standalone AVN label (not hip-specific)
OTHER_ORTHOPAEDIC_PA = [
    "ਰੀੜ੍ਹ ਦੀ ਹੱਡੀ ਦੇ ਟਿਊਮਰ",           # Spinal Tumors
    "ਡORG_REPLACEMENT ਸਰਜਰੀ",           # Discharge Destination Post Surgery (anonymised)
    "avascular necrosis",               # standalone English-label entry in Punjabi dataset
]

# Build merge dict
DISEASE_MERGE_PA = {}
for d in AVASCULAR_NECROSIS_PA:
    DISEASE_MERGE_PA[d] = "Avascular Necrosis"
for d in OSTEOARTHRITIS_PA:
    DISEASE_MERGE_PA[d] = "Osteoarthritis"
for d in HIP_BONE_FRACTURE_PA:
    DISEASE_MERGE_PA[d] = "Hip & Bone Fracture"
for d in OTHER_ORTHOPAEDIC_PA:
    DISEASE_MERGE_PA[d] = "Other Orthopaedic"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("📂 Loading Punjabi medical dataset...")
df = pd.read_csv(DATA_PATH)

df = df[['Pseudonymized_Diagnosis', 'Pseudonymized_symptoms']].copy()
df.columns = ['Disease', 'symptoms']
df = df.dropna()
df['Disease']  = df['Disease'].str.strip()
df['symptoms'] = df['symptoms'].str.strip()

print(f"   Raw rows: {len(df)} | Raw diseases: {df['Disease'].nunique()}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — APPLY MERGE MAP
# ─────────────────────────────────────────────────────────────────────────────
print("\n🔀 Applying disease merge map...")

df['Disease'] = df['Disease'].map(DISEASE_MERGE_PA).fillna("Other Orthopaedic")

print(f"   After merge: {len(df)} rows | {df['Disease'].nunique()} canonical classes")
print(f"   Zero rows dropped — all {len(df)} rows used\n")
print("   Class distribution:")
for disease, count in df['Disease'].value_counts().items():
    bar = '█' * (count // 5)
    print(f"     {count:5d}  {disease:<25} {bar}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — CLEAN & BUILD SYMPTOM MATRIX (Unicode safe, matching original)
# ─────────────────────────────────────────────────────────────────────────────
print("\n🔧 Building symptom matrix...")

def normalize_text(text):
    return unicodedata.normalize("NFKC", text.strip())

symptom_lists = df["symptoms"].apply(
    lambda x: [normalize_text(s) for s in x.split(",") if s.strip()]
)

mlb = MultiLabelBinarizer()
X   = mlb.fit_transform(symptom_lists)
X   = pd.DataFrame(X, columns=mlb.classes_)

print(f"   Unique symptoms: {len(mlb.classes_)}")
print(f"   Matrix shape: {X.shape}")

y = df["Disease"].values

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — ENCODE LABELS
# ─────────────────────────────────────────────────────────────────────────────
le        = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"   ✅ Encoded {len(le.classes_)} diseases")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — SAVE ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────
symptom_list = mlb.classes_.tolist()
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

# Convert to numpy for XGBoost stability
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

results        = {}
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
    "language":      "Punjabi",
    "best_model":    best_name,
    "results":       results,
    "n_symptoms":    len(symptom_list),
    "n_diseases":    len(disease_list),
    "train_size":    len(X_train),
    "test_size":     len(X_test),
    "data_source":   "Punjabi dataset — 4 merged classes, all rows used",
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
print("✅ Done. Punjabi artifacts saved in /model/punjabi")
print(f"   disease_list.json: {disease_list}")