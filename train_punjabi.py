"""
train_punjabi.py — Robust Punjabi Training Pipeline
Fully cleaned, rare-disease safe, XGBoost-compatible
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import joblib
import os
import json
import unicodedata
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "data/updated_result_with_AI_PUNJABI.csv"
MODEL_DIR = "model/punjabi"
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────────────────────────
print("📂 Loading Punjabi medical dataset...")
df = pd.read_csv(DATA_PATH)

df = df[['Pseudonymized_Diagnosis', 'Pseudonymized_symptoms']].copy()
df.columns = ['Disease', 'symptoms']
df = df.dropna()

print(f"   Rows: {len(df)}")
print(f"   Diseases: {df['Disease'].nunique()}")

# ─────────────────────────────────────────────────────────────
# STEP 2 — CLEAN & SPLIT SYMPTOMS (Unicode Safe)
# ─────────────────────────────────────────────────────────────
def normalize_text(text):
    return unicodedata.normalize("NFKC", text.strip())

symptom_lists = df["symptoms"].apply(
    lambda x: [normalize_text(s) for s in x.split(",") if s.strip()]
)

# ─────────────────────────────────────────────────────────────
# STEP 3 — BUILD MATRIX USING MultiLabelBinarizer
# ─────────────────────────────────────────────────────────────
print("\n🔧 Building symptom matrix...")

mlb = MultiLabelBinarizer()
X = mlb.fit_transform(symptom_lists)

X = pd.DataFrame(X, columns=mlb.classes_)
X["Disease"] = df["Disease"].values

print(f"   Unique symptoms: {len(mlb.classes_)}")
print(f"   Matrix shape: {X.shape}")

# ─────────────────────────────────────────────────────────────
# STEP 4 — REMOVE RARE DISEASES
# ─────────────────────────────────────────────────────────────
print("\n🧹 Removing rare diseases...")

MIN_SAMPLES_PER_CLASS = 5

counts = X["Disease"].value_counts()
valid = counts[counts >= MIN_SAMPLES_PER_CLASS].index

X = X[X["Disease"].isin(valid)].reset_index(drop=True)

print(f"   After filtering:")
print(f"   Rows: {len(X)}")
print(f"   Diseases: {X['Disease'].nunique()}")

y = X["Disease"]
X = X.drop("Disease", axis=1)

# ─────────────────────────────────────────────────────────────
# STEP 5 — ENCODE LABELS
# ─────────────────────────────────────────────────────────────
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"   ✅ Encoded {len(le.classes_)} diseases")

# ─────────────────────────────────────────────────────────────
# STEP 6 — SAVE ARTIFACTS
# ─────────────────────────────────────────────────────────────
joblib.dump(mlb.classes_.tolist(), f"{MODEL_DIR}/symptom_list.pkl")
joblib.dump(le, f"{MODEL_DIR}/label_encoder.pkl")

with open(f"{MODEL_DIR}/symptom_list.json", "w", encoding="utf-8") as f:
    json.dump(mlb.classes_.tolist(), f, indent=2, ensure_ascii=False)

with open(f"{MODEL_DIR}/disease_list.json", "w", encoding="utf-8") as f:
    json.dump(le.classes_.tolist(), f, indent=2, ensure_ascii=False)

print("   ✅ Saved artifacts")

# ─────────────────────────────────────────────────────────────
# STEP 7 — TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print(f"\n📊 Train: {len(X_train)} | Test: {len(X_test)}")

# Convert to NumPy for XGBoost stability
X_train = X_train.values
X_test = X_test.values

# ─────────────────────────────────────────────────────────────
# STEP 8 — TRAIN MODELS
# ─────────────────────────────────────────────────────────────
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
}

results = {}
trained_models = {}

for name, model in models.items():
    print(f"   Training {name}...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)

    model.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, model.predict(X_test))

    results[name] = {
        "cv_mean": round(cv_scores.mean(), 4),
        "cv_std": round(cv_scores.std(), 4),
        "test_acc": round(test_acc, 4)
    }

    trained_models[name] = model

    print(f"   {name}: CV={cv_scores.mean():.4f} ± {cv_scores.std():.4f} | Test={test_acc:.4f}")

# ─────────────────────────────────────────────────────────────
# STEP 9 — SAVE BEST MODEL
# ─────────────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["cv_mean"])
best_model = trained_models[best_name]

joblib.dump(best_model, f"{MODEL_DIR}/best_model.pkl")

metadata = {
    "language": "Punjabi",
    "best_model": best_name,
    "results": results,
    "n_symptoms": len(mlb.classes_),
    "n_diseases": len(le.classes_),
}

with open(f"{MODEL_DIR}/metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print("\n✅ Punjabi model trained and saved to /model/punjabi")
